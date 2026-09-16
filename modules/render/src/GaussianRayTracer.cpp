// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/render/GaussianRayTracer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>

#include <slang-rhi/acceleration-structure-utils.h>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "FrameParams.h"

namespace lrt::render {
namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Result<gpu::Buffer> buffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                           rhi::BufferUsage extra = rhi::BufferUsage::None) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    desc.extraUsage = extra;
    return gpu::Buffer::create(device, desc);
}

template <typename T>
Result<gpu::Buffer> upload(gpu::Device& device, std::vector<T>& values, uint32_t element,
                           const char* label, rhi::BufferUsage extra = rhi::BufferUsage::None) {
    if (values.empty()) {
        values.resize(std::max<size_t>(element / sizeof(T), 1));
    }
    gpu::BufferDesc desc;
    desc.bytes = values.size() * sizeof(T);
    desc.elementBytes = element;
    desc.label = label;
    desc.extraUsage = extra;
    return gpu::Buffer::create(device, desc, values.data());
}

Result<rhi::ComPtr<rhi::IAccelerationStructure>> buildStructure(
    gpu::Device& device, const rhi::AccelerationStructureBuildDesc& build,
    rhi::AccelerationStructureKind kind, const char* label) {
    rhi::AccelerationStructureSizes sizes;
    if (SLANG_FAILED(device.rhi()->getAccelerationStructureSizes(build, &sizes))) {
        return Error::make(ErrorCode::DeviceFailure, "no sizes for acceleration structure '{}'", label);
    }
    auto scratch = buffer(device, sizes.scratchSize, 1, "rt.scratch");
    if (!scratch) return std::move(scratch).error();
    rhi::AccelerationStructureDesc desc;
    desc.kind = kind;
    desc.size = sizes.accelerationStructureSize;
    desc.label = label;
    rhi::ComPtr<rhi::IAccelerationStructure> structure;
    if (SLANG_FAILED(device.rhi()->createAccelerationStructure(desc, structure.writeRef()))) {
        return Error::make(ErrorCode::OutOfMemory, "cannot allocate acceleration structure '{}' ({} bytes)",
                           label, sizes.accelerationStructureSize);
    }
    gpu::CommandBatch batch(device);
    batch.encoder()->buildAccelerationStructure(build, structure, nullptr,
                                                rhi::BufferOffsetPair(scratch->rhi(), 0), 0, nullptr);
    batch.markDirty();
    // Waited for: the scratch buffer goes when this returns.
    LRT_TRY(batch.submit(true));
    return structure;
}

void setProjection(rhi::ShaderCursor p, const Projection& projection, const RenderSettings& settings) {
    p["width"].setData(settings.width);
    p["height"].setData(settings.height);
    p["focalX"].setData(static_cast<float>(projection.focalX));
    p["focalY"].setData(static_cast<float>(projection.focalY));
    p["centreX"].setData(static_cast<float>(projection.centreX));
    p["centreY"].setData(static_cast<float>(projection.centreY));
    p["nearZ"].setData(static_cast<float>(projection.nearZ));
    p["farZ"].setData(static_cast<float>(projection.farZ));
    p["orthographic"].setData(uint32_t{projection.orthographic ? 1u : 0u});
    const std::array<float, 12> rows = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                              "v12", "v13", "v20", "v21", "v22", "v23"};
    for (size_t k = 0; k < 12; ++k) {
        p[kNames[k]].setData(rows[k]);
    }
    p["linearise"].setData(uint32_t{settings.linearise ? 1u : 0u});
    p["bgR"].setData(settings.background[0]);
    p["bgG"].setData(settings.background[1]);
    p["bgB"].setData(settings.background[2]);
    p["bgA"].setData(settings.background[3]);
    p["depthMode"].setData(uint32_t{settings.depth == RenderSettings::Depth::Mean ? 0u : 1u});
    p["depthThreshold"].setData(settings.depthThreshold);
}

}   // namespace

bool GaussianRayTracer::hardwareSupported(const gpu::Device& device) noexcept {
    return device.caps().rayQuery && device.caps().accelerationStructure;
}

Result<GaussianRayTracer> GaussianRayTracer::create(gpu::ShaderLibrary& library,
                                                    RayTracerSettings settings) {
    const gpu::Caps& caps = library.device().caps();
    const bool hardware = hardwareSupported(library.device());
    // The proxies and their structures are one thing, drawing with them
    // another: a device that traces only in a pipeline (CUDA, through OptiX)
    // can build them for somebody else's rays -- a shadow query's -- and
    // cannot run this route's inline kernel. `render` says so where it is
    // asked to draw; `prepare` and `shadowScene` work either way.
    const bool structuresOnly = !hardware && caps.accelerationStructure && caps.rayTracing;
    if (settings.route == RayTracingRoute::Hardware && !hardware && !structuresOnly) {
        return Error::make(ErrorCode::Unsupported,
                           "the Hardware ray tracing route needs acceleration structures; "
                           "{} on '{}' has none", caps.apiName, caps.adapterName);
    }
    GaussianRayTracer r;
    r.device_ = &library.device();
    r.settings_ = settings;
    if (settings.route == RayTracingRoute::Auto) {
        // Metal, measured (M5 Pro, train_7k, 1080p): compute 343 ms a frame
        // and 168 ms to build, hardware 435 ms and 731 ms -- every non-opaque
        // candidate is a trip out of Metal's traversal. Vulkan: hardware,
        // not measured here. CUDA: compute (no RayQuery).
        const bool preferHardware = hardware && library.device().backend() != gpu::Backend::Metal;
        r.settings_.route = preferHardware ? RayTracingRoute::Hardware : RayTracingRoute::ComputeBvh;
    }
    r.settings_.maxSegments = std::max<uint32_t>(settings.maxSegments, 1);
    r.settings_.chunkSplats = std::clamp<uint32_t>(settings.chunkSplats, 1, uint32_t{1} << 24);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    LRT_TRY(make(r.frames_kernel_, "lrt/rt/rt_frames", "rtFrames"));
    LRT_TRY(make(r.shade_, "lrt/rt/rt_shade", "rtShade"));
    if (r.settings_.route == RayTracingRoute::Hardware) {
        LRT_TRY(make(r.proxy_, "lrt/rt/rt_proxy", "rtProxy"));
        if (!structuresOnly) {
            LRT_TRY(make(r.renderSplit_, "lrt/rt/rt_render", "rtRenderSplit"));
            LRT_TRY(make(r.render_, "lrt/rt/rt_render", "rtRender"));
        }
    } else {
        auto sort = gpu::RadixSort::create(library);
        if (!sort) return std::move(sort).error();
        r.sort_ = std::move(*sort);
        LRT_TRY(make(r.bvhLeaves_, "lrt/rt/bvh_leaves", "bvhLeaves"));
        LRT_TRY(make(r.bvhHierarchy_, "lrt/rt/bvh_hierarchy", "bvhHierarchy"));
        LRT_TRY(make(r.bvhRefit_, "lrt/rt/bvh_refit", "bvhRefit"));
        LRT_TRY(make(r.count_, "lrt/reference/count_nonzero", "countNonzero"));
        LRT_TRY(make(r.bvhRender_, "lrt/rt/rt_bvh_render", "rtBvhRender"));
    }
    return r;
}

const GaussianRayTracer::Cloud* GaussianRayTracer::find(const scene::GpuSplats& splats) const {
    const CloudKey key{&splats, splats.positions.rhi(), splats.count, splats.restPerColour};
    const auto found = std::find_if(clouds_.begin(), clouds_.end(),
                                    [&](const Cloud& c) { return c.key == key; });
    return found == clouds_.end() ? nullptr : &*found;
}

Result<void> GaussianRayTracer::rebuild(std::span<const SplatInstance> instances) {
    gpu::Device& device = *device_;
    const bool hardware = settings_.route == RayTracingRoute::Hardware;
    std::vector<Cloud> clouds;
    uint64_t splats = 0;
    uint64_t nodes = 0;
    uint32_t chunks = 0;
    for (const SplatInstance& instance : instances) {
        const scene::GpuSplats* cloud = instance.splats;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        const CloudKey key{cloud, cloud->positions.rhi(), cloud->count, cloud->restPerColour};
        if (std::any_of(clouds.begin(), clouds.end(), [&](const Cloud& c) { return c.key == key; })) {
            continue;
        }
        Cloud entry;
        entry.key = key;
        entry.base = static_cast<uint32_t>(splats);
        entry.firstChunk = chunks;
        entry.chunks = hardware ? (cloud->count + settings_.chunkSplats - 1) / settings_.chunkSplats : 0;
        entry.nodeBase = static_cast<uint32_t>(nodes);
        splats += cloud->count;
        nodes += cloud->count - 1;
        chunks += entry.chunks;
        clouds.push_back(entry);
    }
    if (splats > 0xFFFFFFFFull / 4) {
        return Error::make(ErrorCode::InvalidArgument,
                           "{} splats is more than the ray tracer indexes", splats);
    }
    auto frames = buffer(device, splats * 4, 16, "rt.frames");
    if (!frames) return std::move(frames).error();
    {
        gpu::CommandBatch batch(device);
        for (const Cloud& cloud : clouds) {
            const scene::GpuSplats& source = *cloud.key.cloud;
            frames_kernel_.dispatch(batch, {source.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(source.positions.rhi());
                cursor["shape"].setBinding(source.shape.rhi());
                cursor["frames"].setBinding(frames->rhi());
                cursor["params"]["count"].setData(source.count);
                cursor["params"]["base"].setData(cloud.base);
            });
        }
        LRT_TRY(batch.submit(true));
    }

    std::vector<rhi::ComPtr<rhi::IAccelerationStructure>> blas;
    gpu::Buffer boxes, children, leaves;
    if (hardware) {
        blas.reserve(chunks);
        for (const Cloud& cloud : clouds) {
            LRT_TRY(buildHardware(cloud, blas));
        }
    } else {
        auto madeBoxes = buffer(device, std::max<uint64_t>(nodes, 1) * 2, 16, "bvh.boxes");
        if (!madeBoxes) return std::move(madeBoxes).error();
        auto madeChildren = buffer(device, std::max<uint64_t>(nodes, 1) * 2, 4, "bvh.children");
        if (!madeChildren) return std::move(madeChildren).error();
        auto madeLeaves = buffer(device, splats, 4, "bvh.leaves");
        if (!madeLeaves) return std::move(madeLeaves).error();
        boxes = std::move(*madeBoxes);
        children = std::move(*madeChildren);
        leaves = std::move(*madeLeaves);
        for (const Cloud& cloud : clouds) {
            LRT_TRY(buildBvh(cloud, boxes, children, leaves));
        }
    }

    clouds_ = std::move(clouds);
    blas_ = std::move(blas);
    bvhBoxes_ = std::move(boxes);
    bvhChildren_ = std::move(children);
    bvhLeaves_buffer_ = std::move(leaves);
    splats_ = static_cast<uint32_t>(splats);
    frames_ = std::move(*frames);
    tlas_ = nullptr;
    return ok();
}

Result<void> GaussianRayTracer::buildHardware(const Cloud& cloud,
                                              std::vector<rhi::ComPtr<rhi::IAccelerationStructure>>& blas) {
    gpu::Device& device = *device_;
    const scene::GpuSplats& source = *cloud.key.cloud;
    const rhi::BufferUsage asInput = rhi::BufferUsage::AccelerationStructureBuildInput;
    // The proxies only live until their structures are built.
    auto vertices = buffer(device, uint64_t{source.count} * 12, 16, "rt.proxyVertices", asInput);
    if (!vertices) return std::move(vertices).error();
    auto indices = buffer(device, uint64_t{source.count} * 60, 4, "rt.proxyIndices", asInput);
    if (!indices) return std::move(indices).error();
    {
        gpu::CommandBatch batch(device);
        proxy_.dispatch(batch, {source.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(source.positions.rhi());
            cursor["shape"].setBinding(source.shape.rhi());
            cursor["vertices"].setBinding(vertices->rhi());
            cursor["indices"].setBinding(indices->rhi());
            cursor["params"]["count"].setData(source.count);
            cursor["params"]["base"].setData(settings_.chunkSplats);
        });
        LRT_TRY(batch.submit(true));
    }
    for (uint32_t c = 0; c < cloud.chunks; ++c) {
        const uint32_t first = c * settings_.chunkSplats;
        const uint32_t count = std::min(settings_.chunkSplats, source.count - first);
        // A chunk is a window of both buffers, and the proxy kernel wrote its
        // indices relative to the chunk: vertex and primitive indices are
        // local, and no backend is asked to trust a vertex count larger than
        // the chunk (Metal took max(vertices, indices) / 3 as the triangle
        // count and read past the chunk's indices).
        rhi::AccelerationStructureBuildInput input = {};
        input.type = rhi::AccelerationStructureBuildInputType::Triangles;
        input.triangles.vertexBuffers[0] = rhi::BufferOffsetPair(vertices->rhi(), uint64_t{first} * 12 * 16);
        input.triangles.vertexBufferCount = 1;
        input.triangles.vertexFormat = rhi::Format::RGB32Float;
        input.triangles.vertexCount = count * 12;
        input.triangles.vertexStride = 16;
        input.triangles.indexBuffer = rhi::BufferOffsetPair(indices->rhi(), uint64_t{first} * 60 * 4);
        input.triangles.indexFormat = rhi::IndexFormat::Uint32;
        input.triangles.indexCount = count * 60;
        // Not opaque: every proxy entered is offered to the kernel.
        input.triangles.flags = rhi::AccelerationStructureGeometryFlags::NoDuplicateAnyHitInvocation;
        rhi::AccelerationStructureBuildDesc build;
        build.inputs = &input;
        build.inputCount = 1;
        build.flags = rhi::AccelerationStructureBuildFlags::PreferFastTrace;
        auto structure = buildStructure(device, build, rhi::AccelerationStructureKind::BottomLevel, "rt.blas");
        if (!structure) return std::move(structure).error();
        blas.push_back(std::move(*structure));
    }
    return ok();
}

Result<void> GaussianRayTracer::buildBvh(const Cloud& cloud, gpu::Buffer& boxes, gpu::Buffer& children,
                                         gpu::Buffer& leaves) {
    gpu::Device& device = *device_;
    const scene::GpuSplats& source = *cloud.key.cloud;
    const uint32_t n = source.count;
    auto leafBoxes = buffer(device, uint64_t{n} * 2, 16, "bvh.leafBoxes");
    if (!leafBoxes) return std::move(leafBoxes).error();
    gpu::SortBuffers sorting;
    const auto assign = [&](gpu::Buffer& into, const char* label) -> Result<void> {
        auto made = buffer(device, n, 4, label);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    LRT_TRY(assign(sorting.keysLo, "bvh.keys"));
    LRT_TRY(assign(sorting.values, "bvh.order"));
    LRT_TRY(assign(sorting.scratchKeysLo, "bvh.keys2"));
    LRT_TRY(assign(sorting.scratchValues, "bvh.order2"));
    auto changed = buffer(device, std::max<uint32_t>(n, 2) - 1, 4, "bvh.changed");
    if (!changed) return std::move(changed).error();
    auto changedCount = buffer(device, 1, 4, "bvh.changedCount");
    if (!changedCount) return std::move(changedCount).error();

    const auto setBvh = [&](rhi::ShaderCursor p) {
        p["count"].setData(n);
        p["nodeBase"].setData(cloud.nodeBase);
        p["leafBase"].setData(cloud.base);
        p["boundsLoX"].setData(source.bounds.min[0]);
        p["boundsLoY"].setData(source.bounds.min[1]);
        p["boundsLoZ"].setData(source.bounds.min[2]);
        p["boundsHiX"].setData(source.bounds.max[0]);
        p["boundsHiY"].setData(source.bounds.max[1]);
        p["boundsHiZ"].setData(source.bounds.max[2]);
    };
    {
        gpu::CommandBatch batch(device);
        bvhLeaves_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(source.positions.rhi());
            cursor["shape"].setBinding(source.shape.rhi());
            cursor["leafBoxes"].setBinding(leafBoxes->rhi());
            cursor["mortonKeys"].setBinding(sorting.keysLo.rhi());
            cursor["mortonValues"].setBinding(sorting.values.rhi());
            setBvh(cursor["params"]);
        });
        LRT_TRY(sort_.sort(batch, sorting, n, 30));
        bvhHierarchy_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["sortedKeys"].setBinding(sorting.keysLo.rhi());
            cursor["sortedValues"].setBinding(sorting.values.rhi());
            cursor["children"].setBinding(children.rhi());
            cursor["leaves"].setBinding(leaves.rhi());
            setBvh(cursor["params"]);
        });
        LRT_TRY(batch.submit(true));
    }
    if (n < 2) {
        return ok();
    }
    // Refit until a pass changes nothing: the tree's depth in passes, read
    // back as one counter every kPassesPerCheck.
    static constexpr uint32_t kPassesPerCheck = 8;
    static constexpr uint32_t kMaxPasses = 256;
    for (uint32_t passes = 0; passes < kMaxPasses; passes += kPassesPerCheck) {
        gpu::CommandBatch batch(device);
        for (uint32_t k = 0; k < kPassesPerCheck; ++k) {
            bvhRefit_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["leafBoxes"].setBinding(leafBoxes->rhi());
                cursor["children"].setBinding(children.rhi());
                cursor["leaves"].setBinding(leaves.rhi());
                cursor["boxes"].setBinding(boxes.rhi());
                cursor["changed"].setBinding(changed->rhi());
                setBvh(cursor["params"]);
            });
        }
        count_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["values"].setBinding(changed->rhi());
            cursor["total"].setBinding(changedCount->rhi());
            cursor["params"]["count"].setData(n - 1);
        });
        LRT_TRY(batch.submit(true));
        uint32_t still = 0;
        LRT_TRY(changedCount->read(device, 0, sizeof(still), &still));
        if (still == 0) {
            return ok();
        }
    }
    return Error::make(ErrorCode::InternalError, "BVH refit of {} particles did not settle in {} passes", n,
                       kMaxPasses);
}

Result<void> GaussianRayTracer::prepareFrame(std::span<const SplatInstance> instances,
                                             const Vec3& eyeWorld, uint32_t shLimit) {
    gpu::Device& device = *device_;
    const bool hardware = settings_.route == RayTracingRoute::Hardware;
    std::vector<rhi::AccelerationStructureInstanceDescGeneric> generic;
    std::vector<float> data;          // per instance: world->cloud rows
    std::vector<uint32_t> indices;    // per instance: first particle, colour offset
    std::vector<uint32_t> bvh;        // ComputeBvh, per instance: node base, leaf base, particles
    struct Shade {
        const scene::GpuSplats* cloud;
        uint32_t                colourStart;
        Vec3                    eye;
        SplatEdit               edit;
    };
    std::vector<Shade> shades;
    uint64_t colours = 0;
    for (const SplatInstance& instance : instances) {
        if (instance.splats == nullptr || instance.splats->count == 0) {
            continue;
        }
        const Cloud* cloud = find(*instance.splats);
        if (cloud == nullptr) {
            return Error(ErrorCode::InternalError, "instance of a cloud the ray tracer did not build");
        }
        const Mat4 toCloud = aofx::xform::inverseAffine(instance.objectToWorld);
        const std::array<float, 12> rows = instance.objectToWorld.rows3x4();
        const std::array<float, 12> toCloudRows = toCloud.rows3x4();
        const auto colourStart = static_cast<uint32_t>(colours);
        shades.push_back({instance.splats, colourStart, toCloud.point(eyeWorld), instance.edit});
        colours += instance.splats->count;
        if (!hardware) {
            data.insert(data.end(), toCloudRows.begin(), toCloudRows.end());
            // Colour index = particle id + offset; unsigned wrap-around makes
            // a negative offset work.
            indices.push_back(cloud->base);
            indices.push_back(colourStart - cloud->base);
            bvh.insert(bvh.end(), {cloud->nodeBase, cloud->base, instance.splats->count});
            continue;
        }
        for (uint32_t c = 0; c < cloud->chunks; ++c) {
            rhi::AccelerationStructureInstanceDescGeneric desc{};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 4; ++column) {
                    desc.transform[row][column] = rows[static_cast<size_t>(row * 4 + column)];
                }
            }
            desc.instanceID = cloud->firstChunk + c;
            desc.instanceMask = 0xFF;
            desc.instanceContributionToHitGroupIndex = 0;
            // Facing is decided in object space, so a mirroring transform
            // does not turn a proxy inside out (Metal measured; the Vulkan and
            // DXR specifications say the same). The proxies are wound
            // counter-clockwise seen from outside with a right-handed cross
            // product, which is the APIs' default front face.
            desc.flags = rhi::AccelerationStructureInstanceFlags::None;
            desc.accelerationStructure = blas_[cloud->firstChunk + c]->getHandle();
            generic.push_back(desc);
            data.insert(data.end(), toCloudRows.begin(), toCloudRows.end());
            indices.push_back(cloud->base + c * settings_.chunkSplats);
            indices.push_back(colourStart - cloud->base);
        }
    }
    if (colours > 0xFFFFFFFFull) {
        return Error::make(ErrorCode::InvalidArgument, "{} instanced splats is more than the ray tracer indexes",
                           colours);
    }
    instanceCount_ = static_cast<uint32_t>(shades.size());
    auto dataBuffer = upload(device, data, 16, "rt.instanceData");
    if (!dataBuffer) return std::move(dataBuffer).error();
    auto indexBuffer = upload(device, indices, 4, "rt.instanceIndices");
    if (!indexBuffer) return std::move(indexBuffer).error();
    instanceData_ = std::move(*dataBuffer);
    instanceIndices_ = std::move(*indexBuffer);

    if (hardware) {
        const rhi::AccelerationStructureInstanceDescType type =
            rhi::getAccelerationStructureInstanceDescType(device.rhi());
        const size_t stride = rhi::getAccelerationStructureInstanceDescSize(type);
        std::vector<uint8_t> native(generic.size() * stride, 0);
        rhi::convertAccelerationStructureInstanceDescs(generic.size(), type, native.data(), stride,
                                                       generic.data(), sizeof(generic[0]));
        auto descs = upload(device, native, 0, "rt.instances", rhi::BufferUsage::AccelerationStructureBuildInput);
        if (!descs) return std::move(descs).error();
        instanceDescs_ = std::move(*descs);
        rhi::AccelerationStructureBuildInput input = {};
        input.type = rhi::AccelerationStructureBuildInputType::Instances;
        input.instances.instanceBuffer = rhi::BufferOffsetPair(instanceDescs_.rhi(), 0);
        input.instances.instanceStride = static_cast<uint32_t>(stride);
        input.instances.instanceCount = static_cast<uint32_t>(generic.size());
        rhi::AccelerationStructureBuildDesc build;
        build.inputs = &input;
        build.inputCount = 1;
        build.flags = rhi::AccelerationStructureBuildFlags::PreferFastBuild;
        auto structure = buildStructure(device, build, rhi::AccelerationStructureKind::TopLevel, "rt.tlas");
        if (!structure) return std::move(structure).error();
        tlas_ = std::move(*structure);
    } else {
        auto bvhBuffer = upload(device, bvh, 4, "bvh.instances");
        if (!bvhBuffer) return std::move(bvhBuffer).error();
        bvhInstances_ = std::move(*bvhBuffer);
    }

    if (colours > colourCapacity_ || !colours_.valid()) {
        auto made = buffer(device, colours, 16, "rt.colours");
        if (!made) return std::move(made).error();
        colours_ = std::move(*made);
        colourCapacity_ = std::max<uint64_t>(colours, 1);
    }
    gpu::CommandBatch batch(device);
    for (const Shade& shade : shades) {
        shade_.dispatch(batch, {shade.cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(shade.cloud->positions.rhi());
            cursor["shape"].setBinding(shade.cloud->shape.rhi());
            cursor["sh"].setBinding(shade.cloud->sh.rhi());
            cursor["colours"].setBinding(colours_.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(shade.cloud->count);
            p["base"].setData(shade.colourStart);
            p["shWords"].setData(shade.cloud->shWords);
            p["restPerColour"].setData(shade.cloud->restPerColour);
            p["shLimit"].setData(shLimit);
            p["v03"].setData(static_cast<float>(shade.eye.x));
            p["v13"].setData(static_cast<float>(shade.eye.y));
            p["v23"].setData(static_cast<float>(shade.eye.z));
            setEdit(p["edit"], shade.edit);
        });
    }
    return batch.submit(false);
}

Result<RayTracerStats> GaussianRayTracer::render(const Camera& camera,
                                                 std::span<const SplatInstance> instances,
                                                 const RenderSettings& settings,
                                                 RenderTargets& targets) {
    return render(projectionFor(camera, settings.width, settings.height), instances, settings, targets);
}

Result<RayTracerStats> GaussianRayTracer::prepare(const Projection& projection,
                                                  std::span<const SplatInstance> instances,
                                                  uint32_t maxShDegree) {
    const auto start = Clock::now();
    RayTracerStats stats;
    stats.route = settings_.route;
    std::vector<CloudKey> wanted;
    for (const SplatInstance& instance : instances) {
        const scene::GpuSplats* cloud = instance.splats;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        const CloudKey key{cloud, cloud->positions.rhi(), cloud->count, cloud->restPerColour};
        if (std::find(wanted.begin(), wanted.end(), key) == wanted.end()) {
            wanted.push_back(key);
        }
        stats.instances += 1;
    }
    bool same = wanted.size() == clouds_.size() && frames_.valid();
    for (size_t k = 0; same && k < wanted.size(); ++k) {
        same = wanted[k] == clouds_[k].key;
    }
    if (!same) {
        LRT_TRY(rebuild(instances));
        stats.rebuilt = true;
    }
    stats.splats = splats_;
    for (const Cloud& cloud : clouds_) {
        stats.chunks += cloud.chunks;
    }
    LRT_TRY(prepareFrame(instances, projection.eyeWorld, maxShDegree));
    stats.buildMs = msSince(start);
    stats.totalMs = stats.buildMs;
    return stats;
}

ShadowScene GaussianRayTracer::shadowScene() const noexcept {
    ShadowScene scene;
    scene.tlas = tlas_.get();
    scene.frames = &frames_;
    scene.colours = &colours_;
    scene.instanceData = &instanceData_;
    scene.instanceIndices = &instanceIndices_;
    scene.instances = instanceCount_;
    return scene;
}

Result<RayTracerStats> GaussianRayTracer::render(const Projection& projection,
                                                 std::span<const SplatInstance> instances,
                                                 const RenderSettings& settings,
                                                 RenderTargets& targets) {
    const auto start = Clock::now();
    RayTracerStats stats;
    stats.route = settings_.route;

    std::vector<CloudKey> wanted;
    for (const SplatInstance& instance : instances) {
        const scene::GpuSplats* cloud = instance.splats;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        const CloudKey key{cloud, cloud->positions.rhi(), cloud->count, cloud->restPerColour};
        if (std::find(wanted.begin(), wanted.end(), key) == wanted.end()) {
            wanted.push_back(key);
        }
        stats.instances += 1;
    }
    bool same = wanted.size() == clouds_.size() && frames_.valid();
    for (size_t k = 0; same && k < wanted.size(); ++k) {
        same = wanted[k] == clouds_[k].key;
    }
    if (!same) {
        LRT_TRY(rebuild(instances));
        stats.rebuilt = true;
    }
    stats.splats = splats_;
    for (const Cloud& cloud : clouds_) {
        stats.chunks += cloud.chunks;
    }
    if (targets.width != settings.width || targets.height != settings.height || !targets.colour.valid()) {
        const uint64_t pixels = uint64_t{settings.width} * settings.height;
        auto colour = buffer(*device_, pixels, 16, "rt.target.colour");
        if (!colour) return std::move(colour).error();
        auto depth = buffer(*device_, pixels, 4, "rt.target.depth");
        if (!depth) return std::move(depth).error();
        targets.colour = std::move(*colour);
        targets.depth = std::move(*depth);
        targets.width = settings.width;
        targets.height = settings.height;
    }
    LRT_TRY(prepareFrame(instances, projection.eyeWorld, settings.maxShDegree));
    stats.buildMs = msSince(start);

    const auto renderStart = Clock::now();
    gpu::CommandBatch batch(*device_);
    const auto common = [&](rhi::ShaderCursor cursor) {
        cursor["frames"].setBinding(frames_.rhi());
        cursor["colours"].setBinding(colours_.rhi());
        cursor["instanceData"].setBinding(instanceData_.rhi());
        cursor["instanceIndices"].setBinding(instanceIndices_.rhi());
        cursor["colour"].setBinding(targets.colour.rhi());
        cursor["depth"].setBinding(targets.depth.rhi());
        rhi::ShaderCursor p = cursor["params"];
        setProjection(p, projection, settings);
        p["maxSegments"].setData(settings_.maxSegments);
        p["instances"].setData(instanceCount_);
        p["splitAt"].setData(settings_.splitAt);
    };
    if (settings_.route == RayTracingRoute::Hardware) {
        if (!device_->caps().rayQuery) {
            return Error(ErrorCode::Unsupported,
                         "the Hardware route draws with an inline ray, which this device has not: its "
                         "structures are still built, for a query that traces them in a pipeline");
        }
        const gpu::ComputeKernel& kernel = settings_.splitAt > 0.0F ? renderSplit_ : render_;
        kernel.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            common(cursor);
            cursor["scene"].setBinding(tlas_.get());
        });
    } else {
        bvhRender_.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            common(cursor);
            cursor["bvhBoxes"].setBinding(bvhBoxes_.rhi());
            cursor["bvhChildren"].setBinding(bvhChildren_.rhi());
            cursor["bvhLeaves"].setBinding(bvhLeaves_buffer_.rhi());
            cursor["bvhInstances"].setBinding(bvhInstances_.rhi());
        });
    }
    LRT_TRY(batch.submit(true));
    stats.renderMs = msSince(renderStart);
    stats.totalMs = msSince(start);
    return stats;
}

}   // namespace lrt::render
