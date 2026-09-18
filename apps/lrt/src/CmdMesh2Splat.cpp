// Copyright (c) 2026 lucabRTrender contributors.
//
// `lrt mesh2splat`: a USD stage of meshes in, a stage of gaussian splats out,
// converted by Electronic Arts' mesh2splat running as an AOFX effect.
//
// WHO DOES WHAT, AND WHY IT IS SPLIT THIS WAY
//
// The effect knows nothing about USD. It is handed a picture of triangles, up
// to three maps, and a handful of numbers, and it writes gaussian records into
// the picture it is given -- which is all an AOFX effect can be handed, and
// which is exactly why it also runs unchanged in openFXplayer.
//
// So the stage is this side's work: open it, read its meshes
// (`usd::MeshStage`), triangulate them on the device (`geom::MeshBuilder`),
// pack the triangles into a picture (`lrt/usd/mesh_pack`), turn every texture
// a material names into rows of linear float4 (`lrt/usd/texture_rows`, through
// `material::TextureStore`), run the effect once per mesh, and write what comes
// back as a `UsdVolParticleField3DGaussianSplat` (`usd::writeParticleFieldStage`).
//
// The pictures live in the AOFX host's own image storage, which on a device
// with unified memory is the same memory a kernel reads. So the triangles are
// written where the effect will read them, with nothing copied: the host's
// share of the work is opening files and counting, and every number is a
// kernel's.
//
// What crosses back to the processor is the records, once, because
// `writeParticleFieldStage` takes them as a `io::RawSplats` -- and the values
// in a USD file are the processor's business by definition.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Commands.h"
#include "aofx/Effect.h"
#include "lrt/aofx/EffectRegistry.h"
#include "lrt/aofx/EffectRender.h"
#include "lrt/geom/Mesh.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/gpu_host/Context.h"
#include "lrt/gpu_host/ImageStorage.h"
#include "lrt/image/Image.h"
#include "lrt/material/TextureStore.h"
#include "lrt/usd/Export.h"
#include "lrt/usd/MeshStage.h"

namespace lrt::cli {
namespace {

/// Entries a row of every picture this command makes. A multiple of four, so
/// that the padding an image adds to its rows is none -- and small enough that
/// a mesh of a few triangles is not a picture one pixel tall and a million
/// wide, which no pool would allocate.
constexpr uint32_t kRowEntries = 4096;

/// A picture tall enough to hold `entries` of them.
[[nodiscard]] image::PixelRect pictureFor(uint64_t entries) {
    const uint64_t rows = (entries + kRowEntries - 1) / kRowEntries;
    return {0, 0, static_cast<int32_t>(kRowEntries), static_cast<int32_t>(std::max<uint64_t>(rows, 1))};
}

/// A slang-rhi view of an image's own pixels: the same memory, no copy. Only
/// where the host's storage put the image on the device, which is every
/// backend gpe adopts.
[[nodiscard]] Result<gpu::Buffer> viewOf(gpu_host::Context& context, const image::ImagePtr& image,
                                         const char* label) {
    gpu_host::ImageStorage* storage = context.sharedStorage();
    if (storage == nullptr) {
        return Error(ErrorCode::DeviceFailure, "the AOFX host has no device image storage");
    }
    const uint64_t buffer = storage->bufferFor(image->address());
    if (buffer == 0) {
        return Error(ErrorCode::DeviceFailure, "an image was allocated on the heap, not the device");
    }
    return context.renderView(buffer, image->sizeBytes(), 16, label);
}

struct Options {
    std::string              stage;
    std::string              output = "splats.usda";
    std::string              prim;
    uint32_t                 resolution = 512;
    uint64_t                 maxSplats = 2000000;
    // Their 0.65 is the width they chose for their own renderer; traced here
    // it leaves a converted surface 30% transparent (docs/decisions.md has
    // the table). 1.0 closes it to 91%, 1.2 to 96%.
    double                   sigma = 1.0;
    double                   flatness = 1.0e-7;
    double                   opacity = 1.0;
    double                   minOpacity = 0.15;
    uint32_t                 maxCells = 1u << 18;
    uint32_t                 textureSize = 0;
    bool                     noTextures = false;
    bool                     normalMapTurns = false;
    bool                     addCamera = true;
    bool                     baked = false;
    std::vector<std::string> paths;
};

/// One map the conversion reads, as a picture. Several materials name the same
/// file, and a metallic and a roughness file become one picture, so these are
/// kept by what went into them.
struct MapKey {
    std::string metallic;   ///< also the only file, for a one-file map
    std::string roughness;
    bool        packed = false;

    friend bool operator<(const MapKey& a, const MapKey& b) {
        if (a.metallic != b.metallic) return a.metallic < b.metallic;
        if (a.roughness != b.roughness) return a.roughness < b.roughness;
        return static_cast<int>(a.packed) < static_cast<int>(b.packed);
    }
};

class Converter {
public:
    Converter(gpu_host::Context& context, gpu::ShaderLibrary& library, const Options& options)
        : context_(&context), library_(&library), options_(&options) {}

    [[nodiscard]] Result<void> prepare() {
        auto pack = gpu::ComputeKernel::create(*library_, "lrt/usd/mesh_pack", "meshPack");
        if (!pack) return std::move(pack).error();
        pack_ = std::move(*pack);
        auto chunks = gpu::ComputeKernel::create(*library_, "lrt/usd/mesh_pack", "streamBoundsChunks");
        if (!chunks) return std::move(chunks).error();
        chunks_ = std::move(*chunks);
        auto reduce = gpu::ComputeKernel::create(*library_, "lrt/scene/bounds_reduce", "boundsReduce");
        if (!reduce) return std::move(reduce).error();
        reduce_ = std::move(*reduce);
        auto rows = gpu::ComputeKernel::create(*library_, "lrt/usd/texture_rows", "textureRows");
        if (!rows) return std::move(rows).error();
        rows_ = std::move(*rows);
        auto textures = material::TextureStore::create(*library_);
        if (!textures) return std::move(textures).error();
        textures_ = std::move(*textures);
        return ok();
    }

    /// Every mesh's triangles into a picture of its own, and the model's box
    /// -- which is what the projection grid is measured against -- folded from
    /// those pictures on the device.
    [[nodiscard]] Result<void> packMeshes(std::vector<usd::StageMesh>& meshes) {
        gpu::Device& device = library_->device();
        streams_.resize(meshes.size());
        triangles_.resize(meshes.size());
        uint64_t chunkTotal = 0;
        std::vector<uint32_t> chunkFirst(meshes.size(), 0);
        std::vector<uint32_t> chunkCount(meshes.size(), 0);
        for (size_t k = 0; k < meshes.size(); ++k) {
            const uint64_t entries = uint64_t{meshes[k].mesh.triangles} * 6;
            chunkFirst[k] = static_cast<uint32_t>(chunkTotal);
            chunkCount[k] = static_cast<uint32_t>((entries + kChunkEntries - 1) / kChunkEntries);
            chunkTotal += chunkCount[k];
        }
        if (chunkTotal == 0) {
            return Error(ErrorCode::InvalidArgument, "no triangles to convert");
        }
        gpu::BufferDesc desc;
        desc.bytes = chunkTotal * 2 * 16;
        desc.elementBytes = 16;
        desc.label = "mesh2splat.extents";
        auto extents = gpu::Buffer::create(device, desc);
        if (!extents) return std::move(extents).error();
        desc.bytes = 2 * 16;
        desc.label = "mesh2splat.bounds";
        auto bounds = gpu::Buffer::create(device, desc);
        if (!bounds) return std::move(bounds).error();

        gpu::CommandBatch batch(device);
        for (size_t k = 0; k < meshes.size(); ++k) {
            const geom::GpuMesh& mesh = meshes[k].mesh;
            triangles_[k] = mesh.triangles;
            const uint64_t entries = uint64_t{mesh.triangles} * 6;
            auto picture = image::Image::create(pictureFor(entries));
            if (!picture) return std::move(picture).error();
            streams_[k] = *picture;
            auto view = viewOf(*context_, streams_[k], "mesh2splat.stream");
            if (!view) return std::move(view).error();

            const geom::GpuPrimvar* normals = mesh.primvar("normals");
            const geom::GpuPrimvar* uvs = mesh.primvar("st");
            const uint32_t stride = static_cast<uint32_t>(streams_[k]->stride());
            const uint32_t chunkThreads = chunkCount[k];
            const auto bind = [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(mesh.positions.rhi());
                cursor["indices"].setBinding(mesh.indices.rhi());
                cursor["triangleCorners"].setBinding(mesh.triangleCorners.rhi());
                cursor["triangleFaces"].setBinding(mesh.triangleFaces.rhi());
                // Every buffer the kernel declares must be bound, there or
                // not: a missing primvar is said with its count, not by
                // leaving a binding empty.
                cursor["normals"].setBinding(normals != nullptr ? normals->values.rhi()
                                                                : mesh.positions.rhi());
                cursor["uvs"].setBinding(uvs != nullptr ? uvs->values.rhi() : mesh.positions.rhi());
                cursor["stream"].setBinding(view->rhi());
                cursor["extents"].setBinding(extents->rhi());
                cursor["pack"]["triangles"].setData(mesh.triangles);
                cursor["pack"]["destFirst"].setData(uint32_t{0});
                cursor["pack"]["normalMode"].setData(normals != nullptr
                                                         ? static_cast<uint32_t>(normals->interpolation)
                                                         : kNoPrimvar);
                cursor["pack"]["normalCount"].setData(normals != nullptr ? normals->count : 0U);
                cursor["pack"]["uvMode"].setData(uvs != nullptr ? static_cast<uint32_t>(uvs->interpolation)
                                                                : kNoPrimvar);
                cursor["pack"]["uvCount"].setData(uvs != nullptr ? uvs->count : 0U);
                cursor["pack"]["points"].setData(mesh.points);
                cursor["pack"]["corners"].setData(mesh.corners);
                cursor["pack"]["destWidth"].setData(kRowEntries);
                cursor["pack"]["destStride"].setData(stride);
                cursor["pack"]["chunkFirst"].setData(uint32_t{0});
                for (uint32_t r = 0; r < 3; ++r) {
                    const std::string world = "toWorld" + std::to_string(r);
                    const std::string normal = "normalTo" + std::to_string(r);
                    cursor["pack"][world.c_str()].setData(meshes[k].toWorld.data() + r * 4, 16);
                    cursor["pack"][normal.c_str()].setData(meshes[k].normalToWorld.data() + r * 4, 16);
                }
                cursor["pack"]["entries"].setData(static_cast<uint32_t>(entries));
                cursor["pack"]["chunkSize"].setData(kChunkEntries);
                cursor["pack"]["chunkCount"].setData(chunkThreads);
            };
            pack_.dispatch(batch, {mesh.triangles, 1, 1}, bind);
            // Every mesh's chunks go into one buffer, each at its own offset,
            // so that one reduce at the end gives the model's box.
            const uint32_t first = chunkFirst[k];
            chunks_.dispatch(batch, {chunkThreads, 1, 1}, [&](rhi::ShaderCursor cursor) {
                bind(cursor);
                cursor["pack"]["chunkFirst"].setData(first);
            });
        }
        const uint32_t total = static_cast<uint32_t>(chunkTotal);
        reduce_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["extents"].setBinding(extents->rhi());
            cursor["result"].setBinding(bounds->rhi());
            cursor["params"]["count"].setData(total * 2);
            cursor["params"]["chunkSize"].setData(kChunkEntries);
            cursor["params"]["chunkCount"].setData(total);
        });
        LRT_TRY(batch.submit(true));
        for (const image::ImagePtr& stream : streams_) {
            stream->deviceWrote();
        }
        auto box = bounds->readAll<float>(device);
        if (!box) return std::move(box).error();
        for (int axis = 0; axis < 3; ++axis) {
            boundsMin_[axis] = (*box)[static_cast<size_t>(axis)];
            boundsMax_[axis] = (*box)[4 + static_cast<size_t>(axis)];
        }
        return ok();
    }

    /// The maps every material names, decoded once and handed over as pictures.
    [[nodiscard]] Result<void> loadTextures(const std::vector<usd::StageMesh>& meshes) {
        if (options_->noTextures) {
            return ok();
        }
        const auto ask = [&](const usd::StageTexture& texture) {
            if (!texture.empty() && !ids_.contains(texture.file)) {
                ids_[texture.file] = textures_->request(
                    texture.file, texture.srgb ? material::ColourSpace::Srgb : material::ColourSpace::Raw);
            }
        };
        for (const usd::StageMesh& mesh : meshes) {
            ask(mesh.material.albedo);
            ask(mesh.material.normal);
            ask(mesh.material.metallicMap);
            ask(mesh.material.roughnessMap);
        }
        if (ids_.empty()) {
            return ok();
        }
        auto loaded = textures_->commit();
        if (!loaded) return std::move(loaded).error();
        sampler_ = textures_->sampler(material::Wrap::Repeat, material::Wrap::Repeat);
        std::printf("mesh2splat: %zu of %zu textures decoded\n", *loaded, ids_.size());
        for (const auto& [file, id] : ids_) {
            const material::TextureInfo& info = textures_->info(id);
            if (!info.loaded) {
                std::fprintf(stderr, "mesh2splat: '%s' did not decode: %s\n", file.c_str(),
                             info.error.c_str());
            }
        }
        return ok();
    }

    /// One map as a picture, cached. Without `pack` it is one file in all four
    /// channels -- an albedo, a normal map. With it, the two files are packed
    /// as glTF packs them, metallic in blue and roughness in green, which is
    /// what the conversion reads, and either of them may be absent.
    [[nodiscard]] Result<image::ImagePtr> mapPicture(const std::string& metallic,
                                                     const std::string& roughness, bool pack = false) {
        const MapKey key{metallic, roughness, pack};
        if (const auto held = maps_.find(key); held != maps_.end()) {
            return held->second;
        }
        const auto sizeOf = [&](const std::string& file, uint32_t& width, uint32_t& height) {
            const auto id = ids_.find(file);
            if (id == ids_.end()) {
                return;
            }
            const material::TextureInfo& info = textures_->info(id->second);
            if (info.loaded) {
                width = std::max(width, info.width);
                height = std::max(height, info.height);
            }
        };
        uint32_t width = 0;
        uint32_t height = 0;
        sizeOf(metallic, width, height);
        sizeOf(roughness, width, height);
        if (width == 0 || height == 0) {
            return image::ImagePtr{};
        }
        if (options_->textureSize > 0) {
            width = std::min(width, options_->textureSize);
            height = std::min(height, options_->textureSize);
        }
        auto picture = image::Image::create({0, 0, static_cast<int32_t>(width), static_cast<int32_t>(height)});
        if (!picture) return std::move(picture).error();
        auto view = viewOf(*context_, *picture, "mesh2splat.map");
        if (!view) return std::move(view).error();

        gpu::CommandBatch batch(library_->device());
        const uint32_t stride = static_cast<uint32_t>((*picture)->stride());
        // `fallback` is what the picture holds where no texture wrote: the
        // defaults a material has when it names no map. Metallic is nothing
        // and roughness is a half, which is what the conversion assumes.
        const std::array<float, 4> defaults{1.0F, 0.5F, 0.0F, 1.0F};
        const auto write = [&](const std::string& file, uint32_t channels, bool clear) {
            const auto id = ids_.find(file);
            const uint32_t which = id != ids_.end() ? id->second : 0;
            rows_.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["rows"].setBinding(view->rhi());
                textures_->bind(cursor["table"]);
                cursor["params"]["id"].setData(which);
                cursor["params"]["sampler"].setData(sampler_);
                cursor["params"]["width"].setData(width);
                cursor["params"]["height"].setData(height);
                cursor["params"]["stride"].setData(stride);
                cursor["params"]["channels"].setData(channels);
                cursor["params"]["fromChannel"].setData(uint32_t{0});
                cursor["params"]["clear"].setData(clear ? 1U : 0U);
                cursor["params"]["fallback"].setData(defaults.data(), 16);
            });
        };
        if (!pack) {
            write(metallic, 15, true);   // a picture of its own: albedo, a normal map
        } else {
            // Blue is metallic, green is roughness: glTF's packing, and what
            // their fragment shader reads. The defaults go down first, so a
            // material that names one map and not the other keeps the
            // default for the one it does not name -- which is what it meant.
            write({}, 0, true);
            if (!metallic.empty()) {
                write(metallic, 4, false);
            }
            if (!roughness.empty()) {
                write(roughness, 2, false);
            }
        }
        LRT_TRY(batch.submit(true));
        (*picture)->deviceWrote();
        maps_[key] = *picture;
        return *picture;
    }

    [[nodiscard]] Result<io::RawSplats> convert(aofx::Effect& effect, std::vector<usd::StageMesh>& meshes) {
        io::RawSplats raw;
        raw.source = options_->stage;
        raw.encoding.x = 0; raw.encoding.y = 1; raw.encoding.z = 2; raw.encoding.opacity = 3;
        raw.encoding.scale0 = 4; raw.encoding.scale1 = 5; raw.encoding.scale2 = 6;
        raw.encoding.rotW = 7; raw.encoding.rotX = 8; raw.encoding.rotY = 9; raw.encoding.rotZ = 10;
        raw.encoding.dc0 = 11; raw.encoding.dc1 = 12; raw.encoding.dc2 = 13;
        raw.encoding.restBase = 17;
        raw.encoding.restPerColour = 0;
        // What the gaussian reflects with, which is what lets a relit cloud
        // show the sheen its mesh had: two more floats a record.
        raw.encoding.metallic = 14;
        raw.encoding.roughness = 15;
        raw.encoding.transmission = 16;
        raw.encoding.floatsPerRecord = 17;
        raw.encoding.opacity_ = io::SplatEncoding::Opacity::Linear;
        raw.encoding.scale_ = io::SplatEncoding::Scale::Linear;
        // The conversion's colours come from a material, in linear light; a
        // cloud carries and blends its colours in the space it was trained
        // in, which for every trainer there is means sRGB.
        raw.encoding.colour = io::SplatEncoding::Colour::LinearLight;
        raw.encoding.rotation = io::SplatEncoding::Rotation::Float;

        uint64_t written = 0;
        uint64_t wanted = 0;
        uint64_t degenerate = 0;
        for (size_t k = 0; k < meshes.size(); ++k) {
            if (triangles_[k] == 0) {
                continue;
            }
            const uint64_t room = options_->maxSplats > written ? options_->maxSplats - written : 0;
            if (room == 0) {
                break;
            }
            const usd::StageMaterial& what = meshes[k].material;
            std::printf("mesh2splat: %s uses %s (albedo '%s', transmission %.3f)\n",
                        meshes[k].path.c_str(), what.path.empty() ? "no material" : what.path.c_str(),
                        what.albedo.file.c_str(), static_cast<double>(what.transmission));
            auto out = runOne(effect, meshes[k], k, room);
            if (!out) return std::move(out).error();
            written += out->written;
            wanted += out->wanted;
            degenerate += out->degenerate;
            raw.records.insert(raw.records.end(), out->records.begin(), out->records.end());
            std::printf("mesh2splat: %s -> %llu splats of %llu wanted (%u triangles)\n",
                        meshes[k].path.c_str(), static_cast<unsigned long long>(out->written),
                        static_cast<unsigned long long>(out->wanted), triangles_[k]);
        }
        if (written == 0) {
            return Error(ErrorCode::InvalidArgument, "the conversion produced no splats");
        }
        if (wanted > written) {
            std::printf("mesh2splat: %llu splats did not fit the budget of %llu; "
                        "raise --max-splats or lower --resolution\n",
                        static_cast<unsigned long long>(wanted - written),
                        static_cast<unsigned long long>(options_->maxSplats));
        }
        if (degenerate > 0) {
            std::printf("mesh2splat: %llu triangles had no frame to stand a gaussian on\n",
                        static_cast<unsigned long long>(degenerate));
        }
        raw.count = static_cast<uint32_t>(written);
        return raw;
    }

private:
    static constexpr uint32_t kChunkEntries = 1024;
    static constexpr uint32_t kNoPrimvar = 0xFFFFFFFF;

    struct OneMesh {
        uint64_t           written = 0;
        uint64_t           wanted = 0;
        uint64_t           degenerate = 0;
        std::vector<float> records;
    };

    [[nodiscard]] Result<OneMesh> runOne(aofx::Effect& effect, const usd::StageMesh& mesh, size_t at,
                                         uint64_t room) {
        const usd::StageMaterial& material = mesh.material;
        auto albedo = options_->noTextures || material.albedo.empty()
                          ? Result<image::ImagePtr>{image::ImagePtr{}}
                          : mapPicture(material.albedo.file, {});
        if (!albedo) return std::move(albedo).error();
        auto normal = options_->noTextures || material.normal.empty()
                          ? Result<image::ImagePtr>{image::ImagePtr{}}
                          : mapPicture(material.normal.file, {});
        if (!normal) return std::move(normal).error();
        const bool anyMr = !options_->noTextures &&
                           (!material.metallicMap.empty() || !material.roughnessMap.empty());
        auto mr = anyMr ? mapPicture(material.metallicMap.file, material.roughnessMap.file, true)
                        : Result<image::ImagePtr>{image::ImagePtr{}};
        if (!mr) return std::move(mr).error();

        // The records the effect writes go into a picture of their own, four
        // entries a splat: position and opacity, the three sizes, the rotation,
        // the colour.
        constexpr uint32_t kRecordEntries = 6;
        const uint64_t budget = std::min<uint64_t>(room, 1ull << 23);
        const image::PixelRect bounds = pictureFor(budget * kRecordEntries);

        streams_[at]->attach("bounds", {boundsMin_[0], boundsMin_[1], boundsMin_[2], boundsMax_[0],
                                        boundsMax_[1], boundsMax_[2]});
        aofx_host::EffectJob job;
        job.bounds = bounds;
        job.instance = "lrt/mesh2splat/" + std::to_string(at);
        job.inputs.push_back({"Mesh", streams_[at]});
        if (*albedo) job.inputs.push_back({"Albedo", *albedo});
        if (*normal) job.inputs.push_back({"Normal", *normal});
        if (*mr) job.inputs.push_back({"MetallicRoughness", *mr});

        const auto number = [&job](const char* name, double value) {
            job.params.push_back(aofx::ParamValue{name, {value}, {}});
        };
        number("triangles", static_cast<double>(triangles_[at]));
        number("resolution", static_cast<double>(options_->resolution));
        number("maxSplats", static_cast<double>(budget));
        number("flatness", options_->flatness);
        number("opacity", options_->opacity);
        number("minOpacity", options_->minOpacity);
        number("maxCells", static_cast<double>(options_->maxCells));
        number("useNormalMap", options_->normalMapTurns ? 1.0 : 0.0);
        // Six entries a splat: the four a gaussian is, and the two that say
        // what it reflects with.
        number("writePbr", 1.0);
        number("transmission", static_cast<double>(material.transmission));
        number("metallic", static_cast<double>(material.metallic));
        number("roughness", static_cast<double>(material.roughness));
        job.params.push_back(aofx::ParamValue{"sigma", {options_->sigma, options_->sigma}, {}});
        const auto colour = [&job](const char* name, const std::array<float, 3>& rgb) {
            job.params.push_back(aofx::ParamValue{name,
                                                 {static_cast<double>(rgb[0]), static_cast<double>(rgb[1]),
                                                  static_cast<double>(rgb[2])},
                                                 {}});
        };
        colour("materialColour", material.baseColour);
        colour("transmissionColour", material.transmissionColour);

        auto rendered = aofx_host::renderEffect(*context_, effect, job);
        if (!rendered) return std::move(rendered).error();
        const image::Image& out = **rendered;
        const std::vector<float>* counted = out.attached("splats");
        if (counted == nullptr || counted->size() < 4) {
            return Error(ErrorCode::DeviceFailure, "the effect did not say how many splats it wrote");
        }
        OneMesh answer;
        answer.written = static_cast<uint64_t>(std::max((*counted)[0], 0.0F));
        answer.wanted = static_cast<uint64_t>(std::max((*counted)[1], 0.0F));
        answer.degenerate = static_cast<uint64_t>(std::max((*counted)[2], 0.0F));
        answer.written = std::min(answer.written, budget);

        // The records as `io::RawSplats` wants them: fourteen floats a splat
        // rather than sixteen, the two lanes the picture pads with left out.
        // A rearrangement, not a decode -- every value here was computed on
        // the device and goes to the exporter's kernel untouched.
        const auto floats = out.floats();
        const auto stride = static_cast<size_t>(out.stride());
        const auto width = static_cast<size_t>(out.bounds().width());
        answer.records.resize(answer.written * 17);
        for (uint64_t splat = 0; splat < answer.written; ++splat) {
            float* record = answer.records.data() + splat * 17;
            const auto entry = [&](uint32_t component) {
                const uint64_t index = splat * kRecordEntries + component;
                return floats.data() + ((index / width) * stride + index % width) * 4;
            };
            const float* position = entry(0);
            const float* sizes = entry(1);
            const float* rotation = entry(2);
            const float* rgb = entry(3);
            record[0] = position[0]; record[1] = position[1]; record[2] = position[2];
            record[3] = position[3];
            record[4] = sizes[0]; record[5] = sizes[1]; record[6] = sizes[2];
            record[7] = rotation[0]; record[8] = rotation[1]; record[9] = rotation[2];
            record[10] = rotation[3];
            record[11] = rgb[0]; record[12] = rgb[1]; record[13] = rgb[2];
            // The fifth entry is (shading normal, metallic) and the sixth
            // (roughness, transmission, u, v): the normal and the texture
            // coordinate have nowhere to live in a ParticleField, and these
            // two do.
            const float* shading = entry(4);
            const float* surface = entry(5);
            record[14] = shading[3];
            record[15] = surface[0];
            record[16] = surface[1];
        }
        return answer;
    }

    gpu_host::Context*                       context_ = nullptr;
    gpu::ShaderLibrary*                      library_ = nullptr;
    const Options*                           options_ = nullptr;
    gpu::ComputeKernel                       pack_, chunks_, reduce_, rows_;
    std::unique_ptr<material::TextureStore>  textures_;
    std::map<std::string, uint32_t>          ids_;
    std::map<MapKey, image::ImagePtr>        maps_;
    std::vector<image::ImagePtr>             streams_;
    std::vector<uint32_t>                    triangles_;
    uint32_t                                 sampler_ = 0;
    std::array<float, 3>                     boundsMin_{0.0F, 0.0F, 0.0F};
    std::array<float, 3>                     boundsMax_{1.0F, 1.0F, 1.0F};
};

}   // namespace

void addMesh2Splat(CLI::App& app) {
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand(
        "mesh2splat", "convert a USD stage of meshes into gaussian splats (Electronic Arts' mesh2splat)");
    cmd->add_option("stage", o->stage, ".usd / .usda / .usdc holding meshes")->required();
    cmd->add_option("-o,--output", o->output, "the ParticleField stage to write (.usda, .usdc, .usd)");
    cmd->add_option("--prim", o->prim, "only meshes at or under this prim path");
    cmd->add_option("--resolution", o->resolution,
                    "cells across the model's longest side: the density of the conversion");
    cmd->add_option("--max-splats", o->maxSplats, "the budget, over the whole stage");
    cmd->add_option("--sigma", o->sigma,
                    "how wide a gaussian is against its cell; mesh2splat's own number is 0.65, which "
                    "leaves a traced surface 30% transparent");
    cmd->add_option("--flatness", o->flatness, "the third size, across the surface");
    cmd->add_option("--opacity", o->opacity, "the opacity every gaussian starts from");
    cmd->add_option("--glass-opacity", o->minOpacity,
                    "what a fully transmitting material keeps: a tint, since a gaussian cannot refract");
    cmd->add_option("--max-cells", o->maxCells, "most cells one triangle may walk");
    cmd->add_option("--texture-size", o->textureSize, "read maps no larger than this (0: their own size)");
    cmd->add_flag("--no-textures", o->noTextures, "ignore the maps; materials keep their constant values");
    cmd->add_flag("--normal-map-turns", o->normalMapTurns,
                  "orient each gaussian by the normal map rather than the surface");
    cmd->add_flag("!--no-camera", o->addCamera, "do not add a camera framing the cloud");
    cmd->add_flag("--baked", o->baked,
                  "show the colours as they are instead of lighting them: the cloud carries an albedo, "
                  "so it is relit by the scene's lights unless this says otherwise");
    cmd->add_option("--path", o->paths, "extra AOFX bundle directories");
    cmd->callback([o] {
        gpu_host::Context* context = gpu_host::installProcessContext();
        if (context == nullptr || context->compute() == nullptr) {
            std::fprintf(stderr, "no GPU compute device for AOFX kernels (gpe has no backend here)\n");
            throw CLI::RuntimeError(1);
        }
        gpu::ShaderLibrary library(context->deviceShared());

        aofx_host::EffectRegistry registry;
        for (const std::string& path : o->paths) {
            registry.addSearchPath(path);
        }
#ifdef LRT_AOFX_BUNDLE_DIR
        registry.addSearchPath(LRT_AOFX_BUNDLE_DIR);
#endif
        registry.scan(context);
        aofx::Effect* effect = registry.find("tv.mediapro.aofx.mesh2splat");
        if (effect == nullptr) {
            std::fprintf(stderr, "no Mesh2Splat bundle on the AOFX search path (try `lrt aofx list`)\n");
            throw CLI::RuntimeError(1);
        }

        // Everything that touches the device happens on the host's own GPU
        // thread, because that is the thread the AOFX host renders on and a
        // device with two callers is the one bug this whole interface exists
        // to prevent. `run` is re-entrant from that thread, so `renderEffect`
        // asking for it again inside this costs nothing.
        uint32_t count = 0;
        Result<void> inside = ok();
        const auto work = [&]() -> Result<void> {
            auto builder = geom::MeshBuilder::create(library);
            if (!builder) return std::move(builder).error();
            auto stage = usd::MeshStage::open(o->stage);
            if (!stage) return std::move(stage).error();
            usd::MeshStageOptions read;
            read.prim = o->prim;
            auto meshes = stage->read(*builder, read);
            if (!meshes) return std::move(meshes).error();

            Converter converter(*context, library, *o);
            LRT_TRY(converter.prepare());
            LRT_TRY(converter.packMeshes(*meshes));
            LRT_TRY(converter.loadTextures(*meshes));
            auto raw = converter.convert(*effect, *meshes);
            if (!raw) return std::move(raw).error();
            count = raw->count;

            usd::ExportOptions options;
            options.maxDegree = 0;
            options.addCamera = o->addCamera;
            // What comes out of a conversion is an albedo the scene is to
            // light, not a capture carrying its own light.
            options.relight = !o->baked;
            return usd::writeParticleFieldStage(library, *raw, o->output, options);
        };
        auto ran = context->run([&] { inside = work(); });
        if (!ran) {
            std::fprintf(stderr, "%s\n", ran.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        if (!inside) {
            std::fprintf(stderr, "%s\n", inside.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        std::printf("mesh2splat: wrote %s (%u splats)\n", o->output.c_str(), count);
    });
}

}   // namespace lrt::cli
