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
#include <set>
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
#include "lrt/usd/StageRenderer.h"

namespace lrt::cli {
namespace {

/// Rest coefficients a colour, by degree: (degree + 1)^2 - 1.
constexpr uint32_t kRestPerDegree[4] = {0, 3, 8, 15};

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
        return Error::make(ErrorCode::DeviceFailure,
                           "the {} picture is on the heap, not the device: {} by {}, {:.1f} MB -- the "
                           "pool would not serve it",
                           label, image->bounds().width(), image->bounds().height(),
                           static_cast<double>(image->sizeBytes()) / (1024.0 * 1024.0));
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
    double                   flatness = 0.1;
    double                   opacity = 1.0;
    double                   minOpacity = 0.6;
    /// A cut-out map reads below this where there is no surface.
    double                   opacityCut = 0.5;
    uint32_t                 maxCells = 1u << 18;
    /// A MAP NO BIGGER THAN THIS, AND A CEILING BY DEFAULT.
    ///
    /// A map travels to the effect as a float4 picture, sixteen bytes a texel
    /// where the file holds one: a 4k map is 268 MB on the device and a car
    /// with fifteen of them does not fit the pool at all. The conversion
    /// samples a map once a cell, and at resolution 512 the whole model is
    /// 512 cells across, so most of a 4k map is thrown away before it is
    /// looked at. 0 reads them at their own size.
    uint32_t                 textureSize = 1024;
    bool                     noTextures = false;
    bool                     normalMapTurns = false;
    bool                     addCamera = true;
    /// Bake the path tracer's answer into the gaussians (the default), or
    /// carry the material and be relit.
    bool                     bake = true;
    uint32_t                 bakeSamples = 64;
    uint32_t                 bakeBounces = 3;
    /// How much of the direction the light leaves in the cloud carries: 0 is
    /// a colour, 1 to 3 are harmonics. Two is where a highlight starts to
    /// look like one.
    uint32_t                 bakeDegree = 2;
    bool                     defaultLights = false;
    /// Carry the skeleton: the gaussians are built in the bind pose and each
    /// keeps the joints that move it.
    bool                     skinned = false;
    /// START:END[:STEP] in time codes; empty is the stage's own range.
    std::string              range;
    double                   time = 0.0;
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
        skins_.resize(meshes.size());
        uv2s_.resize(meshes.size());
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

            // The joints each corner is carried by, in a second picture of
            // exactly the same shape, so the two are addressed alike and the
            // effect needs no second set of dimensions.
            const usd::StageSkinning& skin = meshes[k].skinning;
            gpu::Buffer influences;
            std::optional<gpu::Buffer> skinView;
            if (skin.bound && !skin.influences.empty()) {
                gpu::BufferDesc held;
                held.bytes = skin.influences.size() * 4;
                held.elementBytes = 8;
                held.label = "mesh2splat.influences";
                auto made = gpu::Buffer::create(device, held, skin.influences.data());
                if (!made) return std::move(made).error();
                influences = std::move(*made);
                auto second = image::Image::create(pictureFor(entries));
                if (!second) return std::move(second).error();
                skins_[k] = *second;
                auto held2 = viewOf(*context_, skins_[k], "mesh2splat.skin");
                if (!held2) return std::move(held2).error();
                skinView = std::move(*held2);
            }

            const geom::GpuPrimvar* normals = mesh.primvar("normals");
            const geom::GpuPrimvar* uvs = mesh.primvar("st");
            // The second set of texture coordinates, where a material reads
            // some of its maps by one: a third picture of the same shape.
            const geom::GpuPrimvar* uvs2 = mesh.primvar("st2");
            std::optional<gpu::Buffer> uv2View;
            if (uvs2 != nullptr) {
                auto third = image::Image::create(pictureFor(entries));
                if (!third) return std::move(third).error();
                uv2s_[k] = *third;
                auto held3 = viewOf(*context_, uv2s_[k], "mesh2splat.uv2");
                if (!held3) return std::move(held3).error();
                uv2View = std::move(*held3);
            }
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
                cursor["uvs2"].setBinding(uvs2 != nullptr ? uvs2->values.rhi() : mesh.positions.rhi());
                cursor["influences"].setBinding(influences.valid() ? influences.rhi()
                                                                   : mesh.positions.rhi());
                cursor["skinStream"].setBinding(skinView ? skinView->rhi() : view->rhi());
                cursor["uv2Stream"].setBinding(uv2View ? uv2View->rhi() : view->rhi());
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
                cursor["pack"]["uv2Mode"].setData(uvs2 != nullptr ? static_cast<uint32_t>(uvs2->interpolation)
                                                                  : kNoPrimvar);
                cursor["pack"]["uv2Count"].setData(uvs2 != nullptr ? uvs2->count : 0U);
                cursor["pack"]["points"].setData(mesh.points);
                cursor["pack"]["corners"].setData(mesh.corners);
                cursor["pack"]["destWidth"].setData(kRowEntries);
                cursor["pack"]["destStride"].setData(stride);
                cursor["pack"]["chunkFirst"].setData(uint32_t{0});
                cursor["pack"]["perPoint"].setData(skinView ? skin.perPoint : 0U);
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
        for (const image::ImagePtr& skin : skins_) {
            if (skin) {
                skin->deviceWrote();
            }
        }
        for (const image::ImagePtr& uv2 : uv2s_) {
            if (uv2) {
                uv2->deviceWrote();
            }
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
            // The cut-out too: it was never asked for, and passed only while
            // it happened to be the normal map's file (the sparrow's feathers
            // read their alpha off it). With the normal map repaired into a
            // file of its own, the cut silently went.
            ask(mesh.material.opacityMap);
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
        raw.encoding.restPerColour = options_->bake ? kRestPerDegree[std::min(options_->bakeDegree, 3u)] : 0;
        raw.encoding.restColourOuter = 0;   // rgb per basis, which is how the bake writes them
        // What the gaussian reflects with, which is what lets a relit cloud
        // show the sheen its mesh had: two more floats a record.
        raw.encoding.metallic = 14;
        raw.encoding.roughness = 15;
        raw.encoding.transmission = 16;
        raw.encoding.floatsPerRecord = 17 + raw.encoding.restPerColour * 3;
        raw.encoding.opacity_ = io::SplatEncoding::Opacity::Linear;
        raw.encoding.scale_ = io::SplatEncoding::Scale::Linear;
        // Not baked, the colours come from a material in linear light and the
        // export encodes them into the space a cloud is blended in. Baked,
        // the harmonics are already fitted in that space and already in the
        // form a cloud keeps them (the kernel shifted the constant term), so
        // they pass through as they are.
        raw.encoding.colour = options_->bake ? io::SplatEncoding::Colour::ShDc
                                             : io::SplatEncoding::Colour::LinearLight;
        raw.encoding.rest = io::SplatEncoding::Rest::Float;
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
            std::printf("mesh2splat: %s uses %s (colour %.2f %.2f %.2f, albedo '%s', metallic %.2f, "
                        "roughness %.2f, transmission %.3f)\n",
                        meshes[k].path.c_str(), what.path.empty() ? "no material" : what.path.c_str(),
                        static_cast<double>(what.baseColour[0]), static_cast<double>(what.baseColour[1]),
                        static_cast<double>(what.baseColour[2]), what.albedo.file.c_str(),
                        static_cast<double>(what.metallic), static_cast<double>(what.roughness),
                        static_cast<double>(what.transmission));
            // A PICTURE FOR WHAT THIS MESH CAN WANT, NOT FOR THE WHOLE
            // BUDGET.
            //
            // The output used to be sized for everything still unconverted, a
            // mesh at a time. On the chess pawn, which has two, nobody
            // noticed; on a car of a hundred and sixty it asks the device for
            // a hundred and sixty pictures of a hundred and fifty megabytes,
            // the pool runs out around the thirteenth, and what it serves
            // instead is a heap image the effect cannot be handed.
            //
            // So the first run is sized by a guess, and the effect says how
            // many the mesh wanted whether or not they fit: a mesh that
            // overflows is run again at exactly that, and nothing else pays
            // for it.
            // A ceiling, because a picture is sixteen bytes an entry and six
            // entries a splat: two million is 192 MB, and the device pool
            // holds two gigabytes for every picture a conversion has open at
            // once. Asking for more than this does not fail, it dies --
            // measured on a car whose body wanted ten million at resolution
            // 1024 and took the process with it.
            // AND A MESH THAT WANTS MORE THAN THE CEILING IS CONVERTED IN
            // SLICES, each run starting at the triangle the last one's
            // budget cut into (the effect says which), in the mesh's own
            // order, so the output is the same array a single run of the
            // whole would have written. The sparrow's feathers wanted 6.7 M
            // and got the first 2.1 M: the cards later in the mesh -- half
            // the head and the breast -- were not in the cloud at all.
            constexpr uint64_t kRunCeiling = 2u << 20;
            uint64_t meshWritten = 0;
            uint64_t meshWanted = 0;
            bool     carried = false;
            uint32_t first = 0;
            uint32_t slices = 0;
            while (first < triangles_[k]) {
                const uint64_t left = options_->maxSplats > written ? options_->maxSplats - written : 0;
                if (left == 0) {
                    break;
                }
                const uint64_t guess = std::clamp<uint64_t>(uint64_t{triangles_[k] - first} * 128, 4096,
                                                            std::min(left, kRunCeiling));
                auto out = runOne(effect, meshes[k], k, guess, first);
                if (!out) return std::move(out).error();
                if (out->wanted > out->written && out->written < std::min(left, kRunCeiling)) {
                    const uint64_t again = std::min({out->wanted, left, kRunCeiling});
                    if (again > guess) {
                        out = runOne(effect, meshes[k], k, again, first);
                        if (!out) return std::move(out).error();
                    }
                }
                if (slices == 0) {
                    meshWanted = out->wanted;   // the first run counts everything from here on
                }
                ++slices;
                written += out->written;
                meshWritten += out->written;
                degenerate += out->degenerate;
                raw.records.insert(raw.records.end(), out->records.begin(), out->records.end());
                normals_.insert(normals_.end(), out->normals.begin(), out->normals.end());
                // A MESH NOTHING CARRIES STILL TAKES ITS PLACE IN THE RIG.
                //
                // A stage's skinned meshes are rarely all of them -- the
                // sparrow comes with a cylinder and a plane beside the bird
                // -- and the influences have to stay one to one with the
                // gaussians or the cloud and its rig disagree about who is
                // who. Those gaussians get four joints of no weight, which
                // is what the skinner reads as "leave this one where the
                // bind pose put it".
                if (options_->skinned && out->influences.empty()) {
                    influences_.insert(influences_.end(), out->written * 8, 0.0F);
                } else {
                    influences_.insert(influences_.end(), out->influences.begin(), out->influences.end());
                    carried = carried || !out->influences.empty();
                }
                if (out->wanted <= out->written || out->done <= first || out->done >= triangles_[k]) {
                    break;   // everything from here fit, or nothing more can
                }
                first = static_cast<uint32_t>(out->done);
            }
            wanted += meshWanted;
            std::printf("mesh2splat: %s -> %llu splats of %llu wanted (%u triangles%s)%s\n",
                        meshes[k].path.c_str(), static_cast<unsigned long long>(meshWritten),
                        static_cast<unsigned long long>(meshWanted), triangles_[k],
                        slices > 1 ? (", " + std::to_string(slices) + " slices").c_str() : "",
                        carried ? (", carried by " + meshes[k].skinning.skeleton).c_str() : "");
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

    /// Floats a record: the canonical fourteen, the three the material
    /// reflects with, and the harmonics where a bake writes them.
    [[nodiscard]] uint32_t recordFloats() const {
        return 17 + (options_->bake ? kRestPerDegree[std::min(options_->bakeDegree, 3u)] : 0) * 3;
    }

    struct OneMesh {
        uint64_t           written = 0;
        uint64_t           wanted = 0;
        uint64_t           degenerate = 0;
        /// The first triangle the budget cut into; the triangle count when
        /// everything fit. Where the next slice starts.
        uint64_t           done = 0;
        std::vector<float> records;
        std::vector<float> normals;   ///< xyz a splat, for the bake
        /// The joints a gaussian is carried by: (joint, weight) four times a
        /// splat. Beside the record rather than in it, as the normals are --
        /// `io::SplatEncoding` has no field for a skeleton.
        std::vector<float> influences;
    };

    [[nodiscard]] Result<OneMesh> runOne(aofx::Effect& effect, const usd::StageMesh& mesh, size_t at,
                                         uint64_t room, uint32_t firstTriangle = 0) {
        const usd::StageMaterial& material = mesh.material;
        // A MAP THAT WILL NOT FIT IS A MAP THIS MATERIAL DOES NOT HAVE.
        //
        // The device pool is finite and a stage decides how many maps it
        // wants, so the two can disagree -- and when they do, the material's
        // own constants are a poorer answer than the map and a far better one
        // than no conversion at all. Said once a file, not once a mesh.
        const auto mapOrNone = [&](const std::string& first, const std::string& second, bool pack) {
            auto picture = mapPicture(first, second, pack);
            if (picture) {
                return *picture;
            }
            if (refused_.insert(first + "|" + second).second) {
                std::fprintf(stderr, "mesh2splat: '%s' is left out: %s\n",
                             (first.empty() ? second : first).c_str(), picture.error().toString().c_str());
            }
            return image::ImagePtr{};
        };
        const image::ImagePtr albedoMap =
            options_->noTextures || material.albedo.empty() ? image::ImagePtr{}
                                                            : mapOrNone(material.albedo.file, {}, false);
        const image::ImagePtr normalMap =
            options_->noTextures || material.normal.empty() ? image::ImagePtr{}
                                                            : mapOrNone(material.normal.file, {}, false);
        const bool anyMr = !options_->noTextures &&
                           (!material.metallicMap.empty() || !material.roughnessMap.empty());
        const image::ImagePtr mrMap =
            anyMr ? mapOrNone(material.metallicMap.file, material.roughnessMap.file, true)
                  : image::ImagePtr{};
        // THE CUT-OUT. A map on `opacity` is not a transmission: below the
        // cut the surface is not there at all. Loaded raw -- a mask is not
        // colour and must not be taken through sRGB.
        const image::ImagePtr cutMap =
            options_->noTextures || material.opacityMap.empty()
                ? image::ImagePtr{}
                : mapOrNone(material.opacityMap.file, {}, false);
        const image::ImagePtr* albedo = &albedoMap;
        const image::ImagePtr* normal = &normalMap;
        const image::ImagePtr* mr = &mrMap;

        // The records the effect writes go into a picture of their own, four
        // entries a splat: position and opacity, the three sizes, the rotation,
        // the colour.
        // Four entries a record, six with the PBR channels, eight when the
        // gaussian carries the joints that move it.
        const bool carried = mesh.skinning.bound && skins_[at];
        const uint32_t kRecordEntries = carried ? 8U : 6U;
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
        if (cutMap) job.inputs.push_back({"Opacity", cutMap});
        if (carried) job.inputs.push_back({"Influences", skins_[at]});
        if (uv2s_[at]) job.inputs.push_back({"Texcoord2", uv2s_[at]});

        const auto number = [&job](const char* name, double value) {
            job.params.push_back(aofx::ParamValue{name, {value}, {}});
        };
        number("triangles", static_cast<double>(triangles_[at]));
        number("resolution", static_cast<double>(options_->resolution));
        number("maxSplats", static_cast<double>(budget));
        number("firstTriangle", static_cast<double>(firstTriangle));
        number("flatness", options_->flatness);
        number("opacity", options_->opacity);
        number("minOpacity", options_->minOpacity);
        number("maxCells", static_cast<double>(options_->maxCells));
        number("useNormalMap", options_->normalMapTurns ? 1.0 : 0.0);
        if (cutMap) {
            const char channel = material.opacityMap.channel;
            const double which = channel == 'r'   ? 1.0
                                 : channel == 'g' ? 2.0
                                 : channel == 'b' ? 3.0
                                                  : 4.0;
            number("opacityChannel", which);
            number("opacityCut", options_->opacityCut);
        }
        // Six entries a splat: the four a gaussian is, and the two that say
        // what it reflects with.
        number("writePbr", 1.0);
        number("writeInfluences", carried ? 1.0 : 0.0);
        if (uv2s_[at]) {
            // Which maps the material reads by the second set of coordinates.
            const auto bySecond = [&mesh](const usd::StageTexture& texture) {
                return !texture.empty() && texture.uvSet == mesh.uv2 ? 1.0 : 0.0;
            };
            number("albedoUv2", bySecond(material.albedo));
            number("normalUv2", bySecond(material.normal));
            number("mrUv2", bySecond(material.metallicMap.empty() ? material.roughnessMap : material.metallicMap));
            number("opacityUv2", bySecond(material.opacityMap));
        }
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
        answer.done = counted->size() >= 6 ? static_cast<uint64_t>(std::max((*counted)[5], 0.0F))
                                           : uint64_t{triangles_[at]};
        answer.written = std::min(answer.written, budget);

        // The records as `io::RawSplats` wants them: fourteen floats a splat
        // rather than sixteen, the two lanes the picture pads with left out.
        // A rearrangement, not a decode -- every value here was computed on
        // the device and goes to the exporter's kernel untouched.
        const auto floats = out.floats();
        const auto stride = static_cast<size_t>(out.stride());
        const auto width = static_cast<size_t>(out.bounds().width());
        const uint32_t perRecord = recordFloats();
        answer.records.resize(answer.written * perRecord);
        for (uint64_t splat = 0; splat < answer.written; ++splat) {
            float* record = answer.records.data() + splat * perRecord;
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
            // Where the bake stands and which way it looks, kept beside the
            // record rather than in it: the file has no field for a normal.
            answer.normals.push_back(shading[0]);
            answer.normals.push_back(shading[1]);
            answer.normals.push_back(shading[2]);
            if (carried) {
                const float* low = entry(6);
                const float* high = entry(7);
                for (uint32_t k = 0; k < 4; ++k) {
                    answer.influences.push_back(low[k]);
                }
                for (uint32_t k = 0; k < 4; ++k) {
                    answer.influences.push_back(high[k]);
                }
            }
        }
        return answer;
    }

    /// Every splat's shading normal, xyz, in the order the records are in.
    std::vector<float>                       normals_;
    std::vector<float>                       influences_;
    std::set<std::string>                    refused_;   ///< maps the device would not hold

public:
    [[nodiscard]] const std::vector<float>& normals() const noexcept { return normals_; }
    /// (joint, weight) four times a gaussian, empty when nothing carries it.
    [[nodiscard]] const std::vector<float>& influences() const noexcept { return influences_; }

private:
    gpu_host::Context*                       context_ = nullptr;
    gpu::ShaderLibrary*                      library_ = nullptr;
    const Options*                           options_ = nullptr;
    gpu::ComputeKernel                       pack_, chunks_, reduce_, rows_;
    std::unique_ptr<material::TextureStore>  textures_;
    std::map<std::string, uint32_t>          ids_;
    std::map<MapKey, image::ImagePtr>        maps_;
    std::vector<image::ImagePtr>             streams_;
    std::vector<image::ImagePtr>             skins_;
    std::vector<image::ImagePtr>             uv2s_;
    std::vector<uint32_t>                    triangles_;
    uint32_t                                 sampler_ = 0;
    std::array<float, 3>                     boundsMin_{0.0F, 0.0F, 0.0F};
    std::array<float, 3>                     boundsMax_{1.0F, 1.0F, 1.0F};
};

}   // namespace

/// The path tracer's answer at every gaussian, written into its colour.
///
/// One ray a gaussian: from a little way along its normal, back down onto the
/// surface it came from. What the tracer finds there is the same surface the
/// mesh had -- the same material, the same texture, the same normal map --
/// and what it answers is the radiance leaving it along that normal, with
/// this stage's lights, its shadows and its bounces in it. That is the
/// conversion this engine can make and a relighting approximation cannot.
///
/// A direction had to be chosen, since one colour cannot be view-dependent,
/// and the surface's own normal is the one that needs no camera. What it
/// costs is the highlight that would only be seen from elsewhere.
[[nodiscard]] Result<void> bakeInto(io::RawSplats& raw, const std::vector<float>& normals,
                                    const std::string& stage, double time, uint32_t samples,
                                    uint32_t bounces, bool defaultLights, uint32_t degree) {
    if (raw.count == 0 || normals.size() < size_t{raw.count} * 3) {
        return Error(ErrorCode::InternalError, "bake: a normal a gaussian is what it stands on");
    }
    // How far off the surface the ray starts: far enough that it does not
    // begin inside the triangle it is about to hit, and near enough that
    // nothing else fits in between.
    //
    // It is a fraction of the MODEL, and of nothing else. Taken as a fraction
    // of the scene's unit instead -- a thousandth, floored at one -- the
    // chess pawn, which is 66 mm tall in a stage whose unit is a metre, began
    // its rays a millimetre off the surface: thicker than the gold ring under
    // the glass ball and high enough to start inside the ball, so the ray
    // came down onto the wrong surface and the ring baked grey, the marble
    // body's colour, where the mesh reads gold.
    float low[3] = {raw.records[0], raw.records[1], raw.records[2]};
    float high[3] = {low[0], low[1], low[2]};
    for (uint32_t k = 0; k < raw.count; ++k) {
        const float* record = raw.records.data() + size_t{k} * raw.encoding.floatsPerRecord;
        for (int axis = 0; axis < 3; ++axis) {
            low[axis] = std::min(low[axis], record[axis]);
            high[axis] = std::max(high[axis], record[axis]);
        }
    }
    const float span = std::sqrt((high[0] - low[0]) * (high[0] - low[0]) +
                                 (high[1] - low[1]) * (high[1] - low[1]) +
                                 (high[2] - low[2]) * (high[2] - low[2]));
    const float step = (span > 0.0F ? span : 1.0F) * 1.0e-4F;
    std::vector<float> rays(size_t{raw.count} * 8);
    for (uint32_t k = 0; k < raw.count; ++k) {
        const float* record = raw.records.data() + size_t{k} * raw.encoding.floatsPerRecord;
        const float* n = normals.data() + size_t{k} * 3;
        float* ray = rays.data() + size_t{k} * 8;
        for (int axis = 0; axis < 3; ++axis) {
            ray[axis] = record[axis];        // where the gaussian stands
            ray[4 + axis] = n[axis];         // and which way its surface faces
        }
        ray[3] = step;   // how far off the surface a ray starts
        ray[7] = 0.0F;
    }
    auto renderer = usd::StageRenderer::open(stage);
    if (!renderer) return std::move(renderer).error();
    if (defaultLights) {
        LRT_TRY((*renderer)->setDefaultLights(true));
    }
    auto baked = (*renderer)->bakePoints(rays, raw.count, time, samples, bounces, degree);
    if (!baked) return std::move(baked).error();
    const uint32_t coefficients = (degree + 1) * (degree + 1);
    // Straight into the record. The constant term is the DC the cloud keeps
    // its colour in -- the kernel already shifted it to where 3DGS keeps its
    // own -- and the rest are the harmonics, rgb per basis, which is the
    // layout the encoding below declares. Nothing is computed here.
    uint32_t lit = 0;
    for (uint32_t k = 0; k < raw.count; ++k) {
        float* record = raw.records.data() + size_t{k} * raw.encoding.floatsPerRecord;
        const float* point = baked->data() + size_t{k} * coefficients * 4;
        // A GAUSSIAN THE BAKE FOUND NOTHING UNDER IS NOT A BLACK GAUSSIAN.
        //
        // Its coefficients come back as zeros, and zero is not "no colour":
        // the constant term is kept shifted to where 3DGS trains it, so a
        // zero there decodes as `0.5 - 0.5`, which is black. A cloud out of
        // mesh2splat is nearly all discs (the third axis is 1e-7), so one of
        // those seen edge on at a silhouette is a black splinter -- which is
        // what the pawn's gold ring had a fringe of, forty-four of them in
        // 729073. It stands for nothing, so it draws nothing.
        if (!(point[3] > 0.0F)) {
            record[raw.encoding.opacity] = 0.0F;
            continue;
        }
        record[raw.encoding.dc0] = point[0];
        record[raw.encoding.dc1] = point[1];
        record[raw.encoding.dc2] = point[2];
        for (uint32_t c = 1; c < coefficients; ++c) {
            const float* value = point + size_t{c} * 4;
            float* rest = record + raw.encoding.restBase + (c - 1) * 3;
            rest[0] = value[0];
            rest[1] = value[1];
            rest[2] = value[2];
        }
        lit += point[3] > 0.0F ? 1u : 0u;
    }
    std::printf("mesh2splat: baked %u of %u gaussians (%u paths each, %u bounces, degree %u)\n", lit,
                raw.count, samples, bounces, degree);
    if (lit * 2 < raw.count) {
        std::fprintf(stderr,
                     "mesh2splat: more than half the gaussians found no surface under them; the bake "
                     "is unlikely to be what you want\n");
    }
    return ok();
}

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
    cmd->add_option("--flatness", o->flatness,
                    "the third size, as a fraction of the smaller of the other two");
    cmd->add_option("--opacity", o->opacity, "the opacity every gaussian starts from");
    cmd->add_option("--glass-opacity", o->minOpacity,
                    "what a fully transmitting material still stops. Low is a window -- you see what "
                    "stands behind it -- and translucency is not that: the light comes through "
                    "scattered, so the body stays mostly there");
    cmd->add_option("--opacity-cut", o->opacityCut,
                    "a material whose opacity is a map is a cut-out: below this the surface is "
                    "not there and no gaussian is written, so the budget goes where the surface "
                    "is. It is what makes a feather a feather and not the card it is drawn on");
    cmd->add_option("--max-cells", o->maxCells, "most cells one triangle may walk");
    cmd->add_option("--texture-size", o->textureSize,
                    "read maps no larger than this (0: their own size). A map is a float4 picture "
                    "on the device, so a 4k one is 268 MB and a stage with a few does not fit");
    cmd->add_flag("--no-textures", o->noTextures, "ignore the maps; materials keep their constant values");
    cmd->add_flag("--normal-map-turns", o->normalMapTurns,
                  "orient each gaussian by the normal map rather than the surface");
    cmd->add_flag("!--no-camera", o->addCamera, "do not add a camera framing the cloud");
    cmd->add_flag("!--no-bake", o->bake,
                  "do not bake: carry the material instead and let the scene's lights relight the "
                  "cloud every frame. Cheaper to convert, and the cloud can then be put under other "
                  "light; what it loses is the bounce, the shadows and the exactness");
    cmd->add_option("--bake-samples", o->bakeSamples, "paths a gaussian the bake traces");
    cmd->add_option("--bake-bounces", o->bakeBounces, "bounces after the first hit, in the bake");
    cmd->add_option("--bake-degree", o->bakeDegree,
                    "harmonics the bake fits, 0 to 3: 0 is one colour a gaussian and cannot hold a "
                    "reflection, and each degree costs a pass over the paths");
    cmd->add_flag("--skinned", o->skinned,
                  "carry the skeleton: the gaussians are built in the bind pose and each keeps the "
                  "four joints that move it, so the cloud deforms with the rig instead of being one "
                  "pose. Forces --no-bake: a baked radiance does not turn with a limb");
    cmd->add_option("--range", o->range,
                    "START:END[:STEP] in time codes: the instants a skinned cloud keeps its "
                    "skeleton's transforms at. The stage's own range by default, a code a step");
    cmd->add_flag("--default-lights", o->defaultLights,
                  "bake under a dome and a sun in the session layer, for a stage that brings no "
                  "lights of its own (what lrt view offers)");
    cmd->add_option("--time", o->time,
                    "the USD time code the stage is read at: the pose that becomes gaussians, "
                    "and the instant the bake traces. A skinned stage is posed for it");
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
        // THE SAME INSTANT FOR BOTH. The gaussians come from the mesh at this
        // time and the bake traces the scene at this time, so the rays stand
        // on the surface they were built from. They did not: the conversion
        // read the stage at its default time whatever `--time` said, and a
        // bake at any other instant put its rays where the mesh used to be.
        read.time = o->time;
        read.skinned = o->skinned;
        if (o->skinned && o->bake) {
            // A BAKED RADIANCE DOES NOT TURN WITH A LIMB. What the harmonics
            // hold is the environment and the bounce -- the ground under a
            // paw is in them -- and carrying that up with the leg is the
            // mistake of rotating a lightmap. A cloud a skeleton moves is
            // relit every frame instead, which is right by construction.
            std::printf("mesh2splat: --skinned carries the material, not a bake\n");
            o->bake = false;
        }
            auto meshes = stage->read(*builder, read);
            if (!meshes) return std::move(meshes).error();

            Converter converter(*context, library, *o);
            LRT_TRY(converter.prepare());
            LRT_TRY(converter.packMeshes(*meshes));
            LRT_TRY(converter.loadTextures(*meshes));
            auto raw = converter.convert(*effect, *meshes);
            if (!raw) return std::move(raw).error();
            count = raw->count;

            // THE LIGHT THE MESH HAD, baked into the gaussians.
            //
            // A relit cloud carries the material and is lit again every
            // frame, which is an approximation: one sample a light, no
            // bounce, and a normal a splat never had. A bake asks the path
            // tracer instead -- the same stage, the same lights, the same
            // integrator -- and stores what it answers. What that costs is
            // the light: a baked cloud carries this scene's, and cannot be
            // put under another.
            if (o->bake) {
                LRT_TRY(bakeInto(*raw, converter.normals(), o->stage, o->time, o->bakeSamples,
                                 o->bakeBounces, o->defaultLights, std::min(o->bakeDegree, 3u)));
            }

            // THE RIG, WHEN THE CLOUD KEEPS ONE. Four joints a gaussian came
            // back with the records; what is gathered here is the joints'
            // own transforms at each instant of the range, which is the only
            // thing about an animated cloud that changes from frame to frame.
            usd::SplatSkinning rig;
            if (o->skinned && !converter.influences().empty()) {
                for (const usd::StageMesh& one : *meshes) {
                    if (one.skinning.bound) {
                        rig.skeleton = one.skinning.skeleton;
                        rig.geomBindTransform = one.skinning.geomBindTransform;
                        rig.joints = static_cast<uint32_t>(one.skinning.joints.size());
                        break;
                    }
                }
                rig.influences = converter.influences();
                const auto [begin, end] = (*stage).timeRange();
                double from = begin;
                double to = end;
                double step = 1.0;
                if (!o->range.empty()) {
                    if (std::sscanf(o->range.c_str(), "%lf:%lf:%lf", &from, &to, &step) < 2) {
                        return Error::make(ErrorCode::InvalidArgument,
                                           "'{}': --range wants START:END[:STEP]", o->range);
                    }
                }
                if (!(step > 0.0)) step = 1.0;
                if (to < from) to = from;
                for (double at = from; at <= to + 1e-9; at += step) {
                    rig.times.push_back(at);
                }
                auto moved = (*stage).skeletonTransforms(rig.skeleton, rig.times);
                if (!moved) return std::move(moved).error();
                rig.xforms = std::move(*moved);
                rig.timeCodesPerSecond = (*stage).timeCodesPerSecond();
                std::printf("mesh2splat: carried by %s, %u joints over %zu instants at %g fps\n",
                            rig.skeleton.c_str(), rig.joints, rig.times.size(),
                            rig.timeCodesPerSecond);
            }

            usd::ExportOptions options;
            options.skinning = rig.valid() ? &rig : nullptr;
            options.maxDegree = o->bake ? std::min(o->bakeDegree, 3u) : 0;
            options.addCamera = o->addCamera;
            options.upAxis = (*stage).upAxis();
            // Both baked and not, the cloud is relit -- what differs is what
            // its colours are. Baked, they are the light on the material's
            // body and the frame adds the polish; not baked, they are an
            // albedo and the frame lights them whole.
            // A cloud that carries harmonics carries the light whole, and a
            // frame adds nothing to it. One baked to a single colour has no
            // room for a reflection, so it keeps the material's body and the
            // frame puts the polish back. Not baked at all, the colours are an
            // albedo and the frame lights them.
            // What the cloud holds and what the frame adds. Baked, the cloud
            // carries the light on the material's body -- its harmonics say
            // how that light changes with the direction -- and the frame puts
            // the polish back, which is the one thing neither a colour nor
            // sixteen coefficients can hold: a reflection off a surface of
            // roughness 0.1 is far sharper than that. Not baked, the colours
            // are an albedo and the frame lights them whole.
            options.relight = true;
            options.litBody = o->bake;
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
