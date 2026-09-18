// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/usd/Export.h"

#include "lrt/scene/DecodeParams.h"

#include <algorithm>
#include <cmath>

#include <pxr/base/gf/vec3d.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>

#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace lrt::usd {
namespace {

Result<gpu::Buffer> buffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                           const void* data = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, data);
}

}   // namespace

Result<void> writeParticleFieldStage(gpu::ShaderLibrary& library, const io::RawSplats& raw,
                                     const std::filesystem::path& path,
                                     const ExportOptions& options) {
    const io::SplatEncoding& e = raw.encoding;
    if (raw.count == 0) {
        return Error(ErrorCode::InvalidArgument, "nothing to export");
    }
    static constexpr uint32_t kPerDegree[] = {0, 3, 8, 15};
    const uint32_t keep = std::min(e.restPerColour, kPerDegree[std::min(options.maxDegree, 3u)]);
    gpu::Device& device = library.device();
    auto kernel = gpu::ComputeKernel::create(library, "lrt/scene/splat_export", "splatExport");
    if (!kernel) {
        return std::move(kernel).error();
    }

    VtVec3fArray positions;
    VtQuatfArray orientations;
    VtVec3fArray scales;
    VtFloatArray opacities;
    VtVec3fArray coefficients;
    VtFloatArray metallics;
    VtFloatArray roughnesses;
    VtFloatArray transmissions;
    const bool pbr = e.metallic != io::SplatEncoding::kNoField ||
                     e.roughness != io::SplatEncoding::kNoField ||
                     e.transmission != io::SplatEncoding::kNoField;
    positions.reserve(raw.count);
    const uint32_t perRecord = 1 + keep;
    GfVec3d lo(1e30), hi(-1e30);

    // Slices, as the loader does, so a very large cloud still fits a buffer.
    const uint64_t recordBytes = uint64_t{e.floatsPerRecord} * 4;
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, (uint64_t{256} << 20) / recordBytes));
    for (uint32_t first = 0; first < raw.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, raw.count - first);
        auto rawBuffer = buffer(device, uint64_t{n} * e.floatsPerRecord, 4, "export.raw",
                                raw.records.data() + size_t{first} * e.floatsPerRecord);
        if (!rawBuffer) return std::move(rawBuffer).error();
        auto posOpacity = buffer(device, n, 16, "export.posOpacity");
        auto rotation = buffer(device, n, 16, "export.rotation");
        auto scaleValid = buffer(device, n, 16, "export.scaleValid");
        auto coeff = buffer(device, uint64_t{n} * perRecord, 16, "export.coefficients");
        if (!posOpacity || !rotation || !scaleValid || !coeff) {
            return Error(ErrorCode::OutOfMemory, "cannot allocate export buffers");
        }
        gpu::CommandBatch batch(device);
        kernel->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["raw"].setBinding(rawBuffer->rhi());
            cursor["posOpacity"].setBinding(posOpacity->rhi());
            cursor["rotation"].setBinding(rotation->rhi());
            cursor["scaleValid"].setBinding(scaleValid->rhi());
            cursor["coefficients"].setBinding(coeff->rhi());
            scene::setDecodeParams(cursor, e, n, 0, keep, 1);
        });
        LRT_TRY(batch.submit(true));
        auto po = posOpacity->readAll<float>(device);
        auto ro = rotation->readAll<float>(device);
        auto sv = scaleValid->readAll<float>(device);
        auto co = coeff->readAll<float>(device);
        if (!po || !ro || !sv || !co) {
            return Error(ErrorCode::DeviceFailure, "cannot read export values back");
        }
        for (uint32_t i = 0; i < n; ++i) {
            if ((*sv)[size_t{i} * 4 + 3] < 0.5F) {
                continue;
            }
            const float* pp = po->data() + size_t{i} * 4;
            const float* rr = ro->data() + size_t{i} * 4;
            const float* ss = sv->data() + size_t{i} * 4;
            positions.push_back(GfVec3f(pp[0], pp[1], pp[2]));
            opacities.push_back(pp[3]);
            orientations.push_back(GfQuatf(rr[3], rr[0], rr[1], rr[2]));
            scales.push_back(GfVec3f(ss[0], ss[1], ss[2]));
            for (uint32_t k = 0; k < perRecord; ++k) {
                const float* cc = co->data() + (size_t{i} * perRecord + k) * 4;
                coefficients.push_back(GfVec3f(cc[0], cc[1], cc[2]));
            }
            if (pbr) {
                // Straight out of the record: both are already 0 to 1, and a
                // value is not a thing this file decodes.
                const float* record = raw.records.data() + size_t{first + i} * e.floatsPerRecord;
                metallics.push_back(e.metallic != io::SplatEncoding::kNoField ? record[e.metallic] : 0.0F);
                roughnesses.push_back(e.roughness != io::SplatEncoding::kNoField ? record[e.roughness] : 1.0F);
                transmissions.push_back(
                    e.transmission != io::SplatEncoding::kNoField ? record[e.transmission] : 0.0F);
            }
            for (int axis = 0; axis < 3; ++axis) {
                lo[axis] = std::min(lo[axis], static_cast<double>(pp[axis]));
                hi[axis] = std::max(hi[axis], static_cast<double>(pp[axis]));
            }
        }
    }
    if (positions.empty()) {
        return Error(ErrorCode::InvalidArgument, "no splat survived export");
    }

    UsdStageRefPtr stage = UsdStage::CreateNew(path.string());
    if (!stage) {
        return Error::make(ErrorCode::IoFailure, "cannot create '{}'", path.string());
    }
    UsdGeomSetStageUpAxis(stage, UsdGeomTokens->y);
    UsdGeomXform world = UsdGeomXform::Define(stage, SdfPath("/World"));
    stage->SetDefaultPrim(world.GetPrim());
    auto splats = UsdVolParticleField3DGaussianSplat::Define(stage, SdfPath("/World/Splats"));
    splats.CreatePositionsAttr(VtValue(positions));
    splats.CreateOrientationsAttr(VtValue(orientations));
    splats.CreateScalesAttr(VtValue(scales));
    splats.CreateOpacitiesAttr(VtValue(opacities));
    splats.CreateRadianceSphericalHarmonicsDegreeAttr(VtValue(keep == 15 ? 3 : keep == 8 ? 2 : keep == 3 ? 1 : 0));
    splats.CreateRadianceSphericalHarmonicsCoefficientsAttr(VtValue(coefficients));
    VtVec3fArray extent{GfVec3f(lo), GfVec3f(hi)};
    splats.CreateExtentAttr(VtValue(extent));
    if (options.rotateXDegrees != 0.0) {
        UsdGeomXformCommonAPI(splats.GetPrim()).SetRotate(
            GfVec3f(static_cast<float>(options.rotateXDegrees), 0.0F, 0.0F));
    }

    if (pbr) {
        // What a relit gaussian reflects with, beside the colours it reflects.
        // Primvars rather than attributes of the schema: the schema is USD's
        // and says nothing about a surface, while LrtSplatLightingAPI is ours.
        static const TfToken kMetallic("primvars:lrt:splat:metallic");
        static const TfToken kRoughness("primvars:lrt:splat:roughness");
        static const TfToken kTransmission("primvars:lrt:splat:transmission");
        UsdGeomPrimvarsAPI primvars(splats.GetPrim());
        primvars.CreatePrimvar(kMetallic, SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex)
            .Set(VtValue(metallics));
        primvars.CreatePrimvar(kRoughness, SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex)
            .Set(VtValue(roughnesses));
        primvars.CreatePrimvar(kTransmission, SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex)
            .Set(VtValue(transmissions));
    }

    if (options.relight) {
        // The cloud's colours are an albedo, not radiance: a mesh's material
        // said what the surface reflects, and nothing has lit it yet. The
        // scene's lights do that (LrtSplatLightingAPI), which is what makes a
        // converted mesh sit under the same lights the mesh would have.
        static const TfToken kRelight("primvars:lrt:splat:relight");
        splats.GetPrim().CreateAttribute(kRelight, SdfValueTypeNames->Bool, true).Set(true);
    }

    if (options.addCamera) {
        // The centre where the cloud is drawn: turned with it about x.
        GfVec3d centre = (lo + hi) * 0.5;
        if (options.rotateXDegrees != 0.0) {
            const double a = GfDegreesToRadians(options.rotateXDegrees);
            centre = GfVec3d(centre[0], centre[1] * std::cos(a) - centre[2] * std::sin(a),
                             centre[1] * std::sin(a) + centre[2] * std::cos(a));
        }
        const double size = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-3});
        UsdGeomCamera camera = UsdGeomCamera::Define(stage, SdfPath("/World/Camera"));
        camera.CreateFocalLengthAttr(VtValue(35.0F));
        camera.CreateHorizontalApertureAttr(VtValue(24.576F));
        camera.CreateVerticalApertureAttr(VtValue(13.824F));
        camera.CreateClippingRangeAttr(VtValue(GfVec2f(static_cast<float>(size * 0.001), static_cast<float>(size * 20.0))));
        // Looking down -Z at the centre from in front of it.
        UsdGeomXformCommonAPI(camera.GetPrim()).SetTranslate(
            GfVec3d(centre[0], centre[1], centre[2] + size * 1.2));
    }
    if (!stage->GetRootLayer()->Save()) {
        return Error::make(ErrorCode::IoFailure, "cannot save '{}'", path.string());
    }
    return ok();
}

}   // namespace lrt::usd
