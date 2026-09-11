// Copyright (c) 2026 lucabRTrender contributors.
#include "Material.h"

#include <map>
#include <set>

#include <MaterialXCore/Document.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>

#include "RenderDelegate.h"
#include "lrt/core/Log.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// The MaterialX nodedefs for the USD shading nodes (libraries/bxdf/usd_preview_surface.mtlx).
const std::map<TfToken, TfToken>& usdNodeDefs() {
    static const std::map<TfToken, TfToken> kNames{
        {TfToken("UsdPreviewSurface"), TfToken("ND_UsdPreviewSurface_surfaceshader")},
        {TfToken("UsdUVTexture"), TfToken("ND_UsdUVTexture_23")},
        {TfToken("UsdTransform2d"), TfToken("ND_UsdTransform2d")},
    };
    return kNames;
}

/// UsdPrimvarReader nodes as the geompropvalue nodes they wrap: MaterialX's
/// nodegraphs for them bind the primvar's name through an interface, which
/// its generator needs as a constant.
const std::map<TfToken, TfToken>& primvarReaders() {
    static const std::map<TfToken, TfToken> kNames{
        {TfToken("UsdPrimvarReader_float"), TfToken("ND_geompropvalue_float")},
        {TfToken("UsdPrimvarReader_float2"), TfToken("ND_geompropvalue_vector2")},
        {TfToken("UsdPrimvarReader_float3"), TfToken("ND_geompropvalue_vector3")},
        {TfToken("UsdPrimvarReader_normal"), TfToken("ND_geompropvalue_vector3")},
        {TfToken("UsdPrimvarReader_point"), TfToken("ND_geompropvalue_vector3")},
        {TfToken("UsdPrimvarReader_vector"), TfToken("ND_geompropvalue_vector3")},
        {TfToken("UsdPrimvarReader_float4"), TfToken("ND_geompropvalue_vector4")},
        {TfToken("UsdPrimvarReader_int"), TfToken("ND_geompropvalue_integer")},
        {TfToken("UsdPrimvarReader_string"), TfToken("ND_geompropvalue_string")},
    };
    return kNames;
}

}   // namespace

void HdLrtMaterial::Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr || (*dirtyBits & (DirtyResource | DirtyParams)) == 0) {
        *dirtyBits = Clean;
        return;
    }
    const SdfPath& id = GetId();
    std::shared_ptr<void> document;
    const VtValue resource = sceneDelegate->GetMaterialResource(id);
    if (resource.IsHolding<HdMaterialNetworkMap>()) {
        HdMaterialNetwork2 network = HdConvertToHdMaterialNetwork2(resource.UncheckedGet<HdMaterialNetworkMap>());
        static const TfToken kVarname("varname");
        static const TfToken kFallback("fallback");
        static const TfToken kResult("result");
        std::set<SdfPath> readers;
        for (auto& [path, node] : network.nodes) {
            if (const auto renamed = usdNodeDefs().find(node.nodeTypeId); renamed != usdNodeDefs().end()) {
                node.nodeTypeId = renamed->second;
            } else if (const auto reader = primvarReaders().find(node.nodeTypeId); reader != primvarReaders().end()) {
                node.nodeTypeId = reader->second;
                readers.insert(path);
                const auto rename = [&](const TfToken& from, const TfToken& to) {
                    if (const auto found = node.parameters.find(from); found != node.parameters.end()) {
                        VtValue value = found->second;
                        node.parameters.erase(found);
                        // geomprop is a string; varname may arrive as a token.
                        if (value.IsHolding<TfToken>()) {
                            value = VtValue(value.UncheckedGet<TfToken>().GetString());
                        }
                        node.parameters[to] = value;
                    }
                };
                rename(kVarname, TfToken("geomprop"));
                rename(kFallback, TfToken("default"));
            }
        }
        for (auto& [path, node] : network.nodes) {
            for (auto& [input, connections] : node.inputConnections) {
                for (HdMaterialConnection2& connection : connections) {
                    if (readers.count(connection.upstreamNode) != 0 && connection.upstreamOutputName == kResult) {
                        connection.upstreamOutputName = TfToken("out");
                    }
                }
            }
        }
        const auto terminal = network.terminals.find(HdMaterialTerminalTokens->surface);
        if (terminal != network.terminals.end()) {
            const SdfPath& surfacePath = terminal->second.upstreamNode;
            const auto surface = network.nodes.find(surfacePath);
            if (surface != network.nodes.end()) {
                try {
                    HdMtlxTexturePrimvarData data;
                    MaterialX::DocumentPtr mtlx = HdMtlxCreateMtlxDocumentFromHdNetwork(
                        network, surface->second, surfacePath, id, HdMtlxStdLibraries(), &data);
                    document = mtlx;
                } catch (const std::exception& e) {
                    lrt::log::warn("hdLrt: material {}: {}", id.GetString(), e.what());
                }
            }
        }
    }
    if (!document) {
        lrt::log::warn("hdLrt: material {}: no surface network MaterialX reads; its meshes show displayColor",
                       id.GetString());
    }
    engine->setMaterial(id, std::move(document));
    *dirtyBits = Clean;
}

void HdLrtMaterial::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->removeMaterial(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
