// Copyright (c) 2026 lucabRTrender contributors.
#include "Material.h"

#include <map>
#include <set>

#include <MaterialXCore/Document.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/usd/sdf/assetPath.h>
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

/// Input types as their nodedefs declare them, where an authored value or
/// connection has the same number of floats under another name: USD authors
/// UsdUVTexture's scale and bias as float4, which MaterialX declares color4,
/// and connects a texture's colour output to UsdPreviewSurface's vector
/// normal. hdMtlx types each input by what it was given, the nodedef no
/// longer matches, and the whole material fails ("could not find a nodedef
/// for node 'Surface'"). A float4 and a color4 are one Slang float4, so the
/// input takes the declared name and nothing else changes. So does a value
/// hdMtlx typed as a string because USD's role type has no MaterialX name.
void matchDeclaredTypes(const MaterialX::DocumentPtr& document) {
    const auto numeric = [](const std::string& type) {
        static const std::set<std::string> kNumeric{"float",   "integer", "boolean",  "vector2",  "vector3",
                                                    "vector4", "color3",  "color4",   "matrix33", "matrix44"};
        return kNumeric.count(type) != 0;
    };
    const auto shape = [](const std::string& type) -> int {
        if (type == "vector3" || type == "color3") return 3;
        if (type == "vector4" || type == "color4") return 4;
        return 0;
    };
    for (MaterialX::ElementPtr element : document->traverseTree()) {
        const MaterialX::NodePtr node = element->asA<MaterialX::Node>();
        if (!node) {
            continue;
        }
        MaterialX::NodeDefPtr nodeDef = document->getNodeDef(node->getNodeDefString());
        if (!nodeDef) {
            // The one nodedef of the node's category that outputs its type;
            // with more than one, the node is left as it is.
            MaterialX::NodeDefPtr only;
            size_t count = 0;
            for (const MaterialX::NodeDefPtr& candidate : document->getMatchingNodeDefs(node->getCategory())) {
                if (candidate->getType() == node->getType()) {
                    only = candidate;
                    ++count;
                }
            }
            nodeDef = count == 1 ? only : nullptr;
        }
        if (!nodeDef) {
            continue;
        }
        for (const MaterialX::InputPtr& input : node->getInputs()) {
            const MaterialX::InputPtr declared = nodeDef->getActiveInput(input->getName());
            if (!declared || declared->getType() == input->getType()) {
                continue;
            }
            if (shape(declared->getType()) != 0 && shape(declared->getType()) == shape(input->getType())) {
                input->setType(declared->getType());
            } else if (input->getType() == "string" && numeric(declared->getType()) && input->hasValueString()) {
                // A value of a role type USD has and MaterialX does not
                // (vector3f, point3f, normal3f) reaches it as a string of its
                // numbers: the declared type reads the same text.
                input->setType(declared->getType());
            }
        }
    }
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
                // UsdUVTexture's wrap modes as USD spells them, in MaterialX's
                // enum: "repeat" is "periodic" there, and "useMetadata" (the
                // file's own, which no reader here keeps) its default.
                for (const TfToken& wrap : {TfToken("wrapS"), TfToken("wrapT")}) {
                    const auto found = node.parameters.find(wrap);
                    if (found == node.parameters.end()) {
                        continue;
                    }
                    const std::string mode = found->second.IsHolding<TfToken>()
                                                 ? found->second.UncheckedGet<TfToken>().GetString()
                                                 : found->second.IsHolding<std::string>()
                                                       ? found->second.UncheckedGet<std::string>()
                                                       : std::string();
                    if (mode == "repeat" || mode == "useMetadata") {
                        found->second = VtValue(TfToken("periodic"));
                    }
                }
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
        // A reader of three or four floats is a vector to MaterialX, and a
        // colour input it feeds (UsdPreviewSurface's diffuseColor from
        // displayColor, the common case) would fail its declaration: the
        // reader takes the colour nodedef where what it feeds is a colour.
        const MaterialX::DocumentPtr& libraries = HdMtlxStdLibraries();
        for (auto& [path, node] : network.nodes) {
            const MaterialX::NodeDefPtr nodeDef = libraries->getNodeDef(node.nodeTypeId.GetString());
            for (auto& [input, connections] : node.inputConnections) {
                for (HdMaterialConnection2& connection : connections) {
                    if (readers.count(connection.upstreamNode) == 0) {
                        continue;
                    }
                    if (connection.upstreamOutputName == kResult) {
                        connection.upstreamOutputName = TfToken("out");
                    }
                    const MaterialX::InputPtr declared = nodeDef ? nodeDef->getActiveInput(input.GetString()) : nullptr;
                    if (!declared) {
                        continue;
                    }
                    HdMaterialNode2& reader = network.nodes[connection.upstreamNode];
                    if (declared->getType() == "color3" && reader.nodeTypeId == TfToken("ND_geompropvalue_vector3")) {
                        reader.nodeTypeId = TfToken("ND_geompropvalue_color3");
                    } else if (declared->getType() == "color4" &&
                               reader.nodeTypeId == TfToken("ND_geompropvalue_vector4")) {
                        reader.nodeTypeId = TfToken("ND_geompropvalue_color4");
                    }
                }
            }
        }
        // HdMtlx writes a file input as its authored path, which a relative
        // path ("./textures/wood.jpg") makes relative to the process's
        // directory rather than the layer that wrote it. The value arrives
        // resolved; the resolved path goes in where there is one. A UDIM
        // pattern does not resolve, and keeps its authored path.
        for (auto& [path, node] : network.nodes) {
            for (auto& [name, value] : node.parameters) {
                if (value.IsHolding<SdfAssetPath>()) {
                    const SdfAssetPath& asset = value.UncheckedGet<SdfAssetPath>();
                    if (!asset.GetResolvedPath().empty()) {
                        value = VtValue(SdfAssetPath(asset.GetResolvedPath(), asset.GetResolvedPath()));
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
                    matchDeclaredTypes(mtlx);
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
