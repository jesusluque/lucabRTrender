// Copyright (c) 2026 lucabRTrender contributors.
#include "Material.h"

#include <map>
#include <set>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <pxr/imaging/hd/materialNetworkSchema.h>
#include <pxr/imaging/hd/materialNodeParameterSchema.h>
#include <pxr/imaging/hd/materialNodeSchema.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/sceneIndex.h>
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
        // AN INPUT THE NODEDEF DOES NOT HAVE IS DROPPED, NOT A FAILURE.
        //
        // MaterialX refuses a node whose interface does not match its
        // declaration, and the whole material goes with it. What authors such
        // an input is every day work: Blender's USD exporter writes
        // `inputs:specular` on a UsdPreviewSurface, which the specification
        // does not have (it has `specularColor`), so every fox, every
        // character and every asset that came out of Blender arrived here
        // grey -- "Could not find a nodedef for node 'Surface'". One input
        // nobody declared is worth exactly what it says and no more, and
        // losing the material over it is the wrong trade.
        std::vector<std::string> undeclared;
        for (const MaterialX::InputPtr& input : node->getInputs()) {
            if (!nodeDef->getActiveInput(input->getName())) {
                undeclared.push_back(input->getName());
            }
        }
        for (const std::string& name : undeclared) {
            lrt::log::info("materials: {} has no input {}; it is left out", node->getCategory(), name);
            node->removeInput(name);
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

/// What MaterialX calls the colour space USD named, or empty for one this
/// does not know.
std::string mtlxColourSpace(const TfToken& usd) {
    const std::string& name = usd.GetString();
    if (name == "sRGB" || name == "srgb_texture" || name == "srgb_rec709" || name == "sRGB - Texture") {
        return "srgb_texture";
    }
    if (name == "raw" || name == "Raw" || name == "lin_rec709" || name == "linear") {
        return "lin_rec709";
    }
    if (name == "auto") {
        return "auto";   // not MaterialX's: read by MaterialCompiler, and by nothing else
    }
    return {};
}

/// The node named `name`, wherever it sits in the document.
MaterialX::NodePtr nodeNamed(const MaterialX::DocumentPtr& doc, const std::string& name) {
    if (const MaterialX::NodePtr direct = doc->getNode(name)) {
        return direct;
    }
    for (const MaterialX::NodeGraphPtr& graph : doc->getNodeGraphs()) {
        if (const MaterialX::NodePtr found = graph->getNode(name)) {
            return found;
        }
    }
    return nullptr;
}

/// Puts back the colour space a material network drops.
///
/// WHY THIS IS NEEDED AT ALL
///
/// `HdMaterialNode2::parameters` is a map of names to values and nothing else,
/// so USD's `colorSpace` metadata on `inputs:file` -- and a UsdUVTexture's
/// `sourceColorSpace`, which usdImaging folds into the same place -- is gone by
/// the time a delegate reads `GetMaterialResource`. hdMtlx therefore writes a
/// document whose file inputs say nothing about their colour space, and
/// `MaterialCompiler` reads every texture raw: the chess set's black marble
/// came out pale grey, because 8-bit sRGB values were taken for linear ones.
///
/// It survives in the scene index, which is where a Hydra 2.0 delegate should
/// be reading materials in the first place (`HdMaterialNodeParameterSchema`
/// carries it beside the value). So the network is read for its values, as
/// before, and the scene index for this one thing, which is then written onto
/// the MaterialX input the document ended up with. Nothing is transformed
/// here: `material::MaterialCompiler` reads the attribute and `TextureStore`
/// does the decode on the device.
void applyColourSpaces(HdSceneDelegate* sceneDelegate, const SdfPath& id,
                       const MaterialX::DocumentPtr& doc) {
    const HdSceneIndexBaseRefPtr index = sceneDelegate->GetRenderIndex().GetTerminalSceneIndex();
    if (!index) {
        return;
    }
    const HdSceneIndexPrim prim = index->GetPrim(id);
    if (!prim.dataSource) {
        return;
    }
    const HdMaterialSchema material = HdMaterialSchema::GetFromParent(prim.dataSource);
    if (!material.IsDefined()) {
        return;
    }
    for (const TfToken& context : {TfToken("mtlx"), TfToken()}) {
        const HdMaterialNetworkSchema network = material.GetMaterialNetwork(context);
        if (!network.IsDefined()) {
            continue;
        }
        const HdMaterialNodeContainerSchema nodes = network.GetNodes();
        for (const TfToken& nodeName : nodes.GetNames()) {
            const HdMaterialNodeSchema node = nodes.Get(nodeName);
            if (!node.IsDefined()) {
                continue;
            }
            const MaterialX::NodePtr mxNode =
                nodeNamed(doc, HdMtlxCreateNameFromPath(SdfPath(nodeName.GetString())));
            if (!mxNode) {
                continue;
            }
            const HdMaterialNodeParameterContainerSchema parameters = node.GetParameters();
            for (const TfToken& parameterName : parameters.GetNames()) {
                const HdMaterialNodeParameterSchema parameter = parameters.Get(parameterName);
                const HdTokenDataSourceHandle space = parameter.IsDefined() ? parameter.GetColorSpace()
                                                                            : HdTokenDataSourceHandle();
                if (!space) {
                    continue;
                }
                const std::string mtlx = mtlxColourSpace(space->GetTypedValue(0.0F));
                MaterialX::InputPtr input = mtlx.empty() ? nullptr : mxNode->getInput(parameterName.GetString());
                if (input) {
                    input->setColorSpace(mtlx);
                }
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
                    applyColourSpaces(sceneDelegate, id, mtlx);
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
