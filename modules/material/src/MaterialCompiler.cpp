// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/material/MaterialCompiler.h"

#include <cstdio>
#include <functional>
#include <sstream>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenHw/HwConstants.h>
#include <MaterialXGenHw/Nodes/HwSurfaceNode.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Syntax.h>
#include <MaterialXGenShader/TypeDesc.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXGenSlang/SlangShaderGenerator.h>

namespace mx = MaterialX;

namespace lrt::material {

namespace {

const std::string kFunctionPlaceholder = "LRT_MATERIAL_FUNCTION";

/// The surface node: no light loop. Its BSDF graph runs once, pushing lobes;
/// what it weights them by becomes the material's lobe stack.
class LrtSurfaceNode : public mx::HwSurfaceNode {
public:
    static mx::ShaderNodeImplPtr create() { return std::make_shared<LrtSurfaceNode>(); }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override {
        if (stage.getName() != mx::Stage::PIXEL) {
            return;
        }
        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
        const std::string prefix = mx::HW::T_VERTEX_DATA_INSTANCE + ".";
        const mx::ShaderOutput* output = node.getOutput();
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, true, context, stage);
        shadergen.emitLineEnd(stage);
        shadergen.emitScopeBegin(stage);
        shadergen.emitLine("float3 N = normalize(" + prefix + mx::HW::T_NORMAL_WORLD + ")", stage);
        shadergen.emitLine("float3 V = normalize(" + mx::HW::T_VIEW_POSITION + " - " + prefix +
                               mx::HW::T_POSITION_WORLD + ")",
                           stage);
        shadergen.emitLine("float3 P = " + prefix + mx::HW::T_POSITION_WORLD, stage);
        shadergen.emitLine("float3 L = float3(0.0, 0.0, 0.0)", stage);
        shadergen.emitLine("float occlusion = 1.0", stage);
        shadergen.emitLineBegin(stage);
        shadergen.emitString("float surfaceOpacity = ", stage);
        shadergen.emitInput(node.getInput(getOpacityInputName()), context, stage);
        shadergen.emitLineEnd(stage);
        const std::string outColor = output->getVariable() + ".color";
        const std::string outTransparency = output->getVariable() + ".transparency";
        if (const mx::ShaderInput* bsdfInput = node.getInput(getBsdfInputName())) {
            if (const mx::ShaderNode* bsdf = bsdfInput->getConnectedSibling()) {
                shadergen.emitLine("ClosureData closureData = makeClosureData(CLOSURE_TYPE_REFLECTION, L, V, N, P, "
                                   "occlusion)",
                                   stage);
                shadergen.emitFunctionCall(*bsdf, context, stage);
                shadergen.emitLine("gLrtResult = lrtFinishStack(" + bsdf->getOutput()->getVariable() +
                                       ", gLrtResult.emission, surfaceOpacity)",
                                   stage);
            }
        }
        if (const mx::ShaderInput* edfInput = node.getInput(getEdfInputName())) {
            if (const mx::ShaderNode* edf = edfInput->getConnectedSibling()) {
                shadergen.emitScopeBegin(stage);
                shadergen.emitLine("ClosureData closureData = makeClosureData(CLOSURE_TYPE_EMISSION, L, V, N, P, "
                                   "occlusion)",
                                   stage);
                shadergen.emitFunctionCall(*edf, context, stage);
                shadergen.emitLine(outColor + " += " + edf->getOutput()->getVariable(), stage);
                shadergen.emitScopeEnd(stage);
            }
        }
        shadergen.emitLine(outTransparency + " = float3(1.0 - surfaceOpacity)", stage);
        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);
    }
};

/// The genglsl reference's surface node: the BSDF graph as MaterialX
/// evaluates it for one light, gLrtReferenceL, its response left in
/// gLrtReferenceResponse.
class LrtReferenceSurfaceNode : public mx::HwSurfaceNode {
public:
    static mx::ShaderNodeImplPtr create() { return std::make_shared<LrtReferenceSurfaceNode>(); }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override {
        if (stage.getName() != mx::Stage::PIXEL) {
            return;
        }
        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
        const std::string prefix = mx::HW::T_VERTEX_DATA_INSTANCE + ".";
        const mx::ShaderOutput* output = node.getOutput();
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, true, context, stage);
        shadergen.emitLineEnd(stage);
        shadergen.emitScopeBegin(stage);
        shadergen.emitLine("float3 N = normalize(" + prefix + mx::HW::T_NORMAL_WORLD + ")", stage);
        shadergen.emitLine("float3 V = normalize(" + mx::HW::T_VIEW_POSITION + " - " + prefix +
                               mx::HW::T_POSITION_WORLD + ")",
                           stage);
        shadergen.emitLine("float3 P = " + prefix + mx::HW::T_POSITION_WORLD, stage);
        shadergen.emitLine("float3 L = gLrtReferenceL", stage);
        shadergen.emitLine("float occlusion = 1.0", stage);
        shadergen.emitLineBegin(stage);
        shadergen.emitString("float surfaceOpacity = ", stage);
        shadergen.emitInput(node.getInput(getOpacityInputName()), context, stage);
        shadergen.emitLineEnd(stage);
        if (const mx::ShaderInput* bsdfInput = node.getInput(getBsdfInputName())) {
            if (const mx::ShaderNode* bsdf = bsdfInput->getConnectedSibling()) {
                shadergen.emitLine("ClosureData closureData = makeClosureData(CLOSURE_TYPE_REFLECTION, L, V, N, P, "
                                   "occlusion)",
                                   stage);
                shadergen.emitFunctionCall(*bsdf, context, stage);
                shadergen.emitLine("gLrtReferenceResponse = " + bsdf->getOutput()->getVariable() + ".response",
                                   stage);
            }
        }
        if (const mx::ShaderInput* edfInput = node.getInput(getEdfInputName())) {
            if (const mx::ShaderNode* edf = edfInput->getConnectedSibling()) {
                shadergen.emitScopeBegin(stage);
                shadergen.emitLine("ClosureData closureData = makeClosureData(CLOSURE_TYPE_EMISSION, L, V, N, P, "
                                   "occlusion)",
                                   stage);
                shadergen.emitFunctionCall(*edf, context, stage);
                shadergen.emitLine(output->getVariable() + ".color += " + edf->getOutput()->getVariable(), stage);
                shadergen.emitScopeEnd(stage);
            }
        }
        shadergen.emitLine(output->getVariable() + ".transparency = float3(1.0 - surfaceOpacity)", stage);
        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);
    }
};

class LrtSlangShaderGenerator : public mx::SlangShaderGenerator {
public:
    LrtSlangShaderGenerator(mx::TypeSystemPtr types, ClosureVariant variant)
        : mx::SlangShaderGenerator(types), variant_(variant) {
        const bool lobes = variant == ClosureVariant::Lobes;
        registerImplementation("IM_surface_" + TARGET, lobes ? LrtSurfaceNode::create : LrtReferenceSurfaceNode::create);
        // The closure and shader types live in material_runtime.slang.
        const auto aggregate = [&](mx::TypeDesc type, const std::string& name, const std::string& value) {
            _syntax->registerTypeSyntax(type, std::make_shared<mx::AggregateTypeSyntax>(
                                                  _syntax.get(), name, value, mx::EMPTY_STRING, mx::EMPTY_STRING,
                                                  mx::EMPTY_STRING));
        };
        if (lobes) {
            aggregate(mx::Type::BSDF, "BSDF", "lrt_bsdf_zero()");
        }
        aggregate(mx::Type::VDF, "VDF", "VDF(float3(0.0),float3(1.0))");
        aggregate(mx::Type::SURFACESHADER, "surfaceshader", "surfaceshader(float3(0.0),float3(0.0))");
        aggregate(mx::Type::VOLUMESHADER, "volumeshader", "volumeshader(float3(0.0),float3(0.0))");
        aggregate(mx::Type::DISPLACEMENTSHADER, "displacementshader", "displacementshader(float3(0.0),1.0)");
        aggregate(mx::Type::LIGHTSHADER, "lightshader", "lightshader(float3(0.0),float3(0.0))");
    }

    const ClosureVariant variant_;

    /// Filled by generate: the blob layout, in declaration order.
    mutable std::vector<MaterialSlot> slots;
    mutable uint32_t words = 0;

    mx::ShaderPtr generate(const std::string& name, mx::ElementPtr element, mx::GenContext& context) const override {
        slots.clear();
        words = 0;
        mx::ShaderPtr shader = createShader(name, element, context);
        mx::ScopedFloatFormatting fmt(mx::Value::FloatFormatFixed);
        mx::ShaderStage& ps = shader->getStage(mx::Stage::PIXEL);
        emitMaterial(shader->getGraph(), context, ps);
        replaceTokens(_tokenSubstitutions, ps);
        SlangSyntaxFromGlsl(ps);
        return shader;
    }

private:
    MaterialSlot& addSlot(MaterialSlot::Kind kind, const std::string& variable, uint32_t count) const {
        MaterialSlot slot;
        slot.kind = kind;
        slot.variable = variable;
        slot.offset = words;
        slot.words = count;
        words += count;
        slots.push_back(std::move(slot));
        return slots.back();
    }

    static uint32_t wordsOf(mx::TypeDesc type) {
        if (type == mx::Type::FLOAT || type == mx::Type::INTEGER || type == mx::Type::BOOLEAN) return 1;
        if (type == mx::Type::VECTOR2) return 2;
        if (type == mx::Type::VECTOR3 || type == mx::Type::COLOR3) return 3;
        if (type == mx::Type::VECTOR4 || type == mx::Type::COLOR4) return 4;
        if (type == mx::Type::MATRIX33) return 9;
        if (type == mx::Type::MATRIX44) return 16;
        return 0;
    }

    static std::string readOf(mx::TypeDesc type, uint32_t offset) {
        const auto at = [&](uint32_t k) { return "lrtBlob(" + std::to_string(offset + k) + "u)"; };
        const auto list = [&](uint32_t n) {
            std::string s;
            for (uint32_t k = 0; k < n; ++k) {
                s += (k ? ", " : "") + at(k);
            }
            return s;
        };
        if (type == mx::Type::FLOAT) return at(0);
        if (type == mx::Type::INTEGER) return "int(" + at(0) + ")";
        if (type == mx::Type::BOOLEAN) return "(" + at(0) + " != 0.0)";
        if (type == mx::Type::VECTOR2) return "float2(" + list(2) + ")";
        if (type == mx::Type::VECTOR3 || type == mx::Type::COLOR3) return "float3(" + list(3) + ")";
        if (type == mx::Type::VECTOR4 || type == mx::Type::COLOR4) return "float4(" + list(4) + ")";
        if (type == mx::Type::MATRIX33) return "float3x3(" + list(9) + ")";
        if (type == mx::Type::MATRIX44) return "float4x4(" + list(16) + ")";
        return {};
    }

    static std::vector<float> valueOf(const mx::ValuePtr& value, mx::TypeDesc type) {
        std::vector<float> out(wordsOf(type), 0.0F);
        if (!value) {
            if (type == mx::Type::MATRIX33 || type == mx::Type::MATRIX44) {
                const size_t n = type == mx::Type::MATRIX33 ? 3 : 4;
                for (size_t k = 0; k < n; ++k) out[k * n + k] = 1.0F;
            }
            return out;
        }
        if (value->isA<float>()) out[0] = value->asA<float>();
        else if (value->isA<int>()) out[0] = static_cast<float>(value->asA<int>());
        else if (value->isA<bool>()) out[0] = value->asA<bool>() ? 1.0F : 0.0F;
        else if (value->isA<mx::Vector2>()) { auto v = value->asA<mx::Vector2>(); out = {v[0], v[1]}; }
        else if (value->isA<mx::Vector3>()) { auto v = value->asA<mx::Vector3>(); out = {v[0], v[1], v[2]}; }
        else if (value->isA<mx::Color3>()) { auto v = value->asA<mx::Color3>(); out = {v[0], v[1], v[2]}; }
        else if (value->isA<mx::Vector4>()) { auto v = value->asA<mx::Vector4>(); out = {v[0], v[1], v[2], v[3]}; }
        else if (value->isA<mx::Color4>()) { auto v = value->asA<mx::Color4>(); out = {v[0], v[1], v[2], v[3]}; }
        else if (value->isA<mx::Matrix33>()) {
            auto m = value->asA<mx::Matrix33>();
            for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < 3; ++c) out[r * 3 + c] = m[r][c];
        } else if (value->isA<mx::Matrix44>()) {
            auto m = value->asA<mx::Matrix44>();
            for (size_t r = 0; r < 4; ++r) for (size_t c = 0; c < 4; ++c) out[r * 4 + c] = m[r][c];
        }
        return out;
    }

    static int intInput(const mx::ShaderNode* node, const std::string& name, int fallback) {
        const mx::ShaderInput* input = node != nullptr ? node->getInput(name) : nullptr;
        if (input == nullptr || !input->getValue()) return fallback;
        const mx::ValuePtr v = input->getValue();
        return v->isA<int>() ? v->asA<int>() : fallback;
    }

    void emitMaterial(const mx::ShaderGraph& graph, mx::GenContext& context, mx::ShaderStage& stage) const {
        const bool lobes = variant_ == ClosureVariant::Lobes;
        emitLine(lobes ? "import lrt.material.material_runtime;" : "import lrt.material.material_inputs;", stage, false);
        emitLineBreak(stage);
        emitTypeDefinitions(context, stage);
        emitConstants(context, stage);

        // The vertex data MaterialX's nodes read, filled from the shading point.
        const mx::VariableBlock& vertexData = stage.getInputBlock(mx::HW::VERTEX_DATA);
        emitLine("struct VertexData", stage, false);
        emitScopeBegin(stage);
        emitLine("float4 SV_Position", stage);
        for (size_t i = 0; i < vertexData.size(); ++i) {
            emitLineBegin(stage);
            emitVariableDeclaration(vertexData[i], mx::EMPTY_STRING, context, stage, false);
            emitLineEnd(stage);
        }
        emitScopeEnd(stage, true);
        emitLine("static VertexData " + mx::HW::T_VERTEX_DATA_INSTANCE, stage);
        emitLineBreak(stage);

        // Uniforms as module statics, assigned from the blob.
        std::vector<const mx::ShaderPort*> uniforms;
        for (const auto& it : stage.getUniformBlocks()) {
            const mx::VariableBlock& block = *it.second;
            if (block.getName() == mx::HW::LIGHT_DATA) {
                continue;
            }
            for (size_t i = 0; i < block.size(); ++i) {
                const mx::ShaderPort* port = block[i];
                emitLineBegin(stage);
                // No initial value: materials differing only in values share one source.
                emitVariableDeclaration(port, "static", context, stage, false);
                emitLineEnd(stage);
                uniforms.push_back(port);
            }
        }
        emitLineBreak(stage);
        emitLibraryInclude("stdlib/genslang/lib/mx_math.slang", context, stage);
        emitLine("#define DIRECTIONAL_ALBEDO_METHOD 0", stage, false);
        emitLine("#define AIRY_FRESNEL_ITERATIONS 2", stage, false);
        emitLineBreak(stage);
        if (!lobes) {
            // genglsl's closures call into the environment and transmission
            // code: none and opacity, which read no uniforms.
            emitSpecularEnvironment(context, stage);
            emitTransmissionRender(context, stage);
        }
        _tokenSubstitutions[mx::ShaderGenerator::T_FILE_TRANSFORM_UV] = "mx_transform_uv.glsl";
        _tokenSubstitutions[mx::HW::T_TEX_SAMPLER_SIGNATURE] = "SamplerTexture2D tex_sampler";
        emitFunctionDefinitions(graph, context, stage);

        emitLine("public void " + kFunctionPlaceholder + "(MaterialInputs inputs, uint blob)", stage, false);
        emitFunctionBodyBegin(graph, context, stage);
        emitLine(lobes ? "lrtBeginMaterial(inputs, blob)" : "lrtBeginInputs(inputs, blob)", stage);
        const std::string vd = mx::HW::T_VERTEX_DATA_INSTANCE + ".";
        for (size_t i = 0; i < vertexData.size(); ++i) {
            const mx::ShaderPort* port = vertexData[i];
            const std::string& name = port->getName();
            const mx::TypeDesc type = port->getType();
            std::string value;
            const auto builtin = [&](const std::string& token, const char* field) {
                if (name == token) value = std::string("inputs.") + field;
            };
            builtin(mx::HW::T_POSITION_WORLD, "positionWorld");
            builtin(mx::HW::T_NORMAL_WORLD, "normalWorld");
            builtin(mx::HW::T_TANGENT_WORLD, "tangentWorld");
            builtin(mx::HW::T_BITANGENT_WORLD, "bitangentWorld");
            builtin(mx::HW::T_POSITION_OBJECT, "positionObject");
            builtin(mx::HW::T_NORMAL_OBJECT, "normalObject");
            builtin(mx::HW::T_TANGENT_OBJECT, "tangentObject");
            builtin(mx::HW::T_BITANGENT_OBJECT, "bitangentObject");
            const auto primvar = [&](const std::string& primvarName, const std::string& fallback) {
                MaterialSlot& slot = addSlot(MaterialSlot::Kind::Primvar, name, 1);
                slot.name = primvarName;
                const std::string read = "lrtPrimvar(uint(lrtBlob(" + std::to_string(slot.offset) + "u)), " +
                                         fallback + ")";
                if (type == mx::Type::FLOAT) value = read + ".x";
                else if (type == mx::Type::INTEGER) value = "int(" + read + ".x)";
                else if (type == mx::Type::BOOLEAN) value = "(" + read + ".x != 0.0)";
                else if (type == mx::Type::VECTOR2) value = read + ".xy";
                else if (type == mx::Type::VECTOR3 || type == mx::Type::COLOR3) value = read + ".xyz";
                else value = read;
            };
            if (name.rfind(mx::HW::T_TEXCOORD + "_", 0) == 0) {
                const std::string index = name.substr(mx::HW::T_TEXCOORD.size() + 1);
                primvar(index == "0" ? "st" : "st" + index, "float4(0.0)");
            } else if (name.rfind(mx::HW::T_COLOR + "_", 0) == 0) {
                primvar("displayColor", "inputs.displayColor");
            } else if (name.rfind(mx::HW::T_IN_GEOMPROP + "_", 0) == 0) {
                primvar(name.substr(mx::HW::T_IN_GEOMPROP.size() + 1), "float4(0.0)");
            }
            if (!value.empty()) {
                emitLine(vd + port->getVariable() + " = " + value, stage);
            }
        }
        for (const mx::ShaderPort* port : uniforms) {
            const mx::TypeDesc type = port->getType();
            const std::string& variable = port->getVariable();
            if (port->getName() == mx::HW::T_VIEW_POSITION) {
                emitLine(variable + " = inputs.viewPosition", stage);
                continue;
            }
            if (port->getName() == mx::HW::T_FRAME) {
                emitLine(variable + " = inputs.frame", stage);
                continue;
            }
            if (port->getName() == mx::HW::T_TIME) {
                emitLine(variable + " = inputs.time", stage);
                continue;
            }
            if (type == mx::Type::FILENAME) {
                MaterialSlot& slot = addSlot(MaterialSlot::Kind::Texture, variable, 2);
                slot.name = port->getValue() ? port->getValue()->getValueString() : std::string();
                const mx::ShaderNode* node = port->getNode();
                // MaterialX reads a file in its colorspace, the document's
                // (linear) unless it says srgb_texture; UsdUVTexture says
                // sourceColorSpace instead: auto, raw or sRGB.
                const std::string& space = port->getColorSpace();
                const mx::ShaderInput* source = node != nullptr ? node->getInput("sourceColorSpace") : nullptr;
                const std::string usd = source != nullptr && source->getValue() ? source->getValue()->getValueString()
                                                                                : std::string();
                if (space == "srgb_texture" || space == "g22_rec709" || usd == "sRGB") {
                    slot.space = ColourSpace::Srgb;
                } else if (space.empty() && source != nullptr && (usd.empty() || usd == "auto")) {
                    slot.space = ColourSpace::Auto;
                } else {
                    slot.space = ColourSpace::Raw;
                }
                const auto wrap = [](int mode) {
                    // MaterialX address modes: 0 constant, 1 clamp, 2 periodic, 3 mirror.
                    return mode == 0 ? Wrap::Black : mode == 1 ? Wrap::Clamp : mode == 3 ? Wrap::Mirror : Wrap::Repeat;
                };
                slot.wrapS = wrap(intInput(node, "uaddressmode", 2));
                slot.wrapT = wrap(intInput(node, "vaddressmode", 2));
                slot.filter = intInput(node, "filtertype", 1) == 0 ? Filter::Nearest : Filter::Linear;
                emitLine(variable + " = lrtTextureOf(lrtBlob(" + std::to_string(slot.offset) + "u), lrtBlob(" +
                             std::to_string(slot.offset + 1) + "u))",
                         stage);
                continue;
            }
            const uint32_t count = wordsOf(type);
            if (count == 0) {
                continue;   // strings, shader-typed uniforms: their declared defaults
            }
            MaterialSlot& slot = addSlot(MaterialSlot::Kind::Value, variable, count);
            slot.value = valueOf(port->getValue(), type);
            emitLine(variable + " = " + readOf(type, slot.offset), stage);
        }

        // The graph, as GenSlang orders it.
        const mx::ShaderGraphOutputSocket* outputSocket = graph.getOutputSocket();
        if (graph.hasClassification(mx::ShaderNode::Classification::SHADER | mx::ShaderNode::Classification::SURFACE)) {
            emitFunctionCalls(graph, context, stage, mx::ShaderNode::Classification::TEXTURE);
            for (mx::ShaderGraphOutputSocket* socket : graph.getOutputSockets()) {
                if (socket->getConnection()) {
                    const mx::ShaderNode* upstream = socket->getConnection()->getNode();
                    if (upstream->getParent() == &graph &&
                        (upstream->hasClassification(mx::ShaderNode::Classification::CLOSURE) ||
                         upstream->hasClassification(mx::ShaderNode::Classification::SHADER))) {
                        emitFunctionCall(*upstream, context, stage);
                    }
                }
            }
        } else {
            emitFunctionCalls(graph, context, stage);
        }
        if (const mx::ShaderOutput* connection = outputSocket->getConnection()) {
            const std::string variable = connection->getVariable();
            if (graph.hasClassification(mx::ShaderNode::Classification::SURFACE)) {
                if (lobes) {
                    emitLine("gLrtResult.emission = " + variable + ".color", stage);
                    emitLine("gLrtResult.opacity = saturate(1.0 - dot(" + variable +
                                 ".transparency, float3(1.0 / 3.0)))",
                             stage);
                } else {
                    emitLine("gLrtReferenceEmission = " + variable + ".color", stage);
                }
            } else {
                const mx::TypeDesc type = outputSocket->getType();
                std::string rgb = variable;
                if (type == mx::Type::FLOAT || type == mx::Type::INTEGER) rgb = "float3(float(" + variable + "))";
                else if (type.isFloat2()) rgb = "float3(" + variable + ", 0.0)";
                else if (type.isFloat4()) rgb = variable + ".rgb";
                emitLine((lobes ? "gLrtResult.emission = " : "gLrtReferenceEmission = ") + rgb, stage);
            }
        }
        emitFunctionBodyEnd(graph, context, stage);
    }
};

std::string hexHash(const std::string& text) {
    // FNV-1a, 64 bits: a name for equal source, not a secret.
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : text) {
        h = (h ^ c) * 1099511628211ULL;
    }
    char out[17];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(h));
    return out;
}

}   // namespace

struct MaterialCompiler::Impl {
    mx::DocumentPtr                          libraries;            ///< standard, the engine's images and closures
    mx::DocumentPtr                          referenceLibraries;   ///< standard and the engine's images
    mx::FileSearchPath                       sourcePaths;
    std::shared_ptr<LrtSlangShaderGenerator> generator;
    std::shared_ptr<LrtSlangShaderGenerator> reference;
};

MaterialCompiler::~MaterialCompiler() = default;

Result<std::unique_ptr<MaterialCompiler>> MaterialCompiler::create(
    const std::vector<std::filesystem::path>& materialxRoots, const std::vector<std::filesystem::path>& shaderPaths) {
    try {
        mx::FileSearchPath libraryPaths;
        std::vector<std::filesystem::path> searchPaths;
        for (const auto& root : materialxRoots) {
            libraryPaths.append(mx::FilePath(root.string()));
            searchPaths.push_back(root);
            searchPaths.push_back(root / "libraries");
        }
        mx::DocumentPtr standard = mx::createDocument();
        mx::loadLibraries({"libraries"}, libraryPaths, standard);
        return create(std::shared_ptr<void>(standard), searchPaths, shaderPaths);
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::InternalError, "materials: MaterialX: {}", e.what());
    }
}

Result<std::unique_ptr<MaterialCompiler>> MaterialCompiler::create(
    const std::shared_ptr<void>& libraries, const std::vector<std::filesystem::path>& librarySearchPaths,
    const std::vector<std::filesystem::path>& shaderPaths) {
    auto compiler = std::unique_ptr<MaterialCompiler>(new MaterialCompiler());
    compiler->impl_ = std::make_unique<Impl>();
    Impl& impl = *compiler->impl_;
    try {
        const mx::DocumentPtr standard = std::static_pointer_cast<mx::Document>(libraries);
        if (!standard || standard->getNodeDefs().empty()) {
            return Error(ErrorCode::NotFound, "materials: no MaterialX libraries");
        }
        // The engine's node implementations, found beside its shaders.
        impl.libraries = mx::createDocument();
        impl.libraries->importLibrary(standard);
        impl.referenceLibraries = mx::createDocument();
        impl.referenceLibraries->importLibrary(standard);
        bool found = false;
        for (const auto& shaders : shaderPaths) {
            const std::filesystem::path images = shaders / "lrt/material/mx/lrt_genslang_images.mtlx";
            const std::filesystem::path closures = shaders / "lrt/material/mx/lrt_genslang_closures.mtlx";
            if (std::filesystem::exists(images) && std::filesystem::exists(closures)) {
                mx::DocumentPtr imageDoc = mx::createDocument();
                mx::readFromXmlFile(imageDoc, mx::FilePath(images.string()));
                mx::DocumentPtr closureDoc = mx::createDocument();
                mx::readFromXmlFile(closureDoc, mx::FilePath(closures.string()));
                impl.libraries->importLibrary(imageDoc);
                impl.libraries->importLibrary(closureDoc);
                impl.referenceLibraries->importLibrary(imageDoc);
                found = true;
            }
            impl.sourcePaths.append(mx::FilePath(shaders.string()));
        }
        if (!found) {
            return Error(ErrorCode::NotFound,
                         "materials: lrt/material/mx/lrt_genslang_{images,closures}.mtlx are not in the shader paths");
        }
        for (const auto& path : librarySearchPaths) {
            impl.sourcePaths.append(mx::FilePath(path.string()));
        }
        impl.generator = std::make_shared<LrtSlangShaderGenerator>(mx::TypeSystem::create(), ClosureVariant::Lobes);
        impl.generator->registerTypeDefs(impl.libraries);
        impl.reference =
            std::make_shared<LrtSlangShaderGenerator>(mx::TypeSystem::create(), ClosureVariant::GenglslReference);
        impl.reference->registerTypeDefs(impl.referenceLibraries);
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::InternalError, "materials: MaterialX: {}", e.what());
    }
    return compiler;
}

bool MaterialCompiler::cutsOut(const std::shared_ptr<void>& document) {
    const auto doc = std::static_pointer_cast<mx::Document>(document);
    if (!doc) {
        return false;
    }
    for (const mx::ElementPtr& element : doc->traverseTree()) {
        const mx::NodePtr node = element->asA<mx::Node>();
        if (!node) {
            continue;
        }
        const mx::InputPtr input = node->getInput("opacityThreshold");
        if (!input) {
            continue;
        }
        if (!input->getNodeName().empty() || !input->getNodeGraphString().empty() ||
            !input->getInterfaceName().empty()) {
            return true;   // driven by a graph: assume it cuts
        }
        const mx::ValuePtr value = input->getValue();
        if (value && value->isA<float>() && value->asA<float>() > 0.0F) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<void> MaterialCompiler::libraries() const {
    return impl_->libraries;
}

Result<CompiledMaterial> MaterialCompiler::compileXml(const std::string& xml, const std::string& element,
                                                     ClosureVariant variant) {
    try {
        mx::DocumentPtr doc = mx::createDocument();
        mx::readFromXmlString(doc, xml);
        doc->importLibrary(variant == ClosureVariant::Lobes ? impl_->libraries : impl_->referenceLibraries);
        return compileDocument(doc, element, variant);
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::InvalidArgument, "materials: MaterialX: {}", e.what());
    }
}

Result<CompiledMaterial> MaterialCompiler::compileDocument(const std::shared_ptr<void>& document,
                                                           const std::string& element, ClosureVariant variant) {
    const Impl& impl = *impl_;
    const std::shared_ptr<LrtSlangShaderGenerator>& generator =
        variant == ClosureVariant::Lobes ? impl.generator : impl.reference;
    try {
        const mx::DocumentPtr given = std::static_pointer_cast<mx::Document>(document);
        // A document built elsewhere (hdMtlx) knows the standard libraries but
        // not the engine's implementations.
        mx::DocumentPtr doc = mx::createDocument();
        doc->copyContentFrom(given);
        doc->importLibrary(variant == ClosureVariant::Lobes ? impl.libraries : impl.referenceLibraries);
        mx::TypedElementPtr renderable;
        if (!element.empty()) {
            renderable = doc->getDescendant(element) ? doc->getDescendant(element)->asA<mx::TypedElement>() : nullptr;
        } else {
            const auto found = mx::findRenderableElements(doc);
            if (!found.empty()) {
                renderable = found.front();
            }
        }
        if (!renderable) {
            return Error::make(ErrorCode::NotFound, "materials: no renderable element '{}'", element);
        }
        mx::GenContext context(generator);
        context.registerSourceCodeSearchPath(impl.sourcePaths);
        mx::GenOptions& options = context.getOptions();
        options.shaderInterfaceType = mx::SHADER_INTERFACE_COMPLETE;
        options.hwSpecularEnvironmentMethod = mx::SPECULAR_ENVIRONMENT_NONE;
        options.hwTransmissionRenderMethod = mx::TRANSMISSION_OPACITY;
        options.hwMaxActiveLightSources = 0;
        options.fileTextureVerticalFlip = false;
        options.hwDirectionalAlbedoMethod = mx::DIRECTIONAL_ALBEDO_ANALYTIC;
        mx::ShaderPtr shader = generator->generate("lrt_material", renderable, context);
        CompiledMaterial compiled;
        const std::string source = shader->getSourceCode(mx::Stage::PIXEL);
        const std::string hash = hexHash(source);
        const bool lobes = variant == ClosureVariant::Lobes;
        compiled.module = (lobes ? "lrt_mat_" : "lrt_ref_") + hash;
        compiled.function = (lobes ? "lrt_material_" : "lrt_reference_") + hash;
        compiled.variant = variant;
        compiled.source = source;
        for (size_t at = compiled.source.find(kFunctionPlaceholder); at != std::string::npos;
             at = compiled.source.find(kFunctionPlaceholder, at)) {
            compiled.source.replace(at, kFunctionPlaceholder.size(), compiled.function);
        }
        compiled.slots = generator->slots;
        compiled.words = generator->words;
        return compiled;
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::ShaderFailure, "materials: MaterialX could not generate '{}': {}", element,
                           e.what());
    }
}

std::vector<float> MaterialCompiler::parameters(const CompiledMaterial& material, TextureStore& textures,
                                                const std::function<uint32_t(const std::string&)>& slotOf) {
    std::vector<float> words(material.words, 0.0F);
    for (const MaterialSlot& slot : material.slots) {
        switch (slot.kind) {
        case MaterialSlot::Kind::Value:
            for (size_t k = 0; k < slot.value.size() && k < slot.words; ++k) {
                words[slot.offset + k] = slot.value[k];
            }
            break;
        case MaterialSlot::Kind::Texture:
            words[slot.offset] = slot.name.empty() ? 4294967295.0F
                                                   : static_cast<float>(textures.request(slot.name, slot.space));
            words[slot.offset + 1] = static_cast<float>(textures.sampler(slot.wrapS, slot.wrapT, slot.filter));
            break;
        case MaterialSlot::Kind::Primvar:
            words[slot.offset] = static_cast<float>(slotOf(slot.name));
            break;
        }
    }
    return words;
}

}   // namespace lrt::material
