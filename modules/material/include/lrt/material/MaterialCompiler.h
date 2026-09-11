// Copyright (c) 2026 lucabRTrender contributors.
//
// MaterialX graphs into Slang the engine shades with. A generator derived from
// MaterialX's own Slang generator emits one function per material structure:
// every node keeps its genslang/genglsl implementation except the BSDF nodes,
// which push lobes (lrt/material/mx), and the surface node, which leaves the
// lobes and emission where the shading kernels read them. Inputs are not baked
// in: every one is read from a float blob, so materials that differ only in
// values share one compiled function.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/material/TextureStore.h"

namespace lrt::material {

/// One input of a compiled material, as it sits in the blob.
struct MaterialSlot {
    enum class Kind : uint8_t {
        Value,     ///< `words` floats: scalars, vectors, colours, matrices (row major), ints and bools as floats
        Texture,   ///< two words: texture id, sampler slot
        Primvar,   ///< one word: the scene slot of `name`
    };
    Kind               kind = Kind::Value;
    std::string        variable;   ///< in the generated code
    uint32_t           offset = 0;
    uint32_t           words = 0;
    std::vector<float> value;      ///< Value: the document's
    std::string        name;       ///< Texture: the file; Primvar: the primvar
    ColourSpace        space = ColourSpace::Auto;
    Wrap               wrapS = Wrap::Repeat;
    Wrap               wrapT = Wrap::Repeat;
    Filter             filter = Filter::Linear;
};

/// What a graph's BSDF nodes become.
enum class ClosureVariant : uint8_t {
    Lobes,               ///< the engine's: lobes on a stack a path tracer samples
    GenglslReference,    ///< MaterialX's own genglsl responses for one light direction, to check the lobes against
};

struct CompiledMaterial {
    std::string               module;     ///< "lrt_mat_" + a hash of the source: equal structures, one module
    std::string               function;   ///< its public entry: void function(MaterialInputs, uint blobOffset)
    ClosureVariant            variant = ClosureVariant::Lobes;
    std::string               source;
    std::vector<MaterialSlot> slots;
    uint32_t                  words = 0;
};

class MaterialCompiler {
public:
    /// `searchPaths`: where MaterialX's libraries are (a folder holding
    /// "libraries"), then the engine's shader directories.
    [[nodiscard]] static Result<std::unique_ptr<MaterialCompiler>> create(
        const std::vector<std::filesystem::path>& materialxRoots, const std::vector<std::filesystem::path>& shaderPaths);
    /// The same with libraries already loaded (hdMtlx's HdMtlxStdLibraries(), a
    /// MaterialX::DocumentPtr) and the folders their source files are found in.
    [[nodiscard]] static Result<std::unique_ptr<MaterialCompiler>> create(
        const std::shared_ptr<void>& libraries, const std::vector<std::filesystem::path>& librarySearchPaths,
        const std::vector<std::filesystem::path>& shaderPaths);
    ~MaterialCompiler();

    /// The renderable element `element` (a material or shader node) of a
    /// MaterialX document given as XML, the standard libraries imported.
    [[nodiscard]] Result<CompiledMaterial> compileXml(const std::string& xml, const std::string& element = {},
                                                      ClosureVariant variant = ClosureVariant::Lobes);

    /// The same for a document already built (by hdMtlx): a MaterialX::DocumentPtr.
    [[nodiscard]] Result<CompiledMaterial> compileDocument(const std::shared_ptr<void>& document,
                                                           const std::string& element = {},
                                                           ClosureVariant variant = ClosureVariant::Lobes);

    /// The blob words for `material`: its values, its textures requested from
    /// `textures`, its primvars' scene slots from `slotOf`.
    [[nodiscard]] static std::vector<float> parameters(const CompiledMaterial& material, TextureStore& textures,
                                                       const std::function<uint32_t(const std::string&)>& slotOf);

    /// The standard libraries' document (MaterialX::DocumentPtr), for building documents against.
    [[nodiscard]] std::shared_ptr<void> libraries() const;

private:
    MaterialCompiler() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace lrt::material
