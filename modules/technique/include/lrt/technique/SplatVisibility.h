// Copyright (c) 2026 lucabRTrender contributors.
//
// WHAT A CLOUD CASTS ON THE SPACE AROUND IT, BAKED BY PART, READ BY PRODUCT.
//
// A skinned cloud cannot carry one baked visibility, because the wing moves
// and takes its shadow with it. But one part of a body changes shape very
// little between poses, so each part gets a field of its own -- over a grid
// of probes, in every direction, how much of a ray leaving that probe the
// part's gaussians stop -- baked in the pose the cloud was bound in. At
// render a gaussian asks each part in that part's own current frame and
// multiplies the answers. Once baked, a frame costs a table read a part a
// light a gaussian and no ray at all, which is what lets an animated cloud
// shadow itself at interactive rates; what it gives up is that a part is
// taken as rigid, and that parts occlude independently.
//
// The kernels are splat_visibility.slang's; this is the host that runs the
// bake once and the read every frame.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/algo/PrefixSum.h"
#include "lrt/render/GaussianRayTracer.h"
#include "lrt/scene/GpuClouds.h"

namespace lrt::technique {

struct VisibilityBakeOptions {
    uint32_t grid = 24;      ///< probes along each axis of a part's box
    uint32_t octave = 8;     ///< directions along each side of the octahedral map
    float    cut = 1.0e-3F;  ///< transmittance under which a bake ray stops
    /// The box every part's field covers, as a margin on the cloud's own,
    /// since a part occludes points anywhere the body can reach.
    float    margin = 0.05F;
};

/// How the cloud is partitioned: the part each joint belongs to, and for each
/// part the joint whose skinning transform carries it. Host bookkeeping over
/// a joint hierarchy, not arithmetic on the cloud.
struct VisibilityParts {
    std::vector<uint32_t> jointToPart;   ///< a part index a joint
    std::vector<uint32_t> partJoint;     ///< a joint a part
};

/// The factors the read writes: one float a gaussian a light, what reaches it.
struct VisibilityFactorsJob {
    const scene::GpuSplats* cloud = nullptr;      ///< carries the fields
    const gpu::Buffer*      positions = nullptr;  ///< posed positions, or the cloud's own
    const gpu::Buffer*      skinningXforms = nullptr;   ///< null: the bind pose is the pose
    const gpu::Buffer*      lights = nullptr;
    uint32_t                lightCount = 0;
    uint32_t                base = 0;             ///< first slot in `factors`
    uint64_t                categories = 0;
    std::array<float, 12>   objectToWorld{};
    /// The cloud's space into the skeleton's bind space, rows of a 3x4 (the
    /// `geomBindTransform` the skinner applied); identity for a cloud nothing
    /// moves.
    std::array<float, 12>   geomBind{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
};

/// HOW A RIG BECOMES PARTS: the largest subtrees of the joint hierarchy,
/// split greedily until there are `targetParts`, a subtree smaller than
/// `minJoints` staying with its parent's part. `paths` are the joints as a
/// Skeleton's `joints` attribute holds them ("a/b/c"), in its order, which is
/// the order a cloud's joint indices use. Bookkeeping over names.
[[nodiscard]] VisibilityParts partitionJoints(const std::vector<std::string>& paths, uint32_t targetParts,
                                              uint32_t minJoints = 6);

class SplatVisibility {
public:
    [[nodiscard]] static Result<SplatVisibility> create(gpu::ShaderLibrary& library);

    /// Bakes every part's field into `cloud`'s visibility buffers. `cloud`
    /// is the bind pose; `influences` are `(joint, weight)` pairs, `perSplat`
    /// each. Waits for the device: a bake is not a frame.
    [[nodiscard]] Result<void> bake(scene::GpuSplats& cloud, const gpu::Buffer& influences, uint32_t perSplat,
                                    const VisibilityParts& parts, const VisibilityBakeOptions& options);

    /// Queued into `batch`: `factors[(base + i) * lightCount + k]` for every
    /// gaussian i and light k.
    [[nodiscard]] Result<void> factors(gpu::CommandBatch& batch, const VisibilityFactorsJob& job,
                                       const gpu::Buffer& factors);

    /// The share of the first `count` factors under `cut`, counted on the
    /// device: what a frame's shadowing amounts to, as one number.
    [[nodiscard]] Result<double> shadowedShare(const gpu::Buffer& factors, uint32_t count, float cut = 0.5F);

private:
    gpu::Device*                          device_ = nullptr;
    gpu::ShaderLibrary*                   library_ = nullptr;
    std::optional<gpu::ComputeKernel>     partOf_;
    std::optional<gpu::ComputeKernel>     mark_;
    std::optional<gpu::ComputeKernel>     gather_;
    std::optional<gpu::ComputeKernel>     bake_;
    std::optional<gpu::ComputeKernel>     factors_;
    std::optional<gpu::ComputeKernel>     histogram_;
    std::optional<gpu::ComputeKernel>     ambient_;
    std::optional<gpu::PrefixSum>   prefix_;
    std::optional<render::GaussianRayTracer> tracer_;
};

}   // namespace lrt::technique
