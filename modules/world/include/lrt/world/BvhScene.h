// Copyright (c) 2026 lucabRTrender contributors.
//
// A GpuScene as compute BVHs, for devices without ray tracing hardware (and
// as a second route to check the first): a Karras LBVH per mesh over its
// triangles in object space, rebuilt when the pools are, and one per frame
// over the instances in view space -- the splat ray tracer's build kernels
// (lrt/rt/bvh_hierarchy, bvh_refit) with scene leaves (lrt/world/bvh_scene).
#pragma once

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/algo/RadixSort.h"
#include "lrt/world/GpuScene.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::world {

class BvhScene {
public:
    [[nodiscard]] static Result<BvhScene> create(gpu::ShaderLibrary& library);

    /// Each mesh's LBVH is built when the pools were repacked and refit --
    /// leaf boxes recomputed, internal boxes settled over the same tree --
    /// when it was deformed in place (GpuScene::meshRevision); the top level
    /// is built every call. `refit` false leaves a deformed mesh's tree as it
    /// was, for a check that the refit matters.
    [[nodiscard]] Result<void> build(const GpuScene& scene, bool refit = true);

    [[nodiscard]] const gpu::Buffer& topBoxes() const noexcept { return topBoxes_; }
    [[nodiscard]] const gpu::Buffer& topChildren() const noexcept { return topChildren_; }
    [[nodiscard]] const gpu::Buffer& topLeaves() const noexcept { return topLeaves_; }
    [[nodiscard]] const gpu::Buffer& meshBoxes() const noexcept { return meshBoxes_; }
    [[nodiscard]] const gpu::Buffer& meshChildren() const noexcept { return meshChildren_; }
    [[nodiscard]] const gpu::Buffer& meshLeaves() const noexcept { return meshLeaves_; }

private:
    struct Build {
        uint32_t count = 0;
        uint32_t nodeBase = 0;
        uint32_t leafBase = 0;
        float    lo[3] = {0, 0, 0};
        float    hi[3] = {0, 0, 0};
    };
    /// Hierarchy from sorted codes, then refit until nothing changes.
    [[nodiscard]] Result<void> hierarchy(const Build& build, gpu::SortBuffers& sorting, const gpu::Buffer& leafBoxes,
                                         gpu::Buffer& boxes, gpu::Buffer& children, gpu::Buffer& leaves);
    /// Refit passes over an existing tree until no box changes.
    [[nodiscard]] Result<void> settle(const Build& build, const gpu::Buffer& leafBoxes, gpu::Buffer& boxes,
                                      const gpu::Buffer& children, const gpu::Buffer& leaves);
    std::vector<Build>    meshBuilds_;      ///< per mesh: what its tree was built with
    std::vector<uint64_t> meshRevisions_;   ///< per mesh: the revision its tree holds

    gpu::Device*       device_ = nullptr;
    gpu::RadixSort     sort_;
    gpu::ComputeKernel triangleLeaves_, instanceLeaves_, instanceCodes_, hierarchy_, refit_, count_;
    gpu::ComputeKernel boundsChunks_, boundsReduce_;
    uint64_t           generation_ = ~uint64_t{0};
    gpu::Buffer        topBoxes_, topChildren_, topLeaves_, meshBoxes_, meshChildren_, meshLeaves_;
};

}   // namespace lrt::world
