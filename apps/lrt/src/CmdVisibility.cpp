// Copyright (c) 2026 lucabRTrender contributors.
//
// `lrt visibility`: bake what a skinned cloud casts on the space around it,
// by part, into the cloud's own file. Once baked, a frame shadows the cloud
// with a table read a part a light a gaussian and no ray at all -- and the
// fields move with the joints, so a bird's wing keeps shadowing its body
// however it is posed (docs/decisions.md: the per-part visibility).
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "Commands.h"
#include "lrt/core/Log.h"
#include "lrt/technique/SplatVisibility.h"
#include "lrt/usd/Export.h"
#include "lrt/usd/MeshStage.h"
#include "lrt/usd/StageRenderer.h"

namespace lrt::cli {
namespace {

struct Options {
    std::string stage;                          ///< the cloud's stage
    std::string prim = "/World/Splats";
    std::string skeletonStage;                  ///< the source stage, for the joint hierarchy
    std::string skeletonPrim;                   ///< the Skeleton prim on it (default: the cloud's own record)
    std::string output;                         ///< written here instead of in place
    uint32_t    parts = 12;
    uint32_t    minJoints = 6;
    uint32_t    grid = 24;
    uint32_t    octave = 16;
    float       cut = 1.0e-3F;
    double      time = 0.0;
};

int run(const Options& o) {
    // 1. The rig's hierarchy, from the source stage: which joint is under which.
    auto source = usd::MeshStage::open(o.skeletonStage);
    if (!source) {
        std::fprintf(stderr, "visibility: %s\n", source.error().toString().c_str());
        return 1;
    }
    auto joints = source->joints(o.skeletonPrim);
    if (!joints) {
        std::fprintf(stderr, "visibility: %s\n", joints.error().toString().c_str());
        return 1;
    }
    const technique::VisibilityParts parts = technique::partitionJoints(*joints, o.parts, o.minJoints);
    std::printf("visibility: %zu joints in %zu parts:", joints->size(), parts.partJoint.size());
    for (uint32_t head : parts.partJoint) {
        const std::string& path = (*joints)[head];
        const size_t slash = path.rfind('/');
        std::printf(" %s", (slash == std::string::npos ? path : path.substr(slash + 1)).c_str());
    }
    std::printf("\n");

    // 2. The cloud on the device, and the bake.
    auto renderer = usd::StageRenderer::open(o.stage);
    if (!renderer) {
        std::fprintf(stderr, "visibility: %s\n", renderer.error().toString().c_str());
        return 1;
    }
    technique::VisibilityBakeOptions options;
    options.grid = o.grid;
    options.octave = o.octave;
    options.cut = o.cut;
    auto baked = (*renderer)->bakeVisibility(o.prim, parts, options, o.time);
    if (!baked) {
        std::fprintf(stderr, "visibility: %s\n", baked.error().toString().c_str());
        return 1;
    }
    const double megabytes = static_cast<double>(baked->texels.size()) * 4.0 / (1024.0 * 1024.0);
    std::printf("visibility: %u parts, grid %u, octave %u: %zu words, %.1f MB\n", baked->partCount, o.grid,
                o.octave, baked->texels.size(), megabytes);

    // 3. Into the file: in place, or a copy first.
    std::filesystem::path target = o.stage;
    if (!o.output.empty()) {
        target = o.output;
        std::error_code ec;
        std::filesystem::copy_file(o.stage, target, std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            std::fprintf(stderr, "visibility: cannot copy '%s' to '%s': %s\n", o.stage.c_str(),
                         target.string().c_str(), ec.message().c_str());
            return 1;
        }
    }
    auto written = usd::writeVisibility(target, o.prim, baked->parts, baked->texels, baked->partOf, baked->ambient);
    if (!written) {
        std::fprintf(stderr, "visibility: %s\n", written.error().toString().c_str());
        return 1;
    }
    std::printf("visibility: wrote %s\n", target.string().c_str());
    return 0;
}

}   // namespace

void addVisibility(CLI::App& app) {
    auto o = std::make_shared<Options>();
    CLI::App* cmd = app.add_subcommand(
        "visibility", "bake what a skinned cloud casts on the space around it, by part, into its file");
    cmd->add_option("stage", o->stage, "the cloud's .usd / .usda / .usdc")->required();
    cmd->add_option("--prim", o->prim, "the ParticleField prim");
    cmd->add_option("--skeleton-stage", o->skeletonStage, "the source stage, for the joint hierarchy")
        ->required();
    cmd->add_option("--skeleton-prim", o->skeletonPrim, "the Skeleton prim on that stage")->required();
    cmd->add_option("-o,--output", o->output, "write here instead of in place");
    cmd->add_option("--parts", o->parts, "how many parts to split the rig into");
    cmd->add_option("--min-joints", o->minJoints, "a subtree smaller than this stays with its parent's part");
    cmd->add_option("--grid", o->grid, "probes along each axis of a part's box");
    cmd->add_option("--octave", o->octave,
                    "directions along each side of the octahedral map: 16 is about 11 degrees a texel, 32 "
                    "about 6; 8 is too coarse to show a wing's edge");
    cmd->add_option("--cut", o->cut, "transmittance under which a bake ray stops");
    cmd->add_option("--time", o->time, "the instant the stage is committed at");
    cmd->callback([o] { std::exit(run(*o)); });
}

}   // namespace lrt::cli
