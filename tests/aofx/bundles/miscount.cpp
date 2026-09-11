// Copyright (c) 2026 lucabRTrender contributors.
//
// A plugin with the bug openFXplayer's D30 describes: it runs a kernel that
// declares two buffers with one. Before kernel trailers that rendered
// something; now the host refuses the run and says which kernel and why.
#include <cstdint>

#include "aofx/Entry.h"
#include "aofx_kernels_miscount.h"

namespace {

struct Uniforms {
    uint32_t width = 0;
    uint32_t height = 0;
};

class Miscount final : public aofx::Effect {
public:
    void describe(aofx::EffectDesc& into) override {
        into.identifier = "tv.mediapro.aofx.test.miscount";
        into.label = "Miscount";
        into.inputs.push_back(aofx::ClipDesc{"Source", "Source", false, false, false});
    }
    [[nodiscard]] std::vector<aofx::KernelDesc> kernels() const override {
        return {aofx::KernelDesc{"lrt.test.miscount", "miscountMain", k_miscount, k_miscountBytes}};
    }
    [[nodiscard]] bool process(const aofx::RenderRequest& request) override {
        const aofx::OutputPlane* out = request.output("Color");
        Uniforms uniforms{static_cast<uint32_t>(out->buffer.width),
                          static_cast<uint32_t>(out->buffer.height)};
        const aofx::KernelId kernel = request.gpu->load("lrt.test.miscount");
        return request.gpu->run(kernel, aofx::Grid{uniforms.width, uniforms.height, 1},
                                {out->buffer}, &uniforms, sizeof(uniforms));
    }
};

}   // namespace

AOFX_EXPORT_EFFECTS(Miscount)
