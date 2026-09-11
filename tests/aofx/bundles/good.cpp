// Copyright (c) 2026 openFXplayer contributors.
//
// A bundle that is exactly right, so the loader has something to succeed on.
// Without one, every assertion about refusal could pass by refusing everything.
#include "aofx/Entry.h"

namespace {

class Nothing final : public aofx::Effect {
public:
    void describe(aofx::EffectDesc& into) override {
        into.identifier = "tv.mediapro.aofx.test.nothing";
        into.label = "Nothing";
        into.grouping = "Test";
        into.description = "Loads, and does nothing else.";
        into.inputs.push_back(aofx::ClipDesc{"Source", "Source", false, false, false});

        aofx::ParamDesc amount;
        amount.name = "amount";
        amount.label = "Amount";
        amount.hint = "How much nothing.";
        amount.type = aofx::ParamType::Double;
        amount.dimension = 2;
        amount.defaults = {1.0, 2.0};
        into.params.push_back(amount);

        into.outputs.push_back(
            aofx::PlaneDesc{"Color", "Colour", {"R", "G", "B", "A"}});
        into.outputs.push_back(aofx::PlaneDesc{"Depth", "Depth", {"Z"}});
    }

    [[nodiscard]] std::vector<aofx::KernelDesc> kernels() const override {
        return {};
    }

    [[nodiscard]] bool process(const aofx::RenderRequest&) override { return true; }
};

}   // namespace

AOFX_EXPORT_EFFECTS(Nothing)
