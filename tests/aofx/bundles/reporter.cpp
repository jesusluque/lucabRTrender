// Copyright (c) 2026 lucabRTrender contributors.
//
// A bundle that says what it was handed (ABI 24): the project's format, by
// publishing it, and -- told to fail -- why, in `RenderRequest::complaint`.
// The two things the host fills that an effect cannot find out for itself.
#include "aofx/Entry.h"

namespace {

class Reporter final : public aofx::Effect {
public:
    void describe(aofx::EffectDesc& into) override {
        into.identifier = "tv.mediapro.aofx.test.reporter";
        into.label = "Reporter";
        into.grouping = "Test";
        into.description = "Publishes the project's format; fails, with a reason, when told to.";

        aofx::ParamDesc fail;
        fail.name = "fail";
        fail.label = "Fail";
        fail.type = aofx::ParamType::Boolean;
        fail.defaults = {0.0};
        into.params.push_back(fail);

        into.outputs.push_back(aofx::PlaneDesc{"Color", "Colour", {"R", "G", "B", "A"}});
    }

    [[nodiscard]] std::vector<aofx::KernelDesc> kernels() const override { return {}; }

    [[nodiscard]] bool process(const aofx::RenderRequest& request) override {
        request.gpu->publish(request.instance, "projectWidth", request.projectWidth);
        request.gpu->publish(request.instance, "projectHeight", request.projectHeight);
        if (request.number("fail", 0.0) >= 0.5) {
            request.complaint = "the reporter was told to fail";
            return false;
        }
        return true;
    }
};

}   // namespace

AOFX_EXPORT_EFFECTS(Reporter)
