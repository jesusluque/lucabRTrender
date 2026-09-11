// Copyright (c) 2026 lucabRTrender contributors.
//
// The host side of `aofx::Gpu` for one render: what a plugin can do to the
// machine, and nothing more.
//
// Ported from openFXplayer's sdk_host::EffectRunner for the verbs a renderer
// host needs -- kernels, scratch, keep, read, publish. The media verbs (clip,
// decode, recorder, audio) and the model verbs (model, infer) answer "not
// available in this host" and say why: they are openFXplayer's GStreamer and
// TensorRT/worker stacks, which this engine does not carry.
//
// Must run on the context's GPU thread (inside gpu_host::Context::run).
#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "aofx/Effect.h"

namespace lrt::gpu_host {
class Context;
}

namespace lrt::aofx_host {

class EffectRunner final : public aofx::Gpu {
public:
    explicit EffectRunner(gpu_host::Context& context);
    ~EffectRunner() override;
    EffectRunner(const EffectRunner&) = delete;
    EffectRunner& operator=(const EffectRunner&) = delete;

    [[nodiscard]] aofx::KernelId load(const std::string& name) override;
    bool run(aofx::KernelId kernel, aofx::Grid grid, const std::vector<aofx::Buffer>& buffers,
             const void* uniforms, size_t uniformBytes) override;
    [[nodiscard]] aofx::Buffer scratch(int width, int height) override;
    [[nodiscard]] aofx::Buffer keep(const std::string& key, const void* data,
                                    size_t bytes) override;
    void drop(const std::string& key) override;
    [[nodiscard]] bool read(const aofx::Buffer& buffer, void* into, size_t bytes) override;
    void publish(const std::string& instance, const char* key, double value) override;

    [[nodiscard]] aofx::ClipId clip(const std::string& path) override;
    [[nodiscard]] aofx::ClipInfo clipInfo(aofx::ClipId) const override;
    [[nodiscard]] bool decode(aofx::ClipId, int frame, const aofx::Buffer& out) override;
    [[nodiscard]] bool decodeNext(aofx::ClipId, int direction, const aofx::Buffer& out,
                                  bool& atStart, bool& atEnd) override;
    [[nodiscard]] aofx::ModelId model(const std::string& name) override;
    [[nodiscard]] aofx::ModelIo modelInput(aofx::ModelId, int index) const override;
    [[nodiscard]] aofx::ModelIo modelOutput(aofx::ModelId, int index) const override;
    bool infer(aofx::ModelId, const std::vector<aofx::Buffer>& inputs,
               const std::vector<aofx::Buffer>& outputs) override;
    [[nodiscard]] bool inferred(aofx::ModelId) override;
    bool inferShaped(aofx::ModelId, const std::vector<aofx::Buffer>& inputs,
                     const std::vector<aofx::ModelIo>& inputShapes,
                     const std::vector<aofx::Buffer>& outputs,
                     const std::vector<aofx::ModelIo>& outputShapes) override;

    /// Why the last load or run failed, kept for the caller that turns a
    /// plugin's `false` into a message (openFXplayer's lastComplaint).
    [[nodiscard]] static std::string lastComplaint();
    static void complain(std::string what);

    [[nodiscard]] static std::map<std::string, std::map<std::string, double>> publishedState();
    [[nodiscard]] uint64_t dispatches() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace lrt::aofx_host
