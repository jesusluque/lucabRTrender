// Copyright (c) 2026 lucabRTrender contributors.
//
// Finding AOFX bundles, loading them, and refusing the ones that would break.
//
// Ported from openFXplayer's sdk_host::EffectRegistry. The rules are the same
// and so are the reasons (see that header): the ABI number must match exactly,
// the toolchain tag must match exactly, the four entry points must be visible,
// and every refusal is kept with its reason. What is left out is the
// translation into OpenFX's vocabulary, which only openFXplayer's panels read;
// this host hands out `aofx::EffectDesc` as it is.
#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "aofx/Descriptor.h"

namespace aofx {
class Effect;
}

namespace lrt::gpu_host {
class Context;
}

namespace lrt::aofx_host {

struct BundleReport {
    std::filesystem::path    path;
    bool                     loaded = false;
    std::string              reason;
    std::vector<std::string> effects;
};

class EffectRegistry {
public:
    EffectRegistry();
    ~EffectRegistry();
    EffectRegistry(const EffectRegistry&) = delete;
    EffectRegistry& operator=(const EffectRegistry&) = delete;

    /// $AOFX_PLUGIN_PATH, then the system location, then these.
    void addSearchPath(std::filesystem::path path);
    [[nodiscard]] std::vector<std::filesystem::path> searchPaths() const;

    /// Loads everything on the search path. Null context (no gpe device) loads
    /// nothing: an effect is a kernel, and a kernel needs a device.
    void scan(gpu_host::Context* context);

    [[nodiscard]] const std::vector<BundleReport>& reports() const noexcept;
    [[nodiscard]] const std::map<std::string, aofx::EffectDesc>& descriptions() const noexcept;
    [[nodiscard]] aofx::Effect* find(const std::string& identifier) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

/// The host's own kernel that puts back the channels an effect was not applied
/// to, and the four switches every effect gets (openFXplayer's Channels.h).
inline constexpr const char* kChannelsKernel = "tv.mediapro.aofx.host.channels";
inline constexpr const char* kChannelRedParam = "aofx.channel.r";
inline constexpr const char* kChannelGreenParam = "aofx.channel.g";
inline constexpr const char* kChannelBlueParam = "aofx.channel.b";
inline constexpr const char* kChannelAlphaParam = "aofx.channel.a";

}   // namespace lrt::aofx_host
