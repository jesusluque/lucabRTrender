// Copyright (c) 2026 lucabRTrender contributors.
//
// Ported from openFXplayer src/sdk_host/src/EffectRegistry.cpp. Comments that
// explain a rule are kept short here; the long form, with the incident behind
// each rule, is in that file.
#include "lrt/aofx/EffectRegistry.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <system_error>

#if defined(_WIN32)
#error "Windows is a later port: LoadLibrary/GetProcAddress go here."
#else
#include <dlfcn.h>
#endif

#include "aofx/Effect.h"
#include "aofx/Version.h"
#include "aofx_kernels_channels.h"
#include "gpe/kernels.h"
#include "lrt/core/Log.h"
#include "lrt/gpu_host/Context.h"

namespace fs = std::filesystem;

namespace lrt::aofx_host {
namespace {

constexpr const char* kArchDir =
#if defined(__APPLE__)
    "MacOS";
#elif defined(_WIN32)
    "Win64";
#else
    "Linux-x86-64";
#endif

constexpr const char* kBundleSuffix = ".aofx.bundle";
constexpr const char* kBinarySuffix = ".aofx";

std::vector<fs::path> splitSearchPath(const char* value) {
    std::vector<fs::path> out;
    if (value == nullptr) {
        return out;
    }
#if defined(_WIN32)
    const char separator = ';';
#else
    const char separator = ':';
#endif
    std::string all(value);
    size_t start = 0;
    while (start <= all.size()) {
        const size_t at = all.find(separator, start);
        const std::string piece =
            all.substr(start, at == std::string::npos ? std::string::npos : at - start);
        if (!piece.empty()) {
            out.emplace_back(piece);
        }
        if (at == std::string::npos) {
            break;
        }
        start = at + 1;
    }
    return out;
}

fs::path systemPath() {
#if defined(__APPLE__)
    return "/Library/AOFX/Plugins";
#elif defined(_WIN32)
    return "C:/Program Files/Common Files/AOFX/Plugins";
#else
    return "/usr/AOFX/Plugins";
#endif
}

struct Bundle {
    fs::path                   path;
    void*                      handle = nullptr;
    std::vector<aofx::Effect*> effects;
    std::vector<std::string>   kernelNames;

    ~Bundle() {
        // Kernels before the library: a registered name pointing into an
        // unmapped page is a dispatch reading freed memory inside a driver.
        for (const std::string& name : kernelNames) {
            gpe::unregisterKernel(name);
        }
        if (handle != nullptr) {
            dlclose(handle);
        }
    }
};

struct Entry {
    int (*abiVersion)() = nullptr;
    const char* (*buildTag)() = nullptr;
    int (*effectCount)() = nullptr;
    aofx::Effect* (*effect)(int) = nullptr;
};

bool resolve(void* handle, Entry& into, std::string& why) {
    into.abiVersion = reinterpret_cast<int (*)()>(dlsym(handle, "AofxGetAbiVersion"));
    into.buildTag = reinterpret_cast<const char* (*)()>(dlsym(handle, "AofxGetBuildTag"));
    into.effectCount = reinterpret_cast<int (*)()>(dlsym(handle, "AofxGetEffectCount"));
    into.effect = reinterpret_cast<aofx::Effect* (*)(int)>(dlsym(handle, "AofxGetEffect"));
    if (into.abiVersion == nullptr || into.buildTag == nullptr ||
        into.effectCount == nullptr || into.effect == nullptr) {
        why = "it does not export the AOFX entry points; if this is an AOFX bundle it "
              "was probably built with hidden visibility";
        return false;
    }
    return true;
}

fs::path binaryIn(const fs::path& bundle) {
    std::string stem = bundle.filename().string();
    stem = stem.substr(0, stem.size() - std::strlen(kBundleSuffix));
    return bundle / "Contents" / kArchDir / (stem + kBinarySuffix);
}

}   // namespace

struct EffectRegistry::Impl {
    std::vector<fs::path>                   extraPaths;
    std::vector<std::unique_ptr<Bundle>>    bundles;
    std::vector<BundleReport>               reports;
    std::map<std::string, aofx::Effect*>    byIdentifier;
    std::map<std::string, aofx::EffectDesc> describedBy;
    bool                                    scanned = false;
};

namespace {

void load(EffectRegistry::Impl& impl, const fs::path& bundle) {
    BundleReport report;
    report.path = bundle;

    const fs::path binary = binaryIn(bundle);
    std::error_code ignored;
    if (!fs::exists(binary, ignored)) {
        report.reason = "no binary at " + binary.string();
        impl.reports.push_back(std::move(report));
        return;
    }
    // RTLD_LOCAL: one plugin's statics must not be the next one's.
    void* handle = dlopen(binary.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* why = dlerror();
        report.reason = why == nullptr ? "it would not load" : why;
        impl.reports.push_back(std::move(report));
        return;
    }
    const auto refuse = [&](std::string why) {
        dlclose(handle);
        report.reason = std::move(why);
        impl.reports.push_back(std::move(report));
    };

    Entry entry;
    if (std::string why; !resolve(handle, entry, why)) {
        refuse(std::move(why));
        return;
    }
    if (const int theirs = entry.abiVersion(); theirs != aofx::kAbiVersion) {
        refuse("it was built against AOFX ABI " + std::to_string(theirs) +
               ", and this host speaks " + std::to_string(aofx::kAbiVersion));
        return;
    }
    if (const std::string theirs = entry.buildTag(); theirs != aofx::buildTag()) {
        refuse("it was built with " + theirs + ", and this host with " +
               std::string(aofx::buildTag()) + "; a C++ interface cannot cross that");
        return;
    }

    auto held = std::make_unique<Bundle>();
    held->path = bundle;
    held->handle = handle;

    const int count = entry.effectCount();
    for (int i = 0; i < count; ++i) {
        aofx::Effect* effect = entry.effect(i);
        if (effect == nullptr) {
            continue;
        }
        aofx::EffectDesc desc;
        effect->describe(desc);
        if (desc.identifier.empty()) {
            log::warn("an effect in {} has no identifier and was skipped", bundle.string());
            continue;
        }
        if (impl.byIdentifier.count(desc.identifier) != 0) {
            log::warn("{} offers '{}', which is already loaded; keeping the first",
                      bundle.string(), desc.identifier);
            continue;
        }
        bool kernelsOk = true;
        for (const aofx::KernelDesc& kernel : effect->kernels()) {
            if (!gpe::registerKernel(kernel.name, kernel.entry, kernel.blob, kernel.blobBytes)) {
                log::warn("'{}' brought a kernel '{}' that could not be registered",
                          desc.identifier, kernel.name);
                kernelsOk = false;
                break;
            }
            held->kernelNames.push_back(kernel.name);
        }
        if (!kernelsOk) {
            continue;
        }
        held->effects.push_back(effect);
        impl.byIdentifier.emplace(desc.identifier, effect);
        report.effects.push_back(desc.identifier);
        impl.describedBy.emplace(desc.identifier, std::move(desc));
    }
    report.loaded = true;
    impl.bundles.push_back(std::move(held));
    impl.reports.push_back(std::move(report));
}

}   // namespace

EffectRegistry::EffectRegistry() : impl_(std::make_unique<Impl>()) {}
EffectRegistry::~EffectRegistry() = default;

void EffectRegistry::addSearchPath(fs::path path) { impl_->extraPaths.push_back(std::move(path)); }

std::vector<fs::path> EffectRegistry::searchPaths() const {
    std::vector<fs::path> all = splitSearchPath(std::getenv("AOFX_PLUGIN_PATH"));
    all.push_back(systemPath());
    all.insert(all.end(), impl_->extraPaths.begin(), impl_->extraPaths.end());
    // One directory reached two ways is one directory: compared canonically.
    std::vector<fs::path> out;
    for (const fs::path& one : all) {
        std::error_code ignored;
        fs::path real = fs::weakly_canonical(one, ignored);
        if (real.empty()) {
            real = one;
        }
        const bool seen = std::any_of(out.begin(), out.end(), [&](const fs::path& had) {
            std::error_code also;
            fs::path other = fs::weakly_canonical(had, also);
            return (other.empty() ? had : other) == real;
        });
        if (!seen) {
            out.push_back(one);
        }
    }
    return out;
}

const std::vector<BundleReport>& EffectRegistry::reports() const noexcept {
    return impl_->reports;
}

const std::map<std::string, aofx::EffectDesc>& EffectRegistry::descriptions() const noexcept {
    return impl_->describedBy;
}

void EffectRegistry::scan(gpu_host::Context* context) {
    if (impl_->scanned) {
        return;
    }
    impl_->scanned = true;
    if (context == nullptr || context->compute() == nullptr) {
        log::info("no gpe device; AOFX effects are not available on this machine");
        return;
    }
    // Registering twice (a second registry in one process) is harmless: the
    // name is not gpe's own, so re-registration replaces it with the same blob.
    if (!gpe::registerKernel(kChannelsKernel, "channelsMain", k_channels, k_channelsBytes)) {
        log::error("the channel-restore kernel could not be registered");
    }
    for (const fs::path& root : searchPaths()) {
        std::error_code ignored;
        if (!fs::is_directory(root, ignored)) {
            continue;
        }
        std::vector<fs::path> found;
        for (const fs::directory_entry& entry : fs::directory_iterator(root, ignored)) {
            const std::string name = entry.path().filename().string();
            if (name.size() > std::strlen(kBundleSuffix) &&
                name.compare(name.size() - std::strlen(kBundleSuffix), std::string::npos,
                             kBundleSuffix) == 0) {
                found.push_back(entry.path());
            }
        }
        // Sorted: directory order is the filesystem's, and which of two
        // bundles offering one identifier wins should not depend on it.
        std::sort(found.begin(), found.end());
        for (const fs::path& bundle : found) {
            load(*impl_, bundle);
        }
    }
}

aofx::Effect* EffectRegistry::find(const std::string& identifier) const {
    const auto found = impl_->byIdentifier.find(identifier);
    return found == impl_->byIdentifier.end() ? nullptr : found->second;
}

}   // namespace lrt::aofx_host
