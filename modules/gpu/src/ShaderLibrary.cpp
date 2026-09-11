// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/ShaderLibrary.h"

#include <string_view>

#include "lrt/core/Log.h"
#include "lrt/gpu/Device.h"

namespace lrt::gpu {
namespace {

std::string diagnostics(slang::IBlob* blob) {
    if (blob == nullptr || blob->getBufferSize() == 0) {
        return {};
    }
    return std::string(static_cast<const char*>(blob->getBufferPointer()),
                       blob->getBufferSize());
}

}   // namespace

ShaderLibrary::ShaderLibrary(std::shared_ptr<Device> device) : device_(std::move(device)) {}

Result<std::shared_ptr<const Program>> ShaderLibrary::load(
    const std::string& module, const std::vector<std::string>& entries) {
    std::string key = module;
    for (const std::string& entry : entries) {
        key += "|" + entry;
    }
    const std::lock_guard<std::mutex> held(guard_);
    if (auto found = programs_.find(key); found != programs_.end()) {
        return found->second;
    }

    slang::ISession* session = device_->slangSession();
    rhi::ComPtr<slang::IBlob> diag;
    slang::IModule* loaded = session->loadModule(module.c_str(), diag.writeRef());
    if (loaded == nullptr) {
        return Error::make(ErrorCode::ShaderFailure, "cannot load Slang module '{}':\n{}",
                           module, diagnostics(diag.get()));
    }

    std::vector<rhi::ComPtr<slang::IEntryPoint>> entryPoints;
    std::vector<slang::IComponentType*> components{loaded};
    for (const std::string& entry : entries) {
        rhi::ComPtr<slang::IEntryPoint> found;
        if (SLANG_FAILED(loaded->findEntryPointByName(entry.c_str(), found.writeRef()))) {
            return Error::make(ErrorCode::ShaderFailure, "no entry point '{}' in '{}'", entry,
                               module);
        }
        components.push_back(found.get());
        entryPoints.push_back(found);
    }

    rhi::ComPtr<slang::IComponentType> composed;
    if (SLANG_FAILED(session->createCompositeComponentType(
            components.data(), static_cast<SlangInt>(components.size()), composed.writeRef(),
            diag.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot compose '{}':\n{}", module,
                           diagnostics(diag.get()));
    }
    auto program = std::make_shared<Program>();
    if (SLANG_FAILED(composed->link(program->linked.writeRef(), diag.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot link '{}':\n{}", module,
                           diagnostics(diag.get()));
    }

    if (slang::ProgramLayout* layout = program->linked->getLayout();
        layout != nullptr && layout->getEntryPointCount() > 0) {
        SlangUInt sizes[3] = {1, 1, 1};
        layout->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, sizes);
        for (int i = 0; i < 3; ++i) {
            program->threadGroup[static_cast<size_t>(i)] =
                static_cast<uint32_t>(sizes[i] == 0 ? 1 : sizes[i]);
        }
    }

    rhi::ShaderProgramDesc desc;
    desc.slangGlobalScope = program->linked.get();
    if (SLANG_FAILED(device_->rhi()->createShaderProgram(desc, program->program.writeRef(),
                                                        diag.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot compile '{}' for {}:\n{}", module,
                           toString(device_->backend()), diagnostics(diag.get()));
    }
    if (std::string warnings = diagnostics(diag.get()); !warnings.empty()) {
        log::warn("slang '{}': {}", module, warnings);
    }
    programs_.emplace(key, program);
    return std::shared_ptr<const Program>(program);
}

}   // namespace lrt::gpu
