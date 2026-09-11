// Copyright (c) 2026 openFXplayer contributors.
//
// A bundle with the right ABI number and the wrong standard library.
//
// This is the case the version number cannot catch, and the reason the build
// tag exists at all: the interface is identical, the vtable layout is
// identical, and a std::string crossing the call means something different on
// each side. It loads, it resolves, and it corrupts memory somewhere else
// entirely -- so it has to be refused here.
#include "aofx/Effect.h"
#include "aofx/Version.h"

#if defined(_WIN32)
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
TEST_EXPORT int AofxGetAbiVersion() { return aofx::kAbiVersion; }
TEST_EXPORT const char* AofxGetBuildTag() {
    return "gcc-1.0 some-other-stdlib";
}
TEST_EXPORT int AofxGetEffectCount() { return 0; }
TEST_EXPORT aofx::Effect* AofxGetEffect(int) { return nullptr; }
}
