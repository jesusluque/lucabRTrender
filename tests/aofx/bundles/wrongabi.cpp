// Copyright (c) 2026 openFXplayer contributors.
//
// A bundle from the future: the same interface, a different ABI number.
//
// Written by hand rather than through AOFX_EXPORT_EFFECTS, because the macro
// reports the truth and the whole point here is to lie. This is what a plugin
// built against a later SDK looks like from the host's side, and the host must
// turn it away before calling through a vtable whose shape it does not know.
#include "aofx/Effect.h"
#include "aofx/Version.h"

#if defined(_WIN32)
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
TEST_EXPORT int AofxGetAbiVersion() { return aofx::kAbiVersion + 1; }
TEST_EXPORT const char* AofxGetBuildTag() { return aofx::buildTag(); }
TEST_EXPORT int AofxGetEffectCount() { return 0; }
TEST_EXPORT aofx::Effect* AofxGetEffect(int) { return nullptr; }
}
