// Copyright (c) 2026 openFXplayer contributors.
//
// A bundle whose entry points are not exported.
//
// Written without AOFX_EXPORT_EFFECTS on purpose: the macro marks them visible
// itself, which is exactly what makes it worth using. This is what a plugin
// looks like when somebody exports the symbols by hand and inherits the
// project's hidden visibility -- the bundle loads, dlsym returns null, and the
// effect is simply absent with nothing said anywhere.
//
// It has happened here before, to the OpenFX blur. The loader has to name the
// likely cause, because "it is not in the list" is the least useful thing a
// person can be told.
#include "aofx/Effect.h"
#include "aofx/Version.h"

extern "C" {
int AofxGetAbiVersion() { return aofx::kAbiVersion; }
const char* AofxGetBuildTag() { return aofx::buildTag(); }
int AofxGetEffectCount() { return 0; }
aofx::Effect* AofxGetEffect(int) { return nullptr; }
}
