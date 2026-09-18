#!/usr/bin/env bash
# Fetches the Khronos Fox and turns it into a USD stage with its rig, for
# `lrt mesh2splat --skinned` to convert.
#
# Why a script and not a checked-in asset: the model is CC0 but the rig and
# the glTF conversion are CC-BY, and the engine's demonstration assets live in
# ~/tools/assets beside the chess set rather than in the repository.
#
# Why Blender: nothing else on this machine reads glTF skinning. `guc` -- the
# glTF to USD converter this project would otherwise reach for -- says plainly
# that "all glTF features with the exception of animation and skinning are
# implemented", and animation and skinning are the whole point here. Blender
# is a dependency of this asset and of nothing else: no build, no test and no
# part of the engine needs it.
#
#   Fox: model CC0 by PixelMannen; rig and animation CC-BY 4.0 by tomkranis;
#   glTF conversion CC-BY 4.0 by @AsoboStudio and @scurest.
#   https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox
#
# Usage: scripts/fetch-fox.sh [directory]   (default ~/tools/assets/Fox)
set -euo pipefail

DEST="${1:-$HOME/tools/assets/Fox}"
BLENDER="${LRT_BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
BASE="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Fox"

mkdir -p "$DEST"
for file in glTF-Binary/Fox.glb LICENSE.md README.md; do
    out="$DEST/$(basename "$file")"
    if [[ ! -f "$out" ]]; then
        echo "fetching $(basename "$file")"
        curl -L --fail -o "$out" "$BASE/$file"
    fi
done

if [[ ! -x "$BLENDER" ]]; then
    echo "no Blender at $BLENDER: set LRT_BLENDER, or 'brew install --cask blender'" >&2
    exit 1
fi

# Blender's USD exporter writes the armature as UsdSkel, which is what the
# conversion reads. The clips are baked into one SkelAnimation over the
# scene's frame range, and only the first is active: the glTF has three
# (Survey, Walk, Run) and they drive the same joints, so only one may play.
"$BLENDER" --background --factory-startup --python-expr "
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='$DEST/Fox.glb')
scene = bpy.context.scene
action = None
for candidate in bpy.data.actions:
    action = action or candidate
if action is not None:
    scene.frame_start = int(action.frame_range[0])
    scene.frame_end = int(action.frame_range[1])
print('lrt: frames %d..%d' % (scene.frame_start, scene.frame_end))
bpy.ops.wm.usd_export(filepath='$DEST/Fox.usdc',
                      export_animation=True,
                      export_armatures=True,
                      export_materials=True,
                      export_textures_mode='NEW',
                      generate_preview_surface=True,
                      relative_paths=False)
" 2>&1 | grep -vE '^(Blender|Read prefs|found bundled)' || true

if [[ ! -f "$DEST/Fox.usdc" ]]; then
    echo "the export produced no Fox.usdc" >&2
    exit 1
fi
echo "wrote $DEST/Fox.usdc"
