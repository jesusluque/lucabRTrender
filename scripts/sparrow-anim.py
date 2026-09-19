import bpy, json, re, sys, os
from mathutils import Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
fbx, clip, skel_json, out = argv[0], argv[1], argv[2], argv[3]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)

arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
act = bpy.data.actions['Bird|Bird|' + clip]
ad = arm.animation_data or arm.animation_data_create()
ad.action = act
if hasattr(ad, 'action_slot') and len(act.slots):
    ad.action_slot = act.slots[0]

skel = json.load(open(skel_json))
joints = skel['joints']

def as_matrix(nums):
    # USD writes row-major with vectors on the left; mathutils wants the transpose.
    m = Matrix([nums[0:4], nums[4:8], nums[8:12], nums[12:16]])
    m.transpose()
    return m

U_bind = [as_matrix(r) for r in skel['bind']]

# USD joint names are the Blender bone names with every character USD does not
# allow in an identifier turned into an underscore -- `Spine.001_Pelvis` is
# written `Spine_001_Pelvis`. Map back rather than look the sanitised name up.
def sanitize(s):
    s = re.sub(r'[^A-Za-z0-9_]', '_', s)
    return '_' + s if s and s[0].isdigit() else s

bones = arm.data.bones
blender_name = {sanitize(b.name): b.name for b in bones}
assert len(blender_name) == len(bones), "two bones sanitise to one joint"
names = [blender_name[j.split('/')[-1]] for j in joints]
index = {n: i for i, n in enumerate(names)}
parent_of = [index[bones[n].parent.name] if bones[n].parent else -1 for n in names]

# WHAT HAS TO BE TRUE IS THE SKINNING MATRIX, NOT THE JOINT.
#
# UsdSkel deforms a point by `W(t) . bindTransform^-1`; Blender deforms it by
# `pose.matrix . matrix_local^-1`. So ask for the world transform that makes
# those two equal and read the joint-local samples off it:
#
#     W(t) := pose.matrix(t) . matrix_local^-1 . bindTransform
#
# and then `W(t) . bindTransform^-1` is `pose.matrix(t) . matrix_local^-1`,
# exactly, for every joint and every instant, whatever convention either side
# keeps its bone frames in.
#
# Solving a change of basis from `restTransforms` instead looks right and is
# not: Blender writes `restTransforms` from the rest pose and `bindTransforms`
# from the pose the mesh was bound in, and on this bird **they are different
# poses** -- the rest world chain and the bind transforms differ by 1.989 at
# `Toe.L.010`. A basis solved against the rest does not cancel against the
# bind, so the toes, the head, the eyes and the bill came out mangled while
# the body, whose bones agree, looked fine.
K = [bones[n].matrix_local.inverted() @ U_bind[i] for i, n in enumerate(names)]

a, b = act.frame_range
first, last = int(a), int(b)
scene = bpy.context.scene
scene.frame_start, scene.frame_end = first, last

trans, rots, scales = {}, {}, {}
for f in range(first, last + 1):
    scene.frame_set(f)
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    world = [ev.pose.bones[n].matrix @ K[i] for i, n in enumerate(names)]
    t, r, s_ = [], [], []
    for i in range(len(names)):
        p = parent_of[i]
        local = world[i] if p < 0 else world[p].inverted() @ world[i]
        loc, quat, sca = local.decompose()
        t.append('(%.6f, %.6f, %.6f)' % (loc.x, loc.y, loc.z))
        r.append('(%.6f, %.6f, %.6f, %.6f)' % (quat.w, quat.x, quat.y, quat.z))
        s_.append('(%.4f, %.4f, %.4f)' % (sca.x, sca.y, sca.z))
    trans[f], rots[f], scales[f] = t, r, s_

moved = sum(1 for i in range(len(names)) if rots[first][i] != rots[(first + last) // 2][i])
print('lrt: %d of %d joints differ between the ends' % (moved, len(names)))

# READ BACK WHAT THE FILE WILL HOLD and compare the skinning matrix it gives
# against Blender's own, so the check covers the quantisation as well as the
# algebra. This is the number that says the bird is not deformed.
from mathutils import Quaternion, Vector
def parse(text):
    return [float(x) for x in text.strip('()').split(',')]
for f in (first, (first + last) // 2, last):
    scene.frame_set(f)
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    world = [None] * len(names)
    for i in range(len(names)):
        local = Matrix.LocRotScale(Vector(parse(trans[f][i])),
                                   Quaternion(parse(rots[f][i])),
                                   Vector(parse(scales[f][i])))
        p = parent_of[i]
        world[i] = local if p < 0 else world[p] @ local
    worst, who = 0.0, ''
    for i, n in enumerate(names):
        usd = world[i] @ U_bind[i].inverted()
        blender = ev.pose.bones[n].matrix @ bones[n].matrix_local.inverted()
        d = max(abs(usd[r][c] - blender[r][c]) for r in range(4) for c in range(4))
        if d > worst:
            worst, who = d, n
    print('lrt: f%-3d skinning matrix worst %.3e at %s' % (f, worst, who))

anim = 'Flight'
lines = ['#usda 1.0', '(', '    startTimeCode = %d' % first, '    endTimeCode = %d' % last,
         '    timeCodesPerSecond = 30', ')', '',
         'over "root"', '{', '    over "Bird"', '    {', '        over "Bird"', '        {',
         '            rel skel:animationSource = </root/Bird/Bird/%s>' % anim,
         '            def SkelAnimation "%s"' % anim, '            {',
         '                uniform token[] joints = [' + ', '.join('"%s"' % j for j in joints) + ']']
for name, held, kind in (('translations', trans, 'float3[]'),
                         ('rotations', rots, 'quatf[]'),
                         ('scales', scales, 'half3[]')):
    lines.append('                %s %s.timeSamples = {' % (kind, name))
    for f in range(first, last + 1):
        lines.append('                    %d: [%s],' % (f, ', '.join(held[f])))
    lines.append('                }')
lines += ['            }', '        }', '    }', '}']
open(out, 'w').write('\n'.join(lines) + '\n')
print('lrt: wrote %s, %d joints over %d frames, %d bytes'
      % (out, len(names), last - first + 1, os.path.getsize(out)))
