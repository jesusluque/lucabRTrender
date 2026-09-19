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

U_rest = [as_matrix(r) for r in skel['rest']]

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

def rest_local(i):
    m = bones[names[i]].matrix_local
    p = parent_of[i]
    return m.copy() if p < 0 else bones[names[p]].matrix_local.inverted() @ m

# Blender's bone frame and USD's joint frame differ by a rotation per bone.
# Solve for it from the rest pose the exporter itself wrote, so whatever
# convention it used is the one we reproduce.
C = [None] * len(names)
for i in range(len(names)):
    p = parent_of[i]
    B = rest_local(i)
    C[i] = B.inverted() @ U_rest[i] if p < 0 else B.inverted() @ C[p] @ U_rest[i]

worst = 0.0
for i in range(len(names)):
    p = parent_of[i]
    B = rest_local(i)
    U = (B @ C[i]) if p < 0 else (C[p].inverted() @ B @ C[i])
    worst = max(worst, max(abs(U[r][c] - U_rest[i][r][c]) for r in range(4) for c in range(4)))
print('lrt: rest round trip worst %.2e' % worst)

a, b = act.frame_range
first, last = int(a), int(b)
scene = bpy.context.scene
scene.frame_start, scene.frame_end = first, last

trans, rots, scales = {}, {}, {}
for f in range(first, last + 1):
    scene.frame_set(f)
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    t, r, s = [], [], []
    for i, n in enumerate(names):
        p = parent_of[i]
        pb = ev.pose.bones[n].matrix
        B = pb.copy() if p < 0 else ev.pose.bones[names[p]].matrix.inverted() @ pb
        U = (B @ C[i]) if p < 0 else (C[p].inverted() @ B @ C[i])
        loc, quat, sca = U.decompose()
        t.append('(%.6f, %.6f, %.6f)' % (loc.x, loc.y, loc.z))
        r.append('(%.6f, %.6f, %.6f, %.6f)' % (quat.w, quat.x, quat.y, quat.z))
        s.append('(%.4f, %.4f, %.4f)' % (sca.x, sca.y, sca.z))
    trans[f], rots[f], scales[f] = t, r, s

moved = sum(1 for i in range(len(names)) if rots[first][i] != rots[(first + last) // 2][i])
print('lrt: %d of %d joints differ between the ends' % (moved, len(names)))

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
