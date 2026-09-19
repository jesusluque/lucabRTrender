"""Every clip of the sparrow, as its own USD, without converting it again.

Two layers a clip, and neither needs the conversion re-run:

  clips/<clip>.usda       the SkelAnimation, over the mesh stage
  clips/<clip>_rig.usda   the cloud's `skinningXforms`, over the splat stage

The second is what makes this cheap. A skinned cloud is the same four million
gaussians whatever it is doing; what a clip changes is 609 matrices a frame,
which is a hundred kilobytes. Converting each clip would write 365 MB of
identical gaussians seventy-one times over.

The matrices are the ones `UsdSkelSkeletonQuery::ComputeSkinningTransforms`
answers -- joint world times inverse bind, in the skeleton's own space --
which is exactly Blender's `pose.matrix . matrix_local^-1`, transposed,
because USD writes a matrix for row vectors.

    blender -b --python scripts/sparrow-clips.py -- <fbx> <skel.json> <outdir>
"""
import bpy, json, os, re, sys
from mathutils import Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
fbx, skel_json, outdir = argv[0], argv[1], argv[2]
only = argv[3] if len(argv) > 3 else None

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')

skel = json.load(open(skel_json))
joints = skel['joints']


def as_matrix(nums):
    m = Matrix([nums[0:4], nums[4:8], nums[8:12], nums[12:16]])
    m.transpose()
    return m


U_bind = [as_matrix(r) for r in skel['bind']]


def sanitize(s):
    s = re.sub(r'[^A-Za-z0-9_]', '_', s)
    return '_' + s if s and s[0].isdigit() else s


bones = arm.data.bones
blender_name = {sanitize(b.name): b.name for b in bones}
names = [blender_name[j.split('/')[-1]] for j in joints]
index = {n: i for i, n in enumerate(names)}
parent_of = [index[bones[n].parent.name] if bones[n].parent else -1 for n in names]
K = [bones[n].matrix_local.inverted() @ U_bind[i] for i, n in enumerate(names)]
rest_inverse = [bones[n].matrix_local.inverted() for n in names]

STAGE = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    startTimeCode = %d
    endTimeCode = %d
    timeCodesPerSecond = 30
    subLayers = [
        @./%s_rig.usda@,
        @../Sparrow_gs.usdc@
    ]
)

over "World"
{
    def Scope "Lights"
    {
        def DomeLight "Sky"
        {
            float inputs:intensity = 1.15
            color3f inputs:color = (0.72, 0.79, 0.95)
        }
        def DistantLight "Sun"
        {
            float inputs:intensity = 4.5
            float inputs:angle = 1.2
            color3f inputs:color = (1, 0.94, 0.82)
            double3 xformOp:rotateXYZ = (-52, 0, 36)
            uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
        }
    }
}
"""

os.makedirs(outdir, exist_ok=True)
scene = bpy.context.scene
scene.render.fps = 30

clips = sorted(a.name[len('Bird|Bird|'):] for a in bpy.data.actions
               if a.name.startswith('Bird|Bird|') and not a.name.endswith('_static_pose'))
if only:
    clips = [c for c in clips if c == only]

written = []
for clip in clips:
    act = bpy.data.actions['Bird|Bird|' + clip]
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = act
    if hasattr(ad, 'action_slot') and len(act.slots):
        ad.action_slot = act.slots[0]
    key_act = bpy.data.actions.get('Key|Bird|' + clip)
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.data.shape_keys and key_act:
            kd = o.data.shape_keys.animation_data or o.data.shape_keys.animation_data_create()
            kd.action = key_act
            if hasattr(kd, 'action_slot') and len(key_act.slots):
                kd.action_slot = key_act.slots[0]

    a, b = act.frame_range
    first, last = int(a), int(b)
    scene.frame_start, scene.frame_end = first, last

    anim, rig = {}, {}
    for f in range(first, last + 1):
        scene.frame_set(f)
        ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
        pose = [ev.pose.bones[n].matrix for n in names]
        world = [pose[i] @ K[i] for i in range(len(names))]
        t, r, s = [], [], []
        for i in range(len(names)):
            p = parent_of[i]
            local = world[i] if p < 0 else world[p].inverted() @ world[i]
            loc, quat, sca = local.decompose()
            t.append('(%.6f, %.6f, %.6f)' % (loc.x, loc.y, loc.z))
            r.append('(%.6f, %.6f, %.6f, %.6f)' % (quat.w, quat.x, quat.y, quat.z))
            s.append('(%.4f, %.4f, %.4f)' % (sca.x, sca.y, sca.z))
        anim[f] = (t, r, s)
        # The skinning matrix itself, transposed into USD's row-vector layout.
        rows = []
        for i in range(len(names)):
            m = (pose[i] @ rest_inverse[i]).transposed()
            rows.append('( (%s), (%s), (%s), (%s) )' %
                        tuple(', '.join('%.6g' % m[k][c] for c in range(4)) for k in range(4)))
        rig[f] = rows

    header = ['#usda 1.0', '(', '    startTimeCode = %d' % first,
              '    endTimeCode = %d' % last, '    timeCodesPerSecond = 30', ')', '']
    lines = header + ['over "root"', '{', '    over "Bird"', '    {', '        over "Bird"', '        {',
                      '            rel skel:animationSource = </root/Bird/Bird/Clip>',
                      '            def SkelAnimation "Clip"', '            {',
                      '                uniform token[] joints = [' +
                      ', '.join('"%s"' % j for j in joints) + ']']
    for slot, (name, kind) in enumerate((('translations', 'float3[]'), ('rotations', 'quatf[]'),
                                         ('scales', 'half3[]'))):
        lines.append('                %s %s.timeSamples = {' % (kind, name))
        for f in range(first, last + 1):
            lines.append('                    %d: [%s],' % (f, ', '.join(anim[f][slot])))
        lines.append('                }')
    lines += ['            }', '        }', '    }', '}']
    open('%s/%s.usda' % (outdir, clip), 'w').write('\n'.join(lines) + '\n')

    lines = header + ['over "World"', '{', '    over "Splats"', '    {',
                      '        matrix4d[] primvars:lrt:splat:skinningXforms.timeSamples = {']
    for f in range(first, last + 1):
        lines.append('            %d: [%s],' % (f, ', '.join(rig[f])))
    lines += ['        }', '    }', '}']
    open('%s/%s_rig.usda' % (outdir, clip), 'w').write('\n'.join(lines) + '\n')

    # And the stage that plays it: the clip's rig over the one cloud, lit.
    open('%s/%s_gs.usda' % (outdir, clip), 'w').write(STAGE % (first, last, clip))
    written.append((clip, last - first + 1))
    print('lrt: %-26s %3d frames' % (clip, last - first + 1))

print('lrt: %d clips written to %s' % (len(written), outdir))
