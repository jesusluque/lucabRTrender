#!/usr/bin/env python3
"""A film out of the sparrow's clips, without converting anything again.

`sparrow-clips.py` wrote each clip twice: as a `SkelAnimation` over the mesh
and as the cloud's `skinningXforms`. Both are nothing but time samples, so a
sequence is a concatenation: take each clip's samples in turn, renumber them
end to end, and write one layer. Nothing is resampled and nothing is blended --
what plays is what the animator made, in the order asked for.

    python3 scripts/sparrow-film.py <clips dir> <out dir> [--bird N]

The order is the asset's own: the clips that end in `_start` and `_end` exist
to be chained, and the two `*_to_*` clips are the transitions between flapping
and gliding.
"""
import os, re, sys

# Everything on the ground first -- it stands, eats and hops -- and then it
# leaves: takes off, flaps, glides and flies away. Nothing is chained back to
# the ground, so the shot has one direction and one ending. No roll: a bird
# going somewhere does not do tricks on the way.
SEQUENCE = [
    ('land_idle_B0', None),
    ('land_eat_B1', None),
    ('land_hop_start', None),
    ('land_hop_1', None),
    ('land_hop_2', None),
    ('land_hop_end', None),
    ('land_fly_start_B', None),
    ('air_fly_A2', None),
    ('air_fly_A_to_gliding_A', None),
    ('air_gliding_A1', None),
    ('air_gliding_A_to_fly_A', None),
    ('air_fly_A1', None),
]

# The second bird keeps to the air, out of step with the first.
SECOND = [
    ('air_fly_A1', None),
    ('air_fly_turnR_A0', None),
    ('air_gliding_A0', None),
    ('air_fly_A_to_gliding_A', None),
    ('air_fly_A2', None),
    ('air_fly_revolve_A0', None),
    ('air_fly_A1', None),
    ('air_gliding_A1', None),
]

SAMPLE = re.compile(r'^\s*(\d+):\s*\[(.*)\],\s*$')


def samples(path):
    """The time samples of every array in a layer, by attribute, in order."""
    held, current = [], None
    for line in open(path):
        if '.timeSamples = {' in line:
            name = line.strip().split()[1].split('.timeSamples')[0]
            kind = line.strip().split()[0]
            current = (kind, name, [])
            held.append(current)
            continue
        if current is None:
            continue
        m = SAMPLE.match(line)
        if m:
            current[2].append(m.group(2))
        elif line.strip() == '}':
            current = None
    return held


def concatenate(clips, kind, clipdir):
    """Each clip's samples in turn, renumbered from one."""
    out, frame = None, 1
    for clip, limit in clips:
        path = '%s/%s%s.usda' % (clipdir, clip, '_rig' if kind == 'rig' else '')
        got = samples(path)
        if out is None:
            out = [(k, n, []) for k, n, _ in got]
        for slot, (_, _, values) in enumerate(got):
            take = values if limit is None else values[:limit]
            out[slot][2].extend(take)
    return out, sum(len(c[2]) for c in out[:1])


def write(path, header, body, blocks, closers, first_frame=1):
    lines = ['#usda 1.0', '(', '    startTimeCode = %d' % first_frame,
             '    endTimeCode = %d' % (first_frame + len(blocks[0][2]) - 1),
             '    timeCodesPerSecond = 30', ')', ''] + header
    for kind, name, values in blocks:
        lines.append('%s%s %s.timeSamples = {' % (body, kind, name))
        for i, v in enumerate(values):
            lines.append('%s    %d: [%s],' % (body, first_frame + i, v))
        lines.append('%s}' % body)
    lines += closers
    open(path, 'w').write('\n'.join(lines) + '\n')
    return len(blocks[0][2])


def main():
    clipdir, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    for name, order in (('Film', SEQUENCE), ('Film2', SECOND)):
        rig, _ = concatenate(order, 'rig', clipdir)
        n = write('%s/%s_rig.usda' % (outdir, name),
                  ['over "World"', '{', '    over "Splats"', '    {'],
                  '        ', rig, ['    }', '}'])
        anim, _ = concatenate(order, 'anim', clipdir)
        head = ['over "root"', '{', '    over "Bird"', '    {', '        over "Bird"', '        {',
                '            rel skel:animationSource = </root/Bird/Bird/Clip>',
                '            def SkelAnimation "Clip"', '            {',
                '                uniform token[] joints = [' +
                ', '.join('"%s"' % j for j in joints_of(clipdir, order[0][0])) + ']']
        write('%s/%s_anim.usda' % (outdir, name), head, '                ', anim,
              ['            }', '        }', '    }', '}'])
        print('lrt: %s -> %d frames, %.1f s' % (name, n, n / 30.0))


def joints_of(clipdir, clip):
    for line in open('%s/%s.usda' % (clipdir, clip)):
        if 'uniform token[] joints' in line:
            return re.findall(r'"([^"]+)"', line)
    raise SystemExit('no joints in %s' % clip)


main()
