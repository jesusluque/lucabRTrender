import bpy, sys, os

def vec(v):
    return tuple(v[:3]) if hasattr(v, '__len__') else (v, v, v)

def reduce(n, out=None):
    """Cycles surface node -> a Principled-ish description."""
    d = dict(base=(0.8,0.8,0.8), metallic=0.0, rough=0.5, trans=0.0, ior=1.45,
             emit=(0,0,0), emitk=0.0, tex=None)
    if n is None:
        return d
    t = n.type
    def col(name):
        i = n.inputs.get(name)
        if i is None: return (0.8,0.8,0.8), None
        tex = None
        if i.is_linked:
            src = i.links[0].from_node
            if src.type == 'TEX_IMAGE':
                tex = src
        return vec(i.default_value), tex
    def flt(name, dflt):
        i = n.inputs.get(name)
        if i is None or i.is_linked: return dflt
        return float(i.default_value)

    if t in ('BSDF_GLOSSY', 'BSDF_ANISOTROPIC'):
        d['base'], d['tex'] = col('Color'); d['metallic'] = 1.0
        d['rough'] = flt('Roughness', 0.2)
    elif t in ('BSDF_DIFFUSE', 'BSDF_TRANSLUCENT'):
        d['base'], d['tex'] = col('Color'); d['metallic'] = 0.0
        d['rough'] = max(flt('Roughness', 0.5), 0.5)
    elif t == 'BSDF_SHEEN':
        d['base'], d['tex'] = col('Color'); d['rough'] = 0.8
    elif t in ('BSDF_GLASS', 'BSDF_REFRACTION'):
        d['base'], d['tex'] = col('Color'); d['trans'] = 1.0
        d['rough'] = flt('Roughness', 0.0); d['ior'] = flt('IOR', 1.45)
    elif t == 'BSDF_TRANSPARENT':
        d['base'], d['tex'] = col('Color'); d['trans'] = 1.0; d['rough'] = 0.0; d['ior'] = 1.0
    elif t == 'EMISSION':
        d['emit'], d['tex'] = col('Color'); d['emitk'] = flt('Strength', 1.0)
        d['base'] = d['emit']
    elif t == 'MIX_SHADER':
        f = n.inputs[0]
        k = 0.08 if f.is_linked else float(f.default_value)   # LAYER_WEIGHT -> a fresnel-ish 0.4
        a = reduce(n.inputs[1].links[0].from_node) if n.inputs[1].is_linked else reduce(None)
        b = reduce(n.inputs[2].links[0].from_node) if n.inputs[2].is_linked else reduce(None)
        for key in ('base','emit'):
            d[key] = tuple(a[key][i]*(1-k) + b[key][i]*k for i in range(3))
        for key in ('metallic','rough','trans','ior','emitk'):
            d[key] = a[key]*(1-k) + b[key]*k
        d['tex'] = a['tex'] or b['tex']
    elif t == 'ADD_SHADER':
        a = reduce(n.inputs[0].links[0].from_node) if n.inputs[0].is_linked else reduce(None)
        b = reduce(n.inputs[1].links[0].from_node) if n.inputs[1].is_linked else reduce(None)
        d = dict(a)
        d['emit'] = tuple(a['emit'][i]*a['emitk'] + b['emit'][i]*b['emitk'] for i in range(3))
        d['emitk'] = 1.0 if (a['emitk'] or b['emitk']) else 0.0
        d['base'] = tuple(max(a['base'][i], b['base'][i]) for i in range(3))
        d['metallic'] = max(a['metallic'], b['metallic'])
        d['rough'] = min(a['rough'], b['rough'])
        d['trans'] = max(a['trans'], b['trans'])
        d['tex'] = a['tex'] or b['tex']
    return d

for o in bpy.data.objects:
    if o.type == 'MESH':
        o.data.name = o.name

report = []
for m in bpy.data.materials:
    if not m.use_nodes: continue
    nt = m.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if out is None or not out.inputs['Surface'].is_linked: continue
    surf = out[0] if False else out.inputs['Surface'].links[0].from_node
    if surf.type == 'BSDF_PRINCIPLED': continue
    d = reduce(surf)
    tex = d['tex']
    # keep the image node, drop the rest
    keep = {out}
    if tex:
        keep.add(tex)
        for i in tex.inputs:
            if i.is_linked: keep.add(i.links[0].from_node)
    for n in list(nt.nodes):
        if n not in keep: nt.nodes.remove(n)
    p = nt.nodes.new('ShaderNodeBsdfPrincipled')
    p.location = (out.location.x - 300, out.location.y)
    def setv(name, value):
        if name in p.inputs: p.inputs[name].default_value = value
    setv('Base Color', (*d['base'], 1.0))
    setv('Metallic', d['metallic'])
    setv('Roughness', max(0.02, min(1.0, d['rough'])))
    setv('IOR', d['ior'])
    for name in ('Transmission Weight', 'Transmission'):
        if name in p.inputs: p.inputs[name].default_value = d['trans']; break
    if d['emitk'] > 0:
        for name in ('Emission Color', 'Emission'):
            if name in p.inputs: p.inputs[name].default_value = (*d['emit'], 1.0); break
        if 'Emission Strength' in p.inputs: p.inputs['Emission Strength'].default_value = d['emitk']
    if tex:
        nt.links.new(tex.outputs['Color'], p.inputs['Base Color'])
    nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])
    report.append("  %-20s base=%s metal=%.2f rough=%.2f trans=%.2f emit=%.1f tex=%s" %
                  (m.name, tuple(round(x,3) for x in d['base']), d['metallic'], d['rough'],
                   d['trans'], d['emitk'], bool(tex)))
print("=== converted materials")
print("\n".join(report))

out_path = sys.argv[sys.argv.index('--') + 1]
bpy.ops.wm.usd_export(filepath=out_path,
                      export_materials=True,
                      generate_preview_surface=True,
                      export_textures_mode='NEW',
                      export_uvmaps=True,
                      export_normals=True,
                      evaluation_mode='RENDER',
                      export_cameras=True,
                      export_lights=True,
                      root_prim_path='/root')
print("=== wrote", out_path, os.path.getsize(out_path))
