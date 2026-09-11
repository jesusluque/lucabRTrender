// Copyright (c) 2026 lucabRTrender contributors.
//
// Every instance record into the top-level acceleration structure's instance
// descriptors, in the layout the backend's API reads -- written here because
// the transforms of instanced sets never leave the device. Layouts follow
// slang-rhi's (include/slang-rhi-device.h):
//   0 generic / D3D12 / Vulkan  64 bytes: transform rows [3][4]; id:24|mask:8;
//                                          contribution:24|flags:8; handle u64
//   1 OptiX                     80 bytes: rows [3][4]; id; sbt; mask; flags;
//                                          handle u64; pad[2]
//   2 Metal                     68 bytes: columns [4][3]; options; mask;
//                                          function table offset; structure
//                                          index; user id
module instance_descs;
import scene_types;

struct DescParams {
    uint count;
    uint layout;
    uint words;          // per descriptor
    uint flags;          // AccelerationStructureInstanceFlags (options on Metal) for every instance
    uint doubleSided;    // added for double-sided instances: TriangleFacingCullDisable
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<InstanceRecord> instances;
StructuredBuffer<uint>           handles;   // per mesh: acceleration structure handle, low and high words
RWStructuredBuffer<uint>         descs;
ConstantBuffer<DescParams>       params;

[shader("compute")]
[numthreads(256, 1, 1)]
void instanceDescs(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const InstanceRecord r = instances[i];
    const uint at = i * params.words;
    const float3x4 m = float3x4(r.world0, r.world1, r.world2);
    const uint handleLo = handles[r.mesh * 2];
    const uint handleHi = handles[r.mesh * 2 + 1];
    const uint flags = params.flags | ((r.flags & 1u) != 0 ? params.doubleSided : 0u);
    if (params.layout == 2) {
        for (uint c = 0; c < 4; ++c) {
            for (uint row = 0; row < 3; ++row) {
                descs[at + c * 3 + row] = bit_cast<uint>(m[row][c]);
            }
        }
        descs[at + 12] = flags;
        descs[at + 13] = 0xFF;
        descs[at + 14] = 0;
        descs[at + 15] = handleLo;
        descs[at + 16] = i;
        return;
    }
    for (uint row = 0; row < 3; ++row) {
        for (uint c = 0; c < 4; ++c) {
            descs[at + row * 4 + c] = bit_cast<uint>(m[row][c]);
        }
    }
    if (params.layout == 1) {
        descs[at + 12] = i;
        descs[at + 13] = 0;
        descs[at + 14] = 0xFF;
        descs[at + 15] = flags;
        descs[at + 16] = handleLo;
        descs[at + 17] = handleHi;
        descs[at + 18] = 0;
        descs[at + 19] = 0;
        return;
    }
    descs[at + 12] = (i & 0xFFFFFFu) | (0xFFu << 24);
    descs[at + 13] = (flags & 0xFFu) << 24;
    descs[at + 14] = handleLo;
    descs[at + 15] = handleHi;
}
