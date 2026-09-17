// Copyright (c) 2026 lucabRTrender contributors.
//
// The cloud's shadow tables gathered into one buffer, on the device.
//
// Why at all: rt_shadow_packed.slang says it -- a kernel at Metal's limit of
// 31 buffers cannot bind four more. Why a kernel rather than four copies:
// this engine does no arithmetic on scene data in the host, and a copy that
// the host issues per table is four commands where one dispatch reads and
// writes the same device memory once.
//
// One thread an entry of the destination, so the frames (four rows a
// particle) cost what they are and nothing is read twice.
module rt_shadow_pack;
import rt_shadow_packed;

struct PackParams {
    PackedShadow where;
    uint particles;      ///< frames: 4 entries each; colours: `colourEntries`
    uint colourEntries;  ///< every instance's colours, end to end
    uint total;          ///< entries in the packed buffer
    uint pad0;
};

StructuredBuffer<float4>   frames;
StructuredBuffer<float4>   colours;
StructuredBuffer<float4>   instanceData;
StructuredBuffer<uint>     instanceIndices;
RWStructuredBuffer<float4> packed;
ConstantBuffer<PackParams> pack;

[shader("compute")]
[numthreads(256, 1, 1)]
void rtShadowPack(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= pack.total) {
        return;
    }
    if (at >= pack.where.frames && at < pack.where.frames + pack.particles * 4) {
        packed[at] = frames[at - pack.where.frames];
    } else if (at >= pack.where.colours && at < pack.where.colours + pack.colourEntries) {
        packed[at] = colours[at - pack.where.colours];
    } else if (at >= pack.where.instances && at < pack.where.instances + pack.where.instanceCount * 3) {
        packed[at] = instanceData[at - pack.where.instances];
    } else if (at >= pack.where.indices && at < pack.where.indices + pack.where.instanceCount) {
        // Two uints an instance, reinterpreted into the first two lanes.
        const uint instance = at - pack.where.indices;
        packed[at] = float4(asfloat(instanceIndices[instance * 2]), asfloat(instanceIndices[instance * 2 + 1]),
                            0.0, 0.0);
    } else {
        packed[at] = float4(0.0);
    }
}
