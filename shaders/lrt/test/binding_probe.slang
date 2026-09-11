// Copyright (c) 2026 lucabRTrender contributors.
//
// Which buffer a kernel's parameter actually reads. Every binding is filled
// with a marker of its own and the kernel reports what each name saw, so a
// backend that lands a dispatch's buffers on the wrong parameters shows up
// here rather than as a wrong answer three kernels later.
module binding_probe;

struct BindingProbeParams {
    uint count;
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<uint>            first;
StructuredBuffer<uint>            second;
StructuredBuffer<uint>            third;
RWStructuredBuffer<uint>          fourth;
RWStructuredBuffer<uint>          fifth;
RWStructuredBuffer<uint>          sixth;
RWStructuredBuffer<uint>          seventh;
RWStructuredBuffer<uint>          seen;      // 7: the marker each name read or wrote
ConstantBuffer<BindingProbeParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void bindingProbe(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    seen[0] = first[0];
    seen[1] = second[0];
    seen[2] = third[0];
    seen[3] = fourth[0];
    seen[4] = fifth[0];
    seen[5] = sixth[0];
    seen[6] = seventh[0];
}
