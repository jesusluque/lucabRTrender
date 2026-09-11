// Copyright (c) 2026 lucabRTrender contributors.
//
// A kernel whose value is decided at link time.
module linked;

extern static const uint kLinkedValue;
RWStructuredBuffer<uint> linkedOut;

[shader("compute")]
[numthreads(1, 1, 1)]
void linkedConstant(uint3 tid: SV_DispatchThreadID) {
    if (tid.x == 0) {
        linkedOut[0] = kLinkedValue;
    }
}
