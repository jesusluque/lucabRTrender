// Copyright (c) 2026 lucabRTrender contributors.
//
// The smallest question a launch can be asked: what value did you see?
//
// A number is written before every launch and the launch writes it back. A
// launch that answers with the number of the launch before it read memory
// that was no longer current, and no image would say so -- it would simply be
// the frame before. The same line is asked three ways, because which of the
// three is wrong says where the repair belongs: a constant buffer, which is
// how every per-frame parameter is bound; a read-only buffer, which is how
// every scene resource is bound; and a writable buffer over the *same* memory
// as the read-only one. The compute entry and the ray generation entry ask
// all three of the same device, which separates "this device's memory" from
// "this device's ray tracing".
module uniform_echo;

public struct EchoParams {
    public uint value;
    public uint count;
    uint pad0;
    uint pad1;
};

public ConstantBuffer<EchoParams>  echo;
/// The same two numbers through a buffer of the device's own, whose contents
/// a launch reads as ordinary memory rather than out of its parameter block.
public StructuredBuffer<uint>      echoWords;
/// And again through a writable buffer, whose loads are coherent where a
/// read-only one's may be served from a cache the launch did not invalidate.
public RWStructuredBuffer<uint>    echoRw;
public RWStructuredBuffer<uint>    answers;

public void echoAt(uint at) {
    if (at >= echo.count) {
        return;
    }
    if (at == 0) {
        // One tally a launch: a launch that never ran leaves the answer of
        // the launch before it in place, and only a counter can tell the two
        // apart.
        InterlockedAdd(answers[echo.count], 1u);
    }
    answers[at] = echo.value;
    answers[at + echo.count + 4] = echoWords[0];
    if (at == 0) { answers[echo.count + 1] = echoRw[0]; }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void echoCompute(uint3 tid: SV_DispatchThreadID) {
    echoAt(tid.x);
}
