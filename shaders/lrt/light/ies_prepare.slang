// Copyright (c) 2026 lucabRTrender contributors.
//
// Each IES profile's power: its intensity integrated over the sphere, over
// the solid-angle patches its angle lists define, folded out by the symmetry
// its horizontal range declares -- then divided by 4 pi, so normalize makes
// the mean intensity over the sphere one. A statistic, so a kernel's.
module ies_prepare;
import lrt.light.lights;

uniform uint iesCount;

[shader("compute")]
[numthreads(1, 1, 1)]
void iesPrepare(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= iesCount) {
        return;
    }
    IesRecord r = iesRecords[at];
    const uint nv = r.verticalCount;
    const uint nh = r.horizontalCount;
    const float kPi = 3.14159265358979;
    // How much of the circle the authored horizontal range stands for.
    float fold = 1.0;
    float span = 2.0 * kPi;
    if (nh > 1) {
        const float first = iesValues[r.horizontalOffset];
        const float last = iesValues[r.horizontalOffset + nh - 1];
        span = radians(max(last - first, 1.0e-3));
        fold = last <= 90.0 + 1.0e-3 ? 4.0 : last <= 180.0 + 1.0e-3 ? 2.0 : 1.0;
    }
    float total = 0.0;
    for (uint i = 0; i < nv; ++i) {
        // The patch about node i in theta: half way to each neighbour.
        const float t = radians(iesValues[r.verticalOffset + i]);
        const float tLo = i > 0 ? 0.5 * (t + radians(iesValues[r.verticalOffset + i - 1])) : 0.0;
        const float tHi = i + 1 < nv ? 0.5 * (t + radians(iesValues[r.verticalOffset + i + 1])) : kPi;
        const float band = max(cos(tLo) - cos(tHi), 0.0);
        for (uint j = 0; j < nh; ++j) {
            float dPhi = span;
            if (nh > 1) {
                const float p = radians(iesValues[r.horizontalOffset + j]);
                const float pLo = j > 0 ? 0.5 * (p + radians(iesValues[r.horizontalOffset + j - 1])) : p;
                const float pHi = j + 1 < nh ? 0.5 * (p + radians(iesValues[r.horizontalOffset + j + 1])) : p;
                dPhi = max(pHi - pLo, 0.0);
            }
            total += iesValues[r.candelaOffset + j * nv + i] * r.multiplier * band * dPhi;
        }
    }
    total *= fold;
    r.power = total / (4.0 * kPi);
    iesRecords[at] = r;
}
