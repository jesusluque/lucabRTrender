// Copyright (c) 2026 lucabRTrender contributors.
//
// Real spherical harmonics up to degree 3, as 3DGS trains them, over the packed
// fp16 coefficients (common/packing.slang: rgb per basis, two halves a word).
module sh;
import lrt.common.packing;

static const float kC1 = 0.4886025119029199;
static const float kC2[5] = { 1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                              -1.0925484305920792, 0.5462742152960396 };
static const float kC3[7] = { -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                              0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                              -0.5900435899266435 };

float halfAt(StructuredBuffer<uint> sh, uint row, uint index) {
    const uint word = sh[row + index / 2];
    return (index & 1u) == 0u ? lowHalf(word) : highHalf(word);
}

float3 basis(StructuredBuffer<uint> sh, uint row, uint k) {
    return float3(halfAt(sh, row, k * 3), halfAt(sh, row, k * 3 + 1), halfAt(sh, row, k * 3 + 2));
}

/// The view-dependent part added to the base colour. `degree` <= what is stored.
public float3 evaluateRest(StructuredBuffer<uint> sh, uint row, uint degree, float3 dir) {
    if (degree == 0) {
        return float3(0.0);
    }
    const float x = dir.x, y = dir.y, z = dir.z;
    float3 result = -kC1 * y * basis(sh, row, 0) + kC1 * z * basis(sh, row, 1) -
                    kC1 * x * basis(sh, row, 2);
    if (degree < 2) {
        return result;
    }
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    result += kC2[0] * xy * basis(sh, row, 3) + kC2[1] * yz * basis(sh, row, 4) +
              kC2[2] * (2.0 * zz - xx - yy) * basis(sh, row, 5) +
              kC2[3] * xz * basis(sh, row, 6) + kC2[4] * (xx - yy) * basis(sh, row, 7);
    if (degree < 3) {
        return result;
    }
    result += kC3[0] * y * (3.0 * xx - yy) * basis(sh, row, 8) +
              kC3[1] * xy * z * basis(sh, row, 9) +
              kC3[2] * y * (4.0 * zz - xx - yy) * basis(sh, row, 10) +
              kC3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * basis(sh, row, 11) +
              kC3[4] * x * (4.0 * zz - xx - yy) * basis(sh, row, 12) +
              kC3[5] * z * (xx - yy) * basis(sh, row, 13) +
              kC3[6] * x * (xx - 3.0 * yy) * basis(sh, row, 14);
    return result;
}

public uint degreeOf(uint restPerColour) {
    return restPerColour >= 15 ? 3 : restPerColour >= 8 ? 2 : restPerColour >= 3 ? 1 : 0;
}
