// Copyright (c) 2026 lucabRTrender contributors.
//
// The camera as techniques use it: the same projection every renderer in the
// engine uses (screen = focal * v.xy / v.z + centre, pixel (x, y) bottom row
// first, its centre at + 0.5), a primary ray whose direction has view z = 1,
// and for rasterisation a reversed, infinite perspective depth: near / z.
module camera;

public struct CameraParams {
    public uint  width;
    public uint  height;
    public float focalX;
    public float focalY;
    public float centreX;
    public float centreY;
    public float nearZ;
    public float farZ;
    public uint  orthographic;
    public uint  pad0; public uint pad1; public uint pad2;
};

/// The view-space ray through pixel (px, py): origin and a direction with z = 1.
public void viewRay(CameraParams c, float2 pixel, out float3 origin, out float3 direction) {
    const float2 p = float2((pixel.x - c.centreX) / c.focalX, (pixel.y - c.centreY) / c.focalY);
    if (c.orthographic != 0) {
        origin = float3(p, 0.0);
        direction = float3(0.0, 0.0, 1.0);
    } else {
        origin = float3(0.0);
        direction = float3(p, 1.0);
    }
}

/// View space to clip space for the rasteriser: NDC y up, reversed depth
/// (near is 1, infinitely far is 0).
public float4 viewToClip(CameraParams c, float3 v) {
    const float w = float(c.width);
    const float h = float(c.height);
    if (c.orthographic != 0) {
        return float4((2.0 * c.focalX * v.x + 2.0 * c.centreX - w) / w,
                      (2.0 * c.focalY * v.y + 2.0 * c.centreY - h) / h,
                      saturate((c.farZ - v.z) / (c.farZ - c.nearZ)), 1.0);
    }
    return float4((2.0 * c.focalX * v.x + (2.0 * c.centreX - w) * v.z) / w,
                  (2.0 * c.focalY * v.y + (2.0 * c.centreY - h) * v.z) / h,
                  c.nearZ, v.z);
}
