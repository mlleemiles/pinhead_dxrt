#include "dxrt_header.hlsl"

#ifdef RAYGEN_AND_MISS_SHADERS

[shader("raygeneration")]
void ENTRY_POINT(raygeneration_main)()
{
    uint2 pixelPos = ApplyCheckerboard( DispatchRaysIndex().xy );
    float2 pixelUv = (pixelPos + 0.5) * inv_rt_size;

    // Early out
    float z = GBufLoadLinearDepth(pixelUv);

    [branch]
    if (z > ray_max_t)
    {
        WriteCheckerboardOutput(pixelPos, 1.0);
        return;
    }

    // Pixel data
    float3 N = GBufLoadNormal(pixelPos);
    float3 X = ReconstructWorldPos(pixelUv, z);

    // Flip normal if NoV is suspiciously negative
    float3 V = normalize(cam_pos.xyz - X);
    float NoV = dot(N, V);
    if (NoV < -0.5)
        N = -N;

    // Choose a ray
    STL::Rng::Initialize(pixelPos, dxrt_frame_idx);

    float2 rnd = STL::Rng::GetFloat2();
    float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay(rnd);

    float3x3 mLocalBasis = STL::Geometry::GetBasis(N);
    float3 rayDirection = STL::Geometry::RotateVectorInverse(mLocalBasis, rayLocal);

    // Screen space tracing
    float3 Xoffset = GetXWithOffset(X, N, V, z);

    float screenSpaceHitDist = 0;
    bool isHitFound = TraceRayScreenSpace(Xoffset, z, rayDirection, NoV, screenSpaceHitDist);

    [branch]
    if (isHitFound)
    {
        float ao = REBLUR_FrontEnd_GetNormHitDist(screenSpaceHitDist, z, diff_hit_dist_params, 1.0);
        WriteCheckerboardOutput(pixelPos, ao);
        return;
    }

    float3 Xhit = Xoffset + rayDirection * screenSpaceHitDist;

    // Ray tracing
    RayDesc ray;
    ray.Origin = Xhit; // start where SSRT stopped
    ray.Direction = rayDirection;
    ray.TMin = 0.0;
    ray.TMax = INF;

    RayPayload payload = (RayPayload)0;
    {
        const uint rayFlags = CULLING_FLAGS;
        const uint instanceInclusionMask = EInstanceRayVisibility::FullRays;
        const uint rayContributionToHitGroupIndex = FULL_RAY_ID;
        const uint multiplierForGeometryContributionToHitGroupIndex = 0;
        const uint missShaderIndex = FULL_RAY_ID;

        TraceRay(g_TLAS, rayFlags, instanceInclusionMask, rayContributionToHitGroupIndex, multiplierForGeometryContributionToHitGroupIndex, missShaderIndex, ray, payload);
        Report(_TraceRay);
    }

    float ao = REBLUR_FrontEnd_GetNormHitDist(payload.hitDist, z, diff_hit_dist_params, 1.0);
    WriteCheckerboardOutput(pixelPos, ao);
}

[shader("miss")]
void ENTRY_POINT(miss_main)(inout RayPayload payload)
{
    payload.hitDist = INF;
}

[shader("miss")]
void ENTRY_POINT(shadow_miss_main)(inout ShadowRayPayload payload)
{
    payload.hitDist = INF;
}

#else // RAYGEN_AND_MISS_SHADERS

//
// Full rays
//
[shader("closesthit")]
void ENTRY_POINT(closesthit_main)(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.hitDist = RayTCurrent();
}

[shader("anyhit")]
void ENTRY_POINT(anyhit_main)(inout RayPayload payload,  in BuiltInTriangleIntersectionAttributes attribs)
{
    Report(_AnyHit);
    CommonAnyHitShader(attribs);
}

//
// Shadow rays
//
[shader("closesthit")]
void ENTRY_POINT(shadow_closesthit_main)(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.hitDist = RayTCurrent();
}

[shader("anyhit")]
void ENTRY_POINT(shadow_anyhit_main)(inout ShadowRayPayload payload,  in BuiltInTriangleIntersectionAttributes attribs)
{
    Report(_AnyHit);
    CommonAnyHitShader(attribs);
}

#endif // RAYGEN_AND_MISS_SHADERS
