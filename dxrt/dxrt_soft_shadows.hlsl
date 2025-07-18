#include "dxrt_header.hlsl"

#ifdef RAYGEN_AND_MISS_SHADERS

[shader("raygeneration")]
void ENTRY_POINT(raygeneration_main)()
{
    uint2 pixelPos = DispatchRaysIndex().xy;
    float2 pixelUv = (pixelPos + 0.5) * inv_rt_size;
	
	float3 litVal = gLit[pixelPos.xy].xyz;
	bool isLeaves = (litVal.r > 0.f);

    // Early out
    float z = GBufLoadLinearDepth(pixelUv);

    [branch]
    if (z > ray_max_t)
    {
        gOutput[pixelPos] = float2(NRD_FP16_MAX, NRD_FP16_MAX).xyxy;
        return;
    }

    // Pixel data
    float3 N = GBufLoadNormal(pixelPos);
    float3 X = ReconstructWorldPos(pixelUv, z);
	bool isBackLit = dot(N, sun_direction) < 0.f;
	bool isBackLitLeaves = isBackLit && isLeaves;

    // Flip normal if NoV is suspiciously negative
    float3 V = normalize(cam_pos.xyz - X);
    float NoV = dot(N, V);
    if (NoV < -0.5)
        N = -N;

    // Choose a ray
    STL::Rng::Initialize(pixelPos, dxrt_frame_idx);

#if( USE_CHECKERBOARDING_FOR_SHADOWS == 1 )
    float2 rnd = STL::Rng::GetFloat2();
#else
    float2 rnd = GetRandom(false, 0, gScrambling1spp, 0, 1, 1);
#endif

    rnd = STL::ImportanceSampling::Cosine::GetRay(rnd).xy;
    rnd *= sun_tan_angular_radius;

    float3x3 mLocalBasis = STL::Geometry::GetBasis(sun_direction); // TODO: move to CB
    float3 rayDirection = normalize(mLocalBasis[0] * rnd.x + mLocalBasis[1] * rnd.y + mLocalBasis[2]);

    // NoL < 0 special case
    bool isInShadow = dot(N, sun_direction) < 1e-5;
    [branch]
    if (isInShadow && !isBackLitLeaves)
    {
        gOutput[pixelPos] = SIGMA_FrontEnd_PackShadow(z, 0.0, sun_tan_angular_radius).xyxy;
        return;
    }

    // Screen space tracing
#if( USE_CHECKERBOARDING_FOR_SHADOWS == 1 )
    uint isActivePixel = STL::Sequence::CheckerBoard(pixelPos, dxrt_frame_idx);
    if (isActivePixel)
    {
#endif

    float3 Xoffset = GetXWithOffset(X, N + rayDirection * 0.1, V, z);

    float hitDist = 0;
    bool isHitFound = TraceRayScreenSpace(Xoffset, z, rayDirection, NoV, hitDist, true);
	if (isLeaves) {
		isHitFound = false;
	}

    hitDist *= float(isHitFound); // TODO: ideally, this line is not needed, but IQ is worse if we continue RT from the end point of SSRT

    // Ray tracing
    RayDesc ray;
    ray.Origin = Xoffset + hitDist * rayDirection;
    ray.Direction = rayDirection;
    ray.TMin = 0.0;
    ray.TMax = NRD_FP16_MAX * float(!isHitFound);

#if( USE_TRACE_RAY_INLINE == 1 )
    const uint instanceInclusionMask = ray.TMax == 0.0 ? 0 : EInstanceRayVisibility::ShadowRays;

    RayQuery<CULLING_FLAGS> rayQuery;
    rayQuery.TraceRayInline(g_TLAS, 0, instanceInclusionMask, ray);

    while(rayQuery.Proceed())
    {
        uint triangleBaseIndex = rayQuery.CandidatePrimitiveIndex() * 3;

        uint3 indices;
        indices.x = gIBLocal[triangleBaseIndex].i;
        indices.y = gIBLocal[triangleBaseIndex + 1].i;
        indices.z = gIBLocal[triangleBaseIndex + 2].i;

        float2 uv = GetUV0(indices, rayQuery.CandidateTriangleBarycentrics());

        uint instanceId = rayQuery.CandidateInstanceID();
        uint idx = (instanceId >> 1) + (instanceId & 0x1) * rayQuery.CandidateGeometryIndex();

        float w, h;
        gClp[idx].GetDimensions(w, h);
        float mipNum = log2( max(w, h) );
        float mip = max(mipNum - 7, 0.0); // Use 128x128 (or the best) to avoid cache trashing

        float clip_val = gClp[idx].SampleLevel(gLinearSampler, uv, mip).x;
        if (clip_val >= 0.0)
            rayQuery.CommitNonOpaqueTriangleHit();
    }

    hitDist += rayQuery.CommittedRayT() * float(!isHitFound);
#else
    ShadowRayPayload payload = (ShadowRayPayload)0;
	payload.matOpacity = litVal.r;
    {
    #if defined(__XBOX_SCARLETT)
        const uint rayFlags = CULLING_FLAGS | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    #else
        const uint rayFlags = CULLING_FLAGS;
    #endif
        const uint instanceInclusionMask = ray.TMax == 0.0 ? 0 : EInstanceRayVisibility::DistantShadowRays;
        const uint rayContributionToHitGroupIndex = SHADOW_RAY_ID;
        const uint multiplierForGeometryContributionToHitGroupIndex = 0;
        const uint missShaderIndex = SHADOW_RAY_ID;

        TraceRay(g_TLAS, rayFlags, instanceInclusionMask, rayContributionToHitGroupIndex, multiplierForGeometryContributionToHitGroupIndex, missShaderIndex, ray, payload);
        Report(_TraceRay);
    }

    hitDist += payload.hitDist * float(!isHitFound);
#endif

    hitDist = min(hitDist, NRD_FP16_MAX);

    float2 result = SIGMA_FrontEnd_PackShadow(z, hitDist, sun_tan_angular_radius);

#if( USE_CHECKERBOARDING_FOR_SHADOWS == 1 )
    pixelPos.x &= ~0x1;
    gOutput[pixelPos] = result.xyxy;
    gOutput[pixelPos + uint2(1, 0)] = result.xyxy;
    }
#else
    gOutput[pixelPos] = result.xyxy;
#endif
}

[shader("miss")]
void ENTRY_POINT(miss_main)(inout RayPayload payload)
{
    payload.hitDist = NRD_FP16_MAX;
}

[shader("miss")]
void ENTRY_POINT(shadow_miss_main)(inout ShadowRayPayload payload)
{
    payload.hitDist = NRD_FP16_MAX;
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
	if (RayTCurrent() < payload.matOpacity * 0.05) {
		IgnoreHit();
	} else {
	    Report(_AnyHit);
		CommonAnyHitShader(attribs);
	}
}

#endif // RAYGEN_AND_MISS_SHADERS