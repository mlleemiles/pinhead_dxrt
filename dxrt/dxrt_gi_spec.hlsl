#include "dxrt_header.hlsl"

#ifdef RAYGEN_AND_MISS_SHADERS

#define USE_IMPORTANCE_SAMPLING     1 // ignore rays with 0 throughput
#define ZERO_TROUGHPUT_SAMPLE_NUM   16

[shader("raygeneration")]
void ENTRY_POINT(raygeneration_main)()
{
	#if (GI_USE_CHECKERBOARD == 1)
		uint2 pixelPos = ApplyCheckerboard( DispatchRaysIndex().xy );
	#else
		uint2 pixelPos = ( DispatchRaysIndex().xy );
	#endif
    float2 pixelUv = ( pixelPos + 0.5 ) * inv_rt_size;

    // Early out
    float z = GBufLoadLinearDepth(pixelUv);

    [branch]
    if (z > ray_max_t || pixelPos.x >= rt_size.x || pixelPos.y >= rt_size.y)
    {
		#if (GI_USE_CHECKERBOARD == 1)
			WriteCheckerboardOutput(pixelPos, 0);
		#else
			WriteFullOutput(pixelPos, 0);
		#endif
		
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

    float m = gLit[pixelPos].z;
    float roughness = STL::Math::Sqrt01(m);

    // Choose a ray
    STL::Rng::Initialize(pixelPos, dxrt_frame_idx);

    float3x3 mLocalBasis = STL::Geometry::GetBasis(N);
    float3 Vlocal = STL::Geometry::RotateVector(mLocalBasis, V);

    float3 rayDirection = 0;
    float throughput = 0.0;
    float VoH = 0.0;
    uint sampleNum = 0;
    float trimmingFactor = 1.0; // TODO: no trimming for now

#if( USE_IMPORTANCE_SAMPLING != 0 )
    while( sampleNum < ZERO_TROUGHPUT_SAMPLE_NUM && throughput == 0.0 )
#endif
    {
        float2 rnd = STL::Rng::GetFloat2();

        float3 Hlocal = STL::ImportanceSampling::VNDF::GetRay(rnd, roughness, Vlocal, trimmingFactor);
        float3 H = STL::Geometry::RotateVectorInverse(mLocalBasis, Hlocal);
        rayDirection = reflect(-V, H);

        VoH = abs( dot(-V, H) );

        // It's a part of VNDF sampling - see http://jcgt.org/published/0007/04/01/paper.pdf (paragraph "Usage in Monte Carlo renderer")
        float NoL = saturate( dot(N, rayDirection) );
        throughput = STL::BRDF::GeometryTerm_Smith(roughness, NoL);

        sampleNum++;
    }

    // Material de-modulation
    float3 Rf0 = gSpc[pixelPos].xyz;
    float3 envBRDF0 = STL::BRDF::EnvironmentTerm_Ross(Rf0, abs(NoV), roughness);
    float3 F = STL::BRDF::FresnelTerm_Schlick(Rf0, VoH);
    float3 preserveIntegralCorrectness = throughput * F / max(envBRDF0, 0.001);

    // Screen space tracing
    float3 Xoffset = GetXWithOffset(X, N, V, z);

    float screenSpaceHitDist = 0;
    bool isHitFound = TraceRayScreenSpace(Xoffset, z, rayDirection, NoV, screenSpaceHitDist);

    float3 Xhit = Xoffset + rayDirection * screenSpaceHitDist;

    // Debug stuff
    #if( DEBUG == 2 )
			#if (GI_USE_CHECKERBOARD == 1)
				WriteCheckerboardOutput(pixelPos, isHitFound ? float4(1.0, 0.5, 0, 1) : 0);
			#else
				WriteFullOutput(pixelPos, isHitFound ? float4(1.0, 0.5, 0, 1) : 0);
			#endif
        return;
    #endif

    // If screen-space hit is found just grab lighting from the g-buffer
    [branch]
    if (isHitFound)
    {
        float4 clip = mul(view_proj_mtx, float4(Xhit, 1.0));
        float2 uv = ClipToUv(clip);
        float3 Lsum = gDirectLit.SampleLevel(gNearestSampler, uv, 0).xyz;

        float3 hitN = GBufLoadNormal(uv);
        float3 hitAlbedo = gDif.SampleLevel(gNearestSampler, uv, 0).xyz;
        float3 hitRf0 = gSpc.SampleLevel(gNearestSampler, uv, 0).xyz;
        float m = gLit.SampleLevel(gNearestSampler, uv, 0).z;
        float hitRoughness = STL::Math::Sqrt01(m);

        float3 Lamb = EstimateAmbient(z, roughness, screenSpaceHitDist, -rayDirection, hitN, hitAlbedo, hitRf0, hitRoughness);
        Lsum += Lamb;

        float4 LsumPrev = GetFinalLightingFromPreviousFrame(Xhit, pixelUv);
        Lsum = lerp(Lsum, LsumPrev.xyz, LsumPrev.w);

        float4 result = PrepareOutputForNrd(Lsum * preserveIntegralCorrectness, screenSpaceHitDist, spec_hit_dist_params, z, roughness);
		
		#if (GI_USE_CHECKERBOARD == 1)
			WriteCheckerboardOutput(pixelPos, result);
		#else
			WriteFullOutput(pixelPos, result);
		#endif

        return;
    }

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

    bool isInScreen = payload.hitDist < 0.0;
    float hitDist = abs(payload.hitDist);
    Xhit += rayDirection * hitDist;

    float4 clip = mul(view_proj_mtx, float4(Xhit, 1.0));
    float2 uv = ClipToUv(clip);
    float border = lerp(SCREEN_EDGE_BORDER, 0.001, saturate(roughness * roughness / 0.8));
    float2 f = STL::Math::LinearStep(0.0, border, uv) * STL::Math::LinearStep(1.0, 1.0 - border, uv);
    float inScreenConfidence = f.x * f.y * float(isInScreen);

    // If hit is fully in-screen, just grab lighting from the g-buffer
    [branch]
    if (inScreenConfidence == 1.0)
    {
        float3 Lsum = gDirectLit.SampleLevel(gNearestSampler, uv, 0).xyz;

        float3 hitN = GBufLoadNormal(uv);
        float3 hitAlbedo = gDif.SampleLevel(gNearestSampler, uv, 0).xyz;
        float3 hitRf0 = gSpc.SampleLevel(gNearestSampler, uv, 0).xyz;
        float m = gLit.SampleLevel(gNearestSampler, uv, 0).z;
        float hitRoughness = STL::Math::Sqrt01(m);

        float3 Lamb = EstimateAmbient(z, roughness, hitDist, -rayDirection, hitN, hitAlbedo, hitRf0, hitRoughness);
        Lsum += Lamb;

        float4 LsumPrev = GetFinalLightingFromPreviousFrame(Xhit, pixelUv);
        Lsum = lerp(Lsum, LsumPrev.xyz, LsumPrev.w);

        float4 result = PrepareOutputForNrd(Lsum * preserveIntegralCorrectness, hitDist, spec_hit_dist_params, z, roughness);
		
		#if (GI_USE_CHECKERBOARD == 1)
			WriteCheckerboardOutput(pixelPos, result);
		#else
			WriteFullOutput(pixelPos, result);
		#endif

        return;
    }

    // Sky radiance
    float3 Csky = SampleSkybox(rayDirection);

    // If hit is a miss - just grab the sky color
    if (hitDist == INF)
    {
        float4 result = PrepareOutputForNrd(Csky * preserveIntegralCorrectness, hitDist, spec_hit_dist_params, z, roughness);
		
		#if (GI_USE_CHECKERBOARD == 1)
			WriteCheckerboardOutput(pixelPos, result);
		#else
			WriteFullOutput(pixelPos, result);
		#endif

        return;
    }

    // Hit is out-of-screen or occluded - we must compute lighting
    HitProps hitProps = GetHitProps(payload, ray);
	bool isLeaves = payload.uv.x == -1.f;

    float3 Cdiff, Cspec;
	if (isLeaves) {
		STL::BRDF::DirectLightingLeaves(hitProps.N, sun_direction, -rayDirection, hitProps.Rf0, hitProps.roughness, Cdiff, Cspec);
	} else {
		STL::BRDF::DirectLighting(hitProps.N, sun_direction, -rayDirection, hitProps.Rf0, hitProps.roughness, Cdiff, Cspec);
	}

    float3 Csun = sun_color;
    float3 Cimp = lerp(Csky, Csun, STL::Math::SmoothStep(0.0, 0.2, hitProps.roughness)); // simple sky importance sampling
	float minDist = isLeaves ? 0.02 : 0.0;

    float3 LsumRT = Cdiff * hitProps.albedo * Csun + Cspec * Cimp;
    LsumRT *= STL::Math::Pi(1.0); // to cancel 1/PI in BRDF math
	LsumRT *= CastShadowRayCustom(hitProps.X, LsumRT, sun_direction, minDist, INF);
    LsumRT += hitProps.emission;

    float3 Lamb = EstimateAmbient(z, roughness, hitDist, -rayDirection, hitProps.N, hitProps.albedo, hitProps.Rf0, hitProps.roughness);
    LsumRT += Lamb;

    // If hit is in-screen, mix RT with g-buffer lighting to hide hard edge
    [branch]
    if (inScreenConfidence != 0.0)
    {
        float3 Lsum = gDirectLit.SampleLevel(gNearestSampler, uv, 0).xyz;

        float4 clip = mul(view_proj_mtx, float4(Xhit, 1.0));
        float2 uv = ClipToUv(clip);

        float3 hitN = GBufLoadNormal(uv);
        float3 hitAlbedo = gDif.SampleLevel(gNearestSampler, uv, 0).xyz;
        float3 hitRf0 = gSpc.SampleLevel(gNearestSampler, uv, 0).xyz;
        float m = gLit.SampleLevel(gNearestSampler, uv, 0).z;
        float hitRoughness = STL::Math::Sqrt01(m);

        float3 Lamb = EstimateAmbient(z, roughness, hitDist, -rayDirection, hitN, hitAlbedo, hitRf0, hitRoughness);
        Lsum += Lamb;

        float4 LsumPrev = GetFinalLightingFromPreviousFrame(Xhit, pixelUv);
        Lsum = lerp(Lsum, LsumPrev.xyz, LsumPrev.w);

        LsumRT = lerp(LsumRT, Lsum, inScreenConfidence);
    }
    float4 result = PrepareOutputForNrd(LsumRT * preserveIntegralCorrectness, hitDist, spec_hit_dist_params, z, roughness);
	
	#if (GI_USE_CHECKERBOARD == 1)
		WriteCheckerboardOutput(pixelPos, result);
	#else
		WriteFullOutput(pixelPos, result);
	#endif
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
    float3 X = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float4 clip = mul(view_proj_mtx, float4(X, 1.0));
    float2 uv = ClipToUv(clip);
    float viewZ = GBufLoadLinearDepth(uv);
    float zError = abs(viewZ - clip.w) / ( max( viewZ, abs(clip.w) ) + 1e-6 );
    bool isInScreen = all( saturate(uv) == uv ) && zError < ALLOWED_Z_THRESHOLD;

    float2 f = STL::Math::LinearStep(0.0, SCREEN_EDGE_BORDER, uv) * STL::Math::LinearStep(1.0, 1.0 - SCREEN_EDGE_BORDER, uv);
    float inScreenConfidence = f.x * f.y * float(isInScreen);

    payload.hitDist = RayTCurrent() * (isInScreen ? -1 : 1);

    [branch]
    if (inScreenConfidence != 1.0)
    {
        uint3 indices = GetPrimitiveIndices();
        uint material_flags = GetInstanceMaterialFlags();

        payload.normal = GetWorldNormal(indices, attribs.barycentrics);
        payload.instanceId = GetInstanceID();

        [branch]
        if ((material_flags & RAY_TRACING_FLAG_EMS_ON) && (material_flags & RAY_TRACING_FLAG_EMS_UV1))
            payload.uv = GetUV1(indices, attribs.barycentrics);
        else if(material_flags & RAY_TRACING_FLAG_EMS_ON)
            payload.uv = GetUV0(indices, attribs.barycentrics);
		else if( any(abs(GetColor(indices, attribs.barycentrics) > 0.01)) && (gPerInstanceData[payload.instanceId].x >> 24) )
			payload.uv.x = -1.f;
    }
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
