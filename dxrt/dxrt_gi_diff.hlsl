#include "dxrt_header.hlsl"

#ifdef RAYGEN_AND_MISS_SHADERS

float3 SecondaryRay(float2 pixelPos, float3 X, float3 N, uint i, bool isOriginInScreen, inout float3 rayDirection) {

	float2 pixelUv = ClipToUv(mul(view_proj_mtx, float4(X, 1.0)));
    float3 V = normalize(cam_pos.xyz - X);
    float NoV = dot(N, V);

	float2 rnd = STL::Rng::GetFloat2();
	float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay(rnd);
	
    float3x3 mLocalBasis = STL::Geometry::GetBasis(N);
    rayDirection = STL::Geometry::RotateVectorInverse(mLocalBasis, rayLocal);
	//rayDirection = normalize(rayDirection + N);	//this fixes ray dir sometimes not in the hemisphere of hit
	
    float hitViewZ = mul(view_proj_mtx, float4(X, 1)).w;
    float3 Xoffset = OffsetRay(X, rayDirection);//GetXWithOffset(X, rayDirection, 0, hitViewZ);
	
	// screen space tracer is shit
	/*
	[branch]
	if (isOriginInScreen) {
		Xoffset = GetXWithOffset(X, rayDirection, 0, hitViewZ);
		float screenSpaceHitDist = 0;
		bool isHitFound = false;//TraceRayScreenSpace(Xoffset, hitViewZ, rayDirection, NoV, screenSpaceHitDist);
		
		Xoffset = Xoffset + rayDirection * screenSpaceHitDist;
		// If screen-space hit is found just grab lighting from the g-buffer
		[branch]
		if (isHitFound)
		{
			float4 clip = mul(view_proj_mtx, float4(Xoffset, 1.0));
			float2 uv = ClipToUv(clip);
			float3 Lsum = gDirectLit.SampleLevel(gNearestSampler, uv, 0).xyz;

			float3 hitN = GBufLoadNormal(uv);
			float3 hitAlbedo = gDif.SampleLevel(gNearestSampler, uv, 0).xyz;
			float3 hitRf0 = gSpc.SampleLevel(gNearestSampler, uv, 0).xyz;
			float m = gLit.SampleLevel(gNearestSampler, uv, 0).z;
			float hitRoughness = STL::Math::Sqrt01(m);

			float3 Lamb = EstimateAmbient(hitViewZ, 1.0, screenSpaceHitDist, -rayDirection, hitN, hitAlbedo, hitRf0, hitRoughness);
			Lsum += Lamb;

			float4 LsumPrev = GetFinalLightingFromPreviousFrame(Xoffset, pixelUv);
			Lsum = lerp(Lsum, LsumPrev.xyz, LsumPrev.w);

			return Lsum;

		}
	}
	*/
	
    RayDesc ray;
    ray.Origin = Xoffset;
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
    Xoffset += rayDirection * hitDist;

    // If hit is in-screen just grab lighting from the g-buffer
    [branch]
    if (isInScreen)
    {
        float4 clip = mul(view_proj_mtx, float4(Xoffset, 1.0));
        float2 uv = ClipToUv(clip);
        float3 Lsum = gDirectLit.SampleLevel(gNearestSampler, uv, 0).xyz;

        float3 hitN = GBufLoadNormal(uv);
        float3 hitAlbedo = gDif.SampleLevel(gNearestSampler, uv, 0).xyz;
        float3 hitRf0 = gSpc.SampleLevel(gNearestSampler, uv, 0).xyz;
        float m = gLit.SampleLevel(gNearestSampler, uv, 0).z;
        float hitRoughness = STL::Math::Sqrt01(m);

        float3 Lamb = EstimateAmbient(hitViewZ, 1.0, hitDist, -rayDirection, hitN, hitAlbedo, hitRf0, hitRoughness);
        Lsum += Lamb;

        float4 LsumPrev = GetFinalLightingFromPreviousFrame(Xoffset, pixelUv);
        Lsum = lerp(Lsum, LsumPrev.xyz, LsumPrev.w);

        return Lsum;//* REBLUR_FrontEnd_GetNormHitDist(hitDist, hitViewZ, diff_hit_dist_params, 1.0);
    }

    // Sky radiance
	// We don't want sky bounce because it accumulated over past frames and ambient in bvh is just simple sky amb
    float3 Csky = SampleSky(rayDirection) * SKYBOUNCE_SCALE;

    // If hit is a miss - just grab the sky color
    if (hitDist == INF)
    {
        return Csky;// * REBLUR_FrontEnd_GetNormHitDist(hitDist, hitViewZ, diff_hit_dist_params, 1.0);
    }

    // Hit is out-of-screen or occluded - we must compute lighting
    HitProps hitProps = GetHitProps(payload, ray);

    float3 Cdiff, Cspec;
    STL::BRDF::DirectLighting(hitProps.N, sun_direction, -rayDirection, hitProps.Rf0, hitProps.roughness, Cdiff, Cspec);

    float3 Csun = sun_color;
    float3 Cimp = lerp(Csky, Csun, STL::Math::SmoothStep(0.0, 0.2, hitProps.roughness)); // simple sky importance sampling

    float3 LsumRT = Cdiff * hitProps.albedo * Csun + Cspec * Cimp;
    LsumRT *= STL::Math::Pi(1.0); // to cancel 1/PI in BRDF math
    LsumRT *= CastShadowRayCustom(hitProps.X, LsumRT, sun_direction, 0.0, INF);
    LsumRT += hitProps.emission;

    float3 Lamb = EstimateAmbient(hitViewZ, 1.0, hitDist, -rayDirection, hitProps.N, hitProps.albedo, hitProps.Rf0, hitProps.roughness);
    LsumRT += Lamb;

	return LsumRT;// * REBLUR_FrontEnd_GetNormHitDist(hitDist, hitViewZ, diff_hit_dist_params, 1.0);
}

[shader("raygeneration")]
void ENTRY_POINT(raygeneration_main)()
{
	#if (GI_USE_CHECKERBOARD == 1)
		uint2 pixelPos = ApplyCheckerboard( DispatchRaysIndex().xy );
	#else
		uint2 pixelPos = ( DispatchRaysIndex().xy );
	#endif
    float2 pixelUv = (pixelPos + 0.5) * inv_rt_size;

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

		
	float3 diffuse = 0;
	float totalHitDist = 0;
	[unroll]
	for (uint i = 0; i < NUM_RAYS; ++i)
	{
		// Choose a ray
		STL::Rng::Initialize(pixelPos, dxrt_frame_idx * NUM_RAYS + i);

		float2 rnd = STL::Rng::GetFloat2();
		float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay(rnd);

		float3x3 mLocalBasis = STL::Geometry::GetBasis(N);
		float3 rayDirection = STL::Geometry::RotateVectorInverse(mLocalBasis, rayLocal);

		float3 firstHitX;
		float3 firstHitNormal;
		float firstHitDist;
		float3 secondaryDiffuse = 0;
		
		float3 screenSpaceAlbedo = 0;
		float3 screenSpaceRf0 = 0;
		float screenSpaceRoughness = 0;
		float4 screenSpaceSample = 0;

		// Screen space tracing
		float3 Xoffset = GetXWithOffset(X, N, V, z);

		float screenSpaceHitDist = 0;
		bool isHitFound = TraceRayScreenSpace(Xoffset, z, rayDirection, NoV, screenSpaceHitDist);

		float3 Xhit = Xoffset + rayDirection * screenSpaceHitDist;
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

			float3 Lamb = EstimateAmbient(z, 1.0, screenSpaceHitDist, -rayDirection, hitN, hitAlbedo, hitRf0, hitRoughness);
			Lsum += Lamb;

			float4 LsumPrev = GetFinalLightingFromPreviousFrame(Xhit, pixelUv);
			Lsum = lerp(Lsum, LsumPrev.xyz, LsumPrev.w);
			
			//diffuse += Lsum;
			//totalHitDist += screenSpaceHitDist;
			
			screenSpaceSample.xyz = Lsum;
			screenSpaceSample.w = screenSpaceHitDist;
			screenSpaceAlbedo = hitAlbedo;
			screenSpaceRf0 = hitRf0;
			screenSpaceRoughness = hitRoughness;

			firstHitX = Xhit;
			firstHitNormal = hitN;
			firstHitDist = screenSpaceHitDist;
		}

		// Ray tracing
		RayDesc ray;
		ray.Origin = Xoffset; // start where SSRT stopped
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
		Xoffset += rayDirection * hitDist;
		
		[branch]
		if ( isHitFound && (hitDist > screenSpaceSample.w) ) {
			diffuse += screenSpaceSample.xyz;
			totalHitDist += screenSpaceSample.w;
		
			float3 secondaryDirection, secondaryLum;
			secondaryLum = SecondaryRay(pixelPos, firstHitX, firstHitNormal, i, true, secondaryDirection);
			float3 Cdiff, Cspec;
			STL::BRDF::DirectLighting(firstHitNormal, secondaryDirection, -rayDirection, screenSpaceRf0, screenSpaceRoughness, Cdiff, Cspec);
			diffuse += secondaryLum*(Cdiff)*STL::Math::Pi(1.0)*screenSpaceAlbedo;
			continue;
		}

		// If hit is in-screen just grab lighting from the g-buffer
		[branch]
		if (isInScreen)
		{
			float4 clip = mul(view_proj_mtx, float4(Xoffset, 1.0));
			float2 uv = ClipToUv(clip);
			float3 Lsum = gDirectLit.SampleLevel(gNearestSampler, uv, 0).xyz;

			float3 hitN = GBufLoadNormal(uv);
			float3 hitAlbedo = gDif.SampleLevel(gNearestSampler, uv, 0).xyz;
			float3 hitRf0 = gSpc.SampleLevel(gNearestSampler, uv, 0).xyz;
			float m = gLit.SampleLevel(gNearestSampler, uv, 0).z;
			float hitRoughness = STL::Math::Sqrt01(m);

			float3 Lamb = EstimateAmbient(z, 1.0, hitDist, -rayDirection, hitN, hitAlbedo, hitRf0, hitRoughness);
			Lsum += Lamb;

			float4 LsumPrev = GetFinalLightingFromPreviousFrame(Xoffset, pixelUv);
			Lsum = lerp(Lsum, LsumPrev.xyz, LsumPrev.w);
			
			diffuse += Lsum;
			totalHitDist += hitDist;
			firstHitX = Xoffset;
			firstHitNormal = hitN;
			firstHitDist = hitDist;
			
			float3 secondaryDirection, secondaryLum;
			secondaryLum = SecondaryRay(pixelPos, firstHitX, firstHitNormal, i, true, secondaryDirection);
			float3 Cdiff, Cspec;
			STL::BRDF::DirectLighting(firstHitNormal, secondaryDirection, -rayDirection, hitRf0, hitRoughness, Cdiff, Cspec);
			diffuse += secondaryLum*(Cdiff)*STL::Math::Pi(1.0)*hitAlbedo;
			continue;
		}

		// Sky radiance
		float3 Csky = SampleSky(rayDirection);

		// If hit is a miss - just grab the sky color
		if (hitDist == INF)
		{
			diffuse += Csky;
			totalHitDist += hitDist;
			firstHitDist = hitDist;
			continue;
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

		float3 Lamb = EstimateAmbient(z, 1.0, hitDist, -rayDirection, hitProps.N, hitProps.albedo, hitProps.Rf0, hitProps.roughness);
		LsumRT += Lamb;
		
		diffuse += LsumRT;
		totalHitDist += hitDist;
		firstHitX = hitProps.X;
		firstHitNormal = hitProps.N;
		firstHitDist = hitDist;
		
		float3 secondaryDirection, secondaryLum;
		secondaryLum = SecondaryRay(pixelPos, firstHitX, firstHitNormal, i, false, secondaryDirection);
		if (isLeaves) {
			STL::BRDF::DirectLightingLeaves(firstHitNormal, secondaryDirection, -rayDirection, hitProps.Rf0, hitProps.roughness, Cdiff, Cspec);
		} else {
			STL::BRDF::DirectLighting(firstHitNormal, secondaryDirection, -rayDirection, hitProps.Rf0, hitProps.roughness, Cdiff, Cspec);
		}
		diffuse += secondaryLum*(Cdiff)*STL::Math::Pi(1.0)*hitProps.albedo;
	}
	float4 result = PrepareOutputForNrd(diffuse/NUM_RAYS, totalHitDist/NUM_RAYS, diff_hit_dist_params, z, 1.0);
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
    bool isInScreen = all( saturate(uv) == uv ) && zError < ALLOWED_Z_THRESHOLD && DEBUG < 10 && clip.w > 0;
	
	uint3 indices = GetPrimitiveIndices();
	float normalDiff = dot(GetWorldNormal(indices, attribs.barycentrics), GBufLoadNormal(uv));
	isInScreen = isInScreen && normalDiff > 1e-5;	// at least it should face the same side..

    payload.hitDist = RayTCurrent() * (isInScreen ? -1 : 1);

    [branch]
    if (!isInScreen)
    {
        //uint3 indices = GetPrimitiveIndices();
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
