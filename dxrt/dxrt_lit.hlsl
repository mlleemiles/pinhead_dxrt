#include "dxrt_header.hlsl"

float GetLightIntensity(float3 world_pos, float3 world_normal)
{
    float3 spot_to_point = world_pos - spot_pos.xyz;
    float inv_spot_to_point_distance = rsqrt( dot(spot_to_point, spot_to_point) + 1e-6 );

    // Cosine between light direction and vector from light to intersection point
    float cosL = dot(spot_to_point, spot_dir.xyz) * inv_spot_to_point_distance;

    // Attenuation by distance
    float attenuation = 1.0 / ( 1.0 + dot(spot_to_point, spot_to_point) * 0.05 );

    // Attenuation by spot light angle
    #if 1
        float a = (cosL - spot_cos_alpha) / (1.0 - spot_cos_alpha);
        attenuation *= pow(saturate(a), spot_falloff_end);
    #else
        float b = STL::Math::AcosApprox(spot_cos_alpha) * 1.2;
        float a = STL::Math::AcosApprox(cosL);
        attenuation *= STL::Math::Pow01(1.0 - a / b, 2.0);
        attenuation /= STL::Math::Pi(1.0);
    #endif

    return spot_intenisty * attenuation;
}

float MaxRayConeIntersection(float3 cone_origin, float3 cone_dir, float3 ray_origin, float3 ray_dir, float costheta)
{
    float3 co = ray_origin - cone_origin;
    float CoD = dot(co, cone_dir);
    float RoD = dot(ray_dir, cone_dir);
    float costhetaSq = costheta * costheta;

    float a = RoD * RoD - costhetaSq;
    float b = 2.0 * (RoD * CoD - dot(ray_dir, co) * costhetaSq);
    float c = CoD * CoD - dot(co, co) * costhetaSq;

    float det = b * b - 4.0 * a * c;
    if (det < EPS)
        return 0.0;

    det = sqrt(det);
    float d1 = 0.5 / a;
    float d2 = det * d1;
    float t1 = -b * d1 - d2;
    float t2 = -b * d1 + d2;

    // Avoid getting samples from The Evil Twin
    float3 p1 = ray_origin + ray_dir * t1;
    if (dot(p1 - cone_origin, cone_dir) < EPS)
        t1 = 0;

    float3 p2 = ray_origin + ray_dir * t2;
    if (dot(p2 - cone_origin, cone_dir) < EPS)
        t2 = 0;

    return max(t1, t2);
}

#ifdef RAYGEN_AND_MISS_SHADERS

void WriteOutput(float2 pixelPos, float4 val)
{
    [branch]
    if (dispatch_mode == DISPATCH_MODE_CHECKERBOARDING)
    {
        WriteCheckerboardOutput(pixelPos, val);
    }
    else
    {
        gOutput[pixelPos] = val;
    }
}

[shader("raygeneration")]
void ENTRY_POINT(raygeneration_main)()
{
    uint2 pixelPos = DispatchRaysIndex().xy;
    float2 rectSize = rt_size * (dispatch_mode == DISPATCH_MODE_QUARTER_RES ? 0.5 : 1.0);
    if (pixelPos.x >= rectSize.x || pixelPos.y >= rectSize.y)
        return;

    uint2 outPixelPos = pixelPos;

    [branch]
    if (dispatch_mode == DISPATCH_MODE_QUARTER_RES)
    {
        pixelPos =  ( pixelPos << 1 ) + uint2( dxrt_frame_idx & 0x1, ( dxrt_frame_idx >> 1 ) & 0x1 );

        // downsample g-buffer
        // gOutput1 - dpt
        // gOutput2 - mv
        // gOutput3 - nrm rgh
        gOutput1[outPixelPos].x = GBufLoadLinearDepth(uint3(pixelPos.x, pixelPos.y, 0));
        gOutput2[outPixelPos].xy = gMotionVectors[pixelPos];
        gOutput3[outPixelPos] = gNrm[pixelPos];
    }
    else
    {
        pixelPos = ApplyCheckerboard( DispatchRaysIndex().xy );
        outPixelPos = pixelPos;
    }

    float2 pixelUv = (pixelPos + 0.5) * inv_rt_size;

    // Early out
    float z = GBufLoadLinearDepth(pixelUv);

    [branch]
    if (z > ray_max_t || spot_intenisty == 0 || pixelPos.x >= rt_size.x || pixelPos.y >= rt_size.y)
    {
        WriteOutput(outPixelPos, 0);
        return;
    }

    // Center data
    float3 N = GBufLoadNormal(pixelPos);
    float3 pos_ws = ReconstructWorldPos(pixelUv, z);

    // Flip normal if NoV is suspiciously negative
    float3 V = normalize(cam_pos.xyz - pos_ws);
    if (dot(N, V) < -0.5)
        N = -N;

    // Choose a ray
    STL::Rng::Initialize(pixelPos, dxrt_frame_idx);

    float3x3 basis = STL::Geometry::GetBasis(N);
    float3 pos_ws_offset = GetXWithOffset(pos_ws, N, V, z);

    float maxt = 0.0;
    float samples = 0.0;
    float3 direction;

    // - We still use 1rpp, but we want to find a direction which is potentially valid
    // - To get proper sampling we must take into account number of "zeros" we found
    // - Actually, FLASHLIGHT_MAX_VIRTUAL_SAMPLES can be set to a really big value, but I just want to avoid a
    //   situation where the loop becomes "semi infinite" for an unknown reason
    // - Setting FLASHLIGHT_MAX_VIRTUAL_SAMPLES = 1 returns old behavior

    while( maxt == 0.0 && samples < FLASHLIGHT_MAX_VIRTUAL_SAMPLES )
    {
        float2 rnd = STL::Rng::GetFloat2();
        float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay(rnd);

        direction = STL::Geometry::RotateVectorInverse( basis, rayLocal );
        maxt = MaxRayConeIntersection(spot_pos.xyz, spot_dir.xyz, pos_ws_offset, direction, spot_cos_alpha);

        samples += 1.0;
    }

    // Ray tracing
    RayDesc ray;
    ray.Origin = pos_ws_offset;
    ray.Direction = direction;
    ray.TMin = 0;
    ray.TMax = maxt;

    // Debug stuff
    #if( DEBUG == 100 )
        ray.Origin = cam_pos.xyz;
        ray.Direction = -V;
        ray.TMax = INF;
    #endif

    RayPayload payload = (RayPayload)0;
    {
        const uint rayFlags = CULLING_FLAGS;
        const uint instanceInclusionMask = ray.TMax == 0.0 ? 0 : EInstanceRayVisibility::FullRays;
        const uint rayContributionToHitGroupIndex = FULL_RAY_ID;
        const uint multiplierForGeometryContributionToHitGroupIndex = 0;
        const uint missShaderIndex = FULL_RAY_ID;

        TraceRay(g_TLAS, rayFlags, instanceInclusionMask, rayContributionToHitGroupIndex, multiplierForGeometryContributionToHitGroupIndex, missShaderIndex, ray, payload);
        Report(_TraceRay);
    }

    // Lighting
    HitProps hitProps = GetHitProps(payload, ray);

    float3 L = normalize(spot_pos.xyz - hitProps.X);
    float intensity = GetLightIntensity(hitProps.X, hitProps.N);
    float dist_to_flashlight = length(spot_pos.xyz - hitProps.X);

    float3 Cdiff, Cspec;
    STL::BRDF::DirectLighting(hitProps.N, L, -ray.Direction, hitProps.Rf0, hitProps.roughness, Cdiff, Cspec);

    float3 Lsum = (Cdiff * hitProps.albedo + Cspec) * spot_color * intensity;
    Lsum *= STL::Math::Pi(1.0); // to cancel 1/PI in BRDF math
    Lsum *= float( payload.hitDist != INF );
    Lsum *= CastShadowRay(hitProps.X, Lsum, L, max(dist_to_flashlight - FLASHLIGHT_MAGIC_OFFSET, 0.0));
    Lsum /= samples + 0.00001;

    // Debug stuff
    #if( DEBUG == 100 )
        if (pixelUv.x > 0.5)
            Lsum = gDirectLit.SampleLevel(gNearestSampler, pixelUv, 0).xyz;

        // Split screen - vertical line
        float separator = 0.5;
        float verticalLine = saturate(1.0 - abs(pixelUv.x - separator) * rt_size.x / 2.5);
        verticalLine = saturate(verticalLine / 0.5);
        Lsum = lerp(Lsum, float3(1,0,0) * verticalLine, verticalLine);
    #endif

    // Output
    float4 result = PrepareOutputForNrd(Lsum, payload.hitDist, flashlight_hit_dist_params, z, 1.0);
    WriteOutput(outPixelPos, result);
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

[shader("closesthit")]
void ENTRY_POINT(closesthit_main)(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    uint3 indices = GetPrimitiveIndices();

    payload.hitDist = RayTCurrent();
    payload.normal = GetWorldNormal(indices, attribs.barycentrics);
    payload.instanceId = GetInstanceID();
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
