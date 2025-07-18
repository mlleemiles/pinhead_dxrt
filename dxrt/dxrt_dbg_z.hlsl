#include "dxrt_header.hlsl"

#ifdef RAYGEN_AND_MISS_SHADERS

[shader("raygeneration")]
void ENTRY_POINT(raygeneration_main)()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    //float2 uv = (float2)launchIndex / (float2)DispatchRaysDimensions().xy;
    int3 coords = int3(launchIndex, 0);
    float z = GBufLoadLinearDepth(coords);

    float2 pixelUv = (launchIndex + 0.5) * inv_rt_size;
    float3 pos_ws = ReconstructWorldPos(pixelUv, z);

    RayDesc ray;

    float3 N = normalize(gNrm.Load(coords).xyz * 2.f - 1.f);

    float gbuffer_distance = distance(pos_ws, cam_pos.xyz);

    float max_ray_dist = gbuffer_distance * 1.001;
    float min_ray_dst = gbuffer_distance * 1.0 / 1.001;

    ray.TMin = ray_min_t;
    ray.TMax = min(max_ray_dist, ray_max_t);
    ray.Origin = cam_pos.xyz;
    ray.Direction = normalize(pos_ws - cam_pos.xyz);

    DebugRayPayload payload = (DebugRayPayload)0;
    TraceRay(g_TLAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES /*rayFlags*/, 0xFF, FULL_RAY_ID /* ray index*/, 0, FULL_RAY_ID, ray, payload);

    float ray_distance = payload.color.y;

    float3 out_clr = gbuffer_distance < ray_distance ? float3(1,0,0) : float3(0,1,0);

    // red = gbuffer is closer than rayhit
    // green = rayhit is closer than gbuffer

    out_clr *= payload.color.x;

    float is_ok = ray_distance >= min_ray_dst && ray_distance <= max_ray_dist ? 1.0 : 0.0;

    gOutput[launchIndex.xy].xyz = lerp(out_clr, gDif.Load(coords).xyz, is_ok);
}

[shader("miss")]
void ENTRY_POINT(miss_main)(inout DebugRayPayload payload)
{
    payload.color.x = 0.0;
    payload.color.y = 0.0;
}

[shader("miss")]
void ENTRY_POINT(shadow_miss_main)(inout ShadowRayPayload payload)
{
    payload.hitDist = INF;
}

#else // RAYGEN_AND_MISS_SHADERS

[shader("closesthit")]
void ENTRY_POINT(closesthit_main)(inout DebugRayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.color.x = 1.0;
    payload.color.y = RayTCurrent();
}

[shader("anyhit")]
void ENTRY_POINT(anyhit_main)(inout DebugRayPayload payload,  in BuiltInTriangleIntersectionAttributes attribs)
{
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
    CommonAnyHitShader(attribs);
}

#endif // RAYGEN_AND_MISS_SHADERS
