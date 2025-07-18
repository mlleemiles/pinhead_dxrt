#include "dxrt_header.hlsl"

#ifdef RAYGEN_AND_MISS_SHADERS

[shader("raygeneration")]
void ENTRY_POINT(raygeneration_main)()
{
    CommonRayGenShader();
}

[shader("miss")]
void ENTRY_POINT(miss_main)(inout DebugRayPayload payload)
{
    payload.color = 0.0;
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
    uint3 indices = GetPrimitiveIndices();
    float3 pos = GetPosition(indices, attribs.barycentrics);

    float3 pos_ws = mul((float3x3)ObjectToWorld3x4(), pos).xyz;
    payload.color = frac(pos_ws);
}

[shader("anyhit")]
void ENTRY_POINT(anyhit_main)(inout DebugRayPayload payload,  in BuiltInTriangleIntersectionAttributes attribs)
{
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