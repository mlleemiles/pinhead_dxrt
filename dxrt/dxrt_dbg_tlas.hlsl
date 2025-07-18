#include "dxrt_header.hlsl"

#ifdef RAYGEN_AND_MISS_SHADERS

[shader("raygeneration")]
void ENTRY_POINT(raygeneration_main)()
{
    CommonRayGenShader(true);
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
    // TODO @dzhdan - use AnyHit instead
    payload.color = GetInstanceColor();
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
