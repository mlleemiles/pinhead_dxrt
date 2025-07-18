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

//
// Full rays
//
[shader("closesthit")]
void ENTRY_POINT(closesthit_main)(inout DebugRayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    uint pos = PrimitiveIndex() * 3;
    uint i0 = gIBLocal[pos].i;
    uint i1 = gIBLocal[pos + 1].i;
    uint i2 = gIBLocal[pos + 2].i;

    float3 N;
    uint3 indices = GetPrimitiveIndices();
#if 0
    float3 v1 =  GetPosition(indices.x);
    float3 v2 =  GetPosition(indices.y);
    float3 v3 =  GetPosition(indices.z);

    float3 vab = v2 - v1;
    float3 vac = v3 - v1;
    float3 Ngeom = normalize(cross(vab, vac));

    N = Ngeom;
#else
    float3 Nvertex = GetNormal(indices, attribs.barycentrics);

    N = Nvertex;
#endif

    N = mul((float3x3)ObjectToWorld3x4(), N).xyz;
    payload.color = (N + 1) / 2.0;
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