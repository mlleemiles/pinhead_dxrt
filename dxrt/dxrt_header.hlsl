#include "dxrt_commons.hlsl"

#include "stl.hlsl"

/*
TODO:
- add to CB "tanHalfFovY"

- f_non_rt_ambient_amount = 1 is very dangerous and must be avoided, because it adds a lot
of wrong energy. A value < 0.5 is OK (?). But specular part MUST be multiplied by RTSO, not RTAO

- add fog density to CB data

- is " EInstanceRayVisibility::ShadowRays" needed? (no objections if it is)

- provide "vertex data" type in "instanceID" which is so needed for TraceRayInline

- Instance data optimizations:
    7:7:6 Rf0 + 7 roughness + 5 flags = 32 bits
    8:8:8 albedo + 8 scale = 32 bits
    move RAY_TRACING_FLAG_EMS_ON to emission scale
    Decoding:
        float4 temp1 = STL::Packing::UintToRgba(packedMaterial.x, 7, 7, 6, 7);
        float4 temp2 = STL::Packing::UintToRgba(packedMaterial.y, 8, 8, 8, 8);
        temp1 = STL::Color::GammaToLinear(temp1);
        temp2.xyz = STL::Color::GammaToLinear(temp2.xyz);
        albedo = temp1.xyz;
        roughness = temp1.w * temp1.w; // <= there is a mystery (I will try to explain)
        Rf0 = temp2.xyz;
        scale = SomeRemap(temp2.w);
        bits = packedMaterial.x >> 27;

For DZ:
- increase "sun_tan_angular_radius" in foggy environment

- return back "ignore mirror specular"?
*/

/*
for SECONDARY HITS:
    0 - no debug
    1 - show AO / SO
    2 - highlight pixels where screen space tracer found a hit
for PRIMARY HITS (in diffuse debug output only):
    10 - show direct lighting of primary rays
    11 - show albedo
    12 - show Rf0
    13 - show roughness
    14 - show emission
for FLASHLIGHT:
    100 - show direct lighting
for SHADOWS:
    USE_INDIRECT => 0
    SetVarFloat f_non_rt_ambient_amount 0
*/

// Debug
#define DEBUG                               0
#define DEBUG_INTENSITY_SCALE               1.0

// Constants
#define RAY_TRACING_FLAG_EMS_ON             0x01
#define RAY_TRACING_FLAG_SUN_EMS            0x02
#define RAY_TRACING_FLAG_SKY_EMS            0x04
#define RAY_TRACING_FLAG_EMS_UV1            0x08
#define RAY_TRACING_FLAG_EMS_VARLIST        0x10
#define RAY_TRACING_FLAG_EMS_USR            0x20
#define EPS                                 1e-5

// Switches
#define USE_SCREEN_SPACE_TRACE              1 // default = 1
#define USE_INDIRECT                        1 // default = 1
#define USE_AMBIENT                         1 // default = 1
#define USE_EMISSION                        1 // default = 1
#define USE_TRACE_RAY_INLINE                0 // WIP
#define USE_CHECKERBOARDING_FOR_SHADOWS     0 // WIP
#define USE_ANY_HIT_SHADERS                 1 // 0 = debugging purposes

// Settings
#define CULLING_FLAGS                       (RAY_FLAG_CULL_BACK_FACING_TRIANGLES)
#define INF                                 1000.0 // m
#define SCREEN_SPACE_MAX_SAMPLE_NUM         32 // 32-64
#define SCREEN_SPACE_Z_THRESHOLD            0.02 // normalized %
#define ALLOWED_Z_THRESHOLD                 0.01 // normalized %
#define FLASHLIGHT_MAX_VIRTUAL_SAMPLES      32
#define FLASHLIGHT_MAGIC_OFFSET             7.0 // m (TODO: WTF!)
#define SCREEN_EDGE_BORDER                  0.1 // normalized %

#define DISPATCH_MODE_FULL_RES 0
#define DISPATCH_MODE_CHECKERBOARDING 1
#define DISPATCH_MODE_QUARTER_RES 2

#define USE_OEM_DIFF_AS_SKY 1 // 1 for best match with non rt
#define NUM_RAYS 1
#define SKYBOUNCE_SCALE 0.5
#define GI_USE_CHECKERBOARD	0

//========================================================================

cbuffer PerFrame : register(b0)
{
    float4 cam_pos;
    float4 inv_frustum0;
    float4 inv_frustum1;
    row_major float3x4 view_mtx;
    row_major float4x4 view_proj_mtx;

    float4 prev_cam_pos;
    float4 prev_inv_frustum1;
    row_major float3x4 prev_view_mtx;
    row_major float4x4 prev_view_proj_mtx;

    float4 spot_pos;
    float4 spot_dir;

    float spot_cos_alpha;
    float spot_range;
    float spot_falloff_start;
    float spot_falloff_end;

    float3 spot_color;
    float spot_intenisty;

    uint dxrt_rpp;
    uint dxrt_frame_idx;
    float ray_min_t;
    float ray_max_t;

    float3 sun_direction;
    float sun_tan_angular_radius;

    float3 sun_color;
    uint blue_noise_enabled;

    float4  flashlight_hit_dist_params;
    float4  diff_hit_dist_params;
    float4  spec_hit_dist_params;

    float2 jitter;
    float2 inv_rt_size;

    float2 rt_size;
    float weather_blend;
    float skybox_intensity;

    float non_rt_ambient_amount;
    float3 emissive_intensity;
};

cbuffer PerDispatch : register(b1)
{
    uint dispatch_mode;
};

float D3DX_INT_to_FLOAT(int _V, float _Scale)
{
    float Scaled = (float)_V / _Scale;
    return max(Scaled, -1.0f);
}

float4 D3DX_R8G8B8A8_SNORM_to_FLOAT4(uint packedInput)
{
    float4 unpackedOutput;
    int4 signExtendedBits;
    signExtendedBits.x = (int)(packedInput << 24) >> 24;
    signExtendedBits.y = (int)((packedInput << 16) & 0xff000000) >> 24;
    signExtendedBits.z = (int)((packedInput << 8) & 0xff000000) >> 24;
    signExtendedBits.w = (int)(packedInput & 0xff000000) >> 24;
    unpackedOutput.x = D3DX_INT_to_FLOAT(signExtendedBits.x, 127);
    unpackedOutput.y = D3DX_INT_to_FLOAT(signExtendedBits.y, 127);
    unpackedOutput.z = D3DX_INT_to_FLOAT(signExtendedBits.z, 127);
    unpackedOutput.w = D3DX_INT_to_FLOAT(signExtendedBits.w, 127);
    return unpackedOutput;
}

////////////////////////////////////////////////////////////////////////////////////////
float2 Frust_Out_PP(float2 v_pos_in, float4 inv_frustum_params)
{
    return v_pos_in * rt_size * inv_frustum_params.xy + inv_frustum_params.zw;
}

float3 Cam_Dir_Out_PP(float2 v_pos_in, float4 inv_frustum_params)
{
    float3 v_cam_dir = -float3(1.0, 1.0, 1.0);
    v_cam_dir.xy = Frust_Out_PP(v_pos_in, inv_frustum_params);
    return v_cam_dir;
}

float3 ReconstructWorldPos(float2 pixelUv, float z)
{
    float3 cam_dir_cs = Cam_Dir_Out_PP(pixelUv + jitter * inv_rt_size, inv_frustum1);
    float3 cam_dir_ws = mul(cam_dir_cs, view_mtx).xyz;

    return cam_pos.xyz + cam_dir_ws * z;
}

float3 ReconstructPreviousWorldPos(float2 prevPixelUv, float prev_z)
{
    float3 prev_cam_dir_cs = Cam_Dir_Out_PP(prevPixelUv, prev_inv_frustum1); // TODO: + prev jitter?
    float3 prev_cam_dir_ws = mul(prev_cam_dir_cs, prev_view_mtx).xyz;

    return prev_cam_pos.xyz + prev_cam_dir_ws * prev_z;
}

////////////////////////////////////////////////////////////////////////////////////////
#if defined(VTX_FORMAT_COMMON)
    #define ENTRY_POINT(x) vtx_common_##x

    struct SVertexFmt
    {
        float3 position;
        uint normal_packed;
        half2 uv[2];
        uint tangent_packed;
        uint color_packed;
    };

    float3 GetPosition(in SVertexFmt vtx)
    {
        return vtx.position;
    }

    float3 GetNormal(in SVertexFmt vtx)
    {
        return D3DX_R8G8B8A8_SNORM_to_FLOAT4(vtx.normal_packed).xyz;
    }

    half2 GetUVChannnel0(in SVertexFmt vtx)
    {
        return vtx.uv[0];
    }

    half2 GetUVChannnel1(in SVertexFmt vtx)
    {
        return vtx.uv[1];
    }
	
    float3 GetColor(in SVertexFmt vtx)
    {
		return D3DX_R8G8B8A8_SNORM_to_FLOAT4(vtx.color_packed).xyz;
        //return STL::Packing::UintToRgba(vtx.color_packed, 8, 8, 8, 8).xyz;
    }
#elif defined(VTX_FORMAT_TINY)
    #define ENTRY_POINT(x) vtx_tiny_##x

    struct SVertexFmt
    {
        half4 position;
        uint normal_packed;
        half2 uv;
    };

    float3 GetPosition(in SVertexFmt vtx)
    {
        return float3(vtx.position.x, vtx.position.y, vtx.position.z);
    }

    float3 GetNormal(in SVertexFmt vtx)
    {
        return D3DX_R8G8B8A8_SNORM_to_FLOAT4(vtx.normal_packed).xyz;
    }

    half2 GetUVChannnel0(in SVertexFmt vtx)
    {
        return vtx.uv;
    }

    half2 GetUVChannnel1(in SVertexFmt vtx)
    {
        return vtx.uv;
    }
	
    float3 GetColor(in SVertexFmt vtx)
    {
        return float3(0.0, 0.0, 0.0);
    }
#elif defined(VTX_FORMAT_COMPACT)
    #define ENTRY_POINT(x) vtx_compact_##x

    struct SVertexFmt
    {
        half4 position;
        uint normal_packed;
        half2 uv;
        uint tangent_packed;
    };

    float3 GetPosition(in SVertexFmt vtx)
    {
        return float3(vtx.position.x, vtx.position.y, vtx.position.z);
    }

    float3 GetNormal(in SVertexFmt vtx)
    {
        return D3DX_R8G8B8A8_SNORM_to_FLOAT4(vtx.normal_packed).xyz;
    }

    half2 GetUVChannnel0(in SVertexFmt vtx)
    {
        return vtx.uv;
    }

    half2 GetUVChannnel1(in SVertexFmt vtx)
    {
        return vtx.uv;
    }
	
    float3 GetColor(in SVertexFmt vtx)
    {
        return float3(0.0, 0.0, 0.0);
    }
#elif defined(VTX_FORMAT_SKINNED)
    #define ENTRY_POINT(x) vtx_skinned_##x

    struct SVertexFmt
    {
        float3 position;
        uint blend_weight_packed;
        uint blend_indices;
        uint normal_packed;
        half2 uv[2];
        uint tangent_packed;
        uint color_packed;
    };

    float3 GetPosition(in SVertexFmt vtx)
    {
        return vtx.position;
    }

    float3 GetNormal(in SVertexFmt vtx)
    {
        return D3DX_R8G8B8A8_SNORM_to_FLOAT4(vtx.normal_packed).xyz;
    }

    half2 GetUVChannnel0(in SVertexFmt vtx)
    {
        return vtx.uv[0];
    }

    half2 GetUVChannnel1(in SVertexFmt vtx)
    {
        return vtx.uv[1];
    }
	
    float3 GetColor(in SVertexFmt vtx)
    {
        return D3DX_R8G8B8A8_SNORM_to_FLOAT4(vtx.color_packed).xyz;
    }
#elif defined(VTX_FORMAT_SKNBLAS)
    #define ENTRY_POINT(x) vtx_sknblas_##x

    struct SVertexFmt
    {
        float3 position;
        uint blend_weight_packed;
        uint blend_indices;
        uint normal_packed;
        half2 uv[2];
        uint tangent_packed;
        uint color_packed;
    };

    float3 GetPosition(in SVertexFmt vtx)
    {
        return vtx.position;
    }

    float3 GetNormal(in SVertexFmt vtx)
    {
        return float3(0.0, 0.0, 1.0);
    }

    half2 GetUVChannnel0(in SVertexFmt vtx)
    {
        return half2(0.0, 0.0);
    }

    half2 GetUVChannnel1(in SVertexFmt vtx)
    {
        return half2(0.0, 0.0);
    }
	
    float3 GetColor(in SVertexFmt vtx)
    {
        return float3(0.0, 0.0, 0.0);
    }
#else
    // Dummy vertex format for compiling ray gen and miss shaders
    #define ENTRY_POINT(x) x
    #define RAYGEN_AND_MISS_SHADERS 1

    struct SVertexFmt
    {
    };

    float3 GetPosition(in SVertexFmt vtx)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float3 GetNormal(in SVertexFmt vtx)
    {
        return float3(0.0, 0.0, 1.0);
    }

    half2 GetUVChannnel0(in SVertexFmt vtx)
    {
        return half2(0.0, 0.0);
    }

    half2 GetUVChannnel1(in SVertexFmt vtx)
    {
        return half2(0.0, 0.0);
    }
	
    float3 GetColor(in SVertexFmt vtx)
    {
        return float3(0.0, 0.0, 0.0);
    }
#endif

////////////////////////////////////////////////////////////////////////////////////////

struct RayPayload
{
    float3 normal;
    float2 uv;
    float hitDist;
    uint instanceId;
};

// Used for debug shaders (like dbg_tlas)
// size has to be <= RayPayload
struct DebugRayPayload
{
    float3 color;
};

struct ShadowRayPayload
{
    float hitDist;
	//uint hitCount;
	//bool useSelfShadow;
};

#define FULL_RAY_ID 0
#define SHADOW_RAY_ID 1

enum EInstanceRayVisibility : uint
{
    LightRays = 0x01,
    ShadowRays = 0x02,
    DistantShadowcasterRays = 0x04,

    FullRays = LightRays | ShadowRays,
    DistantShadowRays = ShadowRays | DistantShadowcasterRays,
};

struct IndexBuffer
{
    uint16_t i;
};

// Outputs
RWTexture2D<float4> gOutput             : register(u0);
RWTexture2D<float4> gOutput1            : register(u1);
RWTexture2D<float4> gOutput2            : register(u2);
RWTexture2D<float4> gOutput3            : register(u3);

// Make this user managed pool (able to set from ppfx)
// space 0
Texture2D<float>    gZ                  : register(t0);
Texture2D<float4>   gDif                : register(t1);
Texture2D<float4>   gNrm                : register(t2);
Texture2D<float4>   gSpc                : register(t3);
Texture2D<float4>   gLit                : register(t4); // rgb = trn, ocl, rgh
Texture2D<uint3>    gScrambling1spp     : register(t5);
Texture2D<uint3>    gScrambling32spp    : register(t6);
Texture2D<uint4>    gSobol              : register(t7);
Texture2D<float4>   gSkybox64           : register(t8);
Texture2D<float4>   gSkybox64Next       : register(t9);
Texture2D<float4>   gPrevSceneLit       : register(t10);
Texture2D<float>    gPrevZLinear        : register(t11);
Texture2D<float2>   gMotionVectors      : register(t12);
Texture2D<float3>   gAvgAmbient         : register(t13);
Texture2D<float3>   gDirectLit          : register(t14);
Texture2D<float3>   gGbufEms            : register(t15);
Texture2D<float3>   gOemDif             : register(t16);

// Make this renderer managed pool (alway set by implementation)
// space 1
RaytracingAccelerationStructure g_TLAS  : register(t0, space1);
StructuredBuffer<uint4> gPerInstanceData : register(t1, space1); // sync with rd3d12_raytracing.h

// Unbounded arrays start here, each in its own register space
Texture2D<float> gClp[] : register(t0, space2);
Texture2D<float3> gEms[] : register(t0, space3);

SamplerState gNearestSampler : register(s0);
SamplerState gLinearSampler : register(s1);

// Local root signature
StructuredBuffer<IndexBuffer> gIBLocal : register(t0, space10);  // uint16_t needs /enable-16bit-types @compiler options
StructuredBuffer<SVertexFmt> gVBLocal : register(t1, space10);

////////////////////////////////////////////////////////////////////////////////////////

uint GetInstanceID()
{
    uint instance_id = InstanceID();
    return (instance_id >> 1) + (instance_id & 0x1) * GeometryIndex();
}

float3 UnpackAndDegammaColor(uint pckd)
{
    float3 color;
    color.x = (float)(((pckd) & 0x000000ff)) / 255;
    color.y = (float)(((pckd >> 8) & 0x000000ff)) / 255;
    color.z = (float)(((pckd >> 16) & 0x000000ff)) / 255;

    return pow(color, 2.2);
}

float3 GetInstanceColor()
{
    uint data = gPerInstanceData[GetInstanceID()].x;
    return UnpackAndDegammaColor(data);
}

uint GetInstanceFlags()
{
    uint data = gPerInstanceData[GetInstanceID()].x;
    data = (data >> 24) & 0x000000ff;

    return data;
}

float4 GetInstanceSpecularRoughness()
{
    uint data = gPerInstanceData[GetInstanceID()].y;
    float4 spc_rgh;
    spc_rgh.w = (float)(((data >> 24) & 0x000000ff)) / 255;

    spc_rgh.xyz = UnpackAndDegammaColor(data);
    spc_rgh.w = spc_rgh.w * spc_rgh.w;

    return spc_rgh;
}

float3 GetInstanceEmissive()
{
    uint data = gPerInstanceData[GetInstanceID()].z;
    return UnpackAndDegammaColor(data);
}

uint GetInstanceMaterialFlags()
{
    return USE_EMISSION ? (gPerInstanceData[GetInstanceID()].z >> 24) : 0;
}

float GetInstanceEmissiveScale()
{
    return asfloat(gPerInstanceData[GetInstanceID()].w);
}

uint4 GetInstancePackedMaterial()
{
    return gPerInstanceData[GetInstanceID()];
}

uint3 GetPrimitiveIndices()
{
    uint pos = PrimitiveIndex() * 3;
    uint3 indices;
    indices.x = gIBLocal[pos].i;
    indices.y = gIBLocal[pos + 1].i;
    indices.z = gIBLocal[pos + 2].i;

    return indices;
}

float3 GetColor(uint index)
{
    return GetColor(gVBLocal[index]);
}

half2 GetUV0(uint index)
{
    return GetUVChannnel0(gVBLocal[index]);
}

half2 GetUV1(uint index)
{
    return GetUVChannnel1(gVBLocal[index]);
}

float3 GetPosition(uint index)
{
    return GetPosition(gVBLocal[index]);
}

float3 GetNormal(uint index)
{
    return GetNormal(gVBLocal[index]);
}

float3 GetColor(uint3 indices, float2 _barycentrics)
{
    float3 color0 = GetColor(indices.x);
    float3 color1 = GetColor(indices.y);
    float3 color2 = GetColor(indices.z);

    float3 barycentrics;
    barycentrics.yz = _barycentrics.xy;
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

    float3 bary_uv = (barycentrics.x * color0 + barycentrics.y * color1 + barycentrics.z * color2);
    return bary_uv;
}

float2 GetUV0(uint3 indices, float2 _barycentrics)
{
    half2 uv0 = GetUV0(indices.x);
    half2 uv1 = GetUV0(indices.y);
    half2 uv2 = GetUV0(indices.z);

    float3 barycentrics;
    barycentrics.yz = _barycentrics.xy;
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

    float2 bary_uv = (barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2);
    return bary_uv;
}

float2 GetUV1(uint3 indices, float2 _barycentrics)
{
    half2 uv0 = GetUV1(indices.x);
    half2 uv1 = GetUV1(indices.y);
    half2 uv2 = GetUV1(indices.z);

    float3 barycentrics;
    barycentrics.yz = _barycentrics.xy;
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

    float2 bary_uv = (barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2);
    return bary_uv;
}

float3 GetPosition(uint3 indices, float2 _barycentrics)
{
    float3 v0 = GetPosition(indices.x);
    float3 v1 = GetPosition(indices.y);
    float3 v2 = GetPosition(indices.z);

    float3 barycentrics;
    barycentrics.yz = _barycentrics.xy;
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

    float3 bary_pos = (barycentrics.x * v0 + barycentrics.y * v1 + barycentrics.z * v2);
    return bary_pos;
}

float3 GetNormal(uint3 indices, float2 _barycentrics)
{
    float3 n0 = GetNormal(indices.x);
    float3 n1 = GetNormal(indices.y);
    float3 n2 = GetNormal(indices.z);

    float3 barycentrics;
    barycentrics.yz = _barycentrics.xy;
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

    float3 bary_norm = normalize(barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2);
    return HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? bary_norm : -bary_norm;
}

float3 GetWorldNormal(uint3 indices, float2 _barycentrics)
{
    float3 N = GetNormal(indices, _barycentrics);
	float3 N2 = GetColor(indices, _barycentrics);
	N = normalize(N + N2);

    return mul((float3x3)ObjectToWorld3x4(), N).xyz;
}

float GBufLoadLinearDepth(int3 coords)
{
    // Sample_zbuffer
    return gZ.Load(coords);
}

float GBufLoadLinearDepth(float2 uv)
{
    // Sample_zbuffer
    return gZ.SampleLevel(gNearestSampler, uv, 0);
}

float3 GBufLoadNormal(uint2 pixelPos)
{
    float3 n = gNrm[pixelPos].xyz;
    return normalize(n * 2.f - 1.f);
}

float3 GBufLoadNormal(float2 uv)
{
    float3 n = gNrm.SampleLevel(gNearestSampler, uv, 0).xyz;
    return normalize(n * 2.f - 1.f);
}

///////////////////////////////////////////////////////////////////////////////////

void CommonRayGenShader(bool use_mirror = true)
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 pixel_uv = ((float2)launchIndex + 0.5) * inv_rt_size;
    int3 coords = int3(launchIndex, 0);

    float z = GBufLoadLinearDepth(coords);
    float3 pos_ws = ReconstructWorldPos(pixel_uv, z);
    float3 cam_dir = normalize(pos_ws - cam_pos.xyz);

    RayDesc ray;
    ray.TMin = ray_min_t;
    ray.TMax = INF;
    ray.Origin = cam_pos.xyz;
    ray.Direction = cam_dir;

    if (use_mirror)
    {
        if (pixel_uv.y > 0.5)
        {
            ray.Direction *= float3(-1.0, -1.0, -1.0);
            ray.Direction.y = (ray.Direction.y) - 0.5 * 0.5;
            ray.Direction.y = -ray.Direction.y;
        }
        else
            ray.Direction.y = (ray.Direction.y) - 0.5 * 0.5;
    }

    DebugRayPayload payload = (DebugRayPayload)0;
    TraceRay(g_TLAS, 0 /*rayFlags*/, 0xFF, FULL_RAY_ID/* ray index*/, 0, FULL_RAY_ID, ray, payload);

    float3 col = STL::Color::LinearToSrgb(payload.color);

    gOutput[launchIndex.xy] = float4(col, 1);
}

void CommonAnyHitShader(in BuiltInTriangleIntersectionAttributes attribs)
{
    #if( USE_ANY_HIT_SHADERS == 0 )
        AcceptHitAndEndSearch();
    #else
        uint3 indices = GetPrimitiveIndices();
        float2 uv = GetUV0(indices, attribs.barycentrics);
        uint idx = GetInstanceID();

        float w, h;
        gClp[idx].GetDimensions(w, h); // TODO: if I only had it as a constant...
        float mipNum = log2( max(w, h) );
        float mip = max(mipNum - 7, 0.0); // Use 128x128 (or the best) to avoid cache trashing

        float clip_val = gClp[idx].SampleLevel(gLinearSampler, uv, mip).x;

        if (clip_val < 0.0)
            IgnoreHit(); // plane disappears
    #endif
}

#if defined(ENABLE_RT_DBG)
    RWStructuredBuffer<SRtDbg> gDbg : register(u10);

    void Report(uint what, uint2 pos, uint2 size)
    {
        uint id = (size.x * pos.y + pos.x);

        if (id > 1)
        {
            InterlockedAdd(gDbg[id].counter[what], 1);

            InterlockedMax(gDbg[0].counter[what], gDbg[id].counter[what]); // 0th pixel keeps max value
            InterlockedAdd(gDbg[1].counter[what], 1);   // 1st pixel keeps sum of values
        }
    }

    void Report(uint what)
    {
        Report(what, DispatchRaysIndex().xy, DispatchRaysDimensions().xy);
    }
#else
    void Report(uint what)
    {}

    void Report(uint what, uint2 pos, uint2 size)
    {}
#endif

//=================================================================================================================================
// Misc
//=================================================================================================================================

#define COMPILER_DXC
#include "NRD.hlsl"

float _REBLUR_GetHitDist( float normHitDist, float viewZ, float4 hitDistParams, float linearRoughness )
{
    float f = _REBLUR_GetHitDistanceNormalization( viewZ, hitDistParams, linearRoughness );

    return normHitDist * f;
}

float2 ClipToUv(float4 clip)
{
    // NOTE: clip must be always jittered using current jitter!
    return (clip.xy / clip.w) * float2(0.5, -0.5) + 0.5 - jitter * inv_rt_size;
}

float3 GetXWithOffset(float3 X, float3 N, float3 V, float z)
{
    X -= cam_pos.xyz;

    // Moves the ray origin further from surface to prevent self-intersections. Minimizes the distance for best results ( taken from RT Gems "A Fast and Robust Method for Avoiding Self-Intersection" )
    int3 o = int3( N * 256.0 );
    float3 a = asfloat( asint( X ) + ( X < 0.0 ? -o : o ) );
    float3 b = X + N * ( 1.0 / 65536.0 );
    float3 Xoffset = abs( X ) < ( 1.0f / 32.0 ) ? b : a;

    Xoffset += cam_pos.xyz;

    // The part above solves problems if RT is used for everything (including primary rays), but if raster is here...
    return Xoffset + (N + V) * (0.002 * z + 0.001);
}

float3 OffsetRay(float3 X, float3 N)
{
    X -= cam_pos.xyz;

    // Moves the ray origin further from surface to prevent self-intersections. Minimizes the distance for best results ( taken from RT Gems "A Fast and Robust Method for Avoiding Self-Intersection" )
    int3 o = int3( N * 256.0 );
    float3 a = asfloat( asint( X ) + ( X < 0.0 ? -o : o ) );
    float3 b = X + N * ( 1.0 / 65536.0 );
    float3 Xoffset = abs( X ) < ( 1.0f / 32.0 ) ? b : a;

    return Xoffset += cam_pos.xyz;
}

uint2 ApplyCheckerboard(uint2 halfResPos)
{
    //     Even frame (0)  Odd frame (1)   ...
    //         B W             W B
    //         W B             B W

    /*
        Even frames
        |.W|.W|.W|
        |W.|W.|W.|
        |.W|.W|.W|

        Odd frames
        |W.|W.|W.|
        |.W|.W|.W|
        |W.|W.|W.|
    */

    uint offset = ~(halfResPos.y ^ dxrt_frame_idx);

    return uint2( (halfResPos.x << 1) + (offset & 0x1), halfResPos.y );
}

void WriteCheckerboardOutput(uint2 pixelPos, float4 data)
{
    #if( USE_INDIRECT == 0 )
        data.xyz = 0;
    #endif

    pixelPos.x >>=1;

    gOutput[pixelPos] = data;
}

void WriteFullOutput(uint2 pixelPos, float4 data)
{
    #if( USE_INDIRECT == 0 )
        data.xyz = 0;
    #endif

    gOutput[pixelPos] = data;
}

float2 OctXY(float2 xy, float y)
{
    float2 inv = 1.0 - abs(xy.yx);
    inv = (xy < 0.0) ? -inv : inv;
    return (y <= 0.0) ? inv : xy;
}

float2 DirNWS2OctUV_bx2(float3 dir_n_ws)
{
    dir_n_ws /= dot(1.0, abs(dir_n_ws));
    return OctXY(dir_n_ws.xz, dir_n_ws.y);
}

float2 DirNWS2OctUV_border_corrected(float3 dir_n_ws, float mip, float2 oem_edge_size, float4 oem_uv_offsets)
{
    const float2 magic_const = float2(0.0, 3.1415 / 2.0 * 0.75);

    float mip_factor = pow(2.0, mip);

    float2 uv_bx2 = DirNWS2OctUV_bx2(dir_n_ws);

    float2 uv_bx4 = abs(uv_bx2) * 2.0 - 1.0;
    float2 uv_masks_inv = 1.0 - abs(uv_bx4);
    uv_masks_inv = saturate(abs(uv_masks_inv) * oem_edge_size / mip_factor); //* 0.125 = 0.25 (2x bx2) * 0.5 (full pixel size)
    float2 uv_masks = 1.0 - uv_masks_inv;
    
    float4 uv_offsets = (uv_bx2.xyxy < 0.0) ? oem_uv_offsets.xyzw : oem_uv_offsets.zwxy; //xy: upper, zw: lower
    uv_offsets *= lerp(magic_const.xxyy, magic_const.yyxx, abs(uv_bx2).xyxy);
    float2 uv_offset = (abs(uv_bx2.yx) > 0.5) ? uv_offsets.zw : uv_offsets.xy;
    uv_offset *= mip_factor;

    return (uv_bx2 + uv_offset * uv_masks.yx) * 0.25 + 0.5;	//form <-1.0, 1.0> to <0.25, 0.75>
}

float3 SampleOEMDif(float3 dirNWS)
{
    const float2 oem_dif_edge_size = float2(32.0, 32.0) * 0.25;
    const float4 oem_dif_uv_offsets = float4(1.0/32.0, 1.0/32.0, -1.0/32.0, -1.0/32.0) * 2.0;
    float2 uvOS = DirNWS2OctUV_border_corrected(dirNWS, 0.0, oem_dif_edge_size, oem_dif_uv_offsets);

    return gOemDif.SampleLevel(gLinearSampler, uvOS, 0).xyz;
}

float3 SampleSkyboxInternal(float3 dirWSN)
{
    dirWSN /= dot(abs(dirWSN), 1.0); // p / (|p.x|+|p.y|+|p.z|)

    float2 skyOctUV = 0.5;
    skyOctUV += dirWSN.x * 0.5;
    skyOctUV += dirWSN.z * float2(-0.5, 0.5);

    float3 sky = gSkybox64.SampleLevel(gLinearSampler, skyOctUV, 0).xyz;
    float3 sky_next = gSkybox64Next.SampleLevel(gLinearSampler, skyOctUV, 0).xyz;
    sky = lerp(sky, sky_next, saturate(weather_blend));

    return sky * skybox_intensity;
}

float3 SampleSky(float3 dirWSN)
{
#if USE_OEM_DIFF_AS_SKY
    float3 Csky = SampleOEMDif(dirWSN);
#else
    float3 Csky = SampleSkyboxInternal(dirWSN);
    Csky *= STL::Math::Pow01(dirWSN.y + 1.0, 8.0); // fix sky, because you use a single-side oct-packing
#endif

    return Csky;
}

float3 SampleSkybox(float3 dirWSN)
{
    float3 Csky = SampleSkyboxInternal(dirWSN);
    Csky *= STL::Math::Pow01(dirWSN.y + 1.0, 8.0); // fix sky, because you use a single-side oct-packing
    return Csky;
}

struct HitProps
{
    float3 X;
    float3 N;
    float3 albedo;
    float3 Rf0;
    float3 emission;
    float roughness;

    bool IsEmissive()
    { return dot(emission, 0.33333) != 0; }
};

HitProps GetHitProps(RayPayload payload, RayDesc ray)
{
    HitProps props = (HitProps)0;

    // Geometry
    props.X = ray.Origin + ray.Direction * abs(payload.hitDist);
    props.N = payload.normal;

    // Material
    uint4 packedMaterial = gPerInstanceData[payload.instanceId];

    float3 temp1 = STL::Packing::UintToRgba(packedMaterial.x, 8, 8, 8, 8).xyz;
    float4 temp2 = STL::Packing::UintToRgba(packedMaterial.y, 8, 8, 8, 8);
    temp1 = STL::Color::GammaToLinear(temp1);
    temp2 = STL::Color::GammaToLinear(temp2);

    props.albedo = temp1;
    props.Rf0 = temp2.xyz;
    props.roughness = temp2.w * temp2.w; // TODO: ^2 or ^1? ^2 shows better (use DEBUG = 6 in diffuse GI)

    // Emission
    #if( USE_EMISSION == 1 )
        packedMaterial.w = abs(payload.hitDist) == INF ? 0 : packedMaterial.w; // TODO: this is needed only for debug visualization
        uint rtFlags = packedMaterial.z >> 24;

        [branch]
        if ((rtFlags & RAY_TRACING_FLAG_EMS_ON) != 0 && packedMaterial.w != 0)
        {
            float3 emsFactor = emissive_intensity;

            [branch]
            if (rtFlags & RAY_TRACING_FLAG_SKY_EMS)
            {
                float3 Csky = SampleSkybox(-props.N);

                // mlysniewski:
                // TODO: sample hsh - get occluded sky factor. Not sure if this wont be a killer

                // mlysniewski: WTF is this... But we do it in raster, so I have to live with it
                float emsFactorLum = dot(0.33333, Csky);
                emsFactor = lerp(emsFactorLum, Csky, 1.5);
            }
            else if (rtFlags & RAY_TRACING_FLAG_SUN_EMS)
                emsFactor = sun_color;

            emsFactor *= asfloat(packedMaterial.w);

            float w, h;
            gEms[payload.instanceId].GetDimensions( w, h ); // TODO: if I only had it as a constant...
            float mipNum = log2( max( w , h ) );
            float mip = max(mipNum - 5.0, 0.0); // Use 32x32 (or the best) to avoid cache trashing

            props.emission = gEms[payload.instanceId].SampleLevel(gLinearSampler, payload.uv, mip).xyz * emsFactor;
        }
    #endif

    return props;
}

float CastShadowRay(float3 X, float3 Lsum, float3 direction, float maxDist)
{
    bool is_shadow_needed = STL::Color::Luminance(Lsum) != 0.0;
    float hitViewZ = mul(view_proj_mtx, float4(X, 1)).w;
    float3 Xoffset = GetXWithOffset(X, direction, 0, hitViewZ);

    RayDesc ray;
    ray.Origin = Xoffset;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = maxDist * float(is_shadow_needed);

    #if( USE_TRACE_RAY_INLINE == 1 )
        const uint instanceInclusionMask = ray.TMax == 0.0 ? 0 : EInstanceRayVisibility::ShadowRays;

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | CULLING_FLAGS> rayQuery;
        rayQuery.TraceRayInline(g_TLAS, 0, instanceInclusionMask, ray);

        while(rayQuery.Proceed())
        {
            uint triangleBaseIndex = rayQuery.CandidatePrimitiveIndex() * 3;

            uint3 indices;
            indices.x = gIBLocal[triangleBaseIndex].i;
            indices.y = gIBLocal[triangleBaseIndex + 1].i;
            indices.z = gIBLocal[triangleBaseIndex + 2].i;

            float2 uv = GetUV0(indices, rayQuery.CandidateTriangleBarycentrics());

            uint idx = rayQuery.CandidateInstanceID();
            idx = (idx >> 1) + (idx & 0x1) * rayQuery.CandidateGeometryIndex();

            float w, h;
            gClp[idx].GetDimensions(w, h);
            float mipNum = log2( max( w , h ) );
            float mip = max(mipNum - 7, 0.0); // Use 128x128 (or the best) to avoid cache trashing

            float clip_val = gClp[idx].SampleLevel(gLinearSampler, uv, mip).x;
            if (clip_val >= 0.0)
                return 0;
        }

        return rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0 : 1;
    #else
        ShadowRayPayload shadowPayload = (ShadowRayPayload)0;
        shadowPayload.hitDist = maxDist;
		//shadowPayload.useSelfShadow = false;
        {
            const uint rayFlags = CULLING_FLAGS | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
            const uint instanceInclusionMask = ray.TMax == 0.0 ? 0 : EInstanceRayVisibility::ShadowRays;
            const uint rayContributionToHitGroupIndex = SHADOW_RAY_ID;
            const uint multiplierForGeometryContributionToHitGroupIndex = 0;
            const uint missShaderIndex = SHADOW_RAY_ID;

            TraceRay(g_TLAS, rayFlags, instanceInclusionMask, rayContributionToHitGroupIndex, multiplierForGeometryContributionToHitGroupIndex, missShaderIndex, ray, shadowPayload);
            Report(_TraceRay);
        }

        return shadowPayload.hitDist >= maxDist;
    #endif
}

float CastShadowRayCustom(float3 X, float3 Lsum, float3 direction, float minDist, float maxDist)
{
    bool is_shadow_needed = STL::Color::Luminance(Lsum) != 0.0;
    float hitViewZ = mul(view_proj_mtx, float4(X, 1)).w;
	//GetXWithOffset makes shadow ray extremely view dependant  
    float3 Xoffset = OffsetRay(X, direction);//X + 0.001*direction;//GetXWithOffset(X, direction, 0, hitViewZ);

    RayDesc ray;
    ray.Origin = Xoffset;
    ray.Direction = direction;
    ray.TMin = minDist;
    ray.TMax = maxDist * float(is_shadow_needed);

    #if( USE_TRACE_RAY_INLINE == 1 )
        const uint instanceInclusionMask = ray.TMax == 0.0 ? 0 : EInstanceRayVisibility::ShadowRays;

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | CULLING_FLAGS> rayQuery;
        rayQuery.TraceRayInline(g_TLAS, 0, instanceInclusionMask, ray);

        while(rayQuery.Proceed())
        {
            uint triangleBaseIndex = rayQuery.CandidatePrimitiveIndex() * 3;

            uint3 indices;
            indices.x = gIBLocal[triangleBaseIndex].i;
            indices.y = gIBLocal[triangleBaseIndex + 1].i;
            indices.z = gIBLocal[triangleBaseIndex + 2].i;

            float2 uv = GetUV0(indices, rayQuery.CandidateTriangleBarycentrics());

            uint idx = rayQuery.CandidateInstanceID();
            idx = (idx >> 1) + (idx & 0x1) * rayQuery.CandidateGeometryIndex();

            float w, h;
            gClp[idx].GetDimensions(w, h);
            float mipNum = log2( max( w , h ) );
            float mip = max(mipNum - 7, 0.0); // Use 128x128 (or the best) to avoid cache trashing

            float clip_val = gClp[idx].SampleLevel(gLinearSampler, uv, mip).x;
            if (clip_val >= 0.0)
                return 0;
        }

        return rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0 : 1;
    #else
        ShadowRayPayload shadowPayload = (ShadowRayPayload)0;
		//shadowPayload.useSelfShadow = selfShadow;
        shadowPayload.hitDist = maxDist;
        {
            const uint rayFlags = CULLING_FLAGS | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
            const uint instanceInclusionMask = ray.TMax == 0.0 ? 0 : EInstanceRayVisibility::ShadowRays;
            const uint rayContributionToHitGroupIndex = SHADOW_RAY_ID;
            const uint multiplierForGeometryContributionToHitGroupIndex = 0;
            const uint missShaderIndex = SHADOW_RAY_ID;

            TraceRay(g_TLAS, rayFlags, instanceInclusionMask, rayContributionToHitGroupIndex, multiplierForGeometryContributionToHitGroupIndex, missShaderIndex, ray, shadowPayload);
            Report(_TraceRay);
        }
        return shadowPayload.hitDist >= maxDist;
    #endif
}

float4 PrepareOutputForNrd(float3 Lsum, float hitDist, float4 hitDistParams, float z, float roughness)
{
    float normHitDist = REBLUR_FrontEnd_GetNormHitDist(hitDist, z, hitDistParams, roughness);
    float4 result = REBLUR_FrontEnd_PackRadianceAndHitDist(Lsum, normHitDist);
	
	if (result.w != 0.0) {
		result.w = max(result.w, 1e-7);
	}
	
	float3 rawResult = result.xyz;
	
	result.x = dot(rawResult.xyz, float3(0.25, 0.5, 0.25));
	result.y = dot(rawResult.xyz, float3(0.5, 0.0, -0.5));
	result.z = dot(rawResult.xyz, float3(-0.25, 0.5, -0.25));

    #if( DEBUG == 1 )
        result.xyz = normHitDist;
    #endif

    result.xyz *= DEBUG_INTENSITY_SCALE;

    return result;
}

//=================================================================================================================================
// BLUE NOISE (copied from NRD sample)
//=================================================================================================================================

// SPP - must be POW of 2!
// Virtual 32 spp tuned for NRD purposes (actually, 1 spp but distributed in time)
// Final SPP = sppVirtual x spp (for this value there is a different "gIn_Scrambling_Ranking" texture!)
float2 GetRandom(bool isCheckerboard, uint seed, Texture2D<uint3> texScramblingRanking, uint sampleIndex, const uint sppVirtual, const uint spp)
{
    // WHITE NOISE (for testing purposes)
    float4 white = STL::Rng::GetFloat4( );
    if (blue_noise_enabled == 0)
        return white.xy;

    // BLUE NOISE
    // Based on - https://eheitzresearch.wordpress.com/772-2/
    // Source code and textures can be found here - https://belcour.github.io/blog/research/2019/06/17/sampling-bluenoise.html (but 2D only)
    uint2 pixelPos = DispatchRaysIndex().xy;

    // Sample index
    uint frameIndex = isCheckerboard ? ( dxrt_frame_idx >> 1 ) : dxrt_frame_idx;
    uint virtualSampleIndex = ( frameIndex + seed ) & ( sppVirtual - 1 );
    sampleIndex &= spp - 1;
    sampleIndex += virtualSampleIndex * spp;
    // Offset retarget (advance each "sppVirtual" frames)
    uint2 offset = pixelPos;
    #if 0 // to keep image stable after "sppVirtual" frames...
        offset += uint2( float2( 0.754877669, 0.569840296 ) * gScreenSize * float( dxrt_frame_idx / sppVirtual ) );
    #endif
    // The algorithm
    uint3 A = texScramblingRanking[ offset & 127 ];
    uint rankedSampleIndex = sampleIndex ^ A.z;
    uint4 B = gSobol[ uint2( rankedSampleIndex & 255, 0 ) ];
    float4 blue = ( float4( B ^ A.xyxy ) + 0.5 ) * ( 1.0 / 256.0 );
    // Randomize in [ 0; 1 / 256 ] area to get rid of possible banding
    #if 1
        uint d = STL::Sequence::Bayer4x4ui( pixelPos, dxrt_frame_idx );
        float2 dither = ( float2( d & 3, d >> 2 ) + 0.5 ) * ( 1.0 / 4.0 );
        blue += ( dither.xyxy - 0.5 ) * ( 1.0 / 256.0 );
    #else
        blue += ( white - 0.5 ) * ( 1.0 / 256.0 );
    #endif
    return saturate( blue.xy );
}

//=================================================================================================================================
// Screen space tracer
//=================================================================================================================================

// IMPORTANT: TraceRayScreenSpace assumes that STL::Rng is initialized beforehand
bool TraceRayScreenSpace(float3 p0, float viewZ, float3 rayDirection, float NoV, inout float hitDist, bool isVisibilityCheck = false)
{
    // isVisibilityCheck = false - is a more strict version of the tracer. The tracer will fail if it detects a disocclusion, assuming
    // that RT is capable to resolve the situation by casting a real ray from the last valid position. But taking into account that
    // BVH in DL2 doesn't include grass and foliage, it's just better to be less accurate in lighting but preserve correct visibility
    // in diffuse and specular tracing. It's a must for IQ.
    isVisibilityCheck = false;

    #if USE_SCREEN_SPACE_TRACE
        float tanHalfFovY = tan( STL::Math::DegToRad(0.5 * 90 * 9.0 / 16.0) ); // TODO: add it to CB
        float rayLength = 8.0 * SCREEN_SPACE_MAX_SAMPLE_NUM * inv_rt_size.y * viewZ * tanHalfFovY;
        if (isVisibilityCheck)
            rayLength = max(rayLength, 0.3);

        float randomScale = 1.0 + (STL::Rng::GetFloat2().x * 2.0 - 1.0) * 0.25;
        rayLength *= randomScale;

        [unroll]
        for (float i = 1; i < SCREEN_SPACE_MAX_SAMPLE_NUM; i++)
        {
            float f = i / SCREEN_SPACE_MAX_SAMPLE_NUM;
            float hitDistTemp = rayLength * f * f; // take more samples closer to the origin
            float3 p = p0 + rayDirection * hitDistTemp;

            float4 clip = mul(view_proj_mtx, float4(p, 1));
            float2 uv = ClipToUv(clip);
            float z = GBufLoadLinearDepth(uv);

            float cmp = abs(z - clip.w) / (max(z, abs(clip.w)) + 1e-6); // relative delta (z can't be 0)
            float threshold = SCREEN_SPACE_Z_THRESHOLD * lerp(0.1, 1.0, f) / max(abs(NoV), 0.1);

            if (any( saturate(uv) != uv ) || (cmp >= threshold && z < clip.w && !isVisibilityCheck) || clip.w < 0)
                return false; // end of search if out of screen or occluded with high confidence

            if (!(cmp >= threshold && z < clip.w) || !isVisibilityCheck)
                hitDist = hitDistTemp;

            if (cmp < threshold && z < clip.w)
                return true; // great success!
        }
    #endif

    return false;
}

// Taken out from NRD
float GetSpecMagicCurve(float roughness, float power = 0.25)
{
    // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiIxLjAtMl4oLTE1LjAqeCkiLCJjb2xvciI6IiNGMjE4MTgifSx7InR5cGUiOjAsImVxIjoiKDEtMl4oLTIwMCp4KngpKSooeF4wLjI1KSIsImNvbG9yIjoiIzIyRUQxNyJ9LHsidHlwZSI6MCwiZXEiOiIoMS0yXigtMjAwKngqeCkpKih4XjAuNSkiLCJjb2xvciI6IiMxNzE2MTYifSx7InR5cGUiOjEwMDAsIndpbmRvdyI6WyIwIiwiMSIsIjAiLCIxLjEiXSwic2l6ZSI6WzEwMDAsNTAwXX1d

    float f = 1.0 - exp2( -200.0 * roughness * roughness );
    f *= STL::Math::Pow01( roughness, power );

    return f;
}

float3 EstimateAmbient(float z, float roughness, float hitDist, float3 hitV, float3 hitN, float3 hitAlbedo, float3 hitRf0, float hitRoughness)
{
    // Approximate BRDF
    float NoV = abs( dot(hitN, hitV) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross(hitRf0, NoV, hitRoughness);
    float3 BRDF = hitAlbedo * (1 - F) + F;

    float3 Camb = gAvgAmbient[uint2(0, 0)].xyz;
    Camb *= BRDF;

    // Basic occlusion
    float occlusion = REBLUR_FrontEnd_GetNormHitDist(hitDist, 0.0, diff_hit_dist_params, 1.0); // z = 0, roughness = 1
    occlusion = lerp(1.0 / STL::Math::Pi(1.0), 1.0, occlusion);

    // Kill ambient with distance (becomes less correct)
    occlusion *= exp2(-0.001 * z * z);

    // Reduce ambient if roughness is high, it will be handled by IBL pipeline corrected by AO and SO
    float f = GetSpecMagicCurve(roughness);
    occlusion *= lerp(1.0, 1.0 - non_rt_ambient_amount, f);

    // Optional
    occlusion *= USE_AMBIENT;

    return Camb * occlusion;
}

float4 GetFinalLightingFromPreviousFrame(float3 X, float2 pixelUv)
{
    // Mix with previously denoised frame to get multibounce lighting for free
    float4 clipPrev = mul(prev_view_proj_mtx, float4(X, 1));
    float2 uvPrev = ClipToUv(clipPrev);
    bool isInScreen = all( saturate(uvPrev) == uvPrev ) && clipPrev.w > 0.0;
    float4 result = 0;

    [branch]
    if (isInScreen)
    {
        result.xyz = gPrevSceneLit.SampleLevel(gNearestSampler, uvPrev, 0).xyz;

        // TODO: I have a gut feeling that previous frame can have a different exposure, if so and lighting is multiplied
        // with exposure the following is needed:
        // LsumShadowedPrev.xyz *= currExposure / prevExposure;

        // Fade-out on screen edges
        float2 f = STL::Math::LinearStep(0.0, SCREEN_EDGE_BORDER, uvPrev) * STL::Math::LinearStep(1.0, 1.0 - SCREEN_EDGE_BORDER, uvPrev);
        float confidence = f.x * f.y;

        // Confidence - viewZ
        // No "abs" for clipPrev.w, because if it's negative we have a back-projection!
        float prevViewZ = gPrevZLinear.SampleLevel(gNearestSampler, uvPrev, 0).x;
        float err = abs(prevViewZ - clipPrev.w) / ( max( prevViewZ, abs(clipPrev.w) ) + 1e-6 );
        confidence *= STL::Math::LinearStep( 0.03, 0.01, err );

        // Confidence - ignore back-facing
        // TODO: Instead of storing previous normal we can store previous NoL, if signs do not match we hit the surface from the opposite side

        // Confidence - ignore too short rays
        float4 clip = mul(view_proj_mtx, float4(X, 1));
        float2 uv = ClipToUv(clip);
        float d = length( (uv - pixelUv) * rt_size );
        confidence *= STL::Math::LinearStep( 1.0, 3.0, d );

        // Clear out bad values (just in case)
        confidence *= all( !isnan(result.xyz) && !isinf(result.xyz) );

        // Confidence - ignore mirror specular
        /*
        float diffusiness = hitProps.roughness * hitProps.roughness;
        float lumDiff = lerp( STL::Color::Luminance( hitProps.albedo ), 1.0, diffusiness );
        float lumSpec = lerp( STL::Color::Luminance( hitProps.Rf0 ), 0.0, diffusiness );
        float diffProb = saturate( lumDiff / ( lumDiff + lumSpec + 1e-6 ) );
        confidence *= diffProb;
        */

        result.w = confidence;// * 0.75;
        result.xyz = confidence == 0.0 ? 0.0 : result.xyz; // only real "=" op can cure NAN / INF
    }

    return result;
}
