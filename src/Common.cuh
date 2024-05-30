#pragma once

#include <glm/glm.hpp>
#include "Scene.h"
#include "BVH.h"
#include "App.h"

#define FN_DECL __device__

#define IMAGE_SIZE(Img) \
    ivec2(Width, Height)

#define GLOBAL_ID() \
    uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)

#define INOUT(Type) Type &

#define GET_ATTR(Obj, Attr) \
    Obj->Attr

#define MAX_LENGTH 1e30f
#define PI_F 3.141592653589
#define INVALID_ID -1
#define MIN_ROUGHNESS (0.03f * 0.03f)



namespace commonCu
{
using namespace glm;
using namespace gpupt;

// __device__ bvhNode *BVHBuffer;
// __device__ tlasNode *TLASNodes;

__device__ cudaFramebuffer CurrentFramebuffer;
__device__ u32 Width;
__device__ indexData *IndexDataBuffer;
__device__ u32 Height;
__device__ triangle *TriangleBuffer;
__device__ u32 *IndicesBuffer;
__device__ instance *TLASInstancesBuffer;
__device__ camera *Cameras;
__device__ tracingParameters *Parameters;
__device__ material *Materials;
__device__ cudaTextureObject_t SceneTextures;
__device__ cudaTextureObject_t EnvTextures;
__device__ int EnvironmentsCount;
__device__ int TexturesWidth;
__device__ int TexturesHeight;
__device__ int LightsCount;
__device__ light *Lights;
__device__ float *LightsCDF;
__device__ environment *Environments;
__device__ int EnvTexturesWidth;
__device__ int EnvTexturesHeight;
__device__ float Time;
__device__ OptixTraversableHandle *ShapeASHandles;
__device__ OptixTraversableHandle IASHandle;

struct half4 {half x, y, z, w;};
FN_DECL half4 Vec4ToHalf4(INOUT(vec4) Input)
{
    return {
        __float2half(Input.x),
        __float2half(Input.y),
        __float2half(Input.z),
        __float2half(Input.w)
    };
}

FN_DECL vec4 Half4ToVec4(INOUT(half4) Input)
{
    return vec4(
    __half2float(Input.x),
        __half2float(Input.y),
        __half2float(Input.z),
        __half2float(Input.w)
    );
}

FN_DECL bool IsFinite(float A)
{
    return !isnan(A);
}

FN_DECL bool IsFinite(vec3 A)
{
    return IsFinite(A.x) && IsFinite(A.y) && IsFinite(A.z);
}

struct kernelParams
{
    CUdeviceptr OutputBuffer;
    OptixTraversableHandle Handle;
    OptixTraversableHandle *ShapeASHandles;
    u32 Width;
    u32 Height;
    triangle *TriangleBuffer;
    u32 *IndicesBuffer;
    instance *TLASInstancesBuffer;
    camera *Cameras;
    tracingParameters *Parameters;
    material *Materials;
    cudaTextureObject_t SceneTextures;
    cudaTextureObject_t EnvTextures;
    instance *Instances;
    indexData *IndexDataBuffer;
    int EnvironmentsCount;
    int TexturesWidth;
    int TexturesHeight;
    int LightsCount;
    light *Lights;
    float *LightsCDF;
    environment *Environments;
    int EnvTexturesWidth;
    int EnvTexturesHeight;
    float Time;
    cudaFramebuffer CurrentFramebuffer;
};

struct rayPayload
{
    uint32_t Distance;
    uint32_t PrimitiveIndex;
    uint32_t InstanceIndex;
    uint32_t U;
    uint32_t V;
    uint32_t MaterialIndex;
};


struct randomState
{
    uint64_t State;
    uint64_t Inc;
};

struct sceneIntersection
{
    float Distance;
    float U;
    float V;
    
    uint InstanceIndex;
    uint PrimitiveIndex;
    
    mat4 InstanceTransform;

    vec3 Normal;
    randomState RandomState;
    uint MaterialIndex;
    vec3 Tangent;
    vec3 Bitangent;
};




// Util
FN_DECL float Sum(vec3 A) { 
    return A.x + A.y + A.z; 
}

FN_DECL float Mean(vec3 A) { 
    return Sum(A) / 3; 
}

FN_DECL bool SameHemisphere(vec3 Normal, vec3 Outgoing, vec3 Incoming)
{
    return dot(Normal, Outgoing) * dot(Normal, Incoming) >= 0;
}

FN_DECL float max3 (vec3 v) {
  return max (max (v.x, v.y), v.z);
}

// region utils
FN_DECL float ToLinear(float SRGB) {
  return (SRGB <= 0.04045) ? SRGB / 12.92f
                           : pow((SRGB + 0.055f) / (1.0f + 0.055f), 2.4f);
}

FN_DECL vec4 ToLinear(vec4 SRGB)
{
    return vec4(
        ToLinear(SRGB.x),
        ToLinear(SRGB.y),
        ToLinear(SRGB.z),
        SRGB.w
    );
}
FN_DECL mat3 GetTBN(INOUT(sceneIntersection) Isect, vec3 Normal)
{ 
    return mat3(Isect.Tangent, Isect.Bitangent, Normal);    
}

FN_DECL vec2 SampleTriangle(vec2 UV){
    return vec2(
        1 - sqrt(UV.x),
        UV.y * sqrt(UV.x)
    );
}
FN_DECL int SampleUniform(int Size, float Rand)
{
    // returns a random number inside the range (0 - Size)
    return clamp(int(Rand * Size), 0, Size-1);
}

// FN_DECL int UpperBound(int CDFStart, int CDFCount, float X)
// {
//     int Mid;
//     int Low = CDFStart;
//     int High = CDFStart + CDFCount;
 
//     while (Low < High) {
//         Mid = Low + (High - Low) / 2;
//         if (X >= LightsCDF[Mid]) {
//             Low = Mid + 1;
//         }
//         else {
//             High = Mid;
//         }
//     }
   
//     // if X is greater than arr[n-1]
//     if(Low < CDFStart + CDFCount && LightsCDF[Low] <= X) {
//        Low++;
//     }
 
//     // Return the upper_bound index
//     return Low;
// }
 

// Random
FN_DECL uint AdvanceState(INOUT(randomState) RNG)
{
    uint64_t OldState = RNG.State;
    RNG.State = OldState * 6364136223846793005ul + RNG.Inc;
    uint XorShifted = uint(((OldState >> uint(18)) ^ OldState) >> uint(27));
    uint Rot = uint(OldState >> uint(59));

    return (XorShifted >> Rot) | (XorShifted << ((~Rot + 1u) & 31));
}

FN_DECL randomState CreateRNG(uint64_t Seed)
{
    uint64_t Sequence = 371213;
    randomState State;

    State.State = 0U;
    State.Inc = (Sequence << 1u) | 1u;
    AdvanceState(State);
    State.State += Seed;
    AdvanceState(State);

    return State;
}

FN_DECL int Rand1i(INOUT(randomState) RNG, int n)
{
    return int(AdvanceState(RNG) % n);
}

FN_DECL float RandomUnilateral(INOUT(randomState) RNG)
{
    uint u = (AdvanceState(RNG) >> 9) | 0x3f800000u;
    return uintBitsToFloat(u) - 1.0f;
}

FN_DECL vec2 Random2F(INOUT(randomState) State)
{
    return vec2(RandomUnilateral(State), RandomUnilateral(State));
}

// Geometry

FN_DECL vec3 TransformPoint(mat4 A, vec3 B)
{
    vec4 Res = A * vec4(B, 1); 
    return vec3(Res);
}

FN_DECL vec3 TransformDirection(mat4 A, vec3 B)
{
    vec4 Res = A * vec4(B, 0); 
    return normalize(vec3(Res));
}

FN_DECL vec3 TransformDirection(mat3 A, vec3 B)
{
    vec3 Res = A * B; 
    return normalize(Res);
}

FN_DECL mat3 BasisFromZ(vec3 V)
{
    vec3 Z = normalize(V);
    
    // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    float Sign = Z.z > 0 ? 1.0f : -1.0f;
    float A = -1.0f / (Sign + Z.z);
    float B = Z.x * Z.y * A;
    vec3 X = vec3(1.0f + Sign * Z.x * Z.x * A, Sign * B, -Sign * Z.x);
    vec3 Y = vec3(B, Sign + Z.y * Z.y * A, -Z.y);

    return mat3(X, Y, Z);
}

// Ray
struct ray
{
    vec3 Origin;
    vec3 Direction;
    vec3 InverseDirection;
};


FN_DECL ray GetRay(vec2 ImageUV, vec2 LensUV)
{
    // Parameters->CurrentCamera
    camera &Camera = Cameras[int(Parameters->CurrentCamera)];
    ImageUV.y = 1 - ImageUV.y;
    ray Ray = {};
    Ray.Origin = TransformPoint(Camera.Frame, vec3(0));
    vec3 Target = TransformPoint(inverse(Camera.ProjectionMatrix), vec3(ImageUV * 2.0f - 1.0f, 0.0f));
    Ray.Direction = TransformDirection(Camera.Frame, normalize(Target));      
    return Ray;
}


// Region lights

FN_DECL int UpperBound(int CDFStart, int CDFCount, float X)
{
    int Mid;
    int Low = CDFStart;
    int High = CDFStart + CDFCount;
 
    while (Low < High) {
        Mid = Low + (High - Low) / 2;
        if (X >= LightsCDF[Mid]) {
            Low = Mid + 1;
        }
        else {
            High = Mid;
        }
    }
   
    // if X is greater than arr[n-1]
    if(Low < CDFStart + CDFCount && LightsCDF[Low] <= X) {
       Low++;
    }
 
    // Return the upper_bound index
    return Low;
}
 

FN_DECL int SampleDiscrete(int LightInx, float R)
{
    //Remap R from 0 to the size of the distribution
    int CDFStart = Lights[LightInx].CDFStart;
    int CDFCount = Lights[LightInx].CDFCount;

    float LastValue = LightsCDF[CDFStart + CDFCount-1];

    R = clamp(R * LastValue, 0.0f, LastValue - 0.00001f);
    // Returns the first element in the array that's greater than R.#
    int Inx= UpperBound(CDFStart, CDFCount, R);
    Inx -= CDFStart;
    return clamp(Inx, 0, CDFCount-1);
}

FN_DECL float DistanceSquared(vec3 A, vec3 B)
{
    return dot(A-B, A-B);
}
FN_DECL float SampleUniformPDF(int Size)
{
    // the probability of a uniform distribution is just the inverse of the size.
    return 1.0f / float(Size);
}

FN_DECL vec3 SampleSphere(vec2 UV)
{
  float Z   = 2 * UV.y - 1;
  float R   = sqrt(clamp(1 - Z * Z, 0.0f, 1.0f));
  float Phi = 2 * PI_F * UV.x;
  return vec3(R * cos(Phi), R * sin(Phi), Z);
}

FN_DECL float SampleDiscretePDF(int CDFStart, int CDFCount, int Inx) {
  if (Inx == 0) return LightsCDF[CDFStart];
  return( LightsCDF[CDFStart + Inx]- LightsCDF[CDFStart + Inx - 1]) /
         LightsCDF[CDFStart + CDFCount -1];
}

FN_DECL vec3 SampleLights(INOUT(vec3) Position, float RandL, float RandEl, vec2 RandUV)
{
    // Take a random light index
    int LightID = SampleUniform(int(LightsCount), RandL);
    // Returns a vector that points to a light in the scene.
    if(Lights[LightID].Instance != INVALID_ID)
    {
        instance Instance = TLASInstancesBuffer[Lights[LightID].Instance];
        indexData IndexData = IndexDataBuffer[Instance.Shape];
        uint TriangleStartInx = IndexData.triangleDataStartInx;
        uint TriangleCount = IndexData.TriangleCount;
        
        // Sample an element on the shape
        int Element = SampleDiscrete(LightID, RandEl);
        // // Sample a point on the triangle
        vec2 UV = TriangleCount > 0 ? SampleTriangle(RandUV) : RandUV;
        // // Calculate the position
        triangle Tri = TriangleBuffer[TriangleStartInx + Element];
        vec3 LightPos = 
            vec3(Tri.PositionUvX1) * UV.x + 
            vec3(Tri.PositionUvX2) * UV.y +
            vec3(Tri.PositionUvX0) * (1 - UV.x - UV.y);
        LightPos = TransformPoint(Instance.Transform, LightPos);
        // return the normalized direction
        return normalize(LightPos - Position);
    }
    else if(Lights[LightID].Environment != INVALID_ID)
    {
        environment Env = Environments[Lights[LightID].Environment];
        if (Env.EmissionTexture != INVALID_ID) {
            // auto& emission_tex = scene.textures[environment.emission_tex];
            int SampleInx = SampleDiscrete(LightID, RandEl);
            vec2 UV = vec2(((SampleInx % EnvTexturesWidth) + 0.5f) / EnvTexturesWidth,
                ((SampleInx / EnvTexturesWidth) + 0.5f) / EnvTexturesHeight);
            
            return TransformDirection(Env.Transform, vec3(cos(UV.x * 2 * PI_F) * sin(UV.y * PI_F), 
                        cos(UV.y * PI_F),
                        sin(UV.x * 2 * PI_F) * sin(UV.y * PI_F)));
        } else {
            return SampleSphere(RandUV);
        }      
    }
    else
    {
        return vec3(0,0,0);
    }
}


FN_DECL void IntersectInstance(ray Ray, INOUT(sceneIntersection) Isect, uint InstanceIndex)
{
    int ShapeInx = TLASInstancesBuffer[InstanceIndex].Shape;
    OptixTraversableHandle Handle = ShapeASHandles[ShapeInx];
    rayPayload Payload = {};
    optixTrace(
        // Handle,
        IASHandle,
        make_float3(Ray.Origin.x, Ray.Origin.y, Ray.Origin.z),
        make_float3(Ray.Direction.x, Ray.Direction.y, Ray.Direction.z),
        0.01f,  // tmin
        MAX_LENGTH,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,  // SBT offset for ray type 0
        1,  // SBT stride for ray type
        0,  // missSBTIndex
        Payload.Distance,
        Payload.PrimitiveIndex,
        Payload.InstanceIndex,
        Payload.U,
        Payload.V
    );

    Isect = {};
    Isect.Distance = MAX_LENGTH;
    if(Payload.InstanceIndex == InstanceIndex)
    {
        // Isect.InstanceIndex = InstanceIndex;
        Isect.PrimitiveIndex = Payload.PrimitiveIndex;

        int MeshIndex = TLASInstancesBuffer[Isect.InstanceIndex].Shape;
        indexData IndexData = IndexDataBuffer[MeshIndex];
        uint TriangleStartInx = IndexData.triangleDataStartInx;
        Isect.PrimitiveIndex = Payload.PrimitiveIndex + TriangleStartInx;
            
        Isect.U = uint_as_float(Payload.U);
        Isect.V = uint_as_float(Payload.V);
        Isect.Distance = uint_as_float(Payload.Distance);
        Isect.MaterialIndex = TLASInstancesBuffer[Isect.InstanceIndex].Material;
    }
}


// Returns the probability of choosing a position on the light
FN_DECL float SampleLightsPDF(INOUT(vec3) Position, INOUT(vec3) Direction)
{
    // Initialize the pdf to 0
    float PDF = 0;
    // Loop through all the lights
    for(int i=0; i<LightsCount; i++)
    {
        if(Lights[i].Instance != INVALID_ID)
        {
            float LightPDF = 0.0f;
            vec3 NextPosition = Position;
            for(int Bounce=0; Bounce<1; Bounce++)
            {
                // Check if the ray intersects the light. If it doesn't, we break. 
                ray Ray;
                Ray.Origin = NextPosition;
                Ray.Direction = Direction;
                
                sceneIntersection Isect;
                Isect.Distance = MAX_LENGTH;
                IntersectInstance(Ray, Isect, Lights[i].Instance);
                // LightPDF += 1; 
                if(Isect.Distance == MAX_LENGTH) break;

                mat4 InstanceTransform = TLASInstancesBuffer[Lights[i].Instance].Transform;

                //Get the point on the light
                triangle Tri = TriangleBuffer[Isect.PrimitiveIndex];
                vec3 LightPos = 
                    vec3(Tri.PositionUvX1) * Isect.U + 
                    vec3(Tri.PositionUvX2) * Isect.V +
                    vec3(Tri.PositionUvX0) * (1 - Isect.U - Isect.V);     
                LightPos = TransformPoint(InstanceTransform, LightPos);
                
                vec3 LightNormal = 
                    vec3(Tri.NormalUvY1) * Isect.U + 
                    vec3(Tri.NormalUvY2) * Isect.V +
                    vec3(Tri.NormalUvY0) * (1 - Isect.U - Isect.V);                    
                LightNormal = TransformDirection(InstanceTransform, LightNormal);

                //Find the probability that this point was sampled
                float Area = LightsCDF[Lights[i].CDFStart + Lights[i].CDFCount-1];
                LightPDF += DistanceSquared(LightPos, Position) / 
                            (abs(dot(LightNormal, Direction)) * Area);
                //Continue for the next ray
                NextPosition = LightPos + Direction * 1e-3f;
            }
            PDF += LightPDF;
        }
        else if(Lights[i].Environment != INVALID_ID)
        {
            environment Env = Environments[Lights[i].Environment];
            if (Env.EmissionTexture != INVALID_ID) {
                vec3 WorldDir = TransformDirection(inverse(Env.Transform), Direction);

                vec2 TexCoord = vec2(atan2(WorldDir.z, WorldDir.x) / (2 * PI_F),
                                     acos(clamp(WorldDir.y, -1.0f, 1.0f)) / PI_F);
                if (TexCoord.x < 0) TexCoord.x += 1;
                
                int u = clamp(
                    int(TexCoord.x * EnvTexturesWidth), 0, EnvTexturesWidth - 1);
                int v    = clamp(int(TexCoord.y * EnvTexturesHeight), 0,
                    EnvTexturesHeight - 1);
                float Probability = SampleDiscretePDF(
                                Lights[i].CDFStart, Lights[i].CDFCount, v * EnvTexturesWidth + u);
                float Angle = (2 * PI_F / EnvTexturesWidth) *
                            (PI_F / EnvTexturesHeight) *
                            sin(PI_F * (v + 0.5f) / EnvTexturesHeight);
                PDF += Probability / Angle;
            } else {
                PDF += 1 / (4 * PI_F);
            }            
        }
    }

    // Multiply the PDF with the probability to pick one light in the scene.
    PDF *= SampleUniformPDF(int(LightsCount));
    return PDF;
}




// Cosine
FN_DECL vec3 SampleHemisphereCosine(vec3 Normal, vec2 UV)
{

    float Z = sqrt(UV.y);
    float R = sqrt(1 - Z * Z);
    float Phi = 2 * PI_F * UV.x;
    vec3 LocalDirection = vec3(R * cos(Phi), R * sin(Phi), Z);    
    return TransformDirection(BasisFromZ(Normal), LocalDirection);
}

FN_DECL float SampleHemisphereCosinePDF(INOUT(vec3) Normal, INOUT(vec3) Direction)
{
    // The probability of generating a direction v is proportional to cos(θ) (as in the cosine-weighted hemisphere).
    // The total probability over the hemisphere should be 1. So, to normalize, we divide by the integral of cos⁡cos(θ) over the hemisphere, which is π.

    float CosW = dot(Normal, Direction);
    return (CosW <= 0) ? 0 : CosW / PI_F;
}


// Microfacet & Misc brdf functions
FN_DECL vec3 EtaToReflectivity(vec3 Eta) {
  return ((Eta - 1.0f) * (Eta - 1.0f)) / ((Eta + 1.0f) * (Eta + 1.0f));
}

FN_DECL vec3 FresnelSchlick(INOUT(vec3) Specular, INOUT(vec3) Normal, INOUT(vec3) Outgoing) {
    // Schlick approximation of the Fresnel term
  if (Specular == vec3(0, 0, 0)) return vec3(0, 0, 0);
  float cosine = dot(Normal, Outgoing);
  return Specular +
         (1.0f - Specular) * pow(clamp(1.0f - abs(cosine), 0.0f, 1.0f), 5.0f);
}
FN_DECL float FresnelDielectric(float Eta, vec3 Normal, vec3 Outgoing)
{
    // The Fresnel equations describe how light is reflected and refracted at the interface between different media, such as the transition from air to a dielectric material.

    // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    float CosW = abs(dot(Normal, Outgoing));
    float Sin2 = 1 - CosW * CosW;
    float Eta2 = Eta * Eta;

    float Cos2T = 1 - Sin2 / Eta2;
    if(Cos2T < 0) return 1;

    float T0 = sqrt(Cos2T);
    float T1 = Eta * T0;
    float T2 = Eta * CosW;

    float RS = (CosW - T1) / (CosW + T1);
    float RP = (T0 - T2) / (T0 + T2);

    return (RS * RS + RP * RP) / 2;
}

// Microfacet
FN_DECL vec3 SampleMicrofacet(float Roughness, vec3 Normal, vec2 RN)
{
    float Phi = 2 * PI_F * RN.x;
    float SinTheta = 0.0f;
    float CosTheta = 0.0f;

    float Theta = atan(Roughness * sqrt(RN.y / (1 - RN.y)));
    SinTheta = sin(Theta);
    CosTheta = cos(Theta);

    vec3 LocalHalfVector = vec3(
        cos(Phi) * SinTheta,
        sin(Phi) * SinTheta,
        CosTheta
    );

    //Transform the half vector to world space
    return TransformDirection(BasisFromZ(Normal), LocalHalfVector);
}


FN_DECL float MicrofacetDistribution(float Roughness, vec3 Normal, vec3 Halfway)
{
    float Cosine = dot(Normal, Halfway);
    if(Cosine <= 0) return 0;

    float Roughness2 = Roughness * Roughness;
    float Cosine2 = Cosine * Cosine;
    return Roughness2 / (PI_F * (Cosine2 * Roughness2 + 1 - Cosine2) * (Cosine2 * Roughness2 + 1 - Cosine2));
}


FN_DECL float MicrofacetShadowing1(float Roughness, vec3 Normal, vec3 Halfway, vec3 Direction)
{
    float Cosine = dot(Normal, Direction);
    float Cosine2 = Cosine * Cosine;
    float CosineH = dot(Halfway, Direction);
    
    if(Cosine * CosineH <= 0) return 0;

    float Roughness2 = Roughness * Roughness;
    return 2.0f / (sqrt(((Roughness2 * (1.0f - Cosine2)) + Cosine2) / Cosine2) + 1.0f);
}


FN_DECL float MicrofacetShadowing(float Roughness, vec3 Normal, vec3 Halfway, vec3 Outgoing, vec3 Incoming)
{
    return MicrofacetShadowing1(Roughness, Normal, Halfway, Outgoing) * MicrofacetShadowing1(Roughness, Normal, Halfway, Incoming);
    // Height Correlated shadowing doesn't really work with SSS for some reason ?
    // return 1.0f / (1.0f + MicrofacetShadowing1(Roughness, Normal, Halfway, Outgoing) + MicrofacetShadowing1(Roughness, Normal, Halfway, Incoming));
}

FN_DECL float SampleMicrofacetPDF(float Roughness, vec3 Normal, vec3 Halfway)
{
    float Cosine = dot(Normal, Halfway);
    if(Cosine < 0) return 0;

    return MicrofacetDistribution(Roughness, Normal, Halfway) * Cosine;
}


// PBR Material

FN_DECL vec3 SamplePbr(INOUT(vec3) Colour, float IOR, float Roughness,
    float Metallic, INOUT(vec3) Normal, INOUT(vec3) Outgoing, float Rand0,
    INOUT(vec2) Rand1) {
    vec3 UpNormal    = dot(Normal, Outgoing) <= 0 ? -Normal : Normal;
    vec3 Reflectivity = mix(EtaToReflectivity(vec3(IOR, IOR, IOR)), Colour, Metallic);
    if (Rand0 < Mean(FresnelSchlick(Reflectivity, UpNormal, Outgoing))) {
        vec3 Halfway  = SampleMicrofacet(Roughness, UpNormal, Rand1);
        vec3 Incoming = reflect(-Outgoing, Halfway);
        if (!SameHemisphere(UpNormal, Outgoing, Incoming)) return vec3(0, 0, 0);
        return Incoming;
    } else {
        return SampleHemisphereCosine(UpNormal, Rand1);
    }
}

FN_DECL vec3 SamplePbr(INOUT(vec3) Colour, float IOR,
    float Metallic, INOUT(vec3) Normal, INOUT(vec3) Outgoing, float Rand0) {
    vec3 UpNormal    = dot(Normal, Outgoing) <= 0 ? -Normal : Normal;
    vec3 Reflectivity = mix(EtaToReflectivity(vec3(IOR, IOR, IOR)), Colour, Metallic);
    vec3 Incoming = reflect(-Outgoing, UpNormal);
    if (!SameHemisphere(UpNormal, Outgoing, Incoming)) return vec3(0, 0, 0);
    return Incoming;
}

FN_DECL vec3 EvalPbr(INOUT(vec3) Colour, float IOR, float Roughness, float Metallic, INOUT(vec3) Normal, INOUT(vec3) Outgoing, INOUT(vec3) Incoming) {
    // Evaluate a specular BRDF lobe.
    if (dot(Normal, Incoming) * dot(Normal, Outgoing) <= 0) return vec3(0, 0, 0);

    vec3 Reflectivity = mix(EtaToReflectivity(vec3(IOR, IOR, IOR)), Colour, Metallic);
    vec3 UpNormal = dot(Normal, Outgoing) <= 0 ? -Normal : Normal;
    vec3 F1        = FresnelSchlick(Reflectivity, UpNormal, Outgoing);
    vec3 Halfway   = normalize(Incoming + Outgoing);
    vec3 F         = FresnelSchlick(Reflectivity, Halfway, Incoming);
    float D        = MicrofacetDistribution(Roughness, UpNormal, Halfway);
    float G         = MicrofacetShadowing(Roughness, UpNormal, Halfway, Outgoing, Incoming);

    float Cosine = abs(dot(UpNormal, Incoming));
    vec3 Diffuse = Colour * (1.0f - Metallic) * (1.0f - F1) / vec3(PI_F) *
                abs(dot(UpNormal, Incoming));
    vec3 Specular = F * D * G / (4 * dot(UpNormal, Outgoing) * dot(UpNormal, Incoming));

    return  Diffuse * Cosine + Specular * Cosine;
}

FN_DECL vec3 EvalPbr(INOUT(vec3) Colour, float IOR, float Metallic, INOUT(vec3) Normal, INOUT(vec3) Outgoing, INOUT(vec3) Incoming) {
    // Evaluate a specular BRDF lobe.
    if (dot(Normal, Incoming) * dot(Normal, Outgoing) <= 0) return vec3(0, 0, 0);

    vec3 Reflectivity = mix(EtaToReflectivity(vec3(IOR, IOR, IOR)), Colour, Metallic);
    vec3 UpNormal = dot(Normal, Outgoing) <= 0 ? -Normal : Normal;
    vec3 F         = FresnelSchlick(Reflectivity, UpNormal, Incoming);

    float Cosine = abs(dot(UpNormal, Incoming));
    vec3 Specular = F  / (4 * dot(UpNormal, Outgoing) * dot(UpNormal, Incoming));

    return Specular * Cosine;
}

FN_DECL float SamplePbrPDF(INOUT(vec3) Colour, float IOR, float Roughness, float Metallic, INOUT(vec3) Normal, INOUT(vec3) Outgoing, INOUT(vec3) Incoming) {
  if (dot(Normal, Incoming) * dot(Normal, Outgoing) <= 0) return 0;
  vec3 UpNormal    = dot(Normal, Outgoing) <= 0 ? -Normal : Normal;
  vec3 Halfway      = normalize(Outgoing + Incoming);
  vec3 Reflectivity = mix(EtaToReflectivity(vec3(IOR, IOR, IOR)), Colour, Metallic);
  float F = Mean(FresnelSchlick(Reflectivity, UpNormal, Outgoing));
  return F * SampleMicrofacetPDF(Roughness, UpNormal, Halfway) /
             (4 * abs(dot(Outgoing, Halfway))) +
         (1 - F) * SampleHemisphereCosinePDF(UpNormal, Incoming);
}

FN_DECL float SamplePbrPDF(INOUT(vec3) Colour, float IOR, float Metallic, INOUT(vec3) Normal, INOUT(vec3) Outgoing, INOUT(vec3) Incoming) {
  if (dot(Normal, Incoming) * dot(Normal, Outgoing) <= 0) return 0;
  vec3 UpNormal    = dot(Normal, Outgoing) <= 0 ? -Normal : Normal;
  vec3 Halfway      = normalize(Outgoing + Incoming);
  vec3 Reflectivity = mix(EtaToReflectivity(vec3(IOR, IOR, IOR)), Colour, Metallic);
  float F = Mean(FresnelSchlick(Reflectivity, UpNormal, Outgoing));
  return F /
             (4 * abs(dot(Outgoing, Halfway)));
}


// Matte

FN_DECL vec3 SampleMatte(INOUT(vec3) Colour, INOUT(vec3) Normal, INOUT(vec3) Outgoing, vec2 RN)
{
    vec3 UpNormal = dot(Normal, Outgoing) <= 0 ? -Normal : Normal;
    return SampleHemisphereCosine(UpNormal, RN);
}

FN_DECL vec3 EvalMatte(INOUT(vec3) Colour, INOUT(vec3) Normal, INOUT(vec3) Outgoing, INOUT(vec3) Incoming)
{
    if(dot(Normal, Incoming) * dot(Normal, Outgoing) <= 0) return vec3(0,0,0);
    // Lambertian BRDF:  
    // F(wi, wo) = (DiffuseReflectance / PI) * Cos(Theta)
    //Note :  This does not take the outgoing direction into account : it's perfectly isotropic : it scatters light uniformly in all directions.
    return Colour / vec3(PI_F) * abs(dot(Normal, Incoming));
}

FN_DECL float SampleMattePDF(INOUT(vec3) Colour, INOUT(vec3) Normal, INOUT(vec3) Outgoing, INOUT(vec3) Incoming)
{
    if(dot(Normal, Incoming) * dot(Normal, Outgoing) <= 0) return 0;
    // returns the pdf of a the normal
    vec3 UpNormal = dot(Normal, Outgoing) <= 0 ? -Normal : Normal;
    return SampleHemisphereCosinePDF(UpNormal, Incoming);
}

// Volumetric

FN_DECL vec3 EvalVolumetric(vec3 Colour, vec3 Normal, vec3 Outgoing, vec3 Incoming)
{
    // If Incoming and outgoing are in the same direction, return 0
    // For a volume that's not the case. 
    if(dot(Normal, Incoming) * dot(Normal, Outgoing) >= 0)
    {
        return vec3(0);
    }
    else
    {
        return vec3(1);
    }
}

FN_DECL float SampleVolumetricPDF(vec3 Colour, vec3 Normal, vec3 Outgoing, vec3 Incoming)
{
    if(dot(Normal, Incoming) * dot(Normal, Outgoing) >= 0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

FN_DECL vec3 SampleVolumetric(vec3 OutgoingDir)
{
    return -OutgoingDir;
}

// Transmittance
FN_DECL float SampleTransmittance(vec3 Density, float MaxDistance, float RL, float RD)
{
    // Choose a random channel to use
    int Channel = clamp(int(RL * 3), 0, 2);
    
    // Here we calculate the distance that a ray traverses inside a medium. We sample this distance using the exponential function, 
    //using the density as the mean free path (=average distance a photon can travel through a medium before an interaction, absorbtion or scattering, happens.)
    //Here, the mean free path is simply 1 / density

    // Calculate the distance we travel, using the inverse of the CDF of the exponential function * mean free path.
    float Distance = (Density[Channel] == 0) ? MAX_LENGTH : -log(1 - RD) / Density[Channel];
    
    return min(Distance, MaxDistance);
}

FN_DECL vec3 EvalTransmittance(vec3 Density, float Distance)
{
    // Beer-Lambert law : attenuation of light as it passes through a medium, as a function of the extinction coefficient and of the distance travelled inside the medium.
    return exp(-Density * Distance);
}

FN_DECL float SampleTransmittancePDF(vec3 Density, float Distance, float MaxDistance)
{
    // We use the pdf of the exponential distribution, because we're sampling distances with the exponential distribution.
    // the pdf is pdf(x, delta) = delta * exp(-delta * x)
    //Here, x = distance and delta = rate parameter, inverse of the mean free path, which is (1 / density), so delta = density.

    if(Distance < MaxDistance)
    {
        return Sum(Density * exp(-Density * Distance)) / 3;
    }
    else
    {
        return Sum(exp(-Density * MaxDistance)) / 3;
    }
}

// Glass
FN_DECL vec3 EvalGlass(vec3 Colour, float IOR, float Roughness, vec3 Normal, vec3 Outgoing, vec3 Incoming)
{
    bool Entering = dot(Normal, Outgoing) >= 0;
    vec3 UpNormal = Entering ? Normal : -Normal;
    float RelIOR = Entering ? IOR : 1 / IOR;
    
    if(dot(Normal, Incoming) * dot(Normal, Outgoing) >=0)
    {
        vec3 Halfway = normalize(Incoming + Outgoing);
        float F = FresnelDielectric(RelIOR, Halfway, Outgoing);
        float D = MicrofacetDistribution(Roughness, UpNormal, Halfway);
        float G = MicrofacetShadowing(Roughness, UpNormal, Halfway, Outgoing, Incoming);
        return vec3(1) * F * D * G /
                abs(4 * dot(Normal, Outgoing) * dot(Normal, Incoming)) * 
                abs(dot(Normal, Incoming));
    }
    else
    {
        vec3 Halfway = -normalize(RelIOR *  Incoming + Outgoing) * (Entering ? 1.0f : -1.0f);

        float F = FresnelDielectric(RelIOR, Halfway, Outgoing);
        float D = MicrofacetDistribution(Roughness, UpNormal, Halfway);
        float G = MicrofacetShadowing(Roughness, UpNormal, Halfway, Outgoing, Incoming);
        
        return vec3(1) * 
                abs(
                    (dot(Outgoing, Halfway)*dot(Incoming, Halfway)) / 
                    (dot(Outgoing, Normal) * dot(Incoming, Normal))
                ) * (1 - F) * D * G /
                pow(RelIOR * dot(Halfway, Incoming) + dot(Halfway, Outgoing), 2.0f) * 
                abs(dot(Normal, Incoming));
    }
}

FN_DECL vec3 EvalGlass(vec3 Colour, float IOR, vec3 Normal, vec3 Outgoing, vec3 Incoming)
{
    bool Entering = dot(Normal, Outgoing) >= 0;
    vec3 UpNormal = Entering ? Normal : -Normal;
    float RelIOR = Entering ? IOR : 1 / IOR;
    
    if(dot(Normal, Incoming) * dot(Normal, Outgoing) >=0)
    {
        float F = FresnelDielectric(RelIOR, UpNormal, Outgoing);
        return vec3(1) * F;
    }
    else
    {
        float F = FresnelDielectric(RelIOR, UpNormal, Outgoing);
        return vec3(1) * (1 / (RelIOR * RelIOR)) * (1 - F);
    }
}


FN_DECL vec3 SampleGlass(vec3 Colour, float IOR, float Roughness, vec3 Normal, vec3 Outgoing, float RNL, vec2 RN)
{
    bool Entering = dot(Normal, Outgoing) >= 0;
    vec3 UpNormal = Entering ? Normal : -Normal;
    vec3 Halfway = SampleMicrofacet(Roughness, UpNormal, RN);

    if(RNL < FresnelDielectric(Entering ? IOR : (1/IOR), Halfway, Outgoing))
    {
        vec3 Incoming = reflect(-Outgoing, Halfway);
        if(!SameHemisphere(UpNormal, Outgoing, Incoming)) return vec3(0);
        return Incoming;
    }
    else
    {
        vec3 Incoming = refract(-Outgoing, Halfway, Entering ? (1 / IOR) : IOR);
        if(SameHemisphere(UpNormal, Outgoing, Incoming)) return vec3(0);
        return Incoming;
    }
}

FN_DECL vec3 SampleGlass(vec3 Colour, float IOR, vec3 Normal, vec3 Outgoing, float RNL)
{
    bool Entering = dot(Normal, Outgoing) >= 0;
    vec3 UpNormal = Entering ? Normal : -Normal;
    float RelIOR = Entering ? IOR : (1/IOR);
    if(RNL < FresnelDielectric(RelIOR, UpNormal, Outgoing))
    {
        return reflect(-Outgoing, UpNormal);
    }
    else
    {
        return refract(-Outgoing, UpNormal, 1 / RelIOR);
    }
}

FN_DECL float SampleGlassPDF(vec3 Colour, float IOR, float Roughness, vec3 Normal, vec3 Outgoing, vec3 Incoming)
{
    bool Entering = dot(Normal, Outgoing) >= 0;
    vec3 UpNormal = Entering ? Normal : -Normal;
    float RelIOR = Entering ? IOR : (1 / IOR);
    if(dot(Normal, Incoming) * dot(Normal, Outgoing) >= 0)
    {
        vec3 Halfway = normalize(Incoming + Outgoing);
        return FresnelDielectric(RelIOR, Halfway, Outgoing) * 
               SampleMicrofacetPDF(Roughness, UpNormal, Halfway) / 
               (4 * abs(dot(Outgoing, Halfway)));
    }
    else
    {
        vec3 Halfway = -normalize(RelIOR * Incoming + Outgoing) * (Entering ? 1.0f : -1.0f);
        return (1 - FresnelDielectric(RelIOR, Halfway, Outgoing)) * 
               SampleMicrofacetPDF(Roughness, UpNormal, Halfway) * 
               abs(dot(Halfway, Incoming)) / 
               pow(RelIOR * dot(Halfway, Incoming) + dot(Halfway, Outgoing), 2.0f);
    }
}

FN_DECL float SampleGlassPDF(vec3 Colour, float IOR, vec3 Normal, vec3 Outgoing, vec3 Incoming)
{
    bool Entering = dot(Normal, Outgoing) >= 0;
    vec3 UpNormal = Entering ? Normal : -Normal;
    float RelIOR = Entering ? IOR : (1 / IOR);
    if(dot(Normal, Incoming) * dot(Normal, Outgoing) >= 0)
    {
        return FresnelDielectric(RelIOR, UpNormal, Outgoing);
    }
    else
    {
        return (1 - FresnelDielectric(RelIOR, UpNormal, Outgoing));
    }
}

// Phase

FN_DECL vec3 SamplePhase(INOUT(materialPoint) Material, vec3 Outgoing, float RNL, vec2 RN)
{
    if(Material.Density == vec3(0)) return vec3(0);
    float CosTheta = 0;
    if(abs(Material.Anisotropy) < 1e-3f)
    {
        CosTheta = 1 - 2 * RN.y;
    }
    else
    {
        float Square = (1 - Material.Anisotropy * Material.Anisotropy) / 
                       (1 + Material.Anisotropy - 2 * Material.Anisotropy * RN.y);
        CosTheta = (1 + Material.Anisotropy * Material.Anisotropy - Square * Square) / 
                   (2 * Material.Anisotropy);
    }

    float SinTheta = sqrt(max(0.0f, 1- CosTheta * CosTheta));
    float Phi = 2 * PI_F * RN.x;
    vec3 LocalIncoming = vec3(SinTheta * cos(Phi), SinTheta * sin(Phi), CosTheta);
    return BasisFromZ(-Outgoing) * LocalIncoming;
}

FN_DECL vec3 EvalPhase(INOUT(materialPoint) Material, vec3 Outgoing, vec3 Incoming)
{
    if(Material.Density == vec3(0)) return vec3(0);
    
    float Cosine = -dot(Outgoing, Incoming);
    float Denom = pow(1 + Material.Anisotropy * Material.Anisotropy - 2 * Material.Anisotropy * Cosine, 1.5f);
    float PhaseFunction = (1 - Material.Anisotropy * Material.Anisotropy) 
        / (4 * PI_F * Denom * sqrt(Denom));

    return Material.ScatteringColour * Material.Density * PhaseFunction;
}

FN_DECL float SamplePhasePDF(INOUT(materialPoint) Material, vec3 Outgoing, vec3 Incoming)
{
    if(Material.Density == vec3(0)) return 0;

    float Cosine = -dot(Outgoing, Incoming);
    float Denom = pow(1 + Material.Anisotropy * Material.Anisotropy - 2 * Material.Anisotropy * Cosine, 1.5f);

    return (1 - Material.Anisotropy * Material.Anisotropy) / (4 * PI_F * Denom * sqrt(Denom));
}

// BSDF

FN_DECL bool IsDelta(INOUT(materialPoint) Material)
{
    return 
    ( Material.MaterialType == MATERIAL_TYPE_PBR && Material.Roughness==0) ||
    ( Material.MaterialType == MATERIAL_TYPE_GLASS && Material.Roughness==0) ||
    ( Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC);    
}

FN_DECL vec3 EvalBSDFCos(INOUT(materialPoint) Material, vec3 Normal, vec3 OutgoingDir, vec3 Incoming)
{
    if(Material.MaterialType == MATERIAL_TYPE_MATTE)
    {
        return EvalMatte(Material.Colour, Normal, OutgoingDir, Incoming);
    }
    else if(Material.MaterialType == MATERIAL_TYPE_PBR)
    {
        return EvalPbr(Material.Colour, 1.5, Material.Roughness, Material.Metallic, Normal, OutgoingDir, Incoming);
    }
    else if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        return EvalVolumetric(Material.Colour, Normal, OutgoingDir, Incoming);
    }    
    else if(Material.MaterialType == MATERIAL_TYPE_GLASS)
    {
        return EvalGlass(Material.Colour, 1.5, Material.Roughness, Normal, OutgoingDir, Incoming);
    }    
    else if(Material.MaterialType == MATERIAL_TYPE_SUBSURFACE)
    {
        return EvalGlass(Material.Colour, 1.5, Material.Roughness, Normal, OutgoingDir, Incoming);
    }    
}

FN_DECL float SampleBSDFCosPDF(INOUT(materialPoint) Material, INOUT(vec3) Normal, INOUT(vec3) OutgoingDir, INOUT(vec3) Incoming)
{
    if(Material.MaterialType == MATERIAL_TYPE_MATTE)
    {
        return SampleMattePDF(Material.Colour, Normal, OutgoingDir, Incoming);
    }
    else if(Material.MaterialType == MATERIAL_TYPE_PBR)
    {
        return SamplePbrPDF(Material.Colour, 1.5, Material.Roughness, Material.Metallic, Normal, OutgoingDir, Incoming);
    }
    else if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        return SampleVolumetricPDF(Material.Colour, Normal, OutgoingDir, Incoming);
    }    
    else if(Material.MaterialType == MATERIAL_TYPE_GLASS)
    {
        return SampleGlassPDF(Material.Colour, 1.5, Material.Roughness, Normal, OutgoingDir, Incoming);
    }    
    else if(Material.MaterialType == MATERIAL_TYPE_SUBSURFACE)
    {
        return SampleGlassPDF(Material.Colour, 1.5, Material.Roughness, Normal, OutgoingDir, Incoming);
    }    
}

FN_DECL vec3 SampleBSDFCos(INOUT(materialPoint) Material, INOUT(vec3) Normal, INOUT(vec3) OutgoingDir, float RNL, vec2 RN)
{
    if(Material.MaterialType == MATERIAL_TYPE_MATTE)
    {
        return SampleMatte(Material.Colour, Normal, OutgoingDir, RN);
    }
    else if(Material.MaterialType == MATERIAL_TYPE_PBR)
    {
        return SamplePbr(Material.Colour, 1.5, Material.Roughness, Material.Metallic, Normal, OutgoingDir, RNL, RN);
    }
    else if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        return SampleVolumetric(OutgoingDir);
    }    
    else if(Material.MaterialType == MATERIAL_TYPE_GLASS)
    {
        return SampleGlass(Material.Colour, 1.5, Material.Roughness, Normal, OutgoingDir, RNL, RN);
    }    
    else if(Material.MaterialType == MATERIAL_TYPE_SUBSURFACE)
    {
        return SampleGlass(Material.Colour, 1.5, Material.Roughness, Normal, OutgoingDir, RNL, RN);
    }    
}

// Delta
FN_DECL vec3 EvalDelta(INOUT(materialPoint) Material, vec3 Normal, vec3 OutgoingDir, vec3 Incoming)
{
    if(Material.Roughness != 0) return vec3(0,0,0);

    if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        return EvalVolumetric(Material.Colour, Normal, OutgoingDir, Incoming);
    }    
    if(Material.MaterialType == MATERIAL_TYPE_PBR)
    {
        return EvalPbr(Material.Colour, 1.5, Material.Metallic, Normal, OutgoingDir, Incoming);
    }
    if(Material.MaterialType == MATERIAL_TYPE_GLASS)
    {
        return EvalGlass(Material.Colour, 1.5, Normal, OutgoingDir, Incoming);
    }
}

FN_DECL float SampleDeltaPDF(INOUT(materialPoint) Material, INOUT(vec3) Normal, INOUT(vec3) OutgoingDir, INOUT(vec3) Incoming)
{
    if(Material.Roughness != 0) return 0;

    if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        return SampleVolumetricPDF(Material.Colour, Normal, OutgoingDir, Incoming);
    }
    if(Material.MaterialType == MATERIAL_TYPE_GLASS)
    {
        return SampleGlassPDF(Material.Colour, 1.5, Normal, OutgoingDir, Incoming);
    }
    if(Material.MaterialType == MATERIAL_TYPE_PBR)
    {
        return SamplePbrPDF(Material.Colour, 1.5, Material.Metallic, Normal, OutgoingDir, Incoming);
    }
}

FN_DECL vec3 SampleDelta(INOUT(materialPoint) Material, INOUT(vec3) Normal, INOUT(vec3) OutgoingDir, float RNL)
{
    if(Material.Roughness != 0) return vec3(0,0,0);

    if(Material.MaterialType == MATERIAL_TYPE_GLASS)
    {
        return SampleGlass(Material.Colour, 1.5, Normal, OutgoingDir, RNL);
    }
    if(Material.MaterialType == MATERIAL_TYPE_PBR)
    {
        return SamplePbr(Material.Colour, 1.5, Material.Metallic, Normal, OutgoingDir, RNL);
    }
    if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        return SampleVolumetric(OutgoingDir);
    }

}



// Texture Eval

__device__ vec4 textureSample(cudaTextureObject_t _SceneTextures, vec3 Coords)
{
    // Coords.x = Coordx.x % 1.0f;
    float W;
    if(Coords.x < 0) Coords.x = 1 - Coords.x;
    if(Coords.y < 0) Coords.y = 1 - Coords.y;

    Coords.x = std::modf(Coords.x, &W);
    Coords.y = std::modf(Coords.y, &W);

    int NumLayersX = 8192 / TexturesWidth;
    int LayerInx = Coords.z;
    
    int LocalCoordX = Coords.x * TexturesWidth;
    int LocalCoordY = Coords.y * TexturesHeight;

    int XOffset = (LayerInx % NumLayersX) * TexturesWidth;
    int YOffset = (LayerInx / NumLayersX) * TexturesHeight;

    int CoordX = XOffset + LocalCoordX;
    int CoordY = YOffset + LocalCoordY;

    uchar4 TexValue = tex2D<uchar4>(_SceneTextures, CoordX, CoordY);
    vec4 TexValueF = vec4((float)TexValue.x / 255.0f, (float)TexValue.y / 255.0f, (float)TexValue.z / 255.0f, (float)TexValue.w / 255.0f);
    return TexValueF;
}

__device__ vec4 textureSampleEnv(cudaTextureObject_t _EnvTextures, vec3 Coords)
{
    int NumLayersX = 8192 / EnvTexturesWidth;
    int LayerInx = Coords.z;
    
    int LocalCoordX = Coords.x * EnvTexturesWidth;
    int LocalCoordY = Coords.y * EnvTexturesHeight;

    int XOffset = (LayerInx % NumLayersX) * EnvTexturesWidth;
    int YOffset = (LayerInx / NumLayersX) * EnvTexturesHeight;

    int CoordX = XOffset + LocalCoordX;
    int CoordY = YOffset + LocalCoordY;

    float4 TexValue = tex2D<float4>(_EnvTextures, CoordX, CoordY);
    vec4 TexValueF = vec4(TexValue.x, TexValue.y, TexValue.z, TexValue.w);
    return TexValueF;
}

FN_DECL vec2 EvalTexCoord(INOUT(sceneIntersection) Isect)
{
    uint Element = Isect.PrimitiveIndex;
    triangle Tri = TriangleBuffer[Isect.PrimitiveIndex];

    return
        vec2(Tri.PositionUvX1.w, Tri.NormalUvY1.w) * Isect.U + 
        vec2(Tri.PositionUvX2.w, Tri.NormalUvY2.w) * Isect.V +
        vec2(Tri.PositionUvX0.w, Tri.NormalUvY0.w) * (1 - Isect.U - Isect.V);    
}

FN_DECL vec4 EvalTexture(int Texture, vec2 UV, bool Linear)
{
    if(Texture == INVALID_ID) return vec4(1, 1, 1, 1);
    vec3 texCoord3D = vec3(UV, Texture);
    // vec4 Colour = textureSample(SceneTextures, texCoord3D); 
    vec4 Colour = vec4(1);
    if(Linear) Colour = ToLinear(Colour);
    return Colour;
}

FN_DECL vec4 EvalEnvTexture(int Texture, vec2 UV, bool Linear)
{
    if(Texture == INVALID_ID) return vec4(1, 1, 1, 1);
    vec3 texCoord3D = vec3(UV, Texture);
    vec4 Colour = textureSampleEnv(EnvTextures, texCoord3D); 
    if(Linear) Colour = ToLinear(Colour);
    return Colour;
}

FN_DECL vec3 EvalNormalMap(vec3 Normal, INOUT(sceneIntersection) Isect)
{
    vec2 UV = EvalTexCoord(Isect);
    if(Materials[Isect.MaterialIndex].NormalTexture != INVALID_ID)
    {
        vec3 NormalTex = vec3(2) * vec3(EvalTexture(Materials[Isect.MaterialIndex].NormalTexture, UV, false)) - vec3(1);

        mat3 TBN = GetTBN(Isect, Normal);
        Normal = TBN * normalize(NormalTex);

        return normalize(Normal);
    }
    return Normal;
}

// Misc Eval functions

FN_DECL vec3 EvalShadingPosition(INOUT(vec3) OutgoingDir, INOUT(sceneIntersection) Isect)
{
    uint Element = Isect.PrimitiveIndex;
    triangle Tri = TriangleBuffer[Isect.PrimitiveIndex];

    vec3 Position = 
        vec3(Tri.PositionUvX1) * Isect.U + 
        vec3(Tri.PositionUvX2) * Isect.V +
        vec3(Tri.PositionUvX0) * (1 - Isect.U - Isect.V);
    return TransformPoint(Isect.InstanceTransform, Position);
}
FN_DECL vec3 EvalShadingNormal(INOUT(vec3) OutgoingDir, INOUT(sceneIntersection) Isect)
{
    vec3 Normal = EvalNormalMap(Isect.Normal, Isect);
    if (Materials[Isect.MaterialIndex].MaterialType == MATERIAL_TYPE_GLASS) return Normal;
    return dot(Normal, OutgoingDir) >= 0 ? Normal : -Normal;
}

FN_DECL materialPoint EvalMaterial(INOUT(sceneIntersection) Isect)
{
    material Material = Materials[Isect.MaterialIndex];
    materialPoint Point;

    vec2 TexCoords = EvalTexCoord(Isect);
    vec4 EmissionTexture = EvalTexture(Material.EmissionTexture, TexCoords, true);    
    vec4 ColourTexture = EvalTexture(Material.ColourTexture, TexCoords, true);
    vec4 RoughnessTexture = EvalTexture(Material.RoughnessTexture, TexCoords, false);
    
    Point.MaterialType = int(Material.MaterialType);
    Point.Colour = Material.Colour * vec3(ColourTexture);
    Point.Emission = Material.Emission * vec3(EmissionTexture);
    
    Point.Metallic = Material.Metallic * RoughnessTexture.z;
    Point.Roughness = Material.Roughness * RoughnessTexture.y;
    Point.Roughness = Point.Roughness * Point.Roughness;

    Point.Opacity = Material.Opacity * ColourTexture.w;
    Point.TransmissionDepth = Material.TransmissionDepth;
    Point.ScatteringColour = Material.ScatteringColour;

    Point.Anisotropy = Material.Anisotropy;
    Point.Density = vec3(0,0,0);

    if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC || Material.MaterialType == MATERIAL_TYPE_GLASS || Material.MaterialType == MATERIAL_TYPE_SUBSURFACE)
    {
        Point.Density = -log(clamp(Point.Colour, 0.0001f, 1.0f)) / Point.TransmissionDepth;
    }

    if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        Point.Roughness=0;
    }
    
    if(Point.Roughness < MIN_ROUGHNESS) Point.Roughness=0;
        

    return Point;
}

FN_DECL vec3 EvalEmission(INOUT(materialPoint) Material, INOUT(vec3) Normal, INOUT(vec3) Outgoing) {
  return dot(Normal, Outgoing) >= 0 ? Material.Emission : vec3(0, 0, 0);
}

FN_DECL bool IsVolumetric(INOUT(materialPoint) Material)
{
    return ( (Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC) ||     
             (Material.MaterialType == MATERIAL_TYPE_GLASS) ||
             (Material.MaterialType == MATERIAL_TYPE_SUBSURFACE)
             );
}

// region environment
FN_DECL vec3 EvalEnvironment(INOUT(environment) Env, vec3 Direction)
{
    
    vec3 WorldDir = TransformDirection(inverse(Env.Transform), Direction);

    vec2 TexCoord = vec2(
        atan(WorldDir.x, WorldDir.z) / (2 * PI_F), 
        acos(clamp(WorldDir.y, -1.0f, 1.0f)) / PI_F
    );
    if(TexCoord.x < 0) TexCoord.x += 1.0f; 

    return Env.Emission * vec3(EvalEnvTexture(Env.EmissionTexture, TexCoord, false));
    
}

FN_DECL vec3 EvalEnvironment(vec3 Direction)
{
    vec3 Emission = vec3(0,0,0);
    for(int i=0; i< EnvironmentsCount; i++)
    {
        Emission += EvalEnvironment(Environments[i], Direction);
    }
    return Emission;
}



FN_DECL vec4 SampleCuTextureHalf4(cudaTextureObject_t Texture, ivec2 Coord)
{
    ushort4 Sample = tex2D<ushort4>(Texture, Coord.x, Coord.y);
    return commonCu::Half4ToVec4({
        __ushort_as_half(Sample.x),
        __ushort_as_half(Sample.y),
        __ushort_as_half(Sample.z),
        __ushort_as_half(Sample.w)
    });
}


FN_DECL sceneIntersection MakeFirstIsect(int Sample)
{
    uvec2 Coord = GLOBAL_ID();
    sceneIntersection Isect = {};
    Isect.Distance = MAX_LENGTH;
    float t = Time * float(GLOBAL_ID().x) * 1973.0f;
    Isect.RandomState = CreateRNG(uint(uint(t) + uint(GLOBAL_ID().y) * uint(9277)  + Sample * uint(117191)) | uint(1)); 


    float4 Position = tex2D<float4>(CurrentFramebuffer.PositionTexture, Coord.x, Coord.y);
    vec4 UV = SampleCuTextureHalf4(CurrentFramebuffer.UVTexture, Coord);
    vec4 Normal = SampleCuTextureHalf4(CurrentFramebuffer.NormalTexture, Coord);
    float4 Motion = tex2D<float4>(CurrentFramebuffer.MotionTexture, Coord.x, Coord.y);
    
    if(length(vec3(Position.x, Position.y, Position.z)) != 0)
    {
        Isect.InstanceIndex = uint(UV.w);
        Isect.PrimitiveIndex = uint(Position.w);
        Isect.U = UV.x;
        Isect.V = UV.y;
        Isect.Distance = Motion.z;
        Isect.MaterialIndex = uint(Normal.w);
    }
    

    return Isect;
}



}
