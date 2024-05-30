
#include <optix.h>
#include <optix_device.h>

#include <cuda_fp16.h>
#include "Common.cuh"
#include <cuda_runtime.h>

namespace pathtracing
{
using namespace glm;
using namespace commonCu;

extern "C" {
__constant__ kernelParams KernelParams;
}


FN_DECL sceneIntersection IntersectTLAS(ray Ray, int Sample, int Bounce)
{

    rayPayload Payload;
    optixTrace(
        KernelParams.Handle,
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

    sceneIntersection Isect = {};
    float t = Time * float(GLOBAL_ID().x) * 1973.0f;
    Isect.RandomState = CreateRNG(uint(uint(t) + uint(GLOBAL_ID().y) * uint(9277)  +  uint(Bounce + Sample) * uint(117191)) | uint(1)); 
    Isect.InstanceIndex = Payload.InstanceIndex;

    int MeshIndex = TLASInstancesBuffer[Isect.InstanceIndex].Shape;
    indexData IndexData = IndexDataBuffer[MeshIndex];
    uint TriangleStartInx = IndexData.triangleDataStartInx;
    Isect.PrimitiveIndex = Payload.PrimitiveIndex + TriangleStartInx;

    Isect.U = uint_as_float(Payload.U);
    Isect.V = uint_as_float(Payload.V);
    Isect.Distance = uint_as_float(Payload.Distance);
    Isect.MaterialIndex = TLASInstancesBuffer[Isect.InstanceIndex].Material;
    return Isect;
}

FN_DECL vec3 PathTrace(int Sample, vec2 UV)
{
    float t = Time * float(GLOBAL_ID().x) * 1973.0f;
    randomState RandomState = CreateRNG(uint( uint(t) + uint(GLOBAL_ID().y) * uint(9277) + uint(Sample) * uint(26699)) | uint(1) ); 
    ray Ray = GetRay(UV, vec2(0));



    vec3 Radiance = vec3(0,0,0);
    vec3 Weight = vec3(1,1,1);
    uint OpacityBounces=0;
    materialPoint VolumeMaterial;
    bool HasVolumeMaterial=false;
    


    for(int Bounce=0; Bounce < Parameters->Bounces; Bounce++)
    {
        sceneIntersection Isect = {};
        // if(Bounce==0) Isect = MakeFirstIsect(Sample);
        // else 
            Isect = IntersectTLAS(Ray, Sample, Bounce);

        if(Isect.Distance == MAX_LENGTH)
        {
            // Radiance += Weight * EvalEnvironment(Ray.Direction);
            Radiance += Weight * EvalEnvironment(Ray.Direction);
            // Radiance += Weight * vec3(0.5);
            break;
        }
        // get all the necessary geometry information
        triangle Tri = TriangleBuffer[Isect.PrimitiveIndex];    
        Isect.InstanceTransform = TLASInstancesBuffer[Isect.InstanceIndex].Transform;
        mat4 NormalTransform = TLASInstancesBuffer[Isect.InstanceIndex].NormalTransform;
        vec3 HitNormal = vec3(Tri.NormalUvY1) * Isect.U + vec3(Tri.NormalUvY2) * Isect.V +vec3(Tri.NormalUvY0) * (1 - Isect.U - Isect.V);
        vec4 Tangent = Tri.Tangent1 * Isect.U + Tri.Tangent2 * Isect.V + Tri.Tangent0 * (1 - Isect.U - Isect.V);
        Isect.Normal = TransformDirection(NormalTransform, HitNormal);
        Isect.Tangent = TransformDirection(NormalTransform, vec3(Tangent));
        Isect.Bitangent = TransformDirection(NormalTransform, normalize(cross(Isect.Normal, vec3(Tangent)) * Tangent.w));    
        
        bool StayInVolume=false;
        if(HasVolumeMaterial)
        {
            // If we're in a volume, we sample the distance that the ray's gonna intersect.
            // The transmittance is based on the colour of the object. The higher, the thicker.
            float Distance = SampleTransmittance(VolumeMaterial.Density, Isect.Distance, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState));
            // float Distance = 0.1f;
            Weight *= EvalTransmittance(VolumeMaterial.Density, Distance) / 
                    SampleTransmittancePDF(VolumeMaterial.Density, Distance, Isect.Distance);
            
            
            //If the distance is higher than the next intersection, it means that we're stepping out of the volume
            StayInVolume = Distance < Isect.Distance;
            
            Isect.Distance = Distance;
        }

        if(!StayInVolume)
        {

            vec3 OutgoingDir = -Ray.Direction;
            vec3 Position = EvalShadingPosition(OutgoingDir, Isect);
            vec3 Normal = EvalShadingNormal(OutgoingDir, Isect);

            


            // Material evaluation
            materialPoint Material = EvalMaterial(Isect);
            
            // Opacity
            if(Material.Opacity < 1 && RandomUnilateral(Isect.RandomState) >= Material.Opacity)
            {
                if(OpacityBounces++ > 128) break;
                Ray.Origin = Position + Ray.Direction * 1e-2f;
                Bounce--;
                continue;
            }



            Radiance += Weight * EvalEmission(Material, Normal, OutgoingDir);

            vec3 Incoming = vec3(0);
            if(!IsDelta(Material))
            {
                if(GET_ATTR(Parameters, SamplingMode) == SAMPLING_MODE_LIGHT)
                {
                    Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
                    if(Incoming == vec3(0,0,0)) break;
                    float PDF = SampleLightsPDF(Position, Incoming);
                    if(PDF > 0)
                    {
                        Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) / vec3(PDF);   
                    }
                    else  
                    {
                        break;
                    }
                }
                else if(GET_ATTR(Parameters, SamplingMode) == SAMPLING_MODE_BSDF)
                {
                    Incoming = SampleBSDFCos(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
                    if(Incoming == vec3(0,0,0)) break;
                    Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) / 
                            vec3(SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming));
                }
                else
                {
                    if(RandomUnilateral(Isect.RandomState) > 0.5f)
                    {
                        Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
                        if(Incoming == vec3(0,0,0)) break;
                        float PDF = SampleLightsPDF(Position, Incoming);
                        if(PDF > 0)
                        {
                            Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) / vec3(PDF);   
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        Incoming = SampleBSDFCos(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
                        if(Incoming == vec3(0,0,0)) break;
                        Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) / 
                                vec3(SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming));
                    }
                }

            }
            else
            {
                Incoming = SampleDelta(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState));
                Weight *= EvalDelta(Material, Normal, OutgoingDir, Incoming) / 
                        SampleDeltaPDF(Material, Normal, OutgoingDir, Incoming);                        
            }

            
            // If the hit material is volumetric
            // And the ray keeps going in the same direction (It always does for volumetric materials)
            // we add the volume material into the stack 
            if(IsVolumetric(Material)   && dot(Normal, OutgoingDir) * dot(Normal, Incoming) < 0)
            {
                VolumeMaterial = Material;
                HasVolumeMaterial = !HasVolumeMaterial;
            }

            Ray.Origin = Position + (dot(Normal, Incoming) > 0 ? Normal : -Normal) * 0.001f;
            Ray.Direction = Incoming;
        }
        else
        {
            vec3 Outgoing = -Ray.Direction;
            vec3 Position = Ray.Origin + Ray.Direction * Isect.Distance;

            vec3 Incoming = vec3(0);
            if(RandomUnilateral(Isect.RandomState) > 0.5f)
            {
                // Sample a scattering direction inside the volume using the phase function
                Incoming = SamplePhase(VolumeMaterial, Outgoing, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
            }
            else
            {
                Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));                
            }

            if(Incoming == vec3(0)) break;
        
            Weight *= EvalPhase(VolumeMaterial, Outgoing, Incoming) / 
                    ( 
                    0.5f * SamplePhasePDF(VolumeMaterial, Outgoing, Incoming)
                    + 0.5f * SampleLightsPDF(Position, Incoming)
                    );
                    
            Ray.Origin = Position;
            Ray.Direction = Incoming;
        }

        if(Weight == vec3(0,0,0) || !commonCu::IsFinite(Weight)) break;

        if(Bounce > 3)
        {
            float RussianRouletteProb = min(0.99f, max3(Weight));
            if(RandomUnilateral(Isect.RandomState) >= RussianRouletteProb) break;
            Weight *= 1.0f / RussianRouletteProb;
        }                
    }

    if(!commonCu::IsFinite(Radiance)) Radiance = vec3(0,0,0);
    if(max3(Radiance) > GET_ATTR(Parameters, Clamp)) Radiance = Radiance * (GET_ATTR(Parameters, Clamp) / max3(Radiance)); 


    return Radiance;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 blockIdx = optixGetLaunchIndex();
    const uint3 blockDim = optixGetLaunchDimensions();


    Width = KernelParams.Width;
    Height = KernelParams.Height;
    TriangleBuffer = KernelParams.TriangleBuffer;
    IndicesBuffer = KernelParams.IndicesBuffer;
    TLASInstancesBuffer = KernelParams.Instances;
    Cameras = KernelParams.Cameras;
    Parameters = KernelParams.Parameters;
    Materials = KernelParams.Materials;
    SceneTextures = KernelParams.SceneTextures;
    EnvTextures = KernelParams.EnvTextures;
    LightsCount = KernelParams.LightsCount;
    Lights = KernelParams.Lights;
    LightsCDF = KernelParams.LightsCDF;
    EnvironmentsCount = KernelParams.EnvironmentsCount;
    Environments = KernelParams.Environments;
    TexturesWidth = KernelParams.TexturesWidth;
    TexturesHeight = KernelParams.TexturesHeight;
    EnvTexturesWidth = KernelParams.EnvTexturesWidth;
    EnvTexturesHeight = KernelParams.EnvTexturesHeight;
    Time = KernelParams.Time;
    IndexDataBuffer = KernelParams.IndexDataBuffer;
    ShapeASHandles = KernelParams.ShapeASHandles;
    IASHandle = KernelParams.Handle;
    CurrentFramebuffer = KernelParams.CurrentFramebuffer;
     
    vec2 UV = vec2(
        (float(blockIdx.x) + 0.5f) / float(blockDim.x),
        1 - (float(blockIdx.y) + 0.5f) / float(blockDim.y)
    );
    vec3 color = PathTrace(0, UV);
    
    half4 *OutputPtr = (half4*)KernelParams.OutputBuffer;
    OutputPtr[blockIdx.y * KernelParams.Width + blockIdx.x] = Vec4ToHalf4(vec4(color, 1));
}

}