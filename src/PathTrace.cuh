// #pragma once
// #include "BVH.h"
// #include "App.h"

// #define GLM_FORCE_CUDA
// #include <glm/glm.hpp>

// #include <cuda_fp16.h>
// #include "Common.cuh"

// namespace pathtracing
// {
// using namespace glm;
// using namespace gpupt;
// using namespace commonCu;


// #define DENOISE_RANGE vec2(1, 4)









// __device__ void imageStore(vec4 *Image, ivec2 p, vec4 Colour)
// {
//     p.x = clamp(p.x, 0, int(Width-1));
//     p.y = clamp(p.y, 0, int(Height-1));
     
//     Image[p.y * Width + p.x] = clamp(Colour, vec4(0), vec4(1));
// }

// __device__ vec4 imageLoad(vec4 *Image, ivec2 p)
// {
//     p.x = clamp(p.x, 0, int(Width-1));
//     p.y = clamp(p.y, 0, int(Height-1));
//     return clamp(Image[p.y * Width + p.x], vec4(0), vec4(1));
// }

// // Util
// FN_DECL float Sum(vec3 A) { 
//     return A.x + A.y + A.z; 
// }

// FN_DECL float Mean(vec3 A) { 
//     return Sum(A) / 3; 
// }

// FN_DECL bool SameHemisphere(vec3 Normal, vec3 Outgoing, vec3 Incoming)
// {
//     return dot(Normal, Outgoing) * dot(Normal, Incoming) >= 0;
// }

// FN_DECL float max3 (vec3 v) {
//   return max (max (v.x, v.y), v.z);
// }

// // region utils
// FN_DECL float ToLinear(float SRGB) {
//   return (SRGB <= 0.04045) ? SRGB / 12.92f
//                            : pow((SRGB + 0.055f) / (1.0f + 0.055f), 2.4f);
// }

// FN_DECL vec4 ToLinear(vec4 SRGB)
// {
//     return vec4(
//         ToLinear(SRGB.x),
//         ToLinear(SRGB.y),
//         ToLinear(SRGB.z),
//         SRGB.w
//     );
// }

// FN_DECL mat3 GetTBN(INOUT(sceneIntersection) Isect, vec3 Normal)
// { 
//     return mat3(Isect.Tangent, Isect.Bitangent, Normal);    
// }


// // BVH

// FN_DECL float RayAABBIntersection(ray Ray, vec3 AABBMin, vec3 AABBMax, INOUT(sceneIntersection) Isect)
// {
//     float tx1 = (AABBMin.x - Ray.Origin.x) * Ray.InverseDirection.x, tx2 = (AABBMax.x - Ray.Origin.x) * Ray.InverseDirection.x;
//     float tmin = min( tx1, tx2 ), tmax = max( tx1, tx2 );
//     float ty1 = (AABBMin.y - Ray.Origin.y) * Ray.InverseDirection.y, ty2 = (AABBMax.y - Ray.Origin.y) * Ray.InverseDirection.y;
//     tmin = max( tmin, min( ty1, ty2 ) ), tmax = min( tmax, max( ty1, ty2 ) );
//     float tz1 = (AABBMin.z - Ray.Origin.z) * Ray.InverseDirection.z, tz2 = (AABBMax.z - Ray.Origin.z) * Ray.InverseDirection.z;
//     tmin = max( tmin, min( tz1, tz2 ) ), tmax = min( tmax, max( tz1, tz2 ) );
//     if(tmax >= tmin && tmin < Isect.Distance && tmax > 0) return tmin;
//     else return MAX_LENGTH;    
// }

// FN_DECL void RayTriangleInteresection(ray Ray, INOUT(triangle) Triangle, INOUT(sceneIntersection) Isect, uint InstanceIndex, uint PrimitiveIndex, uint MaterialIndex)
// {
//     vec3 Edge1 = vec3(Triangle.PositionUvX1) - vec3(Triangle.PositionUvX0);
//     vec3 Edge2 = vec3(Triangle.PositionUvX2) - vec3(Triangle.PositionUvX0);

//     vec3 h = cross(Ray.Direction, Edge2);
//     float a = dot(Edge1, h);
//     if(a > -0.00000001f && a < 0.00000001f) return; //Ray is parallel to the triangle
    
//     float f = 1 / a;
//     vec3 s = Ray.Origin - vec3(Triangle.PositionUvX0);
//     float u = f * dot(s, h);
//     if(u < 0 || u > 1) return;

//     vec3 q = cross(s, Edge1);
//     float v = f * dot(Ray.Direction, q);
//     if(v < 0 || u + v > 1) return;
    
//     float t = f * dot(Edge2, q);
//     if(t > 0.00000001f && t < Isect.Distance) {
//         Isect.InstanceIndex = InstanceIndex;
//         Isect.PrimitiveIndex = PrimitiveIndex;
//         Isect.U = u;
//         Isect.V = v;
//         Isect.Distance = t;
//         Isect.MaterialIndex = MaterialIndex;
//     }
// }

// FN_DECL void IntersectBVH(ray Ray, INOUT(sceneIntersection) Isect, uint InstanceIndex, uint MeshIndex)
// {
//     uint NodeInx = 0;
//     uint Stack[64];
//     uint StackPointer=0;
//     bool t=true;
    
//     indexData IndexData = IndexDataBuffer[MeshIndex];
//     uint NodeStartInx = IndexData.BVHNodeDataStartInx;
//     uint TriangleStartInx = IndexData.triangleDataStartInx;
//     uint IndexStartInx = IndexData.IndicesDataStartInx;
//     uint MaterialIndex = TLASInstancesBuffer[InstanceIndex].Material;

//     //We start with the root node of the shape 
//     while(t)
//     {

//         // The current node contains triangles, it's a leaf. 
//         if(BVHBuffer[NodeStartInx + NodeInx].TriangleCount>0)
//         {
//             // For each triangle in the leaf, intersect them
//             for(uint i=0; i<BVHBuffer[NodeStartInx + NodeInx].TriangleCount; i++)
//             {
//                 uint Index = TriangleStartInx + IndicesBuffer[IndexStartInx + int(BVHBuffer[NodeStartInx + NodeInx].LeftChildOrFirst) + i] ;
//                 RayTriangleInteresection(Ray, 
//                                          TriangleBuffer[Index], 
//                                          Isect, 
//                                          InstanceIndex, 
//                                          Index,
//                                          MaterialIndex);
//             }
//             // Go back up the stack and continue to process the next node on the stack
//             if(StackPointer==0) break;
//             else NodeInx = Stack[--StackPointer];
//             continue;
//         }

//         // Get the 2 children of the current node
//         uint Child1 = uint(BVHBuffer[NodeStartInx + NodeInx].LeftChildOrFirst);
//         uint Child2 = uint(BVHBuffer[NodeStartInx + NodeInx].LeftChildOrFirst)+1;

//         // Intersect with the 2 aabb boxes, and get the closest hit
//         float Dist1 = RayAABBIntersection(Ray, BVHBuffer[Child1 + NodeStartInx].AABBMin, BVHBuffer[Child1 + NodeStartInx].AABBMax, Isect);
//         float Dist2 = RayAABBIntersection(Ray, BVHBuffer[Child2 + NodeStartInx].AABBMin, BVHBuffer[Child2 + NodeStartInx].AABBMax, Isect);
//         if(Dist1 > Dist2) {
//             float tmpDist = Dist2;
//             Dist2 = Dist1;
//             Dist1 = tmpDist;

//             uint tmpChild = Child2;
//             Child2 = Child1;
//             Child1 = tmpChild;
//         }

//         if(Dist1 == MAX_LENGTH)
//         {
//             // If we didn't hit any of the 2 child, we can go up the stack
//             if(StackPointer==0) break;
//             else NodeInx = Stack[--StackPointer];
//         }
//         else
//         {
//             // If we did hit, add this child to the stack.
//             NodeInx = Child1;
//             if(Dist2 != MAX_LENGTH)
//             {
//                 Stack[StackPointer++] = Child2;
//             }   
//         }
//     }
// }

// FN_DECL void IntersectInstance(ray Ray, INOUT(sceneIntersection) Isect, uint InstanceIndex)
// {
//     mat4 InverseTransform = TLASInstancesBuffer[InstanceIndex].InverseTransform;
//     Ray.Origin = vec3((InverseTransform * vec4(Ray.Origin, 1)));
//     Ray.Direction = vec3((InverseTransform * vec4(Ray.Direction, 0)));
//     Ray.InverseDirection = 1.0f / Ray.Direction;

//     IntersectBVH(Ray, Isect, TLASInstancesBuffer[InstanceIndex].Index, TLASInstancesBuffer[InstanceIndex].Shape);
// }


// FN_DECL sceneIntersection IntersectTLAS(ray Ray, int Sample, int Bounce)
// {
//     sceneIntersection Isect;
//     Isect.Distance = MAX_LENGTH;
//     float t = Time * float(GLOBAL_ID().x) * 1973.0f;
//     Isect.RandomState = CreateRNG(uint(uint(t) + uint(GLOBAL_ID().y) * uint(9277)  +  uint(Bounce + Sample) * uint(117191)) | uint(1)); 
                
//     Ray.InverseDirection = 1.0f / Ray.Direction;
//     uint NodeInx = 0;
//     uint Stack[64];
//     uint StackPtr=0;
//     while(true)
//     {
//         //If we hit the leaf, check intersection with the bvhs
//         if(TLASNodes[NodeInx].LeftRight==0)
//         {
//             IntersectInstance(Ray, Isect, TLASNodes[NodeInx].BLAS);
            
//             if(StackPtr == 0) break;
//             else NodeInx = Stack[--StackPtr];
//             continue;
//         }

//         //Check if hit any of the children
//         uint Child1 = TLASNodes[NodeInx].LeftRight & 0xffff;
//         uint Child2 = TLASNodes[NodeInx].LeftRight >> 16;
        
//         float Dist1 = RayAABBIntersection(Ray, TLASNodes[Child1].AABBMin, TLASNodes[Child1].AABBMax, Isect);
//         float Dist2 = RayAABBIntersection(Ray, TLASNodes[Child2].AABBMin, TLASNodes[Child2].AABBMax, Isect);
//         if(Dist1 > Dist2) { //Swap if dist 2 is closer
//             float tmpDist = Dist2;
//             Dist2 = Dist1;
//             Dist1 = tmpDist;

//             uint tmpChild = Child2;
//             Child2 = Child1;
//             Child1 = tmpChild;            
//         }
        
//         if(Dist1 == MAX_LENGTH) //We didn't hit a child
//         {
//             if(StackPtr == 0) break; //There's no node left to explore
//             else NodeInx = Stack[--StackPtr]; //Go to the next node in the stack
//         }
//         else //We hit a child
//         {
//             NodeInx = Child1; //Set the current node to the first child
//             if(Dist2 != MAX_LENGTH) Stack[StackPtr++] = Child2; //If we also hit the other node, add it in the stack
//         }

//     }
//     return Isect;
// }

// // Geometry

// FN_DECL ray MakeRay(vec3 Origin, vec3 Direction)
// {
//     ray Ray = {};
//     Ray.Origin = Origin;
//     Ray.Direction = Direction;
//     return Ray;
// }
















// // MIS
// FN_DECL float PowerHeuristic(float PDF0, float PDF1)
// {
//     return (PDF0 * PDF0) / (PDF0 * PDF0 + PDF1 * PDF1);
// }



// FN_DECL vec3 PathTraceMIS(int Sample, vec2 UV, INOUT(vec3) OutNormal)
// {
//     float t = Time * float(GLOBAL_ID().x) * 1973.0f;
//     randomState RandomState = CreateRNG(uint( uint(t) + uint(GLOBAL_ID().y) * uint(9277) + uint(Sample) * uint(26699)) | uint(1) ); 
//     ray Ray = GetRay(UV, vec2(0));    
    

//     vec3 Radiance = vec3(0,0,0);
//     vec3 Weight = vec3(1,1,1);
//     uint OpacityBounces=0;
//     materialPoint VolumeMaterial;
//     bool HasVolumeMaterial=false;

//     bool UseMisIntersection = false;
//     sceneIntersection MisIntersection= {};

//     for(int Bounce=0; Bounce < GET_ATTR(Parameters, Bounces); Bounce++)
//     {
//         sceneIntersection Isect = {};
//         if(Bounce==0) Isect = MakeFirstIsect(Sample);
//         else Isect = UseMisIntersection ?  MisIntersection : IntersectTLAS(Ray, Sample, Bounce);

//         if(Isect.Distance == MAX_LENGTH)
//         {
//             Radiance += Weight * EvalEnvironment(Ray.Direction);
//             if(Bounce==0) OutNormal = vec3(0,0,0);
//             break;
//         }

//         // get all the necessary geometry information
//         triangle Tri = TriangleBuffer[Isect.PrimitiveIndex];    
//         Isect.InstanceTransform = TLASInstancesBuffer[Isect.InstanceIndex].Transform;
//         mat4 NormalTransform = TLASInstancesBuffer[Isect.InstanceIndex].NormalTransform;
//         vec3 HitNormal = vec3(Tri.NormalUvY1) * Isect.U + vec3(Tri.NormalUvY2) * Isect.V +vec3(Tri.NormalUvY0) * (1 - Isect.U - Isect.V);
//         vec4 Tangent = Tri.Tangent1 * Isect.U + Tri.Tangent2 * Isect.V + Tri.Tangent0 * (1 - Isect.U - Isect.V);
//         Isect.Normal = TransformDirection(NormalTransform, HitNormal);
//         Isect.Tangent = TransformDirection(NormalTransform, vec3(Tangent));
//         Isect.Bitangent = TransformDirection(NormalTransform, normalize(cross(Isect.Normal, vec3(Tangent)) * Tangent.w));    

//         bool StayInVolume=false;
//         if(HasVolumeMaterial)
//         {
//             // If we're in a volume, we sample the distance that the ray's gonna intersect.
//             // The transmittance is based on the colour of the object. The higher, the thicker.
//             float Distance = SampleTransmittance(VolumeMaterial.Density, Isect.Distance, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState));
//             // float Distance = 0.1f;
//             Weight *= EvalTransmittance(VolumeMaterial.Density, Distance) / 
//                     SampleTransmittancePDF(VolumeMaterial.Density, Distance, Isect.Distance);
            
            
//             //If the distance is higher than the next intersection, it means that we're stepping out of the volume
//             StayInVolume = Distance < Isect.Distance;
            
//             Isect.Distance = Distance;
//         }

//         if(!StayInVolume)
//         {

//             vec3 OutgoingDir = -Ray.Direction;
//             vec3 Normal = EvalShadingNormal(OutgoingDir, Isect);
//             vec3 Position = EvalShadingPosition(OutgoingDir, Isect);

//             if(Bounce==0)
//             {
//                 OutNormal = Normal;
//             }
            
//             // Material evaluation
//             materialPoint Material = EvalMaterial(Isect);
            
//             // Opacity
//             if(Material.Opacity < 1 && RandomUnilateral(Isect.RandomState) >= Material.Opacity)
//             {
//                 if(OpacityBounces++ > 128) break;
//                 Ray.Origin = Position + Ray.Direction * 1e-2f;
//                 Bounce--;
//                 continue;
//             }

            

//             if(!UseMisIntersection)
//             {
//                 Radiance += Weight * EvalEmission(Material, Normal, OutgoingDir);
//             }

//             vec3 Incoming = vec3(0);
//             if(!IsDelta(Material))
//             {
//                 {
//                     Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
//                     vec3 ShiftedPosition = Position + (dot(Normal, Incoming) > 0 ? Normal : -Normal) * 0.001f;
//                     if (Incoming == vec3(0, 0, 0)) break;
//                     vec3 BSDFCos   = EvalBSDFCos(Material, Normal, OutgoingDir, Incoming);
//                     float LightPDF = SampleLightsPDF(ShiftedPosition, Incoming); 
//                     float BSDFPDF = SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming);
//                     float MisWeight = PowerHeuristic(LightPDF, BSDFPDF) / LightPDF;
//                     if (BSDFCos != vec3(0, 0, 0) && MisWeight != 0) 
//                     {
//                         sceneIntersection Isect = IntersectTLAS(MakeRay(ShiftedPosition, Incoming), Sample, 0); 
//                         vec3 Emission = vec3(0, 0, 0);
//                         if (Isect.Distance == MAX_LENGTH) {
//                             Emission = EvalEnvironment(Incoming);
//                         } else {
//                             materialPoint Material = EvalMaterial(Isect);
//                             vec3 Outgoing = -Incoming;
//                             vec3 ShadingNormal = EvalShadingNormal(Outgoing, Isect);
//                             Emission      = EvalEmission(Material, ShadingNormal, Outgoing);
//                         }
//                         Radiance += Weight * BSDFCos * Emission * MisWeight;
//                     }
//                 }
//                 {
//                     Incoming = SampleBSDFCos(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
//                     vec3 ShiftedPosition = Position + (dot(Normal, Incoming) > 0 ? Normal : -Normal) * 0.001f;
//                     if (Incoming == vec3(0, 0, 0)) break;
//                     vec3 BSDFCos   = EvalBSDFCos(Material, Normal, OutgoingDir, Incoming);
//                     float LightPDF = SampleLightsPDF(ShiftedPosition, Incoming);
//                     float BSDFPDF = SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming);
//                     float MisWeight = PowerHeuristic(BSDFPDF, LightPDF) / BSDFPDF;
//                     if (BSDFCos != vec3(0, 0, 0) && MisWeight != 0) {
//                         MisIntersection = IntersectTLAS(MakeRay(ShiftedPosition, Incoming), Sample, 0); 
//                         vec3 Emission = vec3(0, 0, 0);
//                         if (MisIntersection.Distance == MAX_LENGTH) { 
//                             Emission = EvalEnvironment(Incoming);
//                         } else {
//                             materialPoint Material = EvalMaterial(MisIntersection);
//                             Emission      = Material.Emission;
//                         }
//                         Radiance += Weight * BSDFCos * Emission * MisWeight; 
//                         // // indirect
//                         Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) /
//                                 vec3(SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming));
//                         UseMisIntersection = true;
//                     }
//                 }
//             }
//             else
//             {
//                 Incoming = SampleDelta(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState));
//                 Weight *= EvalDelta(Material, Normal, OutgoingDir, Incoming) / 
//                         SampleDeltaPDF(Material, Normal, OutgoingDir, Incoming);       
//                 UseMisIntersection=false;
//             }

            
//             //If the hit material is volumetric
//             // And the ray keeps going in the same direction (It always does for volumetric materials)
//             // we add the volume material into the stack 
//             if(IsVolumetric(Material)   && dot(Normal, OutgoingDir) * dot(Normal, Incoming) < 0)
//             {
//                 VolumeMaterial = Material;
//                 HasVolumeMaterial = !HasVolumeMaterial;
//             }

//             Ray.Origin = Position + (dot(Normal, Incoming) > 0 ? Normal : -Normal) * 0.001f;
//             Ray.Direction = Incoming;
//         }
//         else
//         {
//             vec3 Outgoing = -Ray.Direction;
//             vec3 Position = Ray.Origin + Ray.Direction * Isect.Distance;
            

//             vec3 Incoming = vec3(0);
//             if(RandomUnilateral(Isect.RandomState)>0.5f)
//             {
//                 // Sample a scattering direction inside the volume using the phase function
//                 Incoming = SamplePhase(VolumeMaterial, Outgoing, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
//                 UseMisIntersection=false;
//             }
//             else
//             {
//                 Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));                
//                 UseMisIntersection=false;
//             }

//             if(Incoming == vec3(0)) break;
        
//             Weight *= EvalPhase(VolumeMaterial, Outgoing, Incoming) / 
//                     ( 
//                     0.5f * SamplePhasePDF(VolumeMaterial, Outgoing, Incoming) + 
//                     0.5f * SampleLightsPDF(Position, Incoming)
//                     );
                    
//             Ray.Origin = Position;
//             Ray.Direction = Incoming;
//         }

//         if(Weight == vec3(0,0,0) || !commonCu::IsFinite(Weight)) break;

//         if(Bounce > 3)
//         {
//             float RussianRouletteProb = min(0.99f, max3(Weight));
//             if(RandomUnilateral(Isect.RandomState) >= RussianRouletteProb) break;
//             Weight *= 1.0f / RussianRouletteProb;
//         }                
//     }

//     if(!commonCu::IsFinite(Radiance)) Radiance = vec3(0,0,0);
//     if(max3(Radiance) > GET_ATTR(Parameters, Clamp)) Radiance = Radiance * (GET_ATTR(Parameters, Clamp) / max3(Radiance)); 
//     return Radiance;
// }

// FN_DECL vec3 PathTrace(int Sample, vec2 UV, INOUT(vec3) OutNormal)
// {
//     float t = Time * float(GLOBAL_ID().x) * 1973.0f;
//     randomState RandomState = CreateRNG(uint( uint(t) + uint(GLOBAL_ID().y) * uint(9277) + uint(Sample) * uint(26699)) | uint(1) ); 
//     ray Ray = GetRay(UV, vec2(0));
    



//     vec3 Radiance = vec3(0,0,0);
//     vec3 Weight = vec3(1,1,1);
//     uint OpacityBounces=0;
//     materialPoint VolumeMaterial;
//     bool HasVolumeMaterial=false;
    


//     for(int Bounce=0; Bounce < GET_ATTR(Parameters, Bounces); Bounce++)
//     {
//         sceneIntersection Isect = {};
//         if(Bounce==0) Isect = MakeFirstIsect(Sample);
//         else Isect = IntersectTLAS(Ray, Sample, Bounce);

//         if(Isect.Distance == MAX_LENGTH)
//         {
//             Radiance += Weight * EvalEnvironment(Ray.Direction);
//             if(Bounce==0) OutNormal = vec3(0,0,0);            
//             break;
//         }

//         // get all the necessary geometry information
//         triangle Tri = TriangleBuffer[Isect.PrimitiveIndex];    
//         Isect.InstanceTransform = TLASInstancesBuffer[Isect.InstanceIndex].Transform;
//         mat4 NormalTransform = TLASInstancesBuffer[Isect.InstanceIndex].NormalTransform;
//         vec3 HitNormal = vec3(Tri.NormalUvY1) * Isect.U + vec3(Tri.NormalUvY2) * Isect.V +vec3(Tri.NormalUvY0) * (1 - Isect.U - Isect.V);
//         vec4 Tangent = Tri.Tangent1 * Isect.U + Tri.Tangent2 * Isect.V + Tri.Tangent0 * (1 - Isect.U - Isect.V);
//         Isect.Normal = TransformDirection(NormalTransform, HitNormal);
//         Isect.Tangent = TransformDirection(NormalTransform, vec3(Tangent));
//         Isect.Bitangent = TransformDirection(NormalTransform, normalize(cross(Isect.Normal, vec3(Tangent)) * Tangent.w));    


//         bool StayInVolume=false;
//         if(HasVolumeMaterial)
//         {
//             // If we're in a volume, we sample the distance that the ray's gonna intersect.
//             // The transmittance is based on the colour of the object. The higher, the thicker.
//             float Distance = SampleTransmittance(VolumeMaterial.Density, Isect.Distance, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState));
//             // float Distance = 0.1f;
//             Weight *= EvalTransmittance(VolumeMaterial.Density, Distance) / 
//                     SampleTransmittancePDF(VolumeMaterial.Density, Distance, Isect.Distance);
            
            
//             //If the distance is higher than the next intersection, it means that we're stepping out of the volume
//             StayInVolume = Distance < Isect.Distance;
            
//             Isect.Distance = Distance;
//         }

//         if(!StayInVolume)
//         {

//             vec3 OutgoingDir = -Ray.Direction;
//             vec3 Position = EvalShadingPosition(OutgoingDir, Isect);
//             vec3 Normal = EvalShadingNormal(OutgoingDir, Isect);
            


//             if(Bounce==0)
//             {
//                 OutNormal = Normal;
//             }

//             // Material evaluation
//             materialPoint Material = EvalMaterial(Isect);
            
//             // Opacity
//             if(Material.Opacity < 1 && RandomUnilateral(Isect.RandomState) >= Material.Opacity)
//             {
//                 if(OpacityBounces++ > 128) break;
//                 Ray.Origin = Position + Ray.Direction * 1e-2f;
//                 Bounce--;
//                 continue;
//             }

            

//             Radiance += Weight * EvalEmission(Material, Normal, OutgoingDir);

//             vec3 Incoming = vec3(0);
//             if(!IsDelta(Material))
//             {
//                 if(GET_ATTR(Parameters, SamplingMode) == SAMPLING_MODE_LIGHT)
//                 {
//                     Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
//                     if(Incoming == vec3(0,0,0)) break;
//                     float PDF = SampleLightsPDF(Position, Incoming);
//                     if(PDF > 0)
//                     {
//                         Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) / vec3(PDF);   
//                     } 
//                     else  
//                     {
//                         break;
//                     }
//                 }
//                 else if(GET_ATTR(Parameters, SamplingMode) == SAMPLING_MODE_BSDF)
//                 {
//                     Incoming = SampleBSDFCos(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
//                     if(Incoming == vec3(0,0,0)) break;
//                     Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) / 
//                             vec3(SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming));
//                 }
//                 else
//                 {
//                     if(RandomUnilateral(Isect.RandomState) > 0.5f)
//                     {
//                         Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
//                         if(Incoming == vec3(0,0,0)) break;
//                         float PDF = SampleLightsPDF(Position, Incoming);
//                         if(PDF > 0)
//                         {
//                             Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) / vec3(PDF);   
//                         }
//                         else
//                         {
//                             break;
//                         }
//                     }
//                     else
//                     {
//                         Incoming = SampleBSDFCos(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
//                         if(Incoming == vec3(0,0,0)) break;
//                         Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) / 
//                                 vec3(SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming));
//                     }
//                 }

//             }
//             else
//             {
//                 Incoming = SampleDelta(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState));
//                 Weight *= EvalDelta(Material, Normal, OutgoingDir, Incoming) / 
//                         SampleDeltaPDF(Material, Normal, OutgoingDir, Incoming);                        
//             }

            
//             //If the hit material is volumetric
//             // And the ray keeps going in the same direction (It always does for volumetric materials)
//             // we add the volume material into the stack 
//             if(IsVolumetric(Material)   && dot(Normal, OutgoingDir) * dot(Normal, Incoming) < 0)
//             {
//                 VolumeMaterial = Material;
//                 HasVolumeMaterial = !HasVolumeMaterial;
//             }

//             Ray.Origin = Position + (dot(Normal, Incoming) > 0 ? Normal : -Normal) * 0.001f;
//             Ray.Direction = Incoming;
//         }
//         else
//         {
//             vec3 Outgoing = -Ray.Direction;
//             vec3 Position = Ray.Origin + Ray.Direction * Isect.Distance;

//             vec3 Incoming = vec3(0);
//             if(RandomUnilateral(Isect.RandomState) > 0.5f)
//             {
//                 // Sample a scattering direction inside the volume using the phase function
//                 Incoming = SamplePhase(VolumeMaterial, Outgoing, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
//             }
//             else
//             {
//                 Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));                
//             }

//             if(Incoming == vec3(0)) break;
        
//             Weight *= EvalPhase(VolumeMaterial, Outgoing, Incoming) / 
//                     ( 
//                     0.5f * SamplePhasePDF(VolumeMaterial, Outgoing, Incoming) + 
//                     0.5f * SampleLightsPDF(Position, Incoming)
//                     );
                    
//             Ray.Origin = Position;
//             Ray.Direction = Incoming;
//         }

//         if(Weight == vec3(0,0,0) || !commonCu::IsFinite(Weight)) break;

//         if(Bounce > 3)
//         {
//             float RussianRouletteProb = min(0.99f, max3(Weight));
//             if(RandomUnilateral(Isect.RandomState) >= RussianRouletteProb) break;
//             Weight *= 1.0f / RussianRouletteProb;
//         }                
//     }

//     if(!commonCu::IsFinite(Radiance)) Radiance = vec3(0,0,0);
//     if(max3(Radiance) > GET_ATTR(Parameters, Clamp)) Radiance = Radiance * (GET_ATTR(Parameters, Clamp) / max3(Radiance)); 


//     return Radiance;
// }

// __global__ void TraceKernel(half4 *RenderImage, cudaFramebuffer _CurrentFramebuffer, int _Width, int _Height,
//                             triangle *_AllTriangles, bvhNode *_AllBVHNodes, u32 *_AllTriangleIndices, indexData *_IndexData, instance *_Instances, tlasNode *_TLASNodes,
//                             camera *_Cameras, tracingParameters* _TracingParams, material *_Materials, cudaTextureObject_t _SceneTextures, int _TexturesWidth, int _TexturesHeight, light *_Lights, float *_LightsCDF, int _LightsCount,
//                             environment *_Environments, int _EnvironmentsCount, cudaTextureObject_t _EnvTextures, int _EnvTexturesWidth, int _EnvTexturesHeight, float _Time)
// {
//     Width = _Width;
//     Height = _Height;
//     TriangleBuffer = _AllTriangles;
//     BVHBuffer = _AllBVHNodes;
//     IndicesBuffer = _AllTriangleIndices;
//     IndexDataBuffer = _IndexData;
//     TLASInstancesBuffer = _Instances;
//     TLASNodes = _TLASNodes;
//     Cameras = _Cameras;
//     Parameters = _TracingParams;
//     Materials = _Materials;
//     SceneTextures = _SceneTextures;
//     EnvTextures = _EnvTextures;
//     LightsCount = _LightsCount;
//     Lights = _Lights;
//     LightsCDF = _LightsCDF;
//     EnvironmentsCount = _EnvironmentsCount;
//     Environments = _Environments;
//     TexturesWidth = _TexturesWidth;
//     TexturesHeight = _TexturesHeight;
//     EnvTexturesWidth = _EnvTexturesWidth;
//     EnvTexturesHeight = _EnvTexturesHeight;
//     Time = _Time;
//     CurrentFramebuffer = _CurrentFramebuffer;
    
 
//     float t = Time * float(GLOBAL_ID().x) * float(GLOBAL_ID().y) * 1973.0f;
//     randomState RandomState = CreateRNG(uint(uint(t) + uint(GLOBAL_ID().y) * uint(9277)  + uint(GLOBAL_ID().x) * uint(117191)) | uint(1)); 
//     vec2 Jitter = Random2F(RandomState);
//     Jitter = Jitter * 2.0f - 1.0f; 
    
//     ivec2 ImageSize = IMAGE_SIZE(RenderImage);
//     int Width = ImageSize.x;
//     int Height = ImageSize.y;

//     uvec2 GlobalID = GLOBAL_ID();
//     vec2 UV = (vec2(GlobalID) + Jitter) / vec2(ImageSize);
//     UV.y = 1 - UV.y;
    
//     if (GlobalID.x < Width && GlobalID.y < Height) {
//         vec3 Normal;
//         float InverseSampleCount = 1.0f / float(GET_ATTR(Parameters, Batch));
//         vec3 Radiance = vec3(0);
//         for(int Sample=0; Sample < GET_ATTR(Parameters, Batch); Sample++)
//         {
//             if(GET_ATTR(Parameters, SamplingMode) == SAMPLING_MODE_MIS)
//             {
//                 Radiance += PathTraceMIS(Sample, UV, Normal) * InverseSampleCount;
//             }
//             else
//             {
//                 Radiance += PathTrace(0, UV, Normal) * InverseSampleCount;
//             }
//         }

//         half4 Output = commonCu::Vec4ToHalf4(vec4(Radiance, 1.0f));
//         RenderImage[GlobalID.y * Width + GlobalID.x] = Output;
//     }
// }


// }