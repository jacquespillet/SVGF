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

};

// Util
FN_DECL bool IsFinite(float A)
{
    return !isnan(A);
}

FN_DECL bool IsFinite(vec3 A)
{
    return IsFinite(A.x) && IsFinite(A.y) && IsFinite(A.z);
}

FN_DECL float Sum(INOUT(vec3) A) { 
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


// BVH

FN_DECL float RayAABBIntersection(ray Ray, vec3 AABBMin, vec3 AABBMax, INOUT(sceneIntersection) Isect)
{
    float tx1 = (AABBMin.x - Ray.Origin.x) * Ray.InverseDirection.x, tx2 = (AABBMax.x - Ray.Origin.x) * Ray.InverseDirection.x;
    float tmin = min( tx1, tx2 ), tmax = max( tx1, tx2 );
    float ty1 = (AABBMin.y - Ray.Origin.y) * Ray.InverseDirection.y, ty2 = (AABBMax.y - Ray.Origin.y) * Ray.InverseDirection.y;
    tmin = max( tmin, min( ty1, ty2 ) ), tmax = min( tmax, max( ty1, ty2 ) );
    float tz1 = (AABBMin.z - Ray.Origin.z) * Ray.InverseDirection.z, tz2 = (AABBMax.z - Ray.Origin.z) * Ray.InverseDirection.z;
    tmin = max( tmin, min( tz1, tz2 ) ), tmax = min( tmax, max( tz1, tz2 ) );
    if(tmax >= tmin && tmin < Isect.Distance && tmax > 0) return tmin;
    else return 1e30f;    
}

FN_DECL void RayTriangleInteresection(ray Ray, triangle Triangle, INOUT(sceneIntersection) Isect, uint InstanceIndex, uint PrimitiveIndex, uint MaterialIndex)
{
    vec3 Edge1 = Triangle.v1 - Triangle.v0;
    vec3 Edge2 = Triangle.v2 - Triangle.v0;

    vec3 h = cross(Ray.Direction, Edge2);
    float a = dot(Edge1, h);
    if(a > -0.000001f && a < 0.000001f) return; //Ray is parallel to the triangle
    
    float f = 1 / a;
    vec3 s = Ray.Origin - Triangle.v0;
    float u = f * dot(s, h);
    if(u < 0 || u > 1) return;

    vec3 q = cross(s, Edge1);
    float v = f * dot(Ray.Direction, q);
    if(v < 0 || u + v > 1) return;
    
    float t = f * dot(Edge2, q);
    if(t > 0.000001f && t < Isect.Distance) {
        Isect.InstanceIndex = InstanceIndex;
        Isect.PrimitiveIndex = PrimitiveIndex;
        Isect.U = u;
        Isect.V = v;
        Isect.Distance = t;
        Isect.MaterialIndex = MaterialIndex;
    }
}

FN_DECL void IntersectBVH(ray Ray, INOUT(sceneIntersection) Isect, uint InstanceIndex, uint MeshIndex)
{
    uint NodeInx = 0;
    uint Stack[64];
    uint StackPointer=0;
    bool t=true;
    
    indexData IndexData = IndexDataBuffer[MeshIndex];
    uint NodeStartInx = IndexData.BVHNodeDataStartInx;
    uint TriangleStartInx = IndexData.triangleDataStartInx;
    uint IndexStartInx = IndexData.IndicesDataStartInx;
    uint MaterialIndex = TLASInstancesBuffer[InstanceIndex].MaterialIndex;

    //We start with the root node of the shape 
    while(t)
    {

        // The current node contains triangles, it's a leaf. 
        if(BVHBuffer[NodeStartInx + NodeInx].TriangleCount>0)
        {
            // For each triangle in the leaf, intersect them
            for(uint i=0; i<BVHBuffer[NodeStartInx + NodeInx].TriangleCount; i++)
            {
                uint Index = TriangleStartInx + IndicesBuffer[IndexStartInx + BVHBuffer[NodeStartInx + NodeInx].LeftChildOrFirst + i] ;
                RayTriangleInteresection(Ray, 
                                         TriangleBuffer[Index], 
                                         Isect, 
                                         InstanceIndex, 
                                         Index,
                                         MaterialIndex);
            }
            // Go back up the stack and continue to process the next node on the stack
            if(StackPointer==0) break;
            else NodeInx = Stack[--StackPointer];
            continue;
        }

        // Get the 2 children of the current node
        uint Child1 = BVHBuffer[NodeStartInx + NodeInx].LeftChildOrFirst;
        uint Child2 = BVHBuffer[NodeStartInx + NodeInx].LeftChildOrFirst+1;

        // Intersect with the 2 aabb boxes, and get the closest hit
        float Dist1 = RayAABBIntersection(Ray, BVHBuffer[Child1 + NodeStartInx].AABBMin, BVHBuffer[Child1 + NodeStartInx].AABBMax, Isect);
        float Dist2 = RayAABBIntersection(Ray, BVHBuffer[Child2 + NodeStartInx].AABBMin, BVHBuffer[Child2 + NodeStartInx].AABBMax, Isect);
        if(Dist1 > Dist2) {
            float tmpDist = Dist2;
            Dist2 = Dist1;
            Dist1 = tmpDist;

            uint tmpChild = Child2;
            Child2 = Child1;
            Child1 = tmpChild;
        }

        if(Dist1 == 1e30f)
        {
            // If we didn't hit any of the 2 child, we can go up the stack
            if(StackPointer==0) break;
            else NodeInx = Stack[--StackPointer];
        }
        else
        {
            // If we did hit, add this child to the stack.
            NodeInx = Child1;
            if(Dist2 != 1e30f)
            {
                Stack[StackPointer++] = Child2;
            }   
        }
    }
}

FN_DECL void IntersectInstance(ray Ray, INOUT(sceneIntersection) Isect, uint InstanceIndex)
{
    mat4 InverseTransform = TLASInstancesBuffer[InstanceIndex].InverseTransform;
    Ray.Origin = vec3((InverseTransform * vec4(Ray.Origin, 1)));
    Ray.Direction = vec3((InverseTransform * vec4(Ray.Direction, 0)));
    Ray.InverseDirection = 1.0f / Ray.Direction;

    IntersectBVH(Ray, Isect, TLASInstancesBuffer[InstanceIndex].Index, TLASInstancesBuffer[InstanceIndex].MeshIndex);
}

FN_DECL void IntersectTLAS(ray Ray, INOUT(sceneIntersection) Isect)
{
    Ray.InverseDirection = 1.0f / Ray.Direction;
    uint NodeInx = 0;
    uint Stack[64];
    uint StackPtr=0;
    while(true)
    {
        //If we hit the leaf, check intersection with the bvhs
        if(TLASNodes[NodeInx].LeftRight==0)
        {
            IntersectInstance(Ray, Isect, TLASNodes[NodeInx].BLAS);
            
            if(StackPtr == 0) break;
            else NodeInx = Stack[--StackPtr];
            continue;
        }

        //Check if hit any of the children
        uint Child1 = TLASNodes[NodeInx].LeftRight & 0xffff;
        uint Child2 = TLASNodes[NodeInx].LeftRight >> 16;
        
        float Dist1 = RayAABBIntersection(Ray, TLASNodes[Child1].AABBMin, TLASNodes[Child1].AABBMax, Isect);
        float Dist2 = RayAABBIntersection(Ray, TLASNodes[Child2].AABBMin, TLASNodes[Child2].AABBMax, Isect);
        if(Dist1 > Dist2) { //Swap if dist 2 is closer
            float tmpDist = Dist2;
            Dist2 = Dist1;
            Dist1 = tmpDist;

            uint tmpChild = Child2;
            Child2 = Child1;
            Child1 = tmpChild;            
        }
        
        if(Dist1 == 1e30f) //We didn't hit a child
        {
            if(StackPtr == 0) break; //There's no node left to explore
            else NodeInx = Stack[--StackPtr]; //Go to the next node in the stack
        }
        else //We hit a child
        {
            NodeInx = Child1; //Set the current node to the first child
            if(Dist2 != 1e30f) Stack[StackPtr++] = Child2; //If we also hit the other node, add it in the stack
        }

    }
}

// Geometry

FN_DECL ray MakeRay(vec3 Origin, vec3 Direction, vec3 InverseDirection)
{
    ray Ray;
    Ray.Origin = Origin;
    Ray.Direction = Direction;
    Ray.InverseDirection = InverseDirection;
    return Ray;
}

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


FN_DECL ray GetRay( vec2 ImageUV, vec2 LensUV)
{
    camera Camera = Cameras[0];

    vec2 Film = Camera.Aspect >= 1 ? 
               vec2(Camera.Film, Camera.Film / Camera.Aspect): 
               vec2(Camera.Film * Camera.Aspect, Camera.Film);
    
    // Point on the film
    vec3 Q = vec3(
        Film.x * (0.5f - ImageUV.x),
        Film.y * (0.5f - ImageUV.y),
        Camera.Lens
    );
    vec3 RayDirection = -normalize(Q);
    vec3 PointOnFocusPlane = RayDirection * Camera.Focus / abs(RayDirection.z);
    
    // Jitter the point on the lens
    vec3 PointOnLens = vec3 (LensUV.x * Camera.Aperture / 2, LensUV.y * Camera.Aperture / 2, 0);

    
    vec3 FinalDirection =normalize(PointOnFocusPlane - PointOnLens);

    //Transform the ray direction and origin
    ray Ray = MakeRay(
        TransformPoint(Camera.Frame, PointOnLens),
        TransformDirection(Camera.Frame, FinalDirection),
        vec3(0)
    );
    return Ray;
}

FN_DECL vec3 EvalShadingPosition(INOUT(vec3) OutgoingDir, INOUT(sceneIntersection) Isect)
{
    uint Element = Isect.PrimitiveIndex;
    triangle Tri = TriangleBuffer[Isect.PrimitiveIndex];

    vec3 Position = 
        Tri.v1 * Isect.U + 
        Tri.v2 * Isect.V +
        Tri.v0 * (1 - Isect.U - Isect.V);
    return TransformPoint(Isect.InstanceTransform, Position);
}

// Eval

FN_DECL materialPoint EvalMaterial(INOUT(sceneIntersection) Isect)
{
    material Material = Materials[Isect.MaterialIndex];
    materialPoint Point;
    Point.MaterialType = Material.MaterialType;
    Point.Colour = Material.Colour;
    Point.Emission = Material.Emission;
    Point.Roughness = Material.Roughness;
    Point.Roughness = Point.Roughness * Point.Roughness;
    Point.Metallic = Material.Metallic;
    return Point;
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
    return 1.0f / (1.0f + MicrofacetShadowing1(Roughness, Normal, Halfway, Outgoing) + MicrofacetShadowing1(Roughness, Normal, Halfway, Incoming));
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

// BSDF

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
}


// Random
FN_DECL uint AdvanceState(INOUT(randomState) RNG)
{
    uint64_t OldState = RNG.State;
    RNG.State = OldState * 6364136223846793005ul + RNG.Inc;
    uint XorShifted = uint(((OldState >> uint(18)) ^ OldState) >> uint(27));
    uint Rot = uint(OldState >> uint(59));

    return (XorShifted >> Rot) | (XorShifted << ((~Rot + 1u) & 31));
}

FN_DECL randomState CreateRNG(uint64_t Seed, uint64_t Sequence)
{
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



MAIN()
{
    INIT()
    
    ivec2 ImageSize = IMAGE_SIZE(RenderImage);
    int Width = ImageSize.x;
    int Height = ImageSize.y;

    uvec2 GlobalID = GLOBAL_ID();
    vec2 UV = vec2(GlobalID) / vec2(ImageSize);
    UV.y = 1 - UV.y;
        
    if (GlobalID.x < Width && GlobalID.y < Height) {
        vec4 PrevCol = imageLoad(RenderImage, ivec2(GlobalID));
        vec4 NewCol;
        for(int Sample=0; Sample < GET_ATTR(Parameters, Batch); Sample++)
        {
            randomState RandomState = CreateRNG(uint(uint(GLOBAL_ID().x) * uint(1973) + uint(GLOBAL_ID().y) * uint(9277) + uint(GET_ATTR(Parameters,CurrentSample) + Sample) * uint(26699)) | uint(1), 371213); 
            ray Ray = GetRay(UV, Random2F(RandomState));
            

            vec3 Radiance = vec3(0,0,0);
            vec3 Weight = vec3(1,1,1);

            for(int Bounce=0; Bounce < 3; Bounce++)
            {
                sceneIntersection Isect;
                Isect.Distance = 1e30f;
                Isect.RandomState = CreateRNG(uint(uint(GLOBAL_ID().x) * uint(1973) + uint(GLOBAL_ID().y) * uint(9277)  +  uint(Bounce + GET_ATTR(Parameters,CurrentSample) + Sample) * uint(117191)) | uint(1), 371213); 

                IntersectTLAS(Ray, Isect);
                if(Isect.Distance == 1e30f)
                {
                    // Radiance += vec3(1,1,1);
                    break;
                }

                uint Element = Isect.PrimitiveIndex;
                triangle Tri = TriangleBuffer[Element];
                triangleExtraData ExtraData = TriangleExBuffer[Element];    
                Isect.InstanceTransform = TLASInstancesBuffer[Isect.InstanceIndex].Transform;
                
                mat4 NormalTransform = TLASInstancesBuffer[Isect.InstanceIndex].NormalTransform;
                
                vec3 Normal = TransformDirection(NormalTransform, ExtraData.Normal1 * Isect.U + ExtraData.Normal2 * Isect.V +ExtraData.Normal0 * (1 - Isect.U - Isect.V));
                vec3 Position = TransformPoint(Isect.InstanceTransform, Tri.v1 * Isect.U + Tri.v2 * Isect.V + Tri.v0 * (1 - Isect.U - Isect.V));
                // vec3 Normal = ExtraData.Normal1 * Isect.U + ExtraData.Normal2 * Isect.V +ExtraData.Normal0 * (1 - Isect.U - Isect.V);
                // vec3 Position = Tri.v1 * Isect.U + Tri.v2 * Isect.V + Tri.v0 * (1 - Isect.U - Isect.V);
                
                Isect.Normal = Normal;
                vec3 OutgoingDir = -Ray.Direction;
                materialPoint Material = EvalMaterial(Isect);

                Radiance += Weight * Material.Emission;
                
                vec3 Incoming = SampleBSDFCos(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));

                if(Incoming == vec3(0,0,0)) break;

                Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) / 
                          SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming);

                
                Ray.Origin = Position;
                Ray.Direction = Incoming;

                if(Weight == vec3(0,0,0) || !IsFinite(Weight)) break;

                if(Bounce > 3)
                {
                    float RussianRouletteProb = min(0.99f, max3(Weight));
                    if(RandomUnilateral(Isect.RandomState) >= RussianRouletteProb) break;
                    Weight *= 1.0f / RussianRouletteProb;
                }                
            }

            if(!IsFinite(Radiance)) Radiance = vec3(0,0,0);
            if(max3(Radiance) > 10) Radiance = Radiance * (10 / max3(Radiance)); 

            float SampleWeight = 1.0f / (float(GET_ATTR(Parameters,CurrentSample) + Sample) + 1);

            NewCol = mix(PrevCol, vec4(Radiance.x, Radiance.y, Radiance.z, 1.0f), SampleWeight);
            PrevCol = NewCol;
        }
        imageStore(RenderImage, ivec2(GlobalID), NewCol);
    }
}