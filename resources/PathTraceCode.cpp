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

FN_DECL void RayTriangleInteresection(ray Ray, triangle Triangle, INOUT(sceneIntersection) Isect, uint InstanceIndex, uint PrimitiveIndex)
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
                                         Index);
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
    return vec3((Res / Res.w));
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


FN_DECL ray GetRay(vec2 ImageUV)
{
    camera Camera = Cameras[0];

    // Point on the film
    vec3 Q = vec3(
        (0.5f - ImageUV.x),
        (0.5f - ImageUV.y ),
        1
    );
    vec3 RayDirection = -normalize(Q);
    vec3 PointOnLens = vec3 (0,0,0);

    //Transform the ray direction and origin
    ray Ray = MakeRay(
        TransformPoint(Camera.Frame, PointOnLens),
        TransformDirection(Camera.Frame, RayDirection),
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

// BSDF

FN_DECL vec3 SampleHemisphereCosine(vec3 Normal, vec2 UV)
{
    // Calculates Phi and theta angles
    float Phi = 2 * PI_F * UV.x; //Azimuthal angle
    float CosTheta = sqrt(1.0f - UV.y);  // Cosine of polar angle
    float SinTheta = sqrt(UV.y);  // Sine of polar angle

    // By taking the square root of UV.y, we essentially map a uniform distribution to a distribution that gives higher weight #
    // to points closer to the normal vector. This weighting ensures that more samples are taken in directions aligned with the surface normal
    
    // Generates a direction from the angles
    vec3 LocalDirection = vec3(
        cos(Phi) * SinTheta, 
        sin(Phi) * CosTheta,
        SinTheta
    );
    return TransformDirection(BasisFromZ(Normal), LocalDirection);
}

FN_DECL float SampleHemisphereCosinePDF(INOUT(vec3) Normal, INOUT(vec3) Direction)
{
    // The probability of generating a direction v is proportional to cos(θ) (as in the cosine-weighted hemisphere).
    // The total probability over the hemisphere should be 1. So, to normalize, we divide by the integral of cos⁡cos(θ) over the hemisphere, which is π.

    float CosW = dot(Normal, Direction);
    return (CosW <= 0) ? 0 : CosW / PI_F;
}

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
        for(int Sample=0; Sample < Parameters.Batch; Sample++)
        {
            ray Ray = GetRay(UV);
            

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
                    Radiance += Weight * vec3(1);
                    // Environment radiance
                    break;
                }

                uint Element = Isect.PrimitiveIndex;
                triangle Tri = TriangleBuffer[Element];
                triangleExtraData ExtraData = TriangleExBuffer[Element];    
                Isect.InstanceTransform = TLASInstancesBuffer[Isect.InstanceIndex].Transform;
                
                mat4 NormalTransform = TLASInstancesBuffer[Isect.InstanceIndex].NormalTransform;
                vec3 Normal = ExtraData.Normal1 * Isect.U + ExtraData.Normal2 * Isect.V +ExtraData.Normal0 * (1 - Isect.U - Isect.V);
                Isect.Normal = TransformDirection(NormalTransform, Normal);
                
                vec4 Colour = ExtraData.Colour1 * Isect.U + ExtraData.Colour2 * Isect.V + ExtraData.Colour0 * (1 - Isect.U - Isect.V);                

                vec3 Position = Tri.v1 * Isect.U + Tri.v2 * Isect.V + Tri.v0 * (1 - Isect.U - Isect.V);


                Weight *= vec3(Colour);
                
                vec3 OutgoingDir = -Ray.Direction;
                vec3 Incoming = SampleHemisphereCosine(Normal, Random2F(Isect.RandomState));
                
                Ray.Origin = Position;
                Ray.Direction = Incoming;

                if(Weight == vec3(0,0,0) || !IsFinite(Weight)) break;
            }

            float SampleWeight = 1.0f / (float(GET_ATTR(Parameters,CurrentSample) + Sample) + 1);

            NewCol = mix(PrevCol, vec4(Radiance.x, Radiance.y, Radiance.z, 1.0f), SampleWeight);
            PrevCol = NewCol;
        }
        imageStore(RenderImage, ivec2(GlobalID), NewCol);
    }
}