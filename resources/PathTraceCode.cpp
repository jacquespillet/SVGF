struct randomState
{
    uint State;
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
FN_DECL bool IsFinite(float A)
{
    return !isnan(A);
}

FN_DECL bool IsFinite(vec3 A)
{
    return IsFinite(A.x) && IsFinite(A.y) && IsFinite(A.z);
}

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



// Random
FN_DECL uint AdvanceState(INOUT(randomState) RNG)
{
    RNG.State ^= RNG.State << 13u;
    RNG.State ^= RNG.State >> 17u;
    RNG.State ^= RNG.State << 5u;
    return RNG.State;    
}

FN_DECL randomState CreateRNG(uint Seed)
{
    randomState State;
    State.State = Seed;
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
    else return MAX_LENGTH;    
}

FN_DECL void RayTriangleInteresection(ray Ray, INOUT(triangle) Triangle, INOUT(sceneIntersection) Isect, uint InstanceIndex, uint PrimitiveIndex, uint MaterialIndex)
{
    vec3 Edge1 = vec3(Triangle.PositionUvX1) - vec3(Triangle.PositionUvX0);
    vec3 Edge2 = vec3(Triangle.PositionUvX2) - vec3(Triangle.PositionUvX0);

    vec3 h = cross(Ray.Direction, Edge2);
    float a = dot(Edge1, h);
    if(a > -0.000001f && a < 0.000001f) return; //Ray is parallel to the triangle
    
    float f = 1 / a;
    vec3 s = Ray.Origin - vec3(Triangle.PositionUvX0);
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
    uint MaterialIndex = TLASInstancesBuffer[InstanceIndex].Material;

    //We start with the root node of the shape 
    while(t)
    {

        // The current node contains triangles, it's a leaf. 
        if(BVHBuffer[NodeStartInx + NodeInx].TriangleCount>0)
        {
            // For each triangle in the leaf, intersect them
            for(uint i=0; i<BVHBuffer[NodeStartInx + NodeInx].TriangleCount; i++)
            {
                uint Index = TriangleStartInx + IndicesBuffer[IndexStartInx + int(BVHBuffer[NodeStartInx + NodeInx].LeftChildOrFirst) + i] ;
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
        uint Child1 = uint(BVHBuffer[NodeStartInx + NodeInx].LeftChildOrFirst);
        uint Child2 = uint(BVHBuffer[NodeStartInx + NodeInx].LeftChildOrFirst)+1;

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

        if(Dist1 == MAX_LENGTH)
        {
            // If we didn't hit any of the 2 child, we can go up the stack
            if(StackPointer==0) break;
            else NodeInx = Stack[--StackPointer];
        }
        else
        {
            // If we did hit, add this child to the stack.
            NodeInx = Child1;
            if(Dist2 != MAX_LENGTH)
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

    IntersectBVH(Ray, Isect, TLASInstancesBuffer[InstanceIndex].Index, TLASInstancesBuffer[InstanceIndex].Shape);
}


FN_DECL sceneIntersection IntersectTLAS(ray Ray, int Sample, int Bounce)
{
    sceneIntersection Isect;
    Isect.Distance = MAX_LENGTH;
    Isect.RandomState = CreateRNG(uint(uint(GLOBAL_ID().x) * uint(1973) + uint(GLOBAL_ID().y) * uint(9277)  +  uint(Bounce + GET_ATTR(Parameters,CurrentSample) + Sample) * uint(117191)) | uint(1)); 
                
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
        
        if(Dist1 == MAX_LENGTH) //We didn't hit a child
        {
            if(StackPtr == 0) break; //There's no node left to explore
            else NodeInx = Stack[--StackPtr]; //Go to the next node in the stack
        }
        else //We hit a child
        {
            NodeInx = Child1; //Set the current node to the first child
            if(Dist2 != MAX_LENGTH) Stack[StackPtr++] = Child2; //If we also hit the other node, add it in the stack
        }

    }
    return Isect;
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

FN_DECL sceneIntersection MakeIsect(int Sample)
{
    sceneIntersection Isect;
    Isect.Distance = MAX_LENGTH;
    Isect.RandomState = CreateRNG(uint(uint(GLOBAL_ID().x) * uint(1973) + uint(GLOBAL_ID().y) * uint(9277) + uint(GET_ATTR(Parameters,CurrentSample) + Sample) * uint(26699)) | uint(1) ); 
    return Isect;
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
    camera Camera = Cameras[int(GET_ATTR(Parameters, CurrentCamera))];

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
        vec3(Tri.PositionUvX1) * Isect.U + 
        vec3(Tri.PositionUvX2) * Isect.V +
        vec3(Tri.PositionUvX0) * (1 - Isect.U - Isect.V);
    return TransformPoint(Isect.InstanceTransform, Position);
}

// Eval
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
    vec4 Colour = textureSample(SceneTextures, texCoord3D); 
    // vec4 Colour = vec4(1);
    if(Linear) Colour = ToLinear(Colour);
    return Colour;
}

FN_DECL vec4 EvalEnvTexture(int Texture, vec2 UV, bool Linear)
{
    if(Texture == INVALID_ID) return vec4(1, 1, 1, 1);
    vec3 texCoord3D = vec3(UV, Texture);
    vec4 Colour = textureSampleEnv(EnvTextures, texCoord3D); 
    // vec4 Colour = vec4(1);
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

FN_DECL vec3 EvalShadingNormal(INOUT(vec3) OutgoingDir, INOUT(sceneIntersection) Isect)
{
    vec3 Normal = EvalNormalMap(Isect.Normal, Isect);
    if (Materials[Isect.MaterialIndex].MaterialType == MATERIAL_TYPE_GLASS) return Normal;
    return dot(Normal, OutgoingDir) >= 0 ? Normal : -Normal;
}

FN_DECL vec3 EvalEmission(INOUT(materialPoint) Material, INOUT(vec3) Normal, INOUT(vec3) Outgoing) {
  return dot(Normal, Outgoing) >= 0 ? Material.Emission : vec3(0, 0, 0);
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

    return Point;
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


// Region lights
// Multiple problems : 
// The LightSample function doesn't work when the lights are scaled at runtime.
// Then, get MIS to work

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
    return Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC;
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
    if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        return EvalVolumetric(Material.Colour, Normal, OutgoingDir, Incoming);
    }    
}

FN_DECL float SampleDeltaPDF(INOUT(materialPoint) Material, INOUT(vec3) Normal, INOUT(vec3) OutgoingDir, INOUT(vec3) Incoming)
{
    if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        return SampleVolumetricPDF(Material.Colour, Normal, OutgoingDir, Incoming);
    }    
}

FN_DECL vec3 SampleDelta(INOUT(materialPoint) Material, INOUT(vec3) Normal, INOUT(vec3) OutgoingDir, float RNL)
{
    if(Material.MaterialType == MATERIAL_TYPE_VOLUMETRIC)
    {
        return SampleVolumetric(OutgoingDir);
    }    
}

// MIS
FN_DECL float PowerHeuristic(float PDF0, float PDF1)
{
    return (PDF0 * PDF0) / (PDF0 * PDF0 + PDF1 * PDF1);
}



FN_DECL vec3 PathTraceMIS(int Sample, vec2 UV)
{
    randomState RandomState = CreateRNG(uint(uint(GLOBAL_ID().x) * uint(1973) + uint(GLOBAL_ID().y) * uint(9277) + uint(GET_ATTR(Parameters,CurrentSample) + Sample) * uint(26699)) | uint(1) ); 
    ray Ray = GetRay(UV, Random2F(RandomState));
    

    vec3 Radiance = vec3(0,0,0);
    vec3 Weight = vec3(1,1,1);
    uint OpacityBounces=0;
    materialPoint VolumeMaterial;
    bool HasVolumeMaterial=false;

    bool UseMisIntersection = false;
    sceneIntersection MisIntersection = MakeIsect(Sample);

    for(int Bounce=0; Bounce < GET_ATTR(Parameters, Bounces); Bounce++)
    {
        sceneIntersection Isect = UseMisIntersection ?  MisIntersection : IntersectTLAS(Ray, Sample, Bounce);
        if(Isect.Distance == MAX_LENGTH)
        {
            Radiance += Weight * EvalEnvironment(Ray.Direction);
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

            

            if(!UseMisIntersection)
            {
                Radiance += Weight * EvalEmission(Material, Normal, OutgoingDir);
            }

            vec3 Incoming = vec3(0);
            if(!IsDelta(Material))
            {
                {
                    Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
                    if (Incoming == vec3{0, 0, 0}) break;
                    vec3 BSDFCos   = EvalBSDFCos(Material, Normal, OutgoingDir, Incoming);
                    float LightPDF = SampleLightsPDF(Position, Incoming); 
                    float BSDFPDF = SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming);
                    float MisWeight = PowerHeuristic(LightPDF, BSDFPDF) / LightPDF;
                    if (BSDFCos != vec3(0, 0, 0) && MisWeight != 0) 
                    {
                        sceneIntersection Isect = IntersectTLAS(MakeRay(Position, Incoming, 1.0f / Incoming), Sample, 0); 
                        vec3 Emission = vec3(0, 0, 0);
                        if (Isect.Distance == MAX_LENGTH) {
                            Emission = EvalEnvironment(Incoming);
                        } else {
                            materialPoint Material = EvalMaterial(Isect);
                            Emission      = EvalEmission(Material, EvalShadingNormal(-Incoming, Isect), -Incoming);
                        }
                        Radiance += Weight * BSDFCos * Emission * MisWeight;
                    }
                }
                {
                    Incoming = SampleBSDFCos(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
                    if (Incoming == vec3{0, 0, 0}) break;
                    vec3 BSDFCos   = EvalBSDFCos(Material, Normal, OutgoingDir, Incoming);
                    float LightPDF = SampleLightsPDF(Position, Incoming);
                    float BSDFPDF = SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming);
                    float MisWeight = PowerHeuristic(BSDFPDF, LightPDF) / BSDFPDF;
                    if (BSDFCos != vec3(0, 0, 0) && MisWeight != 0) {
                        MisIntersection = IntersectTLAS(MakeRay(Position, Incoming, 1.0f / Incoming), Sample, 0); 
                        vec3 Emission = vec3(0, 0, 0);
                        if (Isect.Distance == MAX_LENGTH) {
                            Emission = EvalEnvironment(Incoming);
                        } else {
                            materialPoint Material = EvalMaterial(Isect);
                            Emission      = Material.Emission;
                        }
                        Radiance += Weight * BSDFCos * Emission * MisWeight; 
                    }
                }
                // indirect
                Weight *= EvalBSDFCos(Material, Normal, OutgoingDir, Incoming) /
                        vec3(SampleBSDFCosPDF(Material, Normal, OutgoingDir, Incoming));
                UseMisIntersection = true;
            }
            else
            {
                Incoming = SampleDelta(Material, Normal, OutgoingDir, RandomUnilateral(Isect.RandomState));
                Weight *= EvalDelta(Material, Normal, OutgoingDir, Incoming) / 
                        SampleDeltaPDF(Material, Normal, OutgoingDir, Incoming);       
                UseMisIntersection=false;
            }

            
            //If the hit material is volumetric
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
            if(GET_ATTR(Parameters, CurrentSample) % 2 ==0)
            {
                // Sample a scattering direction inside the volume using the phase function
                Incoming = SamplePhase(VolumeMaterial, Outgoing, RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));
                UseMisIntersection=false;
            }
            else
            {
                Incoming = SampleLights(Position, RandomUnilateral(Isect.RandomState), RandomUnilateral(Isect.RandomState), Random2F(Isect.RandomState));                
                UseMisIntersection=false;
            }

            if(Incoming == vec3(0)) break;
        
            Weight *= EvalPhase(VolumeMaterial, Outgoing, Incoming) / 
                    ( 
                    0.5f * SamplePhasePDF(VolumeMaterial, Outgoing, Incoming) + 
                    0.5f * SampleLightsPDF(Position, Incoming)
                    );
                    
            Ray.Origin = Position;
            Ray.Direction = Incoming;
        }

        if(Weight == vec3(0,0,0) || !IsFinite(Weight)) break;

        if(Bounce > 3)
        {
            float RussianRouletteProb = min(0.99f, max3(Weight));
            if(RandomUnilateral(Isect.RandomState) >= RussianRouletteProb) break;
            Weight *= 1.0f / RussianRouletteProb;
        }                
    }

    if(!IsFinite(Radiance)) Radiance = vec3(0,0,0);
    if(max3(Radiance) > GET_ATTR(Parameters, Clamp)) Radiance = Radiance * (GET_ATTR(Parameters, Clamp) / max3(Radiance)); 
    return Radiance;
}

FN_DECL vec3 PathTrace(int Sample, vec2 UV)
{
    randomState RandomState = CreateRNG(uint(uint(GLOBAL_ID().x) * uint(1973) + uint(GLOBAL_ID().y) * uint(9277) + uint(GET_ATTR(Parameters,CurrentSample) + Sample) * uint(26699)) | uint(1) ); 
    ray Ray = GetRay(UV, Random2F(RandomState));
    

    vec3 Radiance = vec3(0,0,0);
    vec3 Weight = vec3(1,1,1);
    uint OpacityBounces=0;
    materialPoint VolumeMaterial;
    bool HasVolumeMaterial=false;

    bool NextEmission = true;
    sceneIntersection MisIntersection = MakeIsect(Sample);

    for(int Bounce=0; Bounce < GET_ATTR(Parameters, Bounces); Bounce++)
    {
        sceneIntersection Isect = IntersectTLAS(Ray, Sample, Bounce);
        if(Isect.Distance == MAX_LENGTH)
        {
            Radiance += Weight * EvalEnvironment(Ray.Direction);
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
                    if(GET_ATTR(Parameters, CurrentSample) % 2 ==0)
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

            
            //If the hit material is volumetric
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
            if(GET_ATTR(Parameters, CurrentSample) % 2 ==0)
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
                    0.5f * SamplePhasePDF(VolumeMaterial, Outgoing, Incoming) + 
                    0.5f * SampleLightsPDF(Position, Incoming)
                    );
                    
            Ray.Origin = Position;
            Ray.Direction = Incoming;
        }

        if(Weight == vec3(0,0,0) || !IsFinite(Weight)) break;

        if(Bounce > 3)
        {
            float RussianRouletteProb = min(0.99f, max3(Weight));
            if(RandomUnilateral(Isect.RandomState) >= RussianRouletteProb) break;
            Weight *= 1.0f / RussianRouletteProb;
        }                
    }

    if(!IsFinite(Radiance)) Radiance = vec3(0,0,0);
    if(max3(Radiance) > GET_ATTR(Parameters, Clamp)) Radiance = Radiance * (GET_ATTR(Parameters, Clamp) / max3(Radiance)); 


    return Radiance;
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
            vec3 Radiance;
            if(GET_ATTR(Parameters, SamplingMode) == SAMPLING_MODE_MIS)
            {
                Radiance = PathTraceMIS(Sample, UV);
            }
            else
            {
                Radiance = PathTrace(Sample, UV);
            }
            float SampleWeight = 1.0f / (float(GET_ATTR(Parameters,CurrentSample) + Sample) + 1);
            NewCol = mix(PrevCol, vec4(Radiance.x, Radiance.y, Radiance.z, 1.0f), SampleWeight);
            PrevCol = NewCol;    
        }
        imageStore(RenderImage, ivec2(GlobalID), NewCol);
    }
}