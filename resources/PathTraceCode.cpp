struct sceneIntersection
{
    float Distance;
    uint InstanceIndex;
    uint PrimitiveIndex;
    float U;
    float V;
};

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

FN_DECL ray GetRay(vec2 ImageUV)
{
    camera Camera = Cameras[0];

    // Point on the film
    vec3 Q = vec3(
        (0.5f - ImageUV.x),
        (ImageUV.y - 0.5f),
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




MAIN()
{
    INIT()
    
    ivec2 ImageSize = IMAGE_SIZE(RenderImage);
    int Width = ImageSize.x;
    int Height = ImageSize.y;

    uvec2 GlobalID = GLOBAL_ID();
    if (GlobalID.x < Width && GlobalID.y < Height) {

        vec2 UV = vec2(GLOBAL_ID()) / vec2(ImageSize);
        ray Ray = GetRay(UV);

        sceneIntersection Isect;
        Isect.Distance = 1e30f;

        IntersectTLAS(Ray, Isect);
        if(Isect.Distance < 1e30f)
        {
            triangleExtraData ExtraData = TriangleExBuffer[Isect.PrimitiveIndex];    
            vec4 Colour = 
                ExtraData.Colour1 * Isect.U + 
                ExtraData.Colour2 * Isect.V +
                ExtraData.Colour0 * (1 - Isect.U - Isect.V);
            imageStore(RenderImage, ivec2(GLOBAL_ID()), Colour);
        }
    }
}