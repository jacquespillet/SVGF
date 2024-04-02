#pragma once
#include "BVH.h"
#include <glm/glm.hpp>
using namespace glm;
using namespace gpupt;

#define PI_F 3.141592653589

__device__ u32 Width;
__device__ u32 Height;
__device__ triangle *TriangleBuffer;
__device__ triangleExtraData *TriangleExBuffer;
__device__ bvhNode *BVHBuffer;
__device__ u32 *IndicesBuffer;
__device__ indexData *IndexDataBuffer;
__device__ bvhInstance *TLASInstancesBuffer;
__device__ tlasNode *TLASNodes;
__device__ camera *Cameras;

#define MAIN() \
__global__ void TraceKernel(glm::vec4 *RenderImage, int _Width, int _Height, \
                            triangle *_AllTriangles, triangleExtraData *_AllTrianglesEx, bvhNode *_AllBVHNodes, u32 *_AllTriangleIndices, indexData *_IndexData, bvhInstance *_Instances, tlasNode *_TLASNodes,\
                            camera *_Cameras)

#define INIT() \
    Width = _Width; \
    Height = _Height; \
    TriangleBuffer = _AllTriangles; \
    TriangleExBuffer = _AllTrianglesEx; \
    BVHBuffer = _AllBVHNodes; \
    IndicesBuffer = _AllTriangleIndices; \
    IndexDataBuffer = _IndexData; \
    TLASInstancesBuffer = _Instances; \
    TLASNodes = _TLASNodes; \
    Cameras = _Cameras; \


#define IMAGE_SIZE(Img) \
    ivec2(Width, Height)

#define GLOBAL_ID() \
    uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)

#define FN_DECL __device__

#define INOUT(Type) Type &

__device__ void imageStore(vec4 *Image, ivec2 p, vec4 Colour)
{
    Image[p.y * Width + p.x] = Colour;
}

__device__ vec4 imageLoad(vec4 *Image, ivec2 p)
{
    return Image[p.y * Width + p.x];
}

 
#include "../../resources/PathTraceCode.cpp"