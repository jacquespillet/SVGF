#pragma once
#include <glm/glm.hpp>
using namespace glm;

__device__ u32 Width;
__device__ u32 Height;

#define MAIN() \
__global__ void TraceKernel(glm::vec4 *RenderImage, int _Width, int _Height)

#define INIT() \
    Width = _Width; \
    Height = _Height; \

#define IMAGE_SIZE(Img) \
    ivec2(Width, Height)

#define GLOBAL_ID() \
    uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)


__device__ void imageStore(vec4 *Image, ivec2 p, vec4 Colour)
{
    Image[p.y * Width + p.x] = Colour;
}
 
#include "../../resources/PathTraceCode.cpp"