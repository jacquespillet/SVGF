#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace gpupt
{

__global__ void TexToBuffer(
    glm::vec4* cudaBuffer,cudaTextureObject_t texObj, int width, int height
    )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float4 value = tex2D<float4>(texObj, x, y);
        int index = y * width + x;
        cudaBuffer[index] = glm::vec4(value.x, value.y, value.z, value.w);
    }
}

void GLTexToCuBuffer(
                    glm::vec4* CudaBuffer, 
                    cudaTextureObject_t TexObj, 
                    uint32_t Width, 
                    uint32_t Height
                    )
{
    dim3 blockDim(32, 32);
    dim3 gridDim(Width / blockDim.x + 1, Height / blockDim.y + 1, 1);
    TexToBuffer<<<gridDim, blockDim>>>(
        CudaBuffer, TexObj, Width, Height
        );    
}

}