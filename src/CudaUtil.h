#pragma once
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <memory>

#include <glm/vec4.hpp>

struct cudaArray;
struct cudaGraphicsResource;
class textureGL;


namespace gpupt
{
struct cudaTextureMapping
{
    glm::vec4* CudaBuffer;
    cudaArray* CudaTextureArray;
    cudaGraphicsResource* CudaTextureResource;
    cudaTextureObject_t TexObj;

    void Destroy()
    {
        cudaDestroyTextureObject(TexObj);
        cudaGraphicsUnmapResources(1, &CudaTextureResource);
        cudaFree(CudaBuffer);
    }
};

std::shared_ptr<cudaTextureMapping> CreateMapping(std::shared_ptr<textureGL> Tex, bool Write=false)
{
    std::shared_ptr<cudaTextureMapping> Result = std::make_shared<cudaTextureMapping>();

    size_t bufferSize = Tex->Width * Tex->Height * sizeof(glm::vec4);
    cudaMalloc((void**)&Result->CudaBuffer, bufferSize);        

    cudaGraphicsGLRegisterImage(&Result->CudaTextureResource, Tex->TextureID, GL_TEXTURE_2D, Write ? cudaGraphicsRegisterFlagsWriteDiscard :cudaGraphicsRegisterFlagsNone);

    // Map the CUDA buffer to access it in CUDA
    cudaGraphicsMapResources(1, &Result->CudaTextureResource);
    cudaGraphicsSubResourceGetMappedArray(&Result->CudaTextureArray, Result->CudaTextureResource, 0, 0);

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = Result->CudaTextureArray;


    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.readMode = cudaReadModeElementType;
    

    cudaCreateTextureObject(&Result->TexObj, &texRes, &texDesc, nullptr);

    return Result;
}   
}