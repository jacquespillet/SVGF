#pragma once
#include <cuda_gl_interop.h>
#include <memory>
#include "TextureGL.h"

#include <glm/vec4.hpp>
#include <iostream>

struct cudaArray;
struct cudaGraphicsResource;
class textureGL;

#define CUDA_CHECK_ERROR(err) \
    do { \
        cudaError_t error = err; \
        if (error != cudaSuccess) { \
            std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            assert(false); \
        } \
    } while (0)

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
    ~cudaTextureMapping()
    {
        Destroy();
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

std::shared_ptr<cudaTextureMapping> CreateMapping(GLuint TexID, int Width, int Height, uint32_t ElemSize, bool Write=false)
{
    CUDA_CHECK_ERROR(cudaGetLastError());

    std::shared_ptr<cudaTextureMapping> Result = std::make_shared<cudaTextureMapping>();

    size_t bufferSize = Width * Height * ElemSize;
    cudaMalloc((void**)&Result->CudaBuffer, bufferSize);        
    CUDA_CHECK_ERROR(cudaGetLastError());

    cudaGraphicsGLRegisterImage(&Result->CudaTextureResource, TexID, GL_TEXTURE_2D, Write ? cudaGraphicsRegisterFlagsWriteDiscard :cudaGraphicsRegisterFlagsNone);
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Map the CUDA buffer to access it in CUDA
    cudaGraphicsMapResources(1, &Result->CudaTextureResource);
    CUDA_CHECK_ERROR(cudaGetLastError());
    cudaGraphicsSubResourceGetMappedArray(&Result->CudaTextureArray, Result->CudaTextureResource, 0, 0);
    CUDA_CHECK_ERROR(cudaGetLastError());

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = Result->CudaTextureArray;


    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.readMode = cudaReadModeElementType;
    

    cudaCreateTextureObject(&Result->TexObj, &texRes, &texDesc, nullptr);

    CUDA_CHECK_ERROR(cudaGetLastError());
    return Result;
}   
}