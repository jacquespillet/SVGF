#include "BufferCu.cuh"

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>


namespace gpupt
{

bufferCu::bufferCu(int Size, void *InitData) : Size(Size){
    cudaMalloc((void**)&this->Data, Size);
    if(InitData != nullptr) cudaMemcpy(this->Data, InitData, Size, cudaMemcpyHostToDevice);
}

// Destructor
bufferCu::~bufferCu() {
    Destroy();
}

void bufferCu::Destroy()
{
    cudaFree(Data); 
}

// Update SSBO data
void bufferCu::updateData(const void* data, size_t dataSize) {
    cudaMemcpy(this->Data, data, dataSize, cudaMemcpyHostToDevice);
}

void bufferCu::updateData(size_t offset, const void* data, size_t dataSize) {
    cudaMemcpy((void*)((uint8_t*)this->Data + offset), data, dataSize, cudaMemcpyHostToDevice);
}

}
