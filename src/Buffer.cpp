#include "buffer.h"

#if API==API_CU
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#endif

namespace gpupt
{

buffer::buffer(size_t DataSize, const void* InitData) {
#if API==API_GL
    glGenBuffers(1, &BufferID);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferData(GL_SHADER_STORAGE_BUFFER, DataSize, InitData, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
#else
    this->Size = DataSize;
    cudaMalloc((void**)&this->Data, Size);
    if(InitData != nullptr) cudaMemcpy(this->Data, InitData, Size, cudaMemcpyHostToDevice);
#endif
}

void buffer::Reallocate(const void* InitData, size_t DataSize)
{
#if API==API_GL
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferData(GL_SHADER_STORAGE_BUFFER, DataSize, InitData, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
#else
    this->Size = DataSize;
    cudaFree(Data); 
    cudaMalloc((void**)&this->Data, Size);
    if(InitData != nullptr) cudaMemcpy(this->Data, InitData, Size, cudaMemcpyHostToDevice);
#endif
}

buffer::~buffer() {
#if API==API_GL
    if(BufferID != (GLuint)-1)
        Destroy();
#else
    Destroy();
#endif
}

void buffer::Destroy()
{
#if API==API_GL
    glDeleteBuffers(1, &BufferID);
    BufferID = (GLuint)-1;
#else
    cudaFree(Data); 
#endif
}

void buffer::updateData(const void* data, size_t DataSize) {
#if API==API_GL
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, DataSize, data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
#else
    cudaMemcpy(this->Data, data, DataSize, cudaMemcpyHostToDevice);
#endif
}

void buffer::updateData(size_t offset, const void* data, size_t DataSize) {
#if API==API_GL
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, DataSize, data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
#else
    cudaMemcpy((void*)((uint8_t*)this->Data + offset), data, DataSize, cudaMemcpyHostToDevice);
#endif
}

// 


bufferGL::bufferGL(size_t DataSize, const void* InitData) {
    glGenBuffers(1, &BufferID);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferData(GL_SHADER_STORAGE_BUFFER, DataSize, InitData, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void bufferGL::Reallocate(const void* InitData, size_t DataSize)
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferData(GL_SHADER_STORAGE_BUFFER, DataSize, InitData, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

bufferGL::~bufferGL() {
    if(BufferID != (GLuint)-1)
        Destroy();
}

void bufferGL::Destroy()
{
    glDeleteBuffers(1, &BufferID);
    BufferID = (GLuint)-1;
}

void bufferGL::updateData(const void* data, size_t DataSize) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, DataSize, data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void bufferGL::updateData(size_t offset, const void* data, size_t DataSize) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, DataSize, data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}



// 

uniformBufferGL::uniformBufferGL(size_t DataSize, const void* data) {
    glGenBuffers(1, &BufferID);
    glBindBuffer(GL_UNIFORM_BUFFER, BufferID);
    glBufferData(GL_UNIFORM_BUFFER, DataSize, data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBufferBase(GL_UNIFORM_BUFFER, 8, BufferID);
}

uniformBufferGL::~uniformBufferGL() {
    if(BufferID != (GLuint)-1)
        Destroy();
}

void uniformBufferGL::Destroy()
{
    glDeleteBuffers(1, &BufferID);
    BufferID = -1;
}

void uniformBufferGL::updateData(const void* data, size_t DataSize) {
    glBindBuffer(GL_UNIFORM_BUFFER, BufferID);
    glBufferData(GL_UNIFORM_BUFFER, DataSize, data, GL_DYNAMIC_DRAW);
    // glBufferSubData(GL_UNIFORM_BUFFER, 0, DataSize, data);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}


}