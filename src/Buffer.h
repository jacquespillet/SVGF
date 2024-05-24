#pragma once
#include <glad/gl.h>

#include <vector>

namespace gpupt
{
class buffer {
public:
    buffer(size_t dataSize, const void* data = nullptr);
    ~buffer();
    void Destroy();
    void updateData(const void* data, size_t dataSize);
    void updateData(size_t offset, const void* data, size_t dataSize);
    void Reallocate(const void* data, size_t dataSize);

    void *Data;
    uint32_t Size;
};

class bufferGL {
public:
    bufferGL(size_t dataSize, const void* data = nullptr);
    ~bufferGL();
    void Destroy();
    void updateData(const void* data, size_t dataSize);
    void updateData(size_t offset, const void* data, size_t dataSize);
    void Reallocate(const void* data, size_t dataSize);

    GLuint BufferID;
};


class uniformBufferGL
{
public:
    uniformBufferGL(size_t dataSize, const void* data = nullptr);
    ~uniformBufferGL();
    void Destroy();
    void updateData(const void* data, size_t dataSize);
    GLuint BufferID;
};

}