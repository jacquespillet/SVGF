#pragma once
#include <GL/glew.h>
#include <vector>

namespace gpupt
{
class bufferGL {
public:
    bufferGL(size_t dataSize, const void* data = nullptr);
    ~bufferGL();
    void Destroy();
    void updateData(const void* data, size_t dataSize);
    void updateData(size_t offset, const void* data, size_t dataSize);
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