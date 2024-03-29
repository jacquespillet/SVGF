#include "BufferGL.h"

namespace gpupt
{
bufferGL::bufferGL(size_t dataSize, const void* data) {
    glGenBuffers(1, &BufferID);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferData(GL_SHADER_STORAGE_BUFFER, dataSize, data, GL_DYNAMIC_COPY);
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

void bufferGL::updateData(const void* data, size_t dataSize) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, dataSize, data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void bufferGL::updateData(size_t offset, const void* data, size_t dataSize) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, BufferID);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, dataSize, data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

}