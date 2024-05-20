#pragma once
#include <vector>
#include <glad/gl.h>
#include <memory>

namespace gpupt
{
struct cudaTextureMapping;


struct framebufferDescriptor
{
    GLint InternalFormat;
    GLenum Format;
    GLenum Type;
    uint32_t ElemSize;
};

class framebuffer
{
public:
    framebuffer(int Width, int Height, std::vector<framebufferDescriptor> &Descriptors);

    GLuint GetTexture(int Index);

    void Destroy();

    void Bind();
    void Unbind();

    ~framebuffer();


    std::vector<std::shared_ptr<cudaTextureMapping>> CudaMappings;
private:
    GLuint FBO;
    std::vector<GLuint> Textures;
    GLuint DepthTexture;
};

}