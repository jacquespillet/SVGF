#include "TextureGL.h"

#include "assert.h"

namespace gpupt
{

textureGL::textureGL(int Width, int Height, int NChannels) : Width(Width), Height(Height) {
    // Generate texture ID and bind it
    glGenTextures(1, &TextureID);
    glBindTexture(GL_TEXTURE_2D, TextureID);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    
    GLint InternalFormat = (NChannels==3) ? GL_RGB32F : GL_RGBA32F;
    GLenum Format = (NChannels==3) ? GL_RGB : GL_RGBA;
    GLenum Type = GL_FLOAT;

    glTexImage2D(GL_TEXTURE_2D, 0, InternalFormat, Width, Height, 0, Format, Type, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void textureGL::Download(std::vector<uint8_t> &Output) {
    std::vector<float> DataF(Width * Height * 4);
    if(Output.size() != DataF.size()) Output.resize(DataF.size());
    GLuint PBO;
    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_PACK_BUFFER, DataF.size() * sizeof(float), 0, GL_STATIC_READ);
    glBindTexture(GL_TEXTURE_2D, TextureID);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, (void*)(0));
    float * Ptr =(float*) glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    for (size_t i = 0; i < DataF.size(); i++)
    {
        float f = std::min(1.0f, std::max(0.0f, Ptr[i]));
        Output[i] = (uint8_t)(f * 255.0f);
    }
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);    
    glDeleteBuffers(1, &PBO);
}


// Destructor
textureGL::~textureGL() {
    if(TextureID != (GLuint)-1)
        Destroy();
}

void textureGL::Destroy()
{
    glDeleteTextures(1, &TextureID);
    TextureID=(GLuint)-1;
}

}
