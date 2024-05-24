#include "TextureGL.h"

#include "assert.h"

namespace gpupt
{

GLint GetInternalFormat(textureGL::channels Channel, textureGL::types Type)
{
    switch (Channel)
    {
    case textureGL::channels::R:
        if(Type == textureGL::types::Float)
        {
            return GL_R32F;
        }
        if(Type == textureGL::types::Uint8)
        {
            return GL_R8;
        }
        if(Type == textureGL::types::Half)
        {
            return GL_R16F;
        }
        break;
    case textureGL::channels::RGB:
        if(Type == textureGL::types::Float)
        {
            return GL_RGB32F;
        }
        if(Type == textureGL::types::Uint8)
        {
            return GL_RGB8;
        }
        if(Type == textureGL::types::Half)
        {
            return GL_RGB16F;
        }
        break;
    case textureGL::channels::RGBA:
        if(Type == textureGL::types::Float)
        {
            return GL_RGBA32F;
        }
        if(Type == textureGL::types::Uint8)
        {
            return GL_RGBA8;
        }
        if(Type == textureGL::types::Half)
        {
            return GL_RGBA16F;
        }
        break;
    default:
        break;
    }
}

GLenum GetFormat(textureGL::channels Channel, textureGL::types Type)
{
    switch (Channel)
    {
    case textureGL::channels::R:
            return GL_RED;
        break;
    case textureGL::channels::RGB:
            return GL_RGB;
        break;
    case textureGL::channels::RGBA:
            return GL_RGBA;
        break;
    default:
        break;
    }
}

GLenum GetType(textureGL::types Type)
{
    switch (Type)
    {
    case textureGL::types::Uint8:
        return GL_UNSIGNED_BYTE;
        break;
    case textureGL::types::Half:
        return GL_HALF_FLOAT;
        break;
    case textureGL::types::Float:
        return GL_FLOAT;
        break;
    default:
        break;
    }
}



textureGL::textureGL(int Width, int Height, channels Channel, types Type) : Width(Width), Height(Height) {
    // Generate texture ID and bind it
    glGenTextures(1, &TextureID);
    glBindTexture(GL_TEXTURE_2D, TextureID);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    
    GLint InternalFormat = GetInternalFormat(Channel, Type);
    GLenum Format = GetFormat(Channel, Type);
    GLenum GLType = GetType(Type);

    glTexImage2D(GL_TEXTURE_2D, 0, InternalFormat, Width, Height, 0, Format, GLType, nullptr);
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
