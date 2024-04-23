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

void textureGL::Download(void *Ptr) {
    glBindTexture(GL_TEXTURE_2D, TextureID);
    glGetTexImage(GL_TEXTURE_2D,
                0,
                GL_RGBA,
                GL_FLOAT,
                Ptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    GLenum error = glGetError();
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
