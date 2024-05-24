#include "Framebuffer.h"
#include "CudaUtil.h"

namespace gpupt
{

framebuffer::framebuffer(int Width, int Height, std::vector<framebufferDescriptor> &Descriptors)
{
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);

    Textures.resize(Descriptors.size());
    std::vector<GLuint> Attachments(Descriptors.size());

    glGenTextures(Descriptors.size(), &Textures[0]);
    for(int i=0; i<Descriptors.size(); i++)
    {
        glBindTexture(GL_TEXTURE_2D, Textures[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, Descriptors[i].InternalFormat, Width, Height, 0, Descriptors[i].Format, Descriptors[i].Type, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, Textures[i], 0);

        Attachments[i] = GL_COLOR_ATTACHMENT0 + i;
    }
    glDrawBuffers(Attachments.size(), Attachments.data());

    glGenTextures(1, &DepthTexture);
    glBindTexture(GL_TEXTURE_2D, DepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, Width, Height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, DepthTexture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        assert(false);
        exit(0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    CudaMappings.resize(Descriptors.size());
    for(int i=0; i<CudaMappings.size(); i++) 
    {
        CudaMappings[i] = CreateMapping(Textures[i], Width, Height, Descriptors[i].ElemSize, true);
    }


}

void framebuffer::Destroy()
{
    glDeleteTextures(1, &DepthTexture);
    for(int i=0; i<Textures.size(); i++)
    {
        glDeleteTextures(1, &Textures[i]);
    }
    glDeleteFramebuffers(1, &FBO);
}

framebuffer::~framebuffer()
{
    Destroy();
}


void framebuffer::Bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
}
void framebuffer::Unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER,0);
}


GLuint framebuffer::GetTexture(int Index)
{
    return Textures[Index];
}   

}