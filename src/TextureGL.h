#pragma once

#include <GL/glew.h>

namespace gpupt
{

class textureGL {
public:
    textureGL(int Width, int Height, int NChannels);
    ~textureGL();
    void Destroy();

    GLuint TextureID;
    int Width, Height;
};

}
