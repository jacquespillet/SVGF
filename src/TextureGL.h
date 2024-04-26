#pragma once

#include <GL/glew.h>
#include <vector>

namespace gpupt
{

class textureGL {
public:
    textureGL(int Width, int Height, int NChannels);
    ~textureGL();
    void Destroy();
    void Download(std::vector<uint8_t> &Output);
    GLuint TextureID;
    int Width, Height;
};

}
