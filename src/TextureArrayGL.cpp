#include "TextureArrayGL.h"

namespace gpupt
{
    textureArrayGL::textureArrayGL() : TextureID(0) {}

    textureArrayGL::~textureArrayGL() {
        if (TextureID != 0) {
            glDeleteTextures(1, &TextureID);
        }
    }

    void textureArrayGL::CreateTextureArray(int Width, int Height, int Layers) {
        glGenTextures(1, &TextureID);
        glBindTexture(GL_TEXTURE_2D_ARRAY, TextureID);

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Allocate storage for the texture array
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, Width, Height, Layers, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }

    void textureArrayGL::LoadTextureLayer(int layerIndex, const std::vector<uint8_t>& imageData, int Width, int Height) {
        glBindTexture(GL_TEXTURE_2D_ARRAY, TextureID);
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layerIndex, Width, Height, 1, GL_RGBA, GL_UNSIGNED_BYTE, imageData.data());
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }

    void textureArrayGL::Bind(int textureUnit){
        glActiveTexture(GL_TEXTURE0 + textureUnit);
        glBindTexture(GL_TEXTURE_2D_ARRAY, TextureID);
    }

    void textureArrayGL::Unbind() const {
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }

}