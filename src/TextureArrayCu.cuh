#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_texture_types.h>
#include <vector>


namespace gpupt
{

class textureArrayCu {
public:
    textureArrayCu() = default;
    ~textureArrayCu();
    void CreateTextureArray(int Width, int Height, int Layers, bool IsFloat=false);
    void LoadTextureLayer(int layerIndex, const std::vector<uint8_t>& ImageData, int Width, int Height);
    void LoadTextureLayer(int layerIndex, const std::vector<float>& ImageData, int Width, int Height);

    int TotalWidth = 8192;
    int TotalHeight = 8192;
    size_t Pitch;    
    cudaArray* CuArray;
    cudaTextureObject_t  TexObject;
    int Width, Height;
};

}