#include "TextureArrayCu.cuh"

namespace gpupt
{
    textureArrayCu::~textureArrayCu() {
        cudaDeviceSynchronize();
        
        cudaDestroyTextureObject(TexObject);
        cudaFree(CuArray);   
        
    }


    void textureArrayCu::CreateTextureArray(int Width, int Height, int Layers) {
        this->Width = Width;
        this->Height = Height;

        cudaMallocPitch((void**)&CuArray, &Pitch,  TotalWidth*sizeof(uchar4), TotalHeight);
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = CuArray;
        resDesc.res.pitch2D.width = TotalWidth;
        resDesc.res.pitch2D.height = TotalHeight;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
        resDesc.res.pitch2D.pitchInBytes = Pitch;
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        cudaCreateTextureObject(&TexObject, &resDesc, &texDesc, NULL);        
    }

    void textureArrayCu::LoadTextureLayer(int layerIndex, const std::vector<uint8_t>& ImageData, int Width, int Height) {
        static int LayersPerRow = TotalWidth / Width;
        int DestInxX = layerIndex % LayersPerRow;
        int DestInxY = layerIndex / LayersPerRow;
        uint32_t DestX = DestInxX * Width;
        uint32_t DestY = DestInxY * Height;
        uint32_t Dest = (DestY * TotalWidth + DestX) * 4;


        cudaMemcpy2D((uint8_t*)CuArray + Dest, Pitch, ImageData.data(), 512*sizeof(uchar4), 512*sizeof(uchar4), Height, cudaMemcpyHostToDevice);
    }
}