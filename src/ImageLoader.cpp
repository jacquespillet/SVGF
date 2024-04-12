#include "ImageLoader.h"
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#include "stb_image_write.h"
#include <iostream>

namespace gpupt
{

bool IsHDR(const std::string &FileName)
{
    std::string Extension = FileName.substr(FileName.find_last_of(".") + 1);
    return Extension == "hdr" || Extension == "exr";
}

void ImageFromFile(const std::string &FileName, std::vector<uint8_t> &Data, int &Width, int &Height, int &NumChannels)
{
    int ImgWidth, ImgHeight;
    int channels;

    // Load the image
    stbi_uc* Image = stbi_load(FileName.c_str(), &ImgWidth, &ImgHeight, &channels, STBI_rgb_alpha);

    if (Image == nullptr) {
        // Handle error (file not found, invalid format, etc.)
        std::cout << "Failed to load image: " << FileName << std::endl;
        return;
    }

    // If there's a requested size, set target size with it. Otherwise, use the image size
    int TargetWidth = (Width != 0) ? Width : ImgWidth;
    int TargetHeight = (Height != 0) ? Height : ImgHeight;

    // If the target size is not the current size, resize the image
    if(TargetWidth != ImgWidth || TargetHeight != ImgHeight)
    {
        // Resize the image using stbir_resize (part of stb_image_resize.h)
        stbi_uc* ResizedImage = new stbi_uc[Width * Height * 4]; // Assuming RGBA format

        int result = stbir_resize_uint8(Image, ImgWidth, ImgHeight, 0, ResizedImage, TargetWidth, TargetHeight, 0, 4);
        
        stbi_image_free(Image);

        if (!result) {
            // Handle resize error
            std::cout << "Failed to resize image: " << FileName << std::endl;
            delete[] ResizedImage;
            return;
        }

        // Resize the pixel data, and copy to it
        Data.resize(TargetWidth * TargetHeight * 4);
        memcpy(Data.data(), ResizedImage, TargetWidth * TargetHeight * 4);
        delete[] ResizedImage;
    }
    else
    {
        // Resize the pixel data, and copy to it
        Data.resize(TargetWidth * TargetHeight * 4);
        memcpy(Data.data(), Image, TargetWidth * TargetHeight * 4);
        delete[] Image;
    }
} 


void ImageFromFile(const std::string &FileName, std::vector<float> &Data, int &Width, int &Height, int &NumChannels)
{
    int ImgWidth, ImgHeight;
    int Channels;

    // Load the image
    float* Image = stbi_loadf(FileName.c_str(), &ImgWidth, &ImgHeight, &Channels, NumChannels);

    if (Image == nullptr) {
        // Handle error (file not found, invalid format, etc.)
        std::cout << "Failed to load image: " << FileName << std::endl;
        return;
    }

    // If there's a requested size, set target size with it. Otherwise, use the image size
    int TargetWidth = (Width != 0) ? Width : ImgWidth;
    int TargetHeight = (Height != 0) ? Height : ImgHeight;

    // If the target size is not the current size, resize the image
    if(TargetWidth != ImgWidth || TargetHeight != ImgHeight)
    {
        // Check for too high value that will break when resized
        for(size_t i=0; i<ImgWidth * ImgHeight * NumChannels; i++)
        {
            Image[i] = std::min(10000.0f, Image[i]);
        }

        // Resize the image using stbir_resize (part of stb_image_resize.h)
        float* ResizedImage = new float[Width * Height * 4]; // Assuming RGBA format

        int result = stbir_resize_float(Image, ImgWidth, ImgHeight, 0, ResizedImage, TargetWidth, TargetHeight, 0, 4);
        
        stbi_image_free(Image);

        if (!result) {
            // Handle resize error
            std::cout << "Failed to resize image: " << FileName << std::endl;
            delete[] ResizedImage;
            return;
        }

        // Resize the pixel data, and copy to it
        Data.resize(TargetWidth * TargetHeight * 4);
        memcpy(Data.data(), ResizedImage, TargetWidth * TargetHeight * 4 * sizeof(float));
        delete[] ResizedImage;
    }
    else
    {
        // Resize the pixel data, and copy to it
        Data.resize(TargetWidth * TargetHeight * 4);
        memcpy(Data.data(), Image, TargetWidth * TargetHeight * 4 * sizeof(float));
        delete[] Image;
    }

    for(size_t i=0; i<Data.size(); i++)
    {
        if(std::isnan(Data[i]) || std::isinf(Data[i]))
        {
            Data[i] = 0;
        }
    }
} 

void ImageToFile(const std::string &FileName, std::vector<float> &Data, int Width, int Height, int NumChannels)
{
    stbi_write_hdr(FileName.c_str(), Width, Height, NumChannels,Data.data());
    
}



}