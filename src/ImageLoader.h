#pragma once
#include <string>
#include <vector>
namespace gpupt
{
bool IsHDR(const std::string &FileName);

void ImageFromFile(const std::string &FileName, std::vector<uint8_t> &Data, int &Width, int &Height, int &NumChannels);  
void ImageFromFile(const std::string &FileName, std::vector<float> &Data, int &Width, int &Height, int &NumChannels);  

void ImageToFile(const std::string &FileName, std::vector<float> &Data, int Width, int Height, int NumChannels);
}