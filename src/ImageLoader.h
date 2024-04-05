#pragma once
#include <string>
#include <vector>
namespace gpupt
{
void ImageFromFile(const std::string &FileName, std::vector<uint8_t> &Data, int &Width, int &Height, int &NumChannels);  
}