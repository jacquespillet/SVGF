#pragma once
#include <memory>
#include <string>


namespace gpupt
{
struct scene;

void LoadGLTF(std::string FilePath, std::shared_ptr<scene> Scene, bool AddInstances);

}