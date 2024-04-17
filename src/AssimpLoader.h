#pragma once
#include <memory>
#include <string>


namespace gpupt
{
struct scene;

void LoadAssimp(std::string FilePath, scene *Scene, bool DoLoadInstances, bool DoLoadMaterials, bool DoLoadTextures);

}