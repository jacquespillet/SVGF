#pragma once
#include <string>
#include <memory>
namespace gpupt
{
struct scene;

void LoadAsset(std::string FilePath, scene *Scene, bool LoadInstances, bool LoadMaterials, bool LoadTextures, float GlobalScale);

}