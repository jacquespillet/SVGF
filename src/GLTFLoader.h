#pragma once
#include <memory>
#include <string>


namespace gpupt
{
struct scene;

void LoadGLTF(std::string FilePath, scene *Scene, bool LoadInstances, bool LoadMaterials, bool LoadTextures);

}