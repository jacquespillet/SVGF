#pragma once
#include <memory>
#include <string>


namespace gpupt
{
struct scene;

void LoadGLTF(std::string FilePath, scene *Scene, bool LoadInstances, bool LoadMaterials, bool LoadTextures);
void LoadGLTFShapeOnly(std::string FilePath, scene *Scene, int ShapeInx);

}