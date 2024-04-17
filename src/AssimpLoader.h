#pragma once
#include <memory>
#include <string>


namespace gpupt
{
struct scene;

void LoadAssimp(std::string FilePath, scene* Scene, bool AddInstances);
void LoadAssimpShapeOnly(std::string FilePath, scene* Scene, int ShapeInx);

}