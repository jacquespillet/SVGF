#pragma once
#include <memory>
#include <string>


namespace gpupt
{
struct scene;

void LoadAssimp(std::string FilePath, std::shared_ptr<scene> Scene, bool AddInstances);
void LoadAssimpShapeOnly(std::string FilePath, std::shared_ptr<scene> Scene, int ShapeInx);

}