#pragma once
#include <memory>
#include <string>


namespace gpupt
{
struct scene;

void LoadGLTF(std::string FilePath, std::shared_ptr<scene> Scene, bool AddInstances);
void LoadGLTFShapeOnly(std::string FilePath, std::shared_ptr<scene> Scene, int ShapeInx);

}