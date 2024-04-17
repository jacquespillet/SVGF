#include "AssetLoader.h"
#include "GLTFLoader.h"
#include "AssimpLoader.h"
namespace gpupt
{
     
void LoadAsset(std::string FilePath, scene *Scene, bool LoadInstances, bool LoadMaterials, bool LoadTextures)
{
    std::string Extension = FilePath.substr(FilePath.find_last_of(".") + 1);
    if(Extension == "gltf" || Extension == "glb")
    {
        LoadGLTF(FilePath, Scene, LoadInstances, LoadMaterials, LoadTextures);
    }
    else
    {
        LoadAssimp(FilePath, Scene, LoadInstances, LoadMaterials, LoadTextures);
    }
}
}