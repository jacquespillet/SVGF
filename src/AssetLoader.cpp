#include "AssetLoader.h"
#include "GLTFLoader.h"
#include "AssimpLoader.h"
#include "Scene.h"
#include "Buffer.h"
#include "BVH.h"

namespace gpupt
{
     
void LoadAsset(std::string FilePath, scene *Scene, bool LoadInstances, bool LoadMaterials, bool LoadTextures)
{
    std::string Extension = FilePath.substr(FilePath.find_last_of(".") + 1);
    
    int PrevInstancesCount = Scene->Instances.size();
    int PrevShapesCount = Scene->Shapes.size();
    if(Extension == "gltf" || Extension == "glb")
    {
        LoadGLTF(FilePath, Scene, LoadInstances, LoadMaterials, LoadTextures);
    }
    else
    {
        LoadAssimp(FilePath, Scene, LoadInstances, LoadMaterials, LoadTextures);
    }

    for(int i=PrevShapesCount; i<Scene->Shapes.size(); i++)
    {
        Scene->BVH->AddShape(i);
    }

  


    if(LoadTextures) Scene->ReloadTextureArray();  
    if(LoadInstances)
    {
        if(!LoadMaterials)
        {
            Scene->Materials.emplace_back();
            material &NewMaterial = Scene->Materials.back(); 
            NewMaterial.Colour = {0.725f, 0.71f, 0.68f};            
            Scene->MaterialNames.push_back("New Material");
        }
        Scene->MaterialBuffer->Reallocate(Scene->Materials.data(), Scene->Materials.size() * sizeof(material));
        
        for(int i=PrevInstancesCount; i<Scene->Instances.size();  i++)
        {
            if(!LoadMaterials)
                Scene->Instances[i].Material = Scene->Materials.size()-1;
                
            Scene->BVH->AddInstance(i);
        }
    }  
    Scene->CheckNames();

}
}