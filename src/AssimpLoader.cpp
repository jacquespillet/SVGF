#include "AssimpLoader.h"
#include "Scene.h"

#include <iostream>


#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <filesystem>
#include <unordered_map>
#include <glm/gtc/matrix_transform.hpp>

namespace gpupt
{

void LoadGeometry(const aiScene *AScene, scene *Scene)
{
    for(int i=0; i<AScene->mNumMeshes; i++)
    {
        aiMesh* AMesh = AScene->mMeshes[i];

        Scene->Shapes.emplace_back();
        shape &Shape = Scene->Shapes.back();
        Scene->ShapeNames.push_back(AMesh->mName.C_Str());
        // Reserve memory for Positions, Normals, and texture coordinates
        
        Shape.PositionsTmp.reserve(AMesh->mNumVertices);
        Shape.NormalsTmp.reserve(AMesh->mNumVertices);
        Shape.TexCoordsTmp.reserve(AMesh->mNumVertices );

        // Load Positions
        for (unsigned int i = 0; i < AMesh->mNumVertices; ++i) {
            Shape.PositionsTmp.push_back(glm::vec3(AMesh->mVertices[i].x, AMesh->mVertices[i].y, AMesh->mVertices[i].z));

            // Load Normals if they exist
            if (AMesh->HasNormals()) {
                Shape.NormalsTmp.push_back(glm::vec3(AMesh->mNormals[i].x, AMesh->mNormals[i].y, AMesh->mNormals[i].z));
            }

            // Load texture coordinates if they exist
            if (AMesh->HasTextureCoords(0)) {
                Shape.TexCoordsTmp.push_back(glm::vec2(AMesh->mTextureCoords[0][i].x, AMesh->mTextureCoords[0][i].y));
            }
        }

        // Load indices
        Shape.Triangles.reserve(AMesh->mNumFaces * 3);
        for (unsigned int i = 0; i < AMesh->mNumFaces; ++i) {
            aiFace face = AMesh->mFaces[i];
            Shape.IndicesTmp.push_back(glm::ivec3(face.mIndices[0], face.mIndices[1], face.mIndices[2]));
        }
        Shape.PreProcess();
    }
}

void ProcessNode(const aiNode *Node, const aiScene *AScene, scene *Scene, glm::mat4 ParentTransform, int MeshBaseIndex)
{
    int MaterialBaseIndex = Scene->Materials.size();

    glm::mat4 ChildTransform(1);
    ChildTransform[0][0] = Node->mTransformation[0][0]; ChildTransform[0][1] = Node->mTransformation[0][1]; ChildTransform[0][2] = Node->mTransformation[0][2];  ChildTransform[0][3] = Node->mTransformation[0][3];
    ChildTransform[1][0] = Node->mTransformation[1][0]; ChildTransform[1][1] = Node->mTransformation[1][1]; ChildTransform[1][2] = Node->mTransformation[1][2];  ChildTransform[1][3] = Node->mTransformation[1][3];
    ChildTransform[2][0] = Node->mTransformation[2][0]; ChildTransform[2][1] = Node->mTransformation[2][1]; ChildTransform[2][2] = Node->mTransformation[2][2];  ChildTransform[2][3] = Node->mTransformation[2][3];
    ChildTransform[3][0] = Node->mTransformation[3][0]; ChildTransform[3][1] = Node->mTransformation[3][1]; ChildTransform[3][2] = Node->mTransformation[3][2];  ChildTransform[3][3] = Node->mTransformation[3][3];
    glm::mat4 WorldTransform = ParentTransform * ChildTransform;

    // Process all meshes in this node
    for (unsigned int i = 0; i < Node->mNumMeshes; ++i) {
        aiMesh *AMesh = AScene->mMeshes[Node->mMeshes[i]];

        Scene->Instances.emplace_back();
        instance &Instance = Scene->Instances.back();
        Scene->InstanceNames.push_back(Node->mName.C_Str());
        Instance.Transform = WorldTransform;
        Instance.Material = MaterialBaseIndex + AMesh->mMaterialIndex;
        Instance.Shape = MeshBaseIndex + Node->mMeshes[i];
    }

    // Process all children nodes
    for (unsigned int i = 0; i < Node->mNumChildren; ++i) {
        ProcessNode(Node->mChildren[i], AScene, Scene, WorldTransform, MeshBaseIndex);
    }    
}

void LoadInstances(const aiScene *AScene, scene *Scene, int MeshBaseIndex, float GlobalScale)
{
    glm::mat4 RootTransform = glm::scale(glm::mat4(1), glm::vec3(GlobalScale));
    ProcessNode(AScene->mRootNode, AScene, Scene, RootTransform, MeshBaseIndex);
}

std::string FileNameFromPath(const std::string& FullPath) {
    // Find the last occurrence of the directory separator
    size_t lastSeparator = FullPath.find_last_of("/\\");
    if (lastSeparator == std::string::npos)
        lastSeparator = 0; // If no directory separator found, set it to 0 to start from the beginning of the string

    // Find the last occurrence of the extension separator
    size_t extensionStart = FullPath.find_last_of(".");
    if (extensionStart == std::string::npos || extensionStart < lastSeparator)
        extensionStart = FullPath.length(); // If no extension separator found or it's before the last separator, set it to the end of the string

    // Extract the file name without the extension
    return FullPath.substr(lastSeparator + 1, extensionStart - lastSeparator - 1);
}

void LoadMaterials(const aiScene *AScene, scene *Scene, const std::string &Path, bool DoLoadMaterials, bool DoLoadTextures)
{
    std::unordered_map<std::string, int> TexturesMapping;

    for(int i=0; i<AScene->mNumMaterials; i++)
    {
        aiMaterial *AMaterial = AScene->mMaterials[i];
        if(DoLoadMaterials)
        {
            Scene->Materials.emplace_back();
            Scene->MaterialNames.push_back(AScene->mMaterials[i]->GetName().C_Str());
            material &Material = Scene->Materials.back();
            Material.MaterialType = MATERIAL_TYPE_PBR;


            aiColor3D DiffuseColour;
            AMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, DiffuseColour);
            Material.Colour = glm::vec3(DiffuseColour.r, DiffuseColour.g, DiffuseColour.b);

            aiColor3D Emission;
            AMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, Emission);
            Material.Emission = glm::vec3(Emission.r, Emission.g, Emission.b);
            
            AMaterial->Get(AI_MATKEY_METALLIC_FACTOR, Material.Metallic);
            AMaterial->Get(AI_MATKEY_ROUGHNESS_FACTOR, Material.Roughness);
            AMaterial->Get(AI_MATKEY_OPACITY, Material.Opacity);
        }

        if(DoLoadTextures)
        {
            aiString TexturePath;
            AMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &TexturePath);
            if(TexturePath.length != 0)
            {
                if(TexturesMapping.find(TexturePath.C_Str()) == TexturesMapping.end())
                {
                    TexturesMapping[TexturePath.C_Str()] = Scene->Textures.size();

                    Scene->Textures.emplace_back();
                    Scene->TextureNames.push_back(FileNameFromPath(TexturePath.C_Str()));
                    texture &Texture = Scene->Textures.back();
                    Texture.SetFromFile(Path + "/" + TexturePath.C_Str(), Scene->TextureWidth, Scene->TextureHeight);
                }
                if(DoLoadMaterials)
                {
                    material &Material = Scene->Materials.back();
                    Material.ColourTexture = TexturesMapping[TexturePath.C_Str()];
                }
            }
        }
    }
}



std::string PathFromFile(const std::string &FullPath)
{
    std::filesystem::path filePath = FullPath;
    std::filesystem::path directoryPath = filePath.parent_path();
    return directoryPath.string();    
}


void LoadAssimp(std::string FileName, scene *Scene, bool DoLoadInstances, bool DoLoadMaterials, bool DoLoadTextures, float GlobalScale)
{
    Assimp::Importer Importer;
    uint32_t Flags = aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_CalcTangentSpace;
    const aiScene* AScene = Importer.ReadFile(FileName, Flags);
    
    if (!AScene || AScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !AScene->mRootNode) {
        std::string Error = Importer.GetErrorString();
        std::cout << Error << std::endl;
        return;
    }

    std::string Path = PathFromFile(FileName);

    

    int MeshBaseIndex = Scene->Shapes.size();
    LoadGeometry(AScene, Scene);
    if(DoLoadInstances) LoadInstances(AScene, Scene, MeshBaseIndex, GlobalScale);
    if(DoLoadMaterials || DoLoadTextures) LoadMaterials(AScene, Scene, Path, DoLoadMaterials, DoLoadTextures);
    
}    

}