#include "Scene.h"
#include <glm/ext.hpp>

#include "BufferCu.cuh"
#include "BufferGL.h"
#include "App.h"
#include "GLTFLoader.h"
#include "ImageLoader.h"
#include "TextureArrayGL.h"
#include "TextureArrayCu.cuh"

namespace gpupt
{


glm::mat4 instance::GetModelMatrix() const
{
    return ModelMatrix;    
}

std::shared_ptr<scene> CreateCornellBox()
{
    std::shared_ptr<scene> Scene = std::make_shared<scene>();

    glm::uvec2 RenderSize = application::GetSize();


    Scene->Cameras.emplace_back();
    camera &Camera = Scene->Cameras.back();
    Camera.Frame = glm::translate(glm::mat4(1), glm::vec3(0, 1, 3.9f));
    Camera.Lens = 0.035f;
    Camera.Aperture = 0.0f;
    Camera.Focus = 3.9f;
    Camera.Film = 0.024f;
    Camera.Aspect = (float)RenderSize.x / (float)RenderSize.y;
    Scene->CameraNames.push_back("Main Camera");

    Scene->Shapes.emplace_back();
    shape &Floor = Scene->Shapes.back();
    Floor.Positions = { {-1, 0, 1}, {1, 0, 1}, {1, 0, -1}, {-1, 0, -1} };
    Floor.Triangles = { {0, 1, 2}, {2, 3, 0} };
    Floor.TexCoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    Scene->Materials.emplace_back();
    material &FloorMaterial = Scene->Materials.back(); 
    FloorMaterial.Colour = {0.725f, 0.71f, 0.68f};
    FloorMaterial.NormalTexture = 1;
    Scene->Instances.emplace_back();
    instance &FloorInstance = Scene->Instances.back();
    FloorInstance.Shape = (int)Scene->Shapes.size()-1;
    FloorInstance.Material = (int)Scene->Materials.size()-1;
    Scene->ShapeNames.push_back("Floor");
    Scene->InstanceNames.push_back("Floor");
    Scene->MaterialNames.push_back("Floor");


    Scene->Shapes.emplace_back();
    shape& CeilingShape       = Scene->Shapes.back();
    CeilingShape.Positions   = {{-1, 2, 1}, {-1, 2, -1}, {1, 2, -1}, {1, 2, 1}};
    CeilingShape.Triangles   = {{0, 1, 2}, {2, 3, 0}};
    CeilingShape.TexCoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    Scene->Materials.emplace_back();
    auto& CeilingMaterial    = Scene->Materials.back();
    CeilingMaterial.Colour    = {0.725f, 0.71f, 0.68f};    
    Scene->Instances.emplace_back();
    auto& CeilingInstance    = Scene->Instances.back();
    CeilingInstance.Shape    = (int)Scene->Shapes.size() - 1;
    CeilingInstance.Material = (int)Scene->Materials.size()-1;
    Scene->ShapeNames.push_back("Ceiling");
    Scene->InstanceNames.push_back("Ceiling");
    Scene->MaterialNames.push_back("Ceiling");

    Scene->Shapes.emplace_back();
    shape& BackWallShape       = Scene->Shapes.back();
    BackWallShape.Positions   = {{-1, 0, -1}, {1, 0, -1}, {1, 2, -1}, {-1, 2, -1}};
    BackWallShape.Triangles   = {{0, 1, 2}, {2, 3, 0}};
    BackWallShape.TexCoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    Scene->Materials.emplace_back();
    auto& BackWallMaterial    = Scene->Materials.back();
    BackWallMaterial.Colour    = {0.725f, 0.71f, 0.68f};    
    BackWallMaterial.Roughness = 0.1f;
    BackWallMaterial.Metallic = 0.8f;
    BackWallMaterial.MaterialType = MATERIAL_TYPE_PBR;    
    BackWallMaterial.ColourTexture = 0;
    Scene->Instances.emplace_back();
    auto& BackWallInstance    = Scene->Instances.back();
    BackWallInstance.Shape    = (int)Scene->Shapes.size() - 1;
    BackWallInstance.Material = (int)Scene->Materials.size() - 1;  
    Scene->ShapeNames.push_back("BackWall");
    Scene->InstanceNames.push_back("BackWall");
    Scene->MaterialNames.push_back("BackWall");

    Scene->Shapes.emplace_back();
    shape& RightWallShape       = Scene->Shapes.back();
    RightWallShape.Positions   = {{1, 0, -1}, {1, 0, 1}, {1, 2, 1}, {1, 2, -1}};
    RightWallShape.Triangles   = {{0, 1, 2}, {2, 3, 0}};
    RightWallShape.TexCoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    Scene->Materials.emplace_back();
    auto& RightWallMaterial    = Scene->Materials.back();
    RightWallMaterial.Colour    = {0.14f, 0.45f, 0.091f};    
    Scene->Instances.emplace_back();
    auto& RightWallInstance    = Scene->Instances.back();
    RightWallInstance.Shape    = (int)Scene->Shapes.size() - 1;
    RightWallInstance.Material = (int)Scene->Materials.size() - 1;  
    Scene->ShapeNames.push_back("RightWall");
    Scene->InstanceNames.push_back("RightWall");
    Scene->MaterialNames.push_back("RightWall");

    Scene->Shapes.emplace_back();
    shape& LeftWallShape       = Scene->Shapes.back();
    LeftWallShape.Positions   = {{-1, 0, 1}, {-1, 0, -1}, {-1, 2, -1}, {-1, 2, 1}};
    LeftWallShape.Triangles   = {{0, 1, 2}, {2, 3, 0}};
    LeftWallShape.TexCoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    Scene->Materials.emplace_back();
    auto& LeftWallMaterial    = Scene->Materials.back();
    LeftWallMaterial.Colour    = {0.63, 0.065, 0.05f};    
    LeftWallMaterial.Roughness = 1.0f;
    LeftWallMaterial.Metallic = 0.5f;
    LeftWallMaterial.MaterialType = MATERIAL_TYPE_PBR;   
    LeftWallMaterial.RoughnessTexture = 2; 
    Scene->Instances.emplace_back();
    auto& LeftWallInstance    = Scene->Instances.back();
    LeftWallInstance.Shape    = (int)Scene->Shapes.size() - 1;
    LeftWallInstance.Material = (int)Scene->Materials.size() - 1;
    Scene->ShapeNames.push_back("LeftWall");
    Scene->InstanceNames.push_back("LeftWall");
    Scene->MaterialNames.push_back("LeftWall");

    Scene->Shapes.emplace_back();
    auto& ShortBoxShape       = Scene->Shapes.back();
    ShortBoxShape.Positions   = {{0.53f, 0.6f, 0.75f}, {0.7f, 0.6f, 0.17f},
        {0.13f, 0.6f, 0.0f}, {-0.05f, 0.6f, 0.57f}, {-0.05f, 0.0f, 0.57f},
        {-0.05f, 0.6f, 0.57f}, {0.13f, 0.6f, 0.0f}, {0.13f, 0.0f, 0.0f},
        {0.53f, 0.0f, 0.75f}, {0.53f, 0.6f, 0.75f}, {-0.05f, 0.6f, 0.57f},
        {-0.05f, 0.0f, 0.57f}, {0.7f, 0.0f, 0.17f}, {0.7f, 0.6f, 0.17f},
        {0.53f, 0.6f, 0.75f}, {0.53f, 0.0f, 0.75f}, {0.13f, 0.0f, 0.0f},
        {0.13f, 0.6f, 0.0f}, {0.7f, 0.6f, 0.17f}, {0.7f, 0.0f, 0.17f},
        {0.53f, 0.0f, 0.75f}, {0.7f, 0.0f, 0.17f}, {0.13f, 0.0f, 0.0f},
        {-0.05f, 0.0f, 0.57f}};
    ShortBoxShape.Triangles   = {{0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {6, 7, 4},
        {8, 9, 10}, {10, 11, 8}, {12, 13, 14}, {14, 15, 12}, {16, 17, 18},
        {18, 19, 16}, {20, 21, 22}, {22, 23, 20}};
    Scene->Materials.emplace_back();        
    Scene->Materials.back();
    auto& ShortBoxMaterial    = Scene->Materials.back();
    ShortBoxMaterial.Colour = {0.8, 0.8, 0.8};
    Scene->Instances.emplace_back();
    auto& ShortBoxInstance    = Scene->Instances.back();
    ShortBoxInstance.Shape    = (int)Scene->Shapes.size() - 1;
    ShortBoxInstance.Material = (int)Scene->Materials.size() - 1;    
    Scene->ShapeNames.push_back("ShortBox");
    Scene->InstanceNames.push_back("ShortBox");
    Scene->MaterialNames.push_back("ShortBox");

    Scene->Shapes.emplace_back();
    auto& TallBoxShape       = Scene->Shapes.back();
    TallBoxShape.Positions   = {{-0.53f, 1.2f, 0.09f}, {0.04f, 1.2f, -0.09f},
         {-0.14f, 1.2f, -0.67f}, {-0.71f, 1.2f, -0.49f}, {-0.53f, 0.0f, 0.09f},
         {-0.53f, 1.2f, 0.09f}, {-0.71f, 1.2f, -0.49f}, {-0.71f, 0.0f, -0.49f},
         {-0.71f, 0.0f, -0.49f}, {-0.71f, 1.2f, -0.49f}, {-0.14f, 1.2f, -0.67f},
         {-0.14f, 0.0f, -0.67f}, {-0.14f, 0.0f, -0.67f}, {-0.14f, 1.2f, -0.67f},
         {0.04f, 1.2f, -0.09f}, {0.04f, 0.0f, -0.09f}, {0.04f, 0.0f, -0.09f},
         {0.04f, 1.2f, -0.09f}, {-0.53f, 1.2f, 0.09f}, {-0.53f, 0.0f, 0.09f},
         {-0.53f, 0.0f, 0.09f}, {0.04f, 0.0f, -0.09f}, {-0.14f, 0.0f, -0.67f},
         {-0.71f, 0.0f, -0.49f}};
    TallBoxShape.Triangles   = {{0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {6, 7, 4},
         {8, 9, 10}, {10, 11, 8}, {12, 13, 14}, {14, 15, 12}, {16, 17, 18},
         {18, 19, 16}, {20, 21, 22}, {22, 23, 20}};
    Scene->Materials.emplace_back();                 
    auto& TallBoxMaterial   = Scene->Materials.back();
    TallBoxMaterial.Colour = {0.8, 0.8, 0.8};
    Scene->Instances.emplace_back();
    auto& TallBoxInstance    = Scene->Instances.back();
    TallBoxInstance.Shape    = (int)Scene->Shapes.size() - 1;
    TallBoxInstance.Material = (int)Scene->Materials.size() - 1;    
    Scene->ShapeNames.push_back("TallBox");
    Scene->InstanceNames.push_back("TallBox");
    Scene->MaterialNames.push_back("TallBox");

    // LoadGLTF("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\2.0\\MetalRoughSpheres\\glTF\\MetalRoughSpheres.gltf", Scene);
    // LoadGLTF("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\2.0\\ToyCar\\glTF\\ToyCar.gltf", Scene);


    Scene->Shapes.emplace_back();
    shape &LightShape = Scene->Shapes.back();
    LightShape.Positions = {{-0.5f, 1.99f, 0.5f}, {-0.5f, 1.99f, -0.5f}, {0.5f, 1.99f, -0.5f}, {0.5f, 1.99f, 0.5f}};
    LightShape.Triangles = { {0, 1, 2}, {2, 3, 0} };
    Scene->Materials.emplace_back();
    material &LightMaterial = Scene->Materials.back();
    LightMaterial.Emission = {40, 40, 40};    
    Scene->Instances.emplace_back();
    instance &LightInstance = Scene->Instances.back(); 
    LightInstance.Shape = (int)Scene->Shapes.size()-1;
    LightInstance.Material = (int)Scene->Materials.size()-1;
    Scene->ShapeNames.push_back("Light");
    Scene->InstanceNames.push_back("Light");
    Scene->MaterialNames.push_back("Light");

    Scene->Textures.emplace_back();
    texture &Texture = Scene->Textures.back();
    Texture.SetFromFile("resources/textures/Debug.jpg", Scene->TextureWidth, Scene->TextureHeight);
    Scene->TextureNames.push_back("Debug");

    Scene->Textures.emplace_back();
    texture &Normal = Scene->Textures.back();
    Normal.SetFromFile("resources/textures/Normal.jpg", Scene->TextureWidth, Scene->TextureHeight);
    Scene->TextureNames.push_back("Normal");
    
    Scene->Textures.emplace_back();
    texture &Roughness = Scene->Textures.back();
    Roughness.SetFromFile("resources/textures/Roughness.jpg", Scene->TextureWidth, Scene->TextureHeight);
    Scene->TextureNames.push_back("Roughness");

    Scene->ReloadTextureArray();

    // Checkup
    for (size_t i = 0; i < Scene->Shapes.size(); i++)
    {
        if(Scene->Shapes[i].Normals.size() == 0)
        {
            Scene->Shapes[i].Normals.resize(Scene->Shapes[i].Positions.size());
            for (size_t j = 0; j < Scene->Shapes[i].Triangles.size(); j++)
            {
                glm::ivec3 Tri = Scene->Shapes[i].Triangles[j];
                glm::vec3 v0 = Scene->Shapes[i].Positions[Tri.x];
                glm::vec3 v1 = Scene->Shapes[i].Positions[Tri.y];
                glm::vec3 v2 = Scene->Shapes[i].Positions[Tri.z];

                glm::vec3 Normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                Scene->Shapes[i].Normals[Tri.x] = Normal;
                Scene->Shapes[i].Normals[Tri.y] = Normal;
                Scene->Shapes[i].Normals[Tri.z] = Normal;
            }
        }
        if(Scene->Shapes[i].Tangents.size() ==0)
        {
            CalculateTangents(Scene->Shapes[i]);            
        }
        if(Scene->Shapes[i].TexCoords.size() != Scene->Shapes[i].Positions.size()) Scene->Shapes[i].TexCoords.resize(Scene->Shapes[i].Positions.size());
        if(Scene->Shapes[i].Colours.size() != Scene->Shapes[i].Positions.size()) Scene->Shapes[i].Colours.resize(Scene->Shapes[i].Positions.size(), glm::vec4{1,1,1,1});
    }
    

#if API==API_GL
    Scene->CamerasBuffer = std::make_shared<bufferGL>(Scene->Cameras.size() * sizeof(camera), Scene->Cameras.data());
#elif API==API_CU
    Scene->CamerasBuffer = std::make_shared<bufferCu>(Scene->Cameras.size() * sizeof(camera), Scene->Cameras.data());
#endif    
    return Scene;
}


void CalculateTangents(shape &Shape)
{
    std::vector<glm::vec4> tan1(Shape.Positions.size(), glm::vec4(0));
    std::vector<glm::vec4> tan2(Shape.Positions.size(), glm::vec4(0));
    if (Shape.Tangents.size() != Shape.Positions.size()) Shape.Tangents.resize(Shape.Positions.size());
    if(Shape.TexCoords.size() != Shape.Positions.size()) return;

    for(uint64_t i=0; i<Shape.Triangles.size(); i++) {
        glm::vec3 v1 = Shape.Positions[Shape.Triangles[i].x];
        glm::vec3 v2 = Shape.Positions[Shape.Triangles[i].y];
        glm::vec3 v3 = Shape.Positions[Shape.Triangles[i].z];

        glm::vec2 w1 = Shape.TexCoords[Shape.Triangles[i].x];
        glm::vec2 w2 = Shape.TexCoords[Shape.Triangles[i].y];
        glm::vec2 w3 = Shape.TexCoords[Shape.Triangles[i].z];

        float x1 = v2.x - v1.x;
        float x2 = v3.x - v1.x;
        float y1 = v2.y - v1.y;
        float y2 = v3.y - v1.y;
        float z1 = v2.z - v1.z;
        float z2 = v3.z - v1.z;

        float s1 = w2.x - w1.x;
        float s2 = w3.x - w1.x;
        float t1 = w2.y - w1.y;
        float t2 = w3.y - w1.y;

        float r = 1.0F / (s1 * t2 - s2 * t1);
        glm::vec4 sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r, 0);
        glm::vec4 tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r, 0);

        tan1[Shape.Triangles[i].x] += sdir;
        tan1[Shape.Triangles[i].y] += sdir;
        tan1[Shape.Triangles[i].z] += sdir;
        
        tan2[Shape.Triangles[i].x] += tdir;
        tan2[Shape.Triangles[i].y] += tdir;
        tan2[Shape.Triangles[i].z] += tdir;

    }

    for(uint64_t i=0; i<Shape.Positions.size(); i++) { 
        glm::vec3 n = Shape.Normals[i];
        glm::vec3 t = glm::vec3(tan1[i]);

        Shape.Tangents[i] = glm::vec4(glm::normalize((t - n * glm::dot(n, t))), 1);
        
        Shape.Tangents[i].w = (glm::dot(glm::cross(n, t), glm::vec3(tan2[i])) < 0.0F) ? -1.0F : 1.0F;
    }
}

void scene::ReloadTextureArray()
{
#if API==API_GL
    TexArray = std::make_shared<textureArrayGL>();
#elif API==API_CU
    TexArray = std::make_shared<textureArrayCu>();
#endif

    TexArray->CreateTextureArray(TextureWidth, TextureHeight, Textures.size());
    for (size_t i = 0; i < Textures.size(); i++)
    {
        TexArray->LoadTextureLayer(i, Textures[i].Pixels, TextureWidth, TextureHeight);
    }
}

void texture::SetFromFile(const std::string &FileName, int Width, int Height)
{
    int NumChannels=4;
    ImageFromFile(FileName, this->Pixels, Width, Height, NumChannels);
    this->NumChannels = this->Pixels.size() / (Width * Height);
    this->Width = Width;
    this->Height = Height;
}

void texture::SetFromPixels(const std::vector<uint8_t> &PixelData, int Width, int Height)
{

}


}