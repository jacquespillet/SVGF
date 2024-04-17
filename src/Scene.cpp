#include "Scene.h"
#include <glm/ext.hpp>

#include "BufferCu.cuh"
#include "BufferGL.h"
#include "App.h"
#include "GLTFLoader.h"
#include "AssimpLoader.h"
#include "ImageLoader.h"
#include "TextureArrayGL.h"
#include "TextureArrayCu.cuh"
#include "BVH.h"
#include <unordered_map>

namespace gpupt
{


glm::mat4 instance::GetModelMatrix() const
{
    return ModelMatrix;    
}

void EnsureUnicity(std::vector<std::string> &Names, std::string DefaultName)
{
    std::unordered_map<std::string, int> Counts;

    for (std::string& Name : Names) {
        if(Name == "") Name = DefaultName;
        Counts[Name]++;
    }

    for (std::string& Name : Names) {
        if (Counts[Name] > 1) {
            Name = Name + "_" + std::to_string(Counts[Name]--);
        }
    }
}

scene::scene()
{
    glm::uvec2 RenderSize = application::GetSize();


    this->Cameras.emplace_back();
    camera &Camera = this->Cameras.back();
    Camera.Lens = 0.035f;
    Camera.Aperture = 0.0f;
    Camera.Focus = 3.9f;
    Camera.Film = 0.024f;
    Camera.Aspect = (float)RenderSize.x / (float)RenderSize.y;
    Camera.Controlled = 1;  
    this->CameraNames.push_back("Main Camera");
    
    LoadAssimpShapeOnly("resources/models/BaseShapes/Cube/Cube.obj", this, 0);
    LoadAssimpShapeOnly("resources/models/BaseShapes/Cone/Cone.obj", this, 0);
    LoadAssimpShapeOnly("resources/models/BaseShapes/Cylinder/Cylinder.obj", this, 0);
    LoadAssimpShapeOnly("resources/models/BaseShapes/Sphere/Sphere.obj", this, 0);
    LoadAssimpShapeOnly("resources/models/BaseShapes/Torus/Torus.obj", this, 0);
    LoadAssimpShapeOnly("resources/models/BaseShapes/Plane/Plane.obj", this, 0);

    this->Materials.emplace_back();
    material &BaseMaterial = this->Materials.back(); 
    BaseMaterial.Colour = {0.725f, 0.71f, 0.68f};

    
    this->Instances.emplace_back();
    instance &FloorInstance = this->Instances.back();
    FloorInstance.Shape = (int)this->Shapes.size()-1;
    FloorInstance.Material = (int)this->Materials.size()-1;
    FloorInstance.ModelMatrix = glm::scale(glm::vec3(4, 4, 4));
    this->InstanceNames.push_back("Floor");
    this->MaterialNames.push_back("Floor");

    this->Materials.emplace_back();
    material &LightMaterial = this->Materials.back();
    LightMaterial.Emission = {40, 40, 40};    
    
    this->Instances.emplace_back();
    instance &LightInstance = this->Instances.back(); 
    LightInstance.Shape = (int)this->Shapes.size()-1;
    LightInstance.Material = (int)this->Materials.size()-1;
    LightInstance.ModelMatrix = glm::translate(glm::vec3(0, 2, 0));
    this->InstanceNames.push_back("Light");
    this->MaterialNames.push_back("Light");
}

void scene::CheckNames()
{
    EnsureUnicity(this->ShapeNames, "Shape");
    EnsureUnicity(this->InstanceNames, "Instance");
    EnsureUnicity(this->TextureNames, "Texture");
    EnsureUnicity(this->EnvironmentNames, "Environment");
    EnsureUnicity(this->CameraNames, "Camera");
    EnsureUnicity(this->EnvTextureNames, "EnvTexture");
    EnsureUnicity(this->MaterialNames, "Material");
}

void scene::UpdateLights()
{
    this->Lights = GetLights(this);
    if(this->Lights->Lights.size()>0)
    {
    // TODO: Only recreate the buffer if it's bigger.
    #if API==API_CU
        this->LightsBuffer = std::make_shared<bufferCu>(sizeof(light) * this->Lights->Lights.size(), this->Lights->Lights.data());
        this->LightsCDFBuffer = std::make_shared<bufferCu>(sizeof(float) * this->Lights->LightsCDF.size(), this->Lights->LightsCDF.data());
    #else
        this->LightsBuffer = std::make_shared<bufferGL>(sizeof(light) * this->Lights->Lights.size(), this->Lights->Lights.data());
        this->LightsCDFBuffer = std::make_shared<bufferGL>(sizeof(float) * this->Lights->LightsCDF.size(), this->Lights->LightsCDF.data());
    #endif
    };
}

void scene::UploadMaterial(int MaterialInx)
{
    this->MaterialBuffer->updateData((size_t)MaterialInx * sizeof(material), (void*)&this->Materials[MaterialInx], sizeof(material));
}

void scene::PreProcess()
{
    this->ReloadTextureArray();

    // Ensure name unicity
    CheckNames();

    // Checkup
    for (size_t i = 0; i < this->Shapes.size(); i++)
    {
        if(this->Shapes[i].Normals.size() == 0)
        {
            this->Shapes[i].Normals.resize(this->Shapes[i].Positions.size());
            for (size_t j = 0; j < this->Shapes[i].Triangles.size(); j++)
            {
                glm::ivec3 Tri = this->Shapes[i].Triangles[j];
                glm::vec3 v0 = this->Shapes[i].Positions[Tri.x];
                glm::vec3 v1 = this->Shapes[i].Positions[Tri.y];
                glm::vec3 v2 = this->Shapes[i].Positions[Tri.z];

                glm::vec3 Normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                this->Shapes[i].Normals[Tri.x] = Normal;
                this->Shapes[i].Normals[Tri.y] = Normal;
                this->Shapes[i].Normals[Tri.z] = Normal;
            }
        }
        if(this->Shapes[i].Tangents.size() ==0)
        {
            CalculateTangents(this->Shapes[i]);            
        }
        if(this->Shapes[i].TexCoords.size() != this->Shapes[i].Positions.size()) this->Shapes[i].TexCoords.resize(this->Shapes[i].Positions.size());
        if(this->Shapes[i].Colours.size() != this->Shapes[i].Positions.size()) this->Shapes[i].Colours.resize(this->Shapes[i].Positions.size(), glm::vec4{1,1,1,1});
        
        double InverseSize = 1.0 / double(this->Shapes[i].Positions.size()); 
        glm::dvec3 Centroid;
        for(size_t j=0; j < this->Shapes[i].Positions.size(); j++)
        {
            Centroid += glm::dvec3(this->Shapes[i].Positions[j]) * InverseSize;
        }
        this->Shapes[i].Centroid = glm::vec3(Centroid);
    }
    

    BVH = CreateBVH(this); 
    Lights = GetLights(this);
#if API==API_GL
    this->CamerasBuffer = std::make_shared<bufferGL>(this->Cameras.size() * sizeof(camera), this->Cameras.data());
    this->EnvironmentsBuffer = std::make_shared<bufferGL>(this->Environments.size() * sizeof(camera), this->Environments.data());
    this->LightsBuffer = std::make_shared<bufferGL>(sizeof(light) * Lights->Lights.size(), Lights->Lights.data());
    this->LightsCDFBuffer = std::make_shared<bufferGL>(sizeof(float) * Lights->LightsCDF.size(), Lights->LightsCDF.data());
    this->MaterialBuffer = std::make_shared<bufferGL>(sizeof(material) * Materials.size(), Materials.data());
#elif API==API_CU
    this->CamerasBuffer = std::make_shared<bufferCu>(this->Cameras.size() * sizeof(camera), this->Cameras.data());
    this->EnvironmentsBuffer = std::make_shared<bufferCu>(this->Environments.size() * sizeof(environment), this->Environments.data());
    this->LightsBuffer = std::make_shared<bufferCu>(sizeof(light) * Lights->Lights.size(), Lights->Lights.data());
    this->LightsCDFBuffer = std::make_shared<bufferCu>(sizeof(float) * Lights->LightsCDF.size(), Lights->LightsCDF.data());
    this->MaterialBuffer = std::make_shared<bufferCu>(sizeof(material) * Materials.size(), Materials.data());
#endif    
}

std::shared_ptr<scene> CreateCornellBox()
{
    std::shared_ptr<scene> Scene = std::make_shared<scene>();



#if 1


    // Scene->Shapes.emplace_back();
    // shape& CeilingShape       = Scene->Shapes.back();
    // CeilingShape.Positions   = {{-1, 2, 1}, {-1, 2, -1}, {1, 2, -1}, {1, 2, 1}};
    // CeilingShape.Triangles   = {{0, 1, 2}, {2, 3, 0}};
    // CeilingShape.TexCoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    // Scene->Materials.emplace_back();
    // auto& CeilingMaterial    = Scene->Materials.back();
    // CeilingMaterial.Colour    = {0.725f, 0.71f, 0.68f};    
    // Scene->Instances.emplace_back();
    // auto& CeilingInstance    = Scene->Instances.back();
    // CeilingInstance.Shape    = (int)Scene->Shapes.size() - 1;
    // CeilingInstance.Material = (int)Scene->Materials.size()-1;
    // Scene->ShapeNames.push_back("Ceiling");
    // Scene->InstanceNames.push_back("Ceiling");
    // Scene->MaterialNames.push_back("Ceiling");

    // Scene->Shapes.emplace_back();
    // shape& BackWallShape       = Scene->Shapes.back();
    // BackWallShape.Positions   = {{-1, 0, -1}, {1, 0, -1}, {1, 2, -1}, {-1, 2, -1}};
    // BackWallShape.Triangles   = {{0, 1, 2}, {2, 3, 0}};
    // BackWallShape.TexCoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    // Scene->Materials.emplace_back();
    // auto& BackWallMaterial    = Scene->Materials.back();
    // BackWallMaterial.Colour    = {0.725f, 0.71f, 0.68f};    
    // BackWallMaterial.Roughness = 0.1f;
    // BackWallMaterial.Metallic = 0.8f;
    // BackWallMaterial.MaterialType = MATERIAL_TYPE_PBR;    
    // Scene->Instances.emplace_back();
    // auto& BackWallInstance    = Scene->Instances.back();
    // BackWallInstance.Shape    = (int)Scene->Shapes.size() - 1;
    // BackWallInstance.Material = (int)Scene->Materials.size() - 1;  
    // Scene->ShapeNames.push_back("BackWall");
    // Scene->InstanceNames.push_back("BackWall");
    // Scene->MaterialNames.push_back("BackWall");

    // Scene->Shapes.emplace_back();
    // shape& RightWallShape       = Scene->Shapes.back();
    // RightWallShape.Positions   = {{1, 0, -1}, {1, 0, 1}, {1, 2, 1}, {1, 2, -1}};
    // RightWallShape.Triangles   = {{0, 1, 2}, {2, 3, 0}};
    // RightWallShape.TexCoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    // Scene->Materials.emplace_back();
    // auto& RightWallMaterial    = Scene->Materials.back();
    // RightWallMaterial.Colour    = {0.14f, 0.45f, 0.091f};    
    // Scene->Instances.emplace_back();
    // auto& RightWallInstance    = Scene->Instances.back();
    // RightWallInstance.Shape    = (int)Scene->Shapes.size() - 1;
    // RightWallInstance.Material = (int)Scene->Materials.size() - 1;  
    // Scene->ShapeNames.push_back("RightWall");
    // Scene->InstanceNames.push_back("RightWall");
    // Scene->MaterialNames.push_back("RightWall");

    // Scene->Shapes.emplace_back();
    // shape& LeftWallShape       = Scene->Shapes.back();
    // LeftWallShape.Positions   = {{-1, 0, 1}, {-1, 0, -1}, {-1, 2, -1}, {-1, 2, 1}};
    // LeftWallShape.Triangles   = {{0, 1, 2}, {2, 3, 0}};
    // LeftWallShape.TexCoords = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    // Scene->Materials.emplace_back();
    // auto& LeftWallMaterial    = Scene->Materials.back();
    // LeftWallMaterial.Colour    = {0.63, 0.065, 0.05f};    
    // LeftWallMaterial.Roughness = 1.0f;
    // LeftWallMaterial.Metallic = 0.5f;
    // LeftWallMaterial.MaterialType = MATERIAL_TYPE_PBR;   
    // Scene->Instances.emplace_back();
    // auto& LeftWallInstance    = Scene->Instances.back();
    // LeftWallInstance.Shape    = (int)Scene->Shapes.size() - 1;
    // LeftWallInstance.Material = (int)Scene->Materials.size() - 1;
    // Scene->ShapeNames.push_back("LeftWall");
    // Scene->InstanceNames.push_back("LeftWall");
    // Scene->MaterialNames.push_back("LeftWall");

    // Scene->Shapes.emplace_back();
    // auto& ShortBoxShape       = Scene->Shapes.back();
    // ShortBoxShape.Positions   = {{0.53f, 0.6f, 0.75f}, {0.7f, 0.6f, 0.17f},
    //     {0.13f, 0.6f, 0.0f}, {-0.05f, 0.6f, 0.57f}, {-0.05f, 0.0f, 0.57f},
    //     {-0.05f, 0.6f, 0.57f}, {0.13f, 0.6f, 0.0f}, {0.13f, 0.0f, 0.0f},
    //     {0.53f, 0.0f, 0.75f}, {0.53f, 0.6f, 0.75f}, {-0.05f, 0.6f, 0.57f},
    //     {-0.05f, 0.0f, 0.57f}, {0.7f, 0.0f, 0.17f}, {0.7f, 0.6f, 0.17f},
    //     {0.53f, 0.6f, 0.75f}, {0.53f, 0.0f, 0.75f}, {0.13f, 0.0f, 0.0f},
    //     {0.13f, 0.6f, 0.0f}, {0.7f, 0.6f, 0.17f}, {0.7f, 0.0f, 0.17f},
    //     {0.53f, 0.0f, 0.75f}, {0.7f, 0.0f, 0.17f}, {0.13f, 0.0f, 0.0f},
    //     {-0.05f, 0.0f, 0.57f}};
    // ShortBoxShape.Triangles   = {{0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {6, 7, 4},
    //     {8, 9, 10}, {10, 11, 8}, {12, 13, 14}, {14, 15, 12}, {16, 17, 18},
    //     {18, 19, 16}, {20, 21, 22}, {22, 23, 20}};
    // Scene->Materials.emplace_back();        
    // Scene->Materials.back();
    // auto& ShortBoxMaterial    = Scene->Materials.back();
    // ShortBoxMaterial.Colour = {0.8, 0.8, 0.8};
    // ShortBoxMaterial.MaterialType = MATERIAL_TYPE_GLASS;
    // ShortBoxMaterial.ScatteringColour = {0.9, 0.2, 0.4};
    // ShortBoxMaterial.Roughness = 0.1f;
    // Scene->Instances.emplace_back();
    // auto& ShortBoxInstance    = Scene->Instances.back();
    // ShortBoxInstance.Shape    = (int)Scene->Shapes.size() - 1;
    // ShortBoxInstance.Material = (int)Scene->Materials.size() - 1;    
    // Scene->ShapeNames.push_back("ShortBox");
    // Scene->InstanceNames.push_back("ShortBox");
    // Scene->MaterialNames.push_back("ShortBox");

    // // Scene->Shapes.emplace_back();
    // // auto& TallBoxShape       = Scene->Shapes.back();
    // // TallBoxShape.Positions   = {{-0.53f, 1.2f, 0.09f}, {0.04f, 1.2f, -0.09f},
    // //      {-0.14f, 1.2f, -0.67f}, {-0.71f, 1.2f, -0.49f}, {-0.53f, 0.0f, 0.09f},
    // //      {-0.53f, 1.2f, 0.09f}, {-0.71f, 1.2f, -0.49f}, {-0.71f, 0.0f, -0.49f},
    // //      {-0.71f, 0.0f, -0.49f}, {-0.71f, 1.2f, -0.49f}, {-0.14f, 1.2f, -0.67f},
    // //      {-0.14f, 0.0f, -0.67f}, {-0.14f, 0.0f, -0.67f}, {-0.14f, 1.2f, -0.67f},
    // //      {0.04f, 1.2f, -0.09f}, {0.04f, 0.0f, -0.09f}, {0.04f, 0.0f, -0.09f},
    // //      {0.04f, 1.2f, -0.09f}, {-0.53f, 1.2f, 0.09f}, {-0.53f, 0.0f, 0.09f},
    // //      {-0.53f, 0.0f, 0.09f}, {0.04f, 0.0f, -0.09f}, {-0.14f, 0.0f, -0.67f},
    // //      {-0.71f, 0.0f, -0.49f}};
    // // TallBoxShape.Triangles   = {{0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {6, 7, 4},
    // //      {8, 9, 10}, {10, 11, 8}, {12, 13, 14}, {14, 15, 12}, {16, 17, 18},
    // //      {18, 19, 16}, {20, 21, 22}, {22, 23, 20}};
    // // Scene->Materials.emplace_back();                 
    // // auto& TallBoxMaterial   = Scene->Materials.back();
    // // TallBoxMaterial.Colour = {0.8, 0.8, 0.8};
    // // Scene->Instances.emplace_back();
    // // auto& TallBoxInstance    = Scene->Instances.back();
    // // TallBoxInstance.Shape    = (int)Scene->Shapes.size() - 1;
    // // TallBoxInstance.Material = (int)Scene->Materials.size() - 1;    
    // // Scene->ShapeNames.push_back("TallBox");
    // // Scene->InstanceNames.push_back("TallBox");
    // // Scene->MaterialNames.push_back("TallBox");

    // // LoadGLTF("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\2.0\\MetalRoughSpheres\\glTF\\MetalRoughSpheres.gltf", Scene);
    // LoadGLTFShapeOnly("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\2.0\\Suzanne\\glTF\\Suzanne.gltf", Scene, 0);
    // Scene->Materials.emplace_back();        
    // Scene->Materials.back();
    // auto& DuckMaterial    = Scene->Materials.back();
    // DuckMaterial.Colour = {0.8, 0.8, 0.8};
    // DuckMaterial.MaterialType = MATERIAL_TYPE_GLASS;
    // DuckMaterial.ScatteringColour = {0.9, 0.2, 0.4};
    // DuckMaterial.Roughness = 0.1f;
    // Scene->Instances.emplace_back();
    // auto& DuckInstance    = Scene->Instances.back();
    // DuckInstance.Shape    = (int)Scene->Shapes.size() - 1;
    // DuckInstance.Material = (int)Scene->Materials.size() - 1;    
    // DuckInstance.ModelMatrix = glm::scale(glm::vec3(0.25));
    // Scene->InstanceNames.push_back("Duck");
    // Scene->MaterialNames.push_back("Duck");


    // LoadAssimp("C:/Users/jacqu/Documents/Boulot/Models/BaseShapes/Plane/Plane.obj", Scene, true);

    // Scene->Textures.emplace_back();
    // texture &Texture = Scene->Textures.back();
    // Texture.SetFromFile("resources/textures/Debug.jpg", Scene->TextureWidth, Scene->TextureHeight);
    // Scene->TextureNames.push_back("Debug");

    // Scene->Textures.emplace_back();
    // texture &Normal = Scene->Textures.back();
    // Normal.SetFromFile("resources/textures/Normal.jpg", Scene->TextureWidth, Scene->TextureHeight);
    // Scene->TextureNames.push_back("Normal");
    
    // Scene->Textures.emplace_back();
    // texture &Roughness = Scene->Textures.back();
    // Roughness.SetFromFile("resources/textures/Roughness.jpg", Scene->TextureWidth, Scene->TextureHeight);
    // Scene->TextureNames.push_back("Roughness");

#else

    // LoadGLTF("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\2.0\\Sponza\\glTF\\Sponza.gltf", Scene, true);
    LoadAssimp("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\breakfast_room\\breakfast_room.obj", Scene, true);
    // LoadAssimp("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\breakfast_room\\breakfast_room.obj", Scene, true);
    // LoadAssimp("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\salle_de_bain\\salle_de_bain.obj", Scene, true);
    // LoadAssimp("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\bedroom\\iscv2.obj", Scene, true);
    // LoadAssimp("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\fireplace_room\\fireplace_room.obj", Scene, true);
    // LoadAssimp("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\gallery\\gallery.obj", Scene, true);
    // LoadAssimp("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\rungholt\\house.obj", Scene, true);
    // LoadAssimp("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\vokselia_spawn\\vokselia_spawn.obj", Scene, true);
    // LoadGLTF("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\gltf\\mech_drone\\scene.gltf", Scene, true);  
    // LoadGLTF("C:\\Users\\jacqu\\Documents\\Boulot\\Models\\gltf\\spaceship_corridor\\scene.gltf", Scene, true);
    
    Scene->EnvTextures.emplace_back();
    texture &SkyTex = Scene->EnvTextures.back();
    SkyTex.SetFromFile("resources/textures/Sky.hdr", Scene->EnvTextureWidth, Scene->EnvTextureHeight);
    Scene->EnvTextureNames.push_back("Sky");    

    Scene->Environments.emplace_back();
    Scene->EnvironmentNames.push_back("Sky");
    environment &Sky = Scene->Environments.back();
    Sky.Emission = {1,1,1};
    Sky.EmissionTexture = 0;
    Sky.Transform = glm::rotate(glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));


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
    assert(8192 % TextureWidth==0);
    assert(8192 % TextureHeight==0);
    assert(8192 % EnvTextureWidth==0);
    assert(8192 % EnvTextureHeight==0);

#if API==API_GL
    TexArray = std::make_shared<textureArrayGL>();
    EnvTexArray = std::make_shared<textureArrayGL>();
#elif API==API_CU
    TexArray = std::make_shared<textureArrayCu>();
    EnvTexArray = std::make_shared<textureArrayCu>();
#endif

    TexArray->CreateTextureArray(TextureWidth, TextureHeight, Textures.size());
    for (size_t i = 0; i < Textures.size(); i++)
    {
        TexArray->LoadTextureLayer(i, Textures[i].Pixels, TextureWidth, TextureHeight);
    }

    EnvTexArray->CreateTextureArray(EnvTextureWidth, EnvTextureHeight, EnvTextures.size(), true);
    for (size_t i = 0; i < EnvTextures.size(); i++)
    {
        EnvTexArray->LoadTextureLayer(i, EnvTextures[i].PixelsF, EnvTextureWidth, EnvTextureHeight);
    }
}

glm::vec4 texture::Sample(glm::ivec2 Coords)
{
    glm::vec4 Res;
    Res.x = (float)this->Pixels[(Coords.y * Width + Coords.x) * 4 + 0] * 255.0f;
    Res.y = (float)this->Pixels[(Coords.y * Width + Coords.x) * 4 + 1] * 255.0f;
    Res.z = (float)this->Pixels[(Coords.y * Width + Coords.x) * 4 + 2] * 255.0f;
    Res.w = (float)this->Pixels[(Coords.y * Width + Coords.x) * 4 + 3] * 255.0f;
    return Res;
}

glm::vec4 texture::SampleF(glm::ivec2 Coords)
{
    glm::vec4 Res;
    Res.x = this->PixelsF[(Coords.y * Width + Coords.x) * 4 + 0];
    Res.y = this->PixelsF[(Coords.y * Width + Coords.x) * 4 + 1];
    Res.z = this->PixelsF[(Coords.y * Width + Coords.x) * 4 + 2];
    Res.w = this->PixelsF[(Coords.y * Width + Coords.x) * 4 + 3];
    return Res;
}

void texture::SetFromFile(const std::string &FileName, int Width, int Height)
{
    if(IsHDR(FileName))
    {
        int NumChannels=4;
        ImageFromFile(FileName, this->PixelsF, Width, Height, NumChannels);
        this->NumChannels = this->Pixels.size() / (Width * Height);
        this->Width = Width;
        this->Height = Height;
    }
    else
    {
        int NumChannels=4;
        ImageFromFile(FileName, this->Pixels, Width, Height, NumChannels);
        this->NumChannels = this->Pixels.size() / (Width * Height);
        this->Width = Width;
        this->Height = Height;
    }
}

void texture::SetFromPixels(const std::vector<uint8_t> &PixelData, int Width, int Height)
{

}


}