#pragma once
#include <vector>
#include <string>
#include <memory>
#include <glm/glm.hpp>
#include "Tracing.h"

#define MATERIAL_TYPE_MATTE 0
#define MATERIAL_TYPE_PBR   1
#define MATERIAL_TYPE_VOLUMETRIC   2
#define MATERIAL_TYPE_GLASS   3
#define MATERIAL_TYPE_SUBSURFACE   4
#define ENV_TEX_WIDTH 2048


namespace gpupt
{
class bufferCu;
class bufferGL;
class textureArrayGL;
class textureArrayCu;
struct sceneBVH;
struct lights;
struct blas;

static const int InvalidID = -1;

struct camera
{
    glm::mat4 Frame  = glm::mat4(1);
    
    float Lens = 0.050f;
    float Film = 0.036f;
    float Aspect = 1.5f;
    float Focus = 1000;
    
    glm::vec3 Padding0;
    float Aperture = 0;
      
    int Orthographic = 0;
    int Controlled = true;
    glm::ivec2 Padding;
};

struct texture
{
    int Width = 0;
    int Height = 0;
    int NumChannels = 0;
    std::vector<uint8_t> Pixels = {};
    std::vector<float> PixelsF = {};

    void SetFromFile(const std::string &FileName, int Width = -1, int Height = -1);
    void SetFromPixels(const std::vector<uint8_t> &PixelData, int Width = -1, int Height = -1);
    glm::vec4 Sample(glm::ivec2 Coords);
    glm::vec4 SampleF(glm::ivec2 Coords);
};


struct material
{
    glm::vec3 Emission = {};
    float Roughness = 0;
    
    glm::vec3 Colour = {};
    float Metallic = 0;
    
    float Padding;
    float Anisotropy = 0.0f;
    float MaterialType = 0;
    float Opacity = 1;

    glm::vec3 ScatteringColour = {};
    float TransmissionDepth = 0.01f;

    int EmissionTexture = InvalidID;
    int ColourTexture = InvalidID;
    int RoughnessTexture = InvalidID;
    int NormalTexture = InvalidID;
};


struct aabb
{
    glm::vec3 Min =glm::vec3(1e30f);
    float pad0;
    glm::vec3 Max =glm::vec3(-1e30f);
    float pad1;
    float Area();
    void Grow(glm::vec3 Position);
    void Grow(aabb &AABB);
};


struct instance
{
    glm::mat4 Transform;
    glm::mat4 InverseTransform;
    glm::mat4 NormalTransform;
    aabb Bounds;

    uint32_t Shape;
    uint32_t Index=0;
    uint32_t Material;
    uint32_t Selected=0;  
};

struct triangle
{
    glm::vec4 PositionUvX0;
    glm::vec4 PositionUvX1;
    glm::vec4 PositionUvX2;
    
    glm::vec4 NormalUvY0; 
    glm::vec4 NormalUvY1; 
    glm::vec4 NormalUvY2;
    
    glm::vec4 Tangent0;
    glm::vec4 Tangent1;  
    glm::vec4 Tangent2;
    
    glm::vec3 Centroid;
    float padding3; 
};

struct shape
{

    std::vector<glm::vec3> PositionsTmp;
    std::vector<glm::vec3> NormalsTmp;
    std::vector<glm::vec2> TexCoordsTmp;
    std::vector<glm::vec4> TangentsTmp;
    std::vector<glm::ivec3> IndicesTmp;

    std::vector<triangle> Triangles;

    glm::vec3 Centroid;

    blas *BVH;

    void PreProcess();
    void CalculateTangents();
};

struct environment
{
    glm::mat4 Transform;

    glm::vec3 Emission;
    float pad0;

    glm::ivec3 pad1;
    int EmissionTexture = InvalidID;
};

struct scene
{
    
    std::vector<camera> Cameras = {};
    std::vector<instance> Instances = {};
    std::vector<shape> Shapes = {};
    std::vector<material> Materials = {};
    std::vector<texture> Textures = {};
    std::vector<texture> EnvTextures = {};
    std::vector<environment> Environments = {};

    
    std::vector<std::string> CameraNames = {};
    std::vector<std::string> InstanceNames = {};
    std::vector<std::string> ShapeNames = {};
    std::vector<std::string> MaterialNames = {};
    std::vector<std::string> TextureNames = {};
    std::vector<std::string> EnvTextureNames = {};
    std::vector<std::string> EnvironmentNames = {};


    int TextureWidth = 512;
    int TextureHeight = 512;

    int EnvTextureWidth = ENV_TEX_WIDTH;
    int EnvTextureHeight = ENV_TEX_WIDTH/2;

    std::shared_ptr<sceneBVH> BVH;
    std::shared_ptr<lights> Lights;
    
    scene();
    void ReloadTextureArray();
    void UploadMaterial(int MaterialInx);
    void PreProcess();
    void CheckNames();
    void UpdateLights();
    void RemoveInstance(int InstanceInx);

    void CalculateInstanceTransform(int InstanceInx);
#if API==API_GL
    std::shared_ptr<bufferGL> CamerasBuffer;
    std::shared_ptr<bufferGL> EnvironmentsBuffer;
    std::shared_ptr<textureArrayGL> TexArray;
    std::shared_ptr<textureArrayGL> EnvTexArray;
    std::shared_ptr<bufferGL> MaterialBuffer;
#elif API==API_CU
    std::shared_ptr<bufferCu> EnvironmentsBuffer;
    std::shared_ptr<bufferCu> CamerasBuffer;
    std::shared_ptr<textureArrayCu> TexArray;
    std::shared_ptr<textureArrayCu> EnvTexArray;
    std::shared_ptr<bufferCu> MaterialBuffer;
#endif    
};

std::shared_ptr<scene> CreateCornellBox();
void CalculateTangents(shape &Shape);

}