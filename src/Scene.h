#pragma once
#include <vector>
#include <string>
#include <memory>
#include "Tracing.h"

#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

#define MATERIAL_TYPE_MATTE 0
#define MATERIAL_TYPE_PBR   1
#define MATERIAL_TYPE_VOLUMETRIC   2
#define MATERIAL_TYPE_GLASS   3
#define MATERIAL_TYPE_SUBSURFACE   4
#define ENV_TEX_WIDTH 2048
#define TEX_WIDTH 256


namespace gpupt
{
class buffer;
class bufferGL;
class textureArrayGL;
class textureArrayCu;
struct sceneBVH;
struct lights;
struct blas;
struct vertexBuffer;

static const int InvalidID = -1;

struct camera
{
    glm::mat4 Frame  = glm::mat4(1);
    glm::mat4 PreviousFrame = glm::mat4(1);

    float FOV = 60.0f;
    float Aspect = 1.0f;
    void CalculateProj();
    void SetAspect(float Aspect);
    glm::mat4 ProjectionMatrix = glm::perspective(glm::radians(60.0f), 1.0f, 0.001f, 1000.0f);

    int Controlled = true;
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
    void ToFile(std::ofstream &Stream);
    void FromFile(std::ifstream &Stream);
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

    std::shared_ptr<blas> BVH;

    void PreProcess();
    void CalculateTangents();
    void ToFile(std::ofstream &Stream);
    void FromFile(std::ifstream &Stream);
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


    int TextureWidth = TEX_WIDTH;
    int TextureHeight = TEX_WIDTH;

    int EnvTextureWidth = ENV_TEX_WIDTH;
    int EnvTextureHeight = ENV_TEX_WIDTH/2;

    std::shared_ptr<sceneBVH> BVH;
    std::shared_ptr<lights> Lights;

    std::shared_ptr<vertexBuffer> VertexBuffer;
    
    scene();
    void ReloadTextureArray();
    void UploadMaterial(int MaterialInx);
    void PreProcess();
    void CheckNames();
    void UpdateLights();
    void RemoveInstance(int InstanceInx);

    void ToFile(std::string FileName);
    void FromFile(std::string FileName);
    void Clear();
    void ClearInstances();

    void CalculateInstanceTransform(int InstanceInx);
    std::shared_ptr<textureArrayCu> TexArray;
    std::shared_ptr<textureArrayCu> EnvTexArray;
    std::shared_ptr<buffer> CamerasBuffer;
    std::shared_ptr<buffer> EnvironmentsBuffer;
    
    // TODO: Use single buffer here and use cuda interrop
    std::shared_ptr<buffer> MaterialBuffer;
    std::shared_ptr<bufferGL> MaterialBufferGL;
};

void CalculateTangents(shape &Shape);

}