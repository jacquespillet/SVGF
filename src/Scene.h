#pragma once
#include <vector>
#include <string>
#include <memory>
#include <glm/glm.hpp>

#define MATERIAL_TYPE_MATTE 0
#define MATERIAL_TYPE_PBR   1
#define MATERIAL_TYPE_VOLUMETRIC   2

namespace gpupt
{
class bufferCu;
class bufferGL;
class textureArrayGL;
class textureArrayCu;

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
    glm::ivec3 Padding;
};

struct texture
{
    int Width = 0;
    int Height = 0;
    int NumChannels = 0;
    std::vector<uint8_t> Pixels = {};

    void SetFromFile(const std::string &FileName, int Width = -1, int Height = -1);
    void SetFromPixels(const std::vector<uint8_t> &PixelData, int Width = -1, int Height = -1);
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


struct instance
{
    glm::mat4 ModelMatrix;
    int Shape = InvalidID;
    int Material = InvalidID;
    glm::mat4 GetModelMatrix() const;
};


struct shape
{

    std::vector<glm::vec3> Positions;
    std::vector<glm::vec3> Normals;
    std::vector<glm::vec2> TexCoords;
    std::vector<glm::vec4> Colours;
    std::vector<glm::vec4> Tangents;

    std::vector<glm::ivec3> Triangles;
};



struct scene
{
    
    std::vector<camera> Cameras = {};
    std::vector<instance> Instances = {};
    std::vector<shape> Shapes = {};
    std::vector<material> Materials = {};
    std::vector<texture> Textures = {};

    
    std::vector<std::string> CameraNames = {};
    std::vector<std::string> InstanceNames = {};
    std::vector<std::string> ShapeNames = {};
    std::vector<std::string> MaterialNames = {};
    std::vector<std::string> TextureNames = {};


    int TextureWidth = 512;
    int TextureHeight = 512;
    void ReloadTextureArray();

#if API==API_GL
    std::shared_ptr<bufferGL> CamerasBuffer;
    std::shared_ptr<textureArrayGL> TexArray;
#elif API==API_CU
    std::shared_ptr<bufferCu> CamerasBuffer;
    std::shared_ptr<textureArrayCu> TexArray;
#endif    
};

std::shared_ptr<scene> CreateCornellBox();
void CalculateTangents(shape &Shape);

}