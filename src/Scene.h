#pragma once
#include <vector>
#include <string>
#include <memory>
#include <glm/glm.hpp>

#define MATERIAL_TYPE_MATTE 0
#define MATERIAL_TYPE_PBR   1

namespace gpupt
{
class bufferCu;
class bufferGL;

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

struct material
{
    glm::vec3 Emission = {};
    float Roughness = 0;
    
    glm::vec3 Colour = {};
    float Metallic = 0;
    
    int MaterialType = 0;
    glm::ivec3 Padding2;
};


struct instance
{
    glm::vec3 Position;
    glm::vec3 Rotation;
    glm::vec3 Scale = glm::vec3(1);
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
    
    std::vector<std::string> CameraNames = {};
    std::vector<std::string> InstanceNames = {};
    std::vector<std::string> ShapeNames = {};
    std::vector<std::string> MaterialNames = {};


#if API==API_GL
    std::shared_ptr<bufferGL> CamerasBuffer;
#elif API==API_CU
    std::shared_ptr<bufferCu> CamerasBuffer;
#endif    
};

std::shared_ptr<scene> CreateCornellBox();
void CalculateTangents(shape &Shape);

}