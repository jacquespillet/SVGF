#pragma once
#include <memory>
#include "Scene.h"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace gpupt
{
#define SAMPLING_MODE_BSDF 0
#define SAMPLING_MODE_LIGHT 1
#define SAMPLING_MODE_BOTH 2
#define SAMPLING_MODE_MIS 3

struct scene;
class buffer;

struct tracingParameters
{
    int Batch;
    int Bounces;

    float CurrentCamera;    
    float Clamp;

    int SamplingMode;
};

inline tracingParameters GetTracingParameters()
{
    tracingParameters Params;
    Params.Batch = 1;
    Params.Bounces = 3;
    Params.Clamp = 10;
    Params.CurrentCamera=0;
    Params.SamplingMode = SAMPLING_MODE_LIGHT;
    return Params;
}

struct materialPoint
{
    glm::vec3 Emission;
    glm::vec3 Colour;
    int MaterialType;
    float Roughness, Metallic, Opacity;
    
    glm::vec3 ScatteringColour = {};
    float TransmissionDepth = 0.01f;
    glm::vec3 Density;
    float Anisotropy;
};

struct light 
{
    int Instance = -1;
    int CDFCount = 0;
    int Environment = -1;
    int CDFStart = 0;
};

struct lights
{
    std::vector<light> Lights;
    std::vector<float> LightsCDF;
    void Build(scene *Scene);
    void RemoveInstance(scene *Scene, int InstanceInx);
    void RecreateBuffers();
    light &AddLight();
    std::shared_ptr<buffer> LightsBuffer;
    std::shared_ptr<buffer> LightsCDFBuffer;
};


}