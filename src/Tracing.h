#pragma once
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <memory>
#include "Scene.h"


namespace gpupt
{
struct scene;
class bufferCu;
class bufferGL;

struct tracingParameters
{
    int CurrentSample;
    int TotalSamples;
    int Batch;
    int Bounces;

    glm::vec2 Pad;
    float CurrentCamera;    
    float Clamp;
};

inline tracingParameters GetTracingParameters()
{
    tracingParameters Params;
    Params.CurrentSample = 0;
    Params.Batch = 1;
    Params.TotalSamples = 4096;
    Params.Bounces = 5;
    Params.Clamp = 10;
    Params.CurrentCamera=0;
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
#if API==API_GL
    std::shared_ptr<bufferGL> LightsBuffer;
    std::shared_ptr<bufferGL> LightsCDFBuffer;
#elif API==API_CU
    std::shared_ptr<bufferCu> LightsBuffer;
    std::shared_ptr<bufferCu> LightsCDFBuffer;
#endif        

};


}