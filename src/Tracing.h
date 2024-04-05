#pragma once
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <memory>

#define MAX_LIGHTS 32
#define MAX_CDF 512

namespace gpupt
{
struct scene;

struct tracingParameters
{
    int CurrentSample;
    int TotalSamples;
    int Batch;
    int Bounces;

    glm::vec3 Pad;    
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
    return Params;
}

struct materialPoint
{
    glm::vec3 Emission;
    glm::vec3 Colour;
    int MaterialType;
    float Roughness, Metallic, Opacity;
};

struct light 
{
    int Instance = -1;
    int CDFCount = 0;
    glm::ivec2 Pad0;

    float CDF[MAX_CDF];
};

struct lights
{
    glm::uvec3 Pad0;
    uint32_t LightsCount = 0;
    
    light Lights[MAX_LIGHTS];
};

lights GetLights(std::shared_ptr<scene> Scene, tracingParameters &Parameters);

}