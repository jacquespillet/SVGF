#pragma once
#include <glm/vec3.hpp>

namespace gpupt
{

struct tracingParameters
{
    int CurrentSample;
    int TotalSamples;
    int Batch;
    int Pad0;    
};

inline tracingParameters GetTracingParameters()
{
    tracingParameters Params;
    Params.CurrentSample = 0;
    Params.Batch = 1;
    Params.TotalSamples = 4096;
    return Params;
}

struct materialPoint
{
    glm::vec3 Emission;
    glm::vec3 Colour;
    int MaterialType;
    float Roughness, Metallic;
};




}