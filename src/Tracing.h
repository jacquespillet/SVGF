#pragma once

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
    Params.TotalSamples = 256;
    return Params;
}


}