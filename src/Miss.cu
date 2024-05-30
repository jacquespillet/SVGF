
#include <optix.h>
#include <optix_device.h>

#include <cuda_fp16.h>
#include "Common.cuh"

namespace pathtracing
{
using namespace glm;
using namespace commonCu;

extern "C" __global__ void __miss__ms() {
    // OptixRayPayload* payload = reinterpret_cast<OptixRayPayload*>(optixGetPayload_0());
    // payload->color = params.background_color;
    // payload->done = true;
    // uint32_t *Done = reinterpret_cast<uint32_t*>(optixGetPayload_0());
    // *Done = false;
    // optixSetPayload_0(0);

    float Distance = MAX_LENGTH;
    optixSetPayload_0(float_as_uint(Distance));
    optixSetPayload_1(0);
    optixSetPayload_2(0);
    optixSetPayload_3(0);
    optixSetPayload_4(0);    
}
}