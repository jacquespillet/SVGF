
#include <optix.h>
#include <optix_device.h>

#include <cuda_fp16.h>
#include "Common.cuh"

namespace pathtracing
{
using namespace glm;
using namespace commonCu;


extern "C" __global__ void __closesthit__ch() {

    float Distance = optixGetRayTmax();
    optixSetPayload_0(float_as_uint(Distance));

    uint32_t PrimitiveIndex = optixGetPrimitiveIndex();
    optixSetPayload_1(PrimitiveIndex);

    uint32_t InstanceIndex = optixGetInstanceId();
    optixSetPayload_2(InstanceIndex);
    
    float2 UV = optixGetTriangleBarycentrics();
    optixSetPayload_3(float_as_uint(UV.x));
    optixSetPayload_4(float_as_uint(UV.y));
}

}