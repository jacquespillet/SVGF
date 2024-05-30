
#include <optix.h>
#include <optix_device.h>

#include <cuda_fp16.h>
#include "Common.cuh"

namespace pathtracing
{
using namespace glm;
using namespace commonCu;


extern "C" __global__ void __closesthit__ch() {
    // const uint3 idx = optixGetLaunchIndex();

    // // Retrieve intersection data
    // float3 hit_position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    // float3 normal = normalize(optixGetWorldNormal());

    // // Compute the next ray direction, for simplicity, let's assume a diffuse bounce
    // float3 next_direction = normalize(make_float3(curand_normal(&seed), curand_normal(&seed), curand_normal(&seed)));

    // // Pass data to the payload
    // payload->hit_position = hit_position;
    // payload->next_direction = next_direction;
    // payload->throughput = payload->throughput * 0.8f; // assuming a simple diffuse material with 0.8 reflectance
    // uint32_t *Done = reinterpret_cast<uint32_t*>(optixGetPayload_0());
    // *Done = 1;

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