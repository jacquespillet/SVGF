layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (binding = 0, rgba32f) uniform image2D RenderImage;


layout(std430, binding = 8) buffer CameraBuffer {
  camera Cameras[];
};

// BVH
layout(std430, binding = 1)  buffer triangleBuffer
{
    triangle TriangleBuffer[];
};

layout(std430, binding = 2)  buffer triangleExBuffer
{
    triangleExtraData TriangleExBuffer[];
};

layout(std430, binding = 3)  buffer bvhBuffer
{
    bvhNode BVHBuffer[];
};

layout(std430, binding = 4)  buffer indicesBuffer
{
    uint IndicesBuffer[];
};

layout(std430, binding = 5)  buffer indexDataBuffer
{
    indexData IndexDataBuffer[];
};

layout(std430, binding = 6)  buffer tlasInstancesBuffer
{
    bvhInstance TLASInstancesBuffer[];
};

layout(std430, binding = 7)  buffer tlasNodes
{
    tlasNode TLASNodes[];
};

layout(std140, binding = 9) uniform ParametersUBO {
  tracingParameters Parameters;
};

layout(std430, binding = 12) buffer MaterialsBuffer {
  material Materials[];
};

layout(binding=13) uniform sampler2DArray SceneTextures;


layout(std430, binding = 10) buffer LightsUBO {
    uvec3 Pad0;
    uint LightsCount;
    light Lights[];
};

layout(std430, binding = 11) buffer EnvironmentsBuffer {
  environment Environments[];
};
uniform int EnvironmentsCount;