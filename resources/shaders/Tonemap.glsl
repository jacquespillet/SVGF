#version 430


layout (binding = 0, rgba32f) uniform image2D inputImage;
layout (binding = 1, rgba32f) uniform image2D outputImage;

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;


float ToSRGB(float Col) {
  return (Col <= 0.0031308f) ? 12.92f * Col
                             : (1 + 0.055f) * pow(Col, 1 / 2.4f) - 0.055f;
}

vec3 ToSRGB(vec3 Col)
{
    return vec3(
        ToSRGB(Col.x),
        ToSRGB(Col.y),
        ToSRGB(Col.z)
    );
}

void main() {
    uvec2 GlobalID = gl_GlobalInvocationID.xy;
    vec3 Col = ToSRGB(imageLoad(inputImage, ivec2(GlobalID)).xyz);
    imageStore(outputImage, ivec2(GlobalID), vec4(Col, 1));
}