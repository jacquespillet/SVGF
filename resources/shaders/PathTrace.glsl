#version 460

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (binding = 0, rgba32f) uniform image2D RenderImage;


void main()
{
    ivec2 ImageSize = imageSize(RenderImage);
    vec2 UV = vec2(gl_GlobalInvocationID.xy) / vec2(ImageSize);
    imageStore(RenderImage, ivec2(gl_GlobalInvocationID.xy), vec4(UV, 0, 1));    
}