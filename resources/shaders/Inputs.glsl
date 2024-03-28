layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (binding = 0, rgba32f) uniform image2D RenderImage;
