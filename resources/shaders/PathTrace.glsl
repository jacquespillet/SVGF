#version 460
#include Inputs.glsl
#include Macros.glsl

#define INIT()

#define MAIN()  void main()

#define GLOBAL_ID() \
    gl_GlobalInvocationID.xy

#define IMAGE_SIZE(Img) \
    imageSize(Img)

#include ../PathTraceCode.cpp