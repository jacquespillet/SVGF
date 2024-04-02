#version 460
#extension GL_ARB_gpu_shader_int64 : enable

#include Structs.glsl
#include Inputs.glsl
#include Macros.glsl

#define PI_F 3.141592653589


#define INIT()

#define MAIN()  void main()

#define GLOBAL_ID() \
    gl_GlobalInvocationID.xy

#define IMAGE_SIZE(Img) \
    imageSize(Img)


#define FN_DECL

#define INOUT(Type) inout Type

#define GET_ATTR(Obj, Attr) \
    Obj.Attr

#include ../PathTraceCode.cpp