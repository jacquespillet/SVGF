#pragma once
#include "BVH.h"
#include <glm/glm.hpp>
using namespace glm;
using namespace gpupt;

#define PI_F 3.141592653589
#define INVALID_ID -1


#define MAX_LENGTH 1e30f

__device__ u32 Width;
__device__ u32 Height;
__device__ triangle *TriangleBuffer;
__device__ bvhNode *BVHBuffer;
__device__ u32 *IndicesBuffer;
__device__ indexData *IndexDataBuffer;
__device__ instance *TLASInstancesBuffer;
__device__ tlasNode *TLASNodes;
__device__ camera *Cameras;
__device__ tracingParameters *Parameters;
__device__ material *Materials;
__device__ cudaTextureObject_t SceneTextures;
__device__ cudaTextureObject_t EnvTextures;
__device__ int EnvironmentsCount;
__device__ int TexturesWidth;
__device__ int TexturesHeight;
__device__ int LightsCount;
__device__ light *Lights;
__device__ float *LightsCDF;
__device__ environment *Environments;
__device__ int EnvTexturesWidth;
__device__ int EnvTexturesHeight;


#define MAIN() \
__global__ void TraceKernel(glm::vec4 *RenderImage, int _Width, int _Height, \
                            triangle *_AllTriangles, bvhNode *_AllBVHNodes, u32 *_AllTriangleIndices, indexData *_IndexData, instance *_Instances, tlasNode *_TLASNodes,\
                            camera *_Cameras, tracingParameters* _TracingParams, material *_Materials, cudaTextureObject_t _SceneTextures, int _TexturesWidth, int _TexturesHeight, light *_Lights, float *_LightsCDF, int _LightsCount,\
                            environment *_Environments, int _EnvironmentsCount, cudaTextureObject_t _EnvTextures, int _EnvTexturesWidth, int _EnvTexturesHeight)

#define INIT() \
    Width = _Width; \
    Height = _Height; \
    TriangleBuffer = _AllTriangles; \
    BVHBuffer = _AllBVHNodes; \
    IndicesBuffer = _AllTriangleIndices; \
    IndexDataBuffer = _IndexData; \
    TLASInstancesBuffer = _Instances; \
    TLASNodes = _TLASNodes; \
    Cameras = _Cameras; \
    Parameters = _TracingParams; \
    Materials = _Materials; \
    SceneTextures = _SceneTextures; \
    EnvTextures = _EnvTextures; \
    LightsCount = _LightsCount; \
    Lights = _Lights; \
    LightsCDF = _LightsCDF; \
    EnvironmentsCount = _EnvironmentsCount; \
    Environments = _Environments; \
    TexturesWidth = _TexturesWidth; \
    TexturesHeight = _TexturesHeight; \
    EnvTexturesWidth = _EnvTexturesWidth; \
    EnvTexturesHeight = _EnvTexturesHeight; \


#define IMAGE_SIZE(Img) \
    ivec2(Width, Height)

#define GLOBAL_ID() \
    uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)

#define FN_DECL __device__

#define INOUT(Type) Type &

#define GET_ATTR(Obj, Attr) \
    Obj->Attr


__device__ void imageStore(vec4 *Image, ivec2 p, vec4 Colour)
{
    Image[p.y * Width + p.x] = Colour;
}

__device__ vec4 imageLoad(vec4 *Image, ivec2 p)
{
    return Image[p.y * Width + p.x];
}

__device__ vec4 textureSample(cudaTextureObject_t _SceneTextures, glm::vec3 Coords)
{
    // Coords.x = Coordx.x % 1.0f;
    float W;
    if(Coords.x < 0) Coords.x = 1 - Coords.x;
    if(Coords.y < 0) Coords.y = 1 - Coords.y;

    Coords.x = std::modf(Coords.x, &W);
    Coords.y = std::modf(Coords.y, &W);

    int NumLayersX = 8192 / TexturesWidth;
    int LayerInx = Coords.z;
    
    int LocalCoordX = Coords.x * TexturesWidth;
    int LocalCoordY = Coords.y * TexturesHeight;

    int XOffset = (LayerInx % NumLayersX) * TexturesWidth;
    int YOffset = (LayerInx / NumLayersX) * TexturesHeight;

    int CoordX = XOffset + LocalCoordX;
    int CoordY = YOffset + LocalCoordY;

    uchar4 TexValue = tex2D<uchar4>(_SceneTextures, CoordX, CoordY);
    vec4 TexValueF = vec4((float)TexValue.x / 255.0f, (float)TexValue.y / 255.0f, (float)TexValue.z / 255.0f, (float)TexValue.w / 255.0f);
    return TexValueF;
}

__device__ vec4 textureSampleEnv(cudaTextureObject_t _EnvTextures, glm::vec3 Coords)
{
    int NumLayersX = 8192 / EnvTexturesWidth;
    int LayerInx = Coords.z;
    
    int LocalCoordX = Coords.x * EnvTexturesWidth;
    int LocalCoordY = Coords.y * EnvTexturesHeight;

    int XOffset = (LayerInx % NumLayersX) * EnvTexturesWidth;
    int YOffset = (LayerInx / NumLayersX) * EnvTexturesHeight;

    int CoordX = XOffset + LocalCoordX;
    int CoordY = YOffset + LocalCoordY;

    float4 TexValue = tex2D<float4>(_EnvTextures, CoordX, CoordY);
    vec4 TexValueF = vec4(TexValue.x, TexValue.y, TexValue.z, TexValue.w);
    return TexValueF;
}

 
#include "../../resources/PathTraceCode.cpp"


__device__ float ToSRGB(float Col) {
  return (Col <= 0.0031308f) ? 12.92f * Col
                             : (1 + 0.055f) * pow(Col, 1 / 2.4f) - 0.055f;
}

__device__ glm::vec3 ToSRGB(glm::vec3 Col)
{
    return glm::vec3(
        ToSRGB(Col.x),
        ToSRGB(Col.y),
        ToSRGB(Col.z)
    );
}

__global__ void TonemapKernel(glm::vec4 *Input,glm::vec4 *Output, int Width, int Height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Width && y < Height) {    
        int index = y * Width + x;

        glm::vec3 Col = ToSRGB(Input[y * Width + x]);
        Output[y * Width + x] = vec4(Col, 1);    
    }
}