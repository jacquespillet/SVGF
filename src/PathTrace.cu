#pragma once
#include "BVH.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

using namespace glm;
using namespace gpupt;

#define PI_F 3.141592653589
#define INVALID_ID -1
#define MIN_ROUGHNESS (0.03f * 0.03f)

#define MAX_LENGTH 1e30f

#define DENOISE_RANGE vec2(1, 4)


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
__device__ float Time;
__device__ cudaTextureObject_t NormalTexture;
__device__ cudaTextureObject_t PositionTexture;
__device__ cudaTextureObject_t UVTexture;


struct rgba8 { uint8_t r, g, b, a; };

#define MAIN() \
__global__ void TraceKernel(cudaTextureObject_t _PositionTexture, cudaTextureObject_t _NormalTexture, cudaTextureObject_t _UVTexture, vec4 *RenderImage, \
                            vec4 *PreviousImage, vec4 *NormalImage, int _Width, int _Height, \
                            triangle *_AllTriangles, bvhNode *_AllBVHNodes, u32 *_AllTriangleIndices, indexData *_IndexData, instance *_Instances, tlasNode *_TLASNodes,\
                            camera *_Cameras, tracingParameters* _TracingParams, material *_Materials, cudaTextureObject_t _SceneTextures, int _TexturesWidth, int _TexturesHeight, light *_Lights, float *_LightsCDF, int _LightsCount,\
                            environment *_Environments, int _EnvironmentsCount, cudaTextureObject_t _EnvTextures, int _EnvTexturesWidth, int _EnvTexturesHeight, float _Time)

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
    Time = _Time; \
    UVTexture = _UVTexture; \
    PositionTexture = _PositionTexture; \
    NormalTexture = _NormalTexture; \


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
    p.x = clamp(p.x, 0, int(Width-1));
    p.y = clamp(p.y, 0, int(Height-1));
     
    Image[p.y * Width + p.x] = clamp(Colour, vec4(0), vec4(1));
}

__device__ vec4 imageLoad(vec4 *Image, ivec2 p)
{
    p.x = clamp(p.x, 0, int(Width-1));
    p.y = clamp(p.y, 0, int(Height-1));
    return clamp(Image[p.y * Width + p.x], vec4(0), vec4(1));
}

__device__ vec4 textureSample(cudaTextureObject_t _SceneTextures, vec3 Coords)
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

// Texture sampling function with bilinear interpolation
__device__ vec4 textureSample(vec4* texture, int Width, int Height, const vec2& uv) {
    // Convert UV coordinates to texture space
    float x = uv.x * (Width - 1);
    float y = uv.y * (Height - 1);

    // Get the integer coordinates
    int x0 = floor(x);
    int y0 = floor(y);

    // Get the fractional part
    float tx = x - x0;
    float ty = y - y0;

    // Get the four surrounding pixels
    vec4 c00 = imageLoad(texture, ivec2(clamp(x0, 0, Width-1), clamp(y0, 0,Height-1)));
    return c00;
    vec4 c10 = imageLoad(texture, ivec2(clamp(x0 + 1, 0, Width-1), clamp(y0, 0,Height-1)));
    vec4 c01 = imageLoad(texture, ivec2(clamp(x0, 0, Width-1), clamp(y0 + 1, 0,Height-1)));
    vec4 c11 = imageLoad(texture, ivec2(clamp(x0 + 1, 0, Width-1), clamp(y0 + 1, 0,Height-1)));

    // Perform bilinear interpolation
    vec4 c0 = mix(c00, c10, tx);
    vec4 c1 = mix(c01, c11, tx);
    vec4 c = mix(c0, c1, ty);

    return c;
}

__device__ vec4 textureSampleEnv(cudaTextureObject_t _EnvTextures, vec3 Coords)
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

__device__ vec3 ToSRGB(vec3 Col)
{
    return vec3(
        ToSRGB(Col.x),
        ToSRGB(Col.y),
        ToSRGB(Col.z)
    );
}

__global__ void TonemapKernel(vec4 *Input,vec4 *Output, int Width, int Height, int DoClear)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Width && y < Height) {    
        int index = y * Width + x;

        // vec3 Col = DoClear ? vec4(0,0,0,0) : ToSRGB(Input[y * Width + x]);
        vec3 InputColour = Input[y * Width + x];
        vec3 OutputColour = ToSRGB(InputColour);
        // vec3 Col = Input[y * Width + x] + vec4(0.1,0,0,0);
        Output[y * Width + x] = vec4(OutputColour, 1);    
    }
}

__device__ float hash1(float seed) {
    return fract(sin(seed)*43758.5453123);
}

__global__ void SVGFFilterKernel(vec4 *InputColour,vec4 *InputNormal, vec4 *InputFiltered, vec4 *OutputFiltered, int Width, int Height, float time)
{
    vec2 Resolution = vec2(float(Width), float(Height));
    vec2 InvTexResolution = vec2(1.0f / float(Width), 1.0f / float(Height));
    ivec2 FragCoord = ivec2 ( GLOBAL_ID() );

    
    if (FragCoord.x < Width && FragCoord.y < Height) {
        
        vec2 Offset[25];
        Offset[0] = vec2(-2,-2);
        Offset[1] = vec2(-1,-2);
        Offset[2] = vec2(0,-2);
        Offset[3] = vec2(1,-2);
        Offset[4] = vec2(2,-2);
        
        Offset[5] = vec2(-2,-1);
        Offset[6] = vec2(-1,-1);
        Offset[7] = vec2(0,-1);
        Offset[8] = vec2(1,-1);
        Offset[9] = vec2(2,-1);
        
        Offset[10] = vec2(-2,0);
        Offset[11] = vec2(-1,0);
        Offset[12] = vec2(0,0);
        Offset[13] = vec2(1,0);
        Offset[14] = vec2(2,0);
        
        Offset[15] = vec2(-2,1);
        Offset[16] = vec2(-1,1);
        Offset[17] = vec2(0,1);
        Offset[18] = vec2(1,1);
        Offset[19] = vec2(2,1);
        
        Offset[20] = vec2(-2,2);
        Offset[21] = vec2(-1,2);
        Offset[22] = vec2(0,2);
        Offset[23] = vec2(1,2);
        Offset[24] = vec2(2,2);
        
        
        float Kernel[25];
        Kernel[0] = 1.0f/256.0f;
        Kernel[1] = 1.0f/64.0f;
        Kernel[2] = 3.0f/128.0f;
        Kernel[3] = 1.0f/64.0f;
        Kernel[4] = 1.0f/256.0f;
        
        Kernel[5] = 1.0f/64.0f;
        Kernel[6] = 1.0f/16.0f;
        Kernel[7] = 3.0f/32.0f;
        Kernel[8] = 1.0f/16.0f;
        Kernel[9] = 1.0f/64.0f;
        
        Kernel[10] = 3.0f/128.0f;
        Kernel[11] = 3.0f/32.0f;
        Kernel[12] = 9.0f/64.0f;
        Kernel[13] = 3.0f/32.0f;
        Kernel[14] = 3.0f/128.0f;
        
        Kernel[15] = 1.0f/64.0f;
        Kernel[16] = 1.0f/16.0f;
        Kernel[17] = 3.0f/32.0f;
        Kernel[18] = 1.0f/16.0f;
        Kernel[19] = 1.0f/64.0f;
        
        Kernel[20] = 1.0f/256.0f;
        Kernel[21] = 1.0f/64.0f;
        Kernel[22] = 3.0f/128.0f;
        Kernel[23] = 1.0f/64.0f;
        Kernel[24] = 1.0f/256.0f;
        
        vec3 sum = vec3(0.0);
        vec3 sum_f = vec3(0.0);
        float ColourPhi = 1.0;
        float PreviousPhi = 1.0;
        float NormalPhi = 0.5;
        float BlendPhi = 0.25;
        
        vec3 ColourValue = imageLoad(InputColour, ivec2(FragCoord));
        vec3 PreviousValue = imageLoad(InputFiltered, ivec2(FragCoord));
        vec3 NormalValue = imageLoad(InputNormal, ivec2(FragCoord));

        float Angle = 2.0*3.1415926535*hash1(251.12860182*FragCoord.x + 729.9126812*FragCoord.y+5.1839513*time);
        // float Angle = 0;
        mat2 RotationMatrix = mat2(cos(Angle),sin(Angle),-sin(Angle),cos(Angle));
        
        float WeightCurrent = 0.0;
        float WeightPrevious = 0.0;
        
        float denoiseStrength = (DENOISE_RANGE.x + (DENOISE_RANGE.y-DENOISE_RANGE.x)*hash1(641.128752*FragCoord.x + 312.321374*FragCoord.y+1.92357812*time));
        
        for(int i=0; i<25; i++)
        {
            vec2 uv = (vec2(FragCoord)+RotationMatrix*(Offset[i]* denoiseStrength));
            
            vec3 ColourTemp = imageLoad(InputColour, uv);
            vec3 Dist = ColourValue - ColourTemp;
            float Dist2 = dot(Dist,Dist);
            float ColourWeight = min(exp(-(Dist2)/ColourPhi), 1.0);
            
            vec3 NormalTemp = imageLoad(InputNormal, uv);
            Dist = NormalValue - NormalTemp;
            Dist2 = max(dot(Dist,Dist), 0.0);
            float NormalWeight = min(exp(-(Dist2)/NormalPhi), 1.0);
            
            vec3 PreviousTmp = imageLoad(InputFiltered, uv);
            Dist = PreviousValue - PreviousTmp;
            Dist2 = dot(Dist,Dist);
            float PreviousWeight = min(exp(-(Dist2)/PreviousPhi), 1.0);
            
            // new denoised frame
            float weight0 = ColourWeight * NormalWeight;
            sum += ColourTemp*weight0*Kernel[i];
            WeightCurrent += weight0*Kernel[i];
            
            // denoise the previous denoised frame again
            float weight1 = PreviousWeight * NormalWeight;
            sum_f += PreviousTmp * weight1 * Kernel[i];
            WeightPrevious += weight1 * Kernel[i];
        }
        
        // mix in more of the just-denoised frame if it differs significantly from the
        // frame from feedback
        vec3 ptmp = imageLoad(InputFiltered, FragCoord);
        vec3 t = sum/WeightCurrent - ptmp;
        float dist2 = dot(t,t);
        float p_w = min(exp(-(dist2)/BlendPhi), 1.0);
        
        vec4 fragColor = clamp(vec4(mix(sum/WeightCurrent,sum_f/WeightPrevious,p_w),0.0f),0.0f,1.0f);
        imageStore(OutputFiltered, FragCoord, fragColor);
    }
}

__device__ vec3 encodePalYuv(vec3 rgb)
{
    rgb = pow(rgb, vec3(2.0)); // gamma correction
    return vec3(
        dot(rgb, vec3(0.299, 0.587, 0.114)),
        dot(rgb, vec3(-0.14713, -0.28886, 0.436)),
        dot(rgb, vec3(0.615, -0.51499, -0.10001))
    );
}

__device__ vec3 decodePalYuv(vec3 yuv)
{
    vec3 rgb = vec3(
        dot(yuv, vec3(1., 0., 1.13983)),
        dot(yuv, vec3(1., -0.39465, -0.58060)),
        dot(yuv, vec3(1., 2.03211, 0.))
    );
    return pow(rgb, vec3(1.0 / 2.0)); // gamma correction
}


__global__ void TAAFilterKernel(vec4 *InputFiltered, vec4 *Output, int Width, int Height, float time)
{
    vec2 Resolution = vec2(float(Width), float(Height));
    vec2 InvTexResolution = vec2(1.0f / float(Width), 1.0f / float(Height));
    ivec2 FragCoord = ivec2 ( GLOBAL_ID() );
    vec2 uv = vec2(FragCoord) * InvTexResolution;

    if (FragCoord.x < Width && FragCoord.y < Height) {
        vec4 lastColor = textureSample(Output, Width, Height, uv);
        
        vec3 antialiased = lastColor;
        float mixRate = min(lastColor.w, 0.5);
        
        vec2 off = 1.0f / Resolution;
        vec3 in0 = textureSample(InputFiltered, Width, Height, uv);
        
        antialiased = mix(antialiased * antialiased, in0 * in0, mixRate);
        antialiased = sqrt(antialiased);
        
        vec3 in1 = textureSample(InputFiltered, Width, Height, uv + vec2(+off.x, 0.0));
        vec3 in2 = textureSample(InputFiltered, Width, Height, uv + vec2(-off.x, 0.0));
        vec3 in3 = textureSample(InputFiltered, Width, Height, uv + vec2(0.0, +off.y));
        vec3 in4 = textureSample(InputFiltered, Width, Height, uv + vec2(0.0, -off.y));
        vec3 in5 = textureSample(InputFiltered, Width, Height, uv + vec2(+off.x, +off.y));
        vec3 in6 = textureSample(InputFiltered, Width, Height, uv + vec2(-off.x, +off.y));
        vec3 in7 = textureSample(InputFiltered, Width, Height, uv + vec2(+off.x, -off.y));
        vec3 in8 = textureSample(InputFiltered, Width, Height, uv + vec2(-off.x, -off.y));
        
        antialiased = encodePalYuv(antialiased);
        in0 = encodePalYuv(in0);
        in1 = encodePalYuv(in1);
        in2 = encodePalYuv(in2);
        in3 = encodePalYuv(in3);
        in4 = encodePalYuv(in4);
        in5 = encodePalYuv(in5);
        in6 = encodePalYuv(in6);
        in7 = encodePalYuv(in7);
        in8 = encodePalYuv(in8);
        
        vec3 minColor = min(min(min(in0, in1), min(in2, in3)), in4);
        vec3 maxColor = max(max(max(in0, in1), max(in2, in3)), in4);
        minColor = mix(minColor,
        min(min(min(in5, in6), min(in7, in8)), minColor), 0.5);
        maxColor = mix(maxColor,
        max(max(max(in5, in6), max(in7, in8)), maxColor), 0.5);
        
        vec3 preclamping = antialiased;
        antialiased = clamp(antialiased, minColor, maxColor);
        
        mixRate = 1.0 / (1.0 / mixRate + 1.0);
        
        vec3 diff = antialiased - preclamping;
        float clampAmount = dot(diff, diff);
        
        mixRate += clampAmount * 4.0;
        mixRate = clamp(mixRate, 0.05f, 0.5f);
        
        antialiased = decodePalYuv(antialiased);
            
        vec4 fragColor = vec4(antialiased, 1);    
        if(!IsFinite(fragColor)) fragColor = vec4(0,0,0,0);

        imageStore(Output , FragCoord , fragColor);
    }
}


__global__ void BilateralFilterKernel(vec4 *Input, vec4 *Output, int Width, int Height, int Diameter, float SigmaI, float SigmaS)
{

    vec2 Resolution = vec2(float(Width), float(Height));
    vec2 InvTexResolution = vec2(1.0f / float(Width), 1.0f / float(Height));
    ivec2 FragCoord = ivec2 ( GLOBAL_ID() );
    vec2 uv = vec2(FragCoord) * InvTexResolution;

    if (FragCoord.x < Width && FragCoord.y < Height) {
        int halfDiameter = Diameter / 2;
        vec3 Sum(0.0f);
        vec3 Normalization(0.0f);
        vec3 centerValue = Input[FragCoord.y * Width + FragCoord.x];

        for (int i = -halfDiameter; i <= halfDiameter; ++i) {
            for (int j = -halfDiameter; j <= halfDiameter; ++j) {
                int neighborX = min(max(FragCoord.x + j, 0), Width - 1);
                int neighborY = min(max(FragCoord.y + i, 0), Height - 1);
                vec3 neighborValue = Input[neighborY * Width + neighborX];

                float spatialWeight = expf(-(j * j + i * i) / (2 * SigmaS * SigmaS));
                vec3 intensityWeight = exp(-(neighborValue - centerValue) * (neighborValue - centerValue) / (2 * SigmaI * SigmaI));
                vec3 weight = spatialWeight * intensityWeight;

                Sum += neighborValue * weight;
                Normalization += weight;
            }
        }

        vec3 Result = Sum / Normalization;
        Output[FragCoord.y * Width + FragCoord.x] = vec4(Result, 1.0f);     
    }    
}