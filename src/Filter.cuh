#pragma once

#include "App.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>


namespace filter
{
using namespace glm;
using namespace gpupt;

__device__ int Width;
__device__ int Height;


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


// Texture sampling function with bilinear interpolation
__device__ vec4 textureSample(vec4* texture, int Width, int Height, vec2& uv) {
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

__global__ void TonemapKernel(vec4 *Input,vec4 *Output, int _Width, int _Height, int DoClear)
{
    Width = _Width;
    Height = _Height;
        
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


FN_DECL vec4 SampleCuTexture(cudaTextureObject_t Texture, ivec2 Coord)
{
    float4 Sample = tex2D<float4>(Texture, Coord.x, Coord.y);
    return vec4(Sample.x, Sample.y, Sample.z, Sample.w);
}

FN_DECL float GetDepth(cudaTextureObject_t Texture, ivec2 Coord)
{
    float4 MotionVecSample = tex2D<float4>(Texture, Coord.x, Coord.y);
    float Depth = MotionVecSample.z;
    if(Depth == 0.0f) return 1e30f;

    return Depth;
}

FN_DECL bool LoadPreviousData(vec4 *PrevFrame, cudaFramebuffer &CurrentFramebuffer, cudaFramebuffer &PreviousFramebuffer, 
                              uint32_t *HistoryLengths, vec4 *MomentsBuffer,  ivec2 Coord, vec3 CurrentColour, 
                              INOUT(vec3) Colour, INOUT(int) HistoryLength, INOUT(vec2) PreviousMoments)
{
    float4 MotionVectorSample = tex2D<float4>(CurrentFramebuffer.MotionTexture, Coord.x, Coord.y);
    vec2 MotionVector = vec2(MotionVectorSample.x, MotionVectorSample.y);
    ivec2 PrevCoord = Coord + ivec2(MotionVector);


    if(PrevCoord.x < 0 || PrevCoord.x >= Width || PrevCoord.y < 0 || PrevCoord.y >= Height) return false;
    

    // Check if depth is consistent
    float CurrentDepth = GetDepth(CurrentFramebuffer.MotionTexture, Coord);
    float PreviousDepth = GetDepth(PreviousFramebuffer.MotionTexture, PrevCoord);
    if(abs(CurrentDepth - PreviousDepth) > 0.1f) return false;

    // // Check the mesh ID
    int CurrentMeshID = SampleCuTexture(CurrentFramebuffer.UVTexture, Coord).w;
    int PreviousMeshID = SampleCuTexture(PreviousFramebuffer.UVTexture, PrevCoord).w;
    if(CurrentMeshID != PreviousMeshID) return false;

    // Check the normal
    vec3 CurrNormal = SampleCuTexture(CurrentFramebuffer.NormalTexture, Coord);
    vec3 PrevNormal = SampleCuTexture(PreviousFramebuffer.NormalTexture, PrevCoord);
    if(dot(CurrNormal, PrevNormal) < 0.9f) return false;

    // vec3 PreviousColour = imageLoad(PrevFrame, PrevCoord);

    // if(distance(CurrentColour, PreviousColour) > 0.1f) return false;

    Colour = imageLoad(PrevFrame, PrevCoord);
    HistoryLength = int(HistoryLengths[PrevCoord.y * Width + PrevCoord.x]);
    PreviousMoments = vec2(MomentsBuffer[PrevCoord.y * Width + PrevCoord.x]);
    return true;
}

FN_DECL float CalculateLuminance(vec3 Colour)
{
    return 0.2126f * Colour.r + 0.7152f * Colour.g + 0.0722f * Colour.b;
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


__global__ void TAAFilterKernel(vec4 *InputFiltered, vec4 *Output, int _Width, int _Height, float time)
{
    Width = _Width;
    Height = _Height;
    
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
        if(!pathtracing::IsFinite(fragColor)) fragColor = vec4(0,0,0,0);

        imageStore(Output , FragCoord , fragColor);
    }
}

__global__ void TemporalFilter(vec4 *PreviousImage, vec4 *CurrentImage, cudaFramebuffer CurrentFramebuffer, cudaFramebuffer PreviousFramebuffer, uint32_t* HistoryLengths,  vec4 *MomentsBuffer, int _Width, int _Height)
{
    Width = _Width;
    Height = _Height;
        
    vec2 Resolution = vec2(float(Width), float(Height));
    vec2 InvTexResolution = vec2(1.0f / float(Width), 1.0f / float(Height));
    ivec2 FragCoord = ivec2 ( GLOBAL_ID() );
    vec2 uv = vec2(FragCoord) * InvTexResolution;

    if (FragCoord.x < Width && FragCoord.y < Height) {
        vec3 CurrentColour = imageLoad(CurrentImage, FragCoord);
        vec3 PrevCol(0);
        int HistoryLength=1;
        float Alpha = 0;
        vec2 PreviousMoments;
        
        bool CouldLoad = LoadPreviousData(PreviousImage, CurrentFramebuffer, PreviousFramebuffer, HistoryLengths, MomentsBuffer, FragCoord, CurrentColour, PrevCol, HistoryLength, PreviousMoments);
        if(CouldLoad)
        {
            HistoryLength = min(32, HistoryLength + 1 );
            Alpha = 1.0 / HistoryLength;
        }
        else
        {
            Alpha = 1;
            HistoryLength=1;
        }

        // compute first two moments of luminance
        vec2 Moments;
        Moments.x = CalculateLuminance(CurrentColour);
        Moments.y = Moments.r * Moments.r;

        // temporal integration of the Moments
        Moments = mix(PreviousMoments, Moments, Alpha);

        float Variance = max(0.f, Moments.g - Moments.r * Moments.r);

        vec3 NewCol = mix(PrevCol, CurrentColour, Alpha);
        vec2 NewMoments = mix(PreviousMoments, Moments, Alpha);
        
        HistoryLengths[FragCoord.y * Width + FragCoord.x] = HistoryLength;
        imageStore(CurrentImage, ivec2(FragCoord), vec4(NewCol, 1.0f));
        imageStore(MomentsBuffer, ivec2(FragCoord), vec4(NewMoments.x, NewMoments.y, Variance, 1.0f));    
    }
}

FN_DECL float computeWeight(
    float depthCenter,
    float phiDepth,
    vec3 normalCenter,
    vec3 normalP,
    float phiNormal,
    float luminanceIllumCenter,
    float luminanceIllumP,
    float phiIllum
)
{
    const float weightNormal = pow(saturate(dot(normalCenter, normalP)), phiNormal);
    const float weightZ = (phiDepth == 0) ? 0.0f : abs(depthCenter) / phiDepth;
    const float weightLillum = abs(luminanceIllumCenter - luminanceIllumP) / phiIllum;

    const float weightIllum = exp(0.0 - max(weightLillum, 0.0) - max(weightZ, 0.0)) * weightNormal;

    return weightIllum;
}


__global__ void FilterKernel(vec4 *Input, vec4 *Moments, cudaTextureObject_t Motions, cudaTextureObject_t Normals, uint32_t *HistoryLengths, vec4 *Output, int _Width, int _Height, int Step)
{
    Width = _Width;
    Height = _Height;

    float PhiColour = 10.0f;
    float PhiNormal = 128.0f;

    vec2 Resolution = vec2(float(Width), float(Height));
    vec2 InvTexResolution = vec2(1.0f / float(Width), 1.0f / float(Height));
    ivec2 FragCoord = ivec2 ( GLOBAL_ID() );
    vec2 uv = vec2(FragCoord) * InvTexResolution;
    uint32_t Inx = FragCoord.y * Width + FragCoord.x;

    if (FragCoord.x < Width && FragCoord.y < Height) {
        float epsVariance = 1e-10;
        float kernelWeights[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };

        // nt samplers to prevent the compiler from generating code which
        // fetches the sampler descriptor from memory for each texture access
        vec4 IlluminationCenter = imageLoad(Input, FragCoord);
        float lIlluminationCenter = CalculateLuminance(IlluminationCenter);

        // variance, filtered using 3x3 gaussin blur
        float Variance = Moments[Inx].z;

        // number of temporally integrated pixels
        float historyLength = float(HistoryLengths[Inx]);

        float zCenter = SampleCuTexture(Motions, FragCoord).z;
        if (zCenter == 0)
        {
            // not a valid depth => must be envmap => do not filter
            Output[Inx] = IlluminationCenter;
            return;
        }
        vec3 nCenter = SampleCuTexture(Normals, FragCoord);

        float phiLIllumination = PhiColour * sqrt(max(0.0, epsVariance + Variance));
        float PhiDepth = max(zCenter, 1e-8) * Step;

        // explicitly store/accumulate center pixel with weight 1 to prevent issues
        // with the edge-stopping functions
        float sumWIllumination = 1.0;
        vec4 sumIllumination = IlluminationCenter;

        for (int yy = -2; yy <= 2; yy++)
        {
            for (int xx = -2; xx <= 2; xx++)
            {
                vec2 CurrentCoord = FragCoord + ivec2(xx, yy) * Step;
                int FilterInx = CurrentCoord.y * Width + CurrentCoord.x;

                bool inside = (CurrentCoord.x < Width && CurrentCoord.y < Height && CurrentCoord.x >=0 && CurrentCoord.y >=0);

                float kernel = kernelWeights[abs(xx)] * kernelWeights[abs(yy)];

                if (inside && (xx != 0 || yy != 0)) // skip center pixel, it is already accumulated
                {
                    vec4 IlluminationP = imageLoad(Input, CurrentCoord);
                    float lIlluminationP = CalculateLuminance(IlluminationP);
                    float zP = SampleCuTexture(Motions, CurrentCoord).z;
                    vec3 nP = SampleCuTexture(Normals, CurrentCoord);

                    // compute the edge-stopping functions
                    float w = computeWeight(
                        zCenter,
                        PhiDepth * length(vec2(xx, yy)),
                        nCenter,
                        nP,
                        PhiNormal,
                        lIlluminationCenter,
                        lIlluminationP,
                        phiLIllumination
                    );

                    float wIllumination = w * kernel;

                    // alpha channel contains the variance, therefore the weights need to be squared, see paper for the formula
                    sumWIllumination += wIllumination;
                    sumIllumination += vec4(wIllumination,wIllumination,wIllumination, wIllumination * wIllumination) * IlluminationP;
                }
            }
        }

        // renormalization is different for variance, check paper for the formula
        vec4 filteredIllumination = vec4(sumIllumination / vec4(sumWIllumination,sumWIllumination,sumWIllumination, sumWIllumination * sumWIllumination));

        // return filteredIllumination;
        Output[Inx] = filteredIllumination;
    }    
}
}