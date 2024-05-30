#pragma once
#include <memory>
#include "Tracing.h"
#include "CameraController.h"
#include "Timer.h"
#include "SVGF.h"

#define USE_OPTIX 1
#if USE_OPTIX
#include <optix.h>
#endif

namespace gpupt
{
class window;
class shaderGL;
class uniformBufferGL;
class textureGL;
class cudaTextureMapping;
struct scene;
class buffer;
class gui;
class framebuffer;


// struct kernelParams {
//     OptixTraversableHandle handle;
//     float4* output_buffer;
//     int image_width;
//     int image_height;
// };

enum class rasterizeOutputs
{
    Position=0,
    Normal=1,
    UV=2,
    Motion=3
};

struct cudaFramebuffer
{
    unsigned long long PositionTexture, NormalTexture, UVTexture, MotionTexture;
};

class application
{
public:
    void Init();
    void Run();
    void Cleanup();

    static application *Get();
    static glm::uvec2 GetSize();

    void OnResize(uint32_t NewWidth, uint32_t NewHeight);

    
#if USE_OPTIX
    OptixDeviceContext OptixContext;
    void CreateSBT();
    OptixShaderBindingTable SBT;
    OptixPipeline pipeline;
    std::shared_ptr<buffer> KernelParamsBuffer;
#endif
private:
    friend class gui;

    bool Inited=false;

    static std::shared_ptr<application> Singleton;
    std::shared_ptr<window> Window;

    std::shared_ptr<scene> Scene;
    bool ResetRender = false;

    orbitCameraController Controller;
    bool CameraMoved=false;

    tracingParameters Params;
    std::shared_ptr<gui> GUI;

    
    int32_t  RenderResolution;
    uint32_t  RenderWidth;
    uint32_t  RenderHeight;
    uint32_t  RenderWindowWidth;
    uint32_t  RenderWindowHeight;
    float RenderAspectRatio = 1;

    
    enum class SVGFDebugOutputEnum
    {
        FinalOutput,
        RawOutput,
        Normal,
        Motion,
        Position,
        Depth,
        BarycentricCoords,
        TemporalFilter,
        Moments,
        Variance,
        ATrousWaveletFilter
    }SVGFDebugOutput = SVGFDebugOutputEnum::RawOutput;
    bool DebugRasterize=false;
    glm::vec4 DebugTint=glm::vec4(1);

    int SpatialFilterSteps = 3;
    float DepthThreshold = 0.8f;
    float NormalThreshold = 0.9f;
    int HistoryLength = 24;
    float PhiColour = 10.0f;
    float PhiNormal = 128.0f;



    void Rasterize();
    void Trace();
    void TemporalFilter();
    void WaveletFilter();
    void Tonemap();
    void FilterMoments();
    void TAA();

    float Time=0;


    std::shared_ptr<framebuffer> Framebuffer[2];
    std::shared_ptr<shaderGL> GBufferShader;
    // std::shared_ptr<textureGL> TonemapTexture;


    timer Timer;

    std::shared_ptr<buffer> TracingParamsBuffer;

    std::shared_ptr<buffer> RenderBuffer[2];
    std::shared_ptr<buffer> MomentsBuffer[2];
    std::shared_ptr<buffer> FilterBuffer[2];
    std::shared_ptr<buffer> HistoryLengthBuffer;
    // std::shared_ptr<buffer> TonemapBuffer;    
    // std::shared_ptr<buffer> DenoisedBuffer;    
    std::shared_ptr<textureGL> RenderTexture;
    std::shared_ptr<cudaTextureMapping> RenderTextureMapping;

    uint32_t OutputTexture;

    
    int PingPongInx=0;


    void Render();
    void SaveRender(std::string ImagePath);
    void InitGpuObjects();
    void InitImGui();
    void ResizeRenderTextures();
    void CalculateWindowSizes();
    void StartFrame();
    void EndFrame();
};

}