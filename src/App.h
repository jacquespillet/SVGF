#pragma once
#include <memory>
#include "Tracing.h"
#include "CameraController.h"
#include <OpenImageDenoise/oidn.hpp>
#include "Timer.h"
#include "SVGF.h"

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

    // Denoiser
    oidn::DeviceRef Device;
    oidn::FilterRef Filter;
    cudaStream_t Stream;
    

    bool Denoised=false;
    bool DoDenoise=false;
    
    
    bool DoSVGF=true;


    void Tonemap();

    float Time=0;

    std::shared_ptr<framebuffer> Framebuffer[2];
    std::shared_ptr<shaderGL> GBufferShader;
    std::shared_ptr<textureGL> TonemapTexture;


    timer Timer;

    std::shared_ptr<buffer> TracingParamsBuffer;
    std::shared_ptr<buffer> RenderBuffer[2];

    std::shared_ptr<buffer> NormalBuffer;
    std::shared_ptr<buffer> FilterBuffer;
    std::shared_ptr<buffer> HistoryLengthBuffer;
    // std::shared_ptr<buffer> TonemapBuffer;    
    // std::shared_ptr<buffer> DenoisedBuffer;    
    std::shared_ptr<textureGL> RenderTexture;
    std::shared_ptr<cudaTextureMapping> RenderTextureMapping;

    
    int PingPongInx=0;


    void Render();
    void SaveRender(std::string ImagePath);
    void InitGpuObjects();
    void InitImGui();
    void ResizeRenderTextures();
    void CalculateWindowSizes();
    void StartFrame();
    void Denoise();
    void CreateOIDNFilter();
    void EndFrame();
};

}