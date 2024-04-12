#pragma once
#include <memory>
#include "Tracing.h"
#include "CameraController.h"
#include <OpenImageDenoise/oidn.hpp>
#include "Timer.h"


namespace gpupt
{
class window;
class shaderGL;
class uniformBufferGL;
class textureGL;
class bufferCu;
class cudaTextureMapping;
struct scene;
struct sceneBVH;
class bufferGL;
class gui;


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
    std::shared_ptr<sceneBVH> BVH;
    bool ResetRender = false;
    lights Lights;

    orbitCameraController Controller;

    tracingParameters Params;
    std::shared_ptr<gui> GUI;

    
    uint32_t  RenderResolution;
    uint32_t  RenderWidth;
    uint32_t  RenderHeight;
    uint32_t  RenderWindowWidth;
    uint32_t  RenderWindowHeight;
    float RenderAspectRatio = 1;

    // Denoiser
    oidn::DeviceRef Device;
    oidn::FilterRef Filter;
    cudaStream_t Stream;
    std::shared_ptr<bufferCu> DenoisedBuffer;
    bool Denoised=false;
    bool DoDenoise=false;

    std::shared_ptr<textureGL> TonemapTexture;

    timer Timer;

#if API==API_GL
    std::shared_ptr<shaderGL> PathTracingShader;
    std::shared_ptr<shaderGL> TonemapShader;
    std::shared_ptr<textureGL> RenderTexture;
    std::shared_ptr<uniformBufferGL> TracingParamsBuffer;
    std::shared_ptr<bufferGL> MaterialBuffer;
    std::shared_ptr<bufferGL> LightsBuffer;
    std::shared_ptr<bufferGL> LightsCDFBuffer;
    std::shared_ptr<textureGL> DenoisedTexture;
    std::shared_ptr<cudaTextureMapping> RenderMapping;
    std::shared_ptr<cudaTextureMapping> DenoiseMapping;

#elif API==API_CU
    std::shared_ptr<bufferCu> TracingParamsBuffer;
    std::shared_ptr<bufferCu> RenderBuffer;
    std::shared_ptr<bufferCu> TonemapBuffer;    
    std::shared_ptr<textureGL> RenderTexture;
    std::shared_ptr<cudaTextureMapping> RenderTextureMapping;
    std::shared_ptr<bufferCu> MaterialBuffer;
    std::shared_ptr<bufferCu> LightsBuffer;
    std::shared_ptr<bufferCu> LightsCDFBuffer;
#endif

    void Trace();

    void InitGpuObjects();
    void InitImGui();
    void UploadMaterial(int MaterialInx);
    void ResizeRenderTextures();
    void CalculateWindowSizes();
    void StartFrame();
    void Denoise();
    void CreateOIDNFilter();
    void EndFrame();
};

}