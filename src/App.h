#pragma once
#include <memory>
#include "Tracing.h"
#include "CameraController.h"

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

class application
{
public:
    void Init();
    void Run();
    void Cleanup();

    static application *Get();
    static glm::uvec2 GetSize();
private:
    static std::shared_ptr<application> Singleton;
    std::shared_ptr<window> Window;

    bool ResetRender = false;

    orbitCameraController Controller;

    tracingParameters Params;

    std::shared_ptr<scene> Scene;
    std::shared_ptr<sceneBVH> BVH;

    std::shared_ptr<textureGL> TonemapTexture;
#if API==API_GL
    std::shared_ptr<shaderGL> PathTracingShader;
    std::shared_ptr<shaderGL> TonemapShader;

    std::shared_ptr<textureGL> RenderTexture;
    std::shared_ptr<uniformBufferGL> TracingParamsBuffer;
    std::shared_ptr<bufferGL> MaterialBuffer;
#elif API==API_CU
    std::shared_ptr<bufferCu> TracingParamsBuffer;
    std::shared_ptr<bufferCu> RenderBuffer;
    std::shared_ptr<bufferCu> TonemapBuffer;    
    std::shared_ptr<textureGL> RenderTexture;
    std::shared_ptr<cudaTextureMapping> RenderTextureMapping;
    std::shared_ptr<bufferCu> MaterialBuffer;

#endif
    void Trace();

    void InitGpuObjects();
    void InitImGui();
    
    void StartFrame();
    void EndFrame();
};

}