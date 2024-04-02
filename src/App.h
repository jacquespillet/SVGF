#pragma once
#include <memory>
#include "Tracing.h"

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

class application
{
public:
    void Init();
    void Run();
    void Cleanup();

    static application *Get();
private:
    static std::shared_ptr<application> Singleton;
    std::shared_ptr<window> Window;

    tracingParameters Params;

    std::shared_ptr<scene> Scene;
    std::shared_ptr<sceneBVH> BVH;

#if API==API_GL
    std::shared_ptr<shaderGL> PathTracingShader;
    std::shared_ptr<textureGL> RenderTexture;
    std::shared_ptr<uniformBufferGL> TracingParamsBuffer;
#elif API==API_CU
    std::shared_ptr<bufferCu> TracingParamsBuffer;
    std::shared_ptr<bufferCu> RenderBuffer;
    std::shared_ptr<textureGL> RenderTexture;
    std::shared_ptr<cudaTextureMapping> RenderTextureMapping;
#endif
    void Trace();

    void InitGpuObjects();
    void InitImGui();
    
    void StartFrame();
    void EndFrame();
};

}