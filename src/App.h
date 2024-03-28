#pragma once
#include <memory>

namespace gpupt
{
class window;
class shaderGL;
class textureGL;
class bufferCu;
class cudaTextureMapping;

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

#if API==API_GL
    std::shared_ptr<shaderGL> PathTracingShader;
    std::shared_ptr<textureGL> RenderTexture;
#elif API==API_CU
    std::shared_ptr<bufferCu> RenderBuffer;
    std::shared_ptr<textureGL> RenderTexture;
    std::shared_ptr<cudaTextureMapping> RenderTextureMapping;
#endif

    void InitGpuObjects();
    void InitImGui();
    
    void StartFrame();
    void EndFrame();
};

}