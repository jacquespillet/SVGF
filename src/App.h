#pragma once
#include <memory>

namespace gpupt
{
class window;
class shaderGL;
class textureGL;

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

    std::shared_ptr<shaderGL> PathTracingShader;
    std::shared_ptr<textureGL> RenderTexture;

    void InitGpuObjects();
    void InitImGui();
    
    void StartFrame();
    void EndFrame();
};

}