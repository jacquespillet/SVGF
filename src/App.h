#pragma once
#include <memory>

namespace gpupt
{
class window;

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

    void StartFrame();
    void EndFrame();
};

}