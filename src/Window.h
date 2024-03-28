#pragma once
#include <stdint.h>
#include <functional>

struct GLFWwindow;

namespace gpupt
{

class window
{
public:
    window(uint32_t Width, uint32_t Height);

    bool ShouldClose() const;
    void PollEvents() const;
    void Present();
    ~window();
    uint32_t Width, Height;
    GLFWwindow *Handle=nullptr;
};
}