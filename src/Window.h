#pragma once
#include <stdint.h>
#include <functional>
#include <glm/vec2.hpp>

struct GLFWwindow;

namespace gpupt
{

class window
{
public:
    window(uint32_t Width, uint32_t Height);

    std::function<void(window &, glm::ivec2)> OnResize;

    bool ShouldClose() const;
    void PollEvents() const;
    void Present();
    ~window();
    uint32_t Width, Height;
    GLFWwindow *Handle=nullptr;
};
}