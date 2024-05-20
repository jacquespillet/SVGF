#pragma once

#include <memory>
#include <glm/mat4x4.hpp>

namespace gpupt
{

struct orbitCameraController
{
    orbitCameraController();

    void Recalculate();
    bool Update();

    glm::mat4 &GetMatrix();

    float Theta = 0.0f;
    float Phi = 0.0f;
    float Distance = 1.0f;

    glm::vec3 Target = glm::vec3(0,0,0);

    float MouseSpeedX = 1.0f;
    float MouseSpeedY = 1.0f;
    float MouseSpeedWheel = 1.0f;

    glm::vec3 Position;
    glm::vec3 Rotation;

    glm::mat4 ModelMatrix;
    glm::mat4 ViewMatrix;

    glm::mat4 PrevViewMatrix;

    bool Locked = false;
};

}