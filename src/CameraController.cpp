#include "CameraController.h"

#include <glm/ext.hpp>

#include <imgui.h>
#include <iostream>

namespace gpupt
{

orbitCameraController::orbitCameraController()
{
    Position = glm::vec3(1,3,5);
    Rotation = glm::vec3(0,0,0);

    this->Distance = std::sqrt(Position.x * Position.x + Position.y * Position.y + Position.z * Position.z);
    this->Theta = std::acos(Position.y / this->Distance);
    this->Phi = std::atan2(Position.z, Position.x);
    Recalculate();
}

void orbitCameraController::Recalculate()
{
    glm::vec3 Position;
    Position.x = this->Distance * std::sin(this->Theta) * std::cos(this->Phi);
    Position.z = this->Distance * std::sin(this->Theta) * std::sin(this->Phi);
    Position.y = this->Distance * std::cos(this->Theta);
    
    glm::mat4 LookAtMatrix = glm::lookAt(Position + this->Target, this->Target, glm::vec3(0,1,0));
    
    this->ModelMatrix = (glm::inverse(LookAtMatrix));
    this->ViewMatrix = LookAtMatrix;
}

bool orbitCameraController::Update()
{
    if(Locked) return false;
    ImGuiIO &io = ImGui::GetIO();

    bool ShouldRecalculate=false;
    if(io.MouseDownDuration[0]>0 && io.KeyShift)
    {
        float Offset = io.MouseDelta.y * 0.001f * this->MouseSpeedWheel * this->Distance;
        this->Distance -= Offset;
        if(Distance < 0.1f) Distance = 0.1f;
        ShouldRecalculate=true;
    }
    else if(io.MouseDownDuration[0]>0)
    {
        this->Phi += io.MouseDelta.x * 0.001f * this->MouseSpeedX;
        this->Theta -= io.MouseDelta.y * 0.001f * this->MouseSpeedY;
        ShouldRecalculate=true;
    }
    else if(io.MouseDownDuration[1]>0)
    {
        glm::vec3 Right = glm::column(ModelMatrix, 0);
        glm::vec3 Up = glm::column(ModelMatrix, 1);

        this->Target -= Right * io.MouseDelta.x * 0.01f * this->MouseSpeedX;
        this->Target += Up * io.MouseDelta.y * 0.01f * this->MouseSpeedY;
        ShouldRecalculate=true;
    }
    
    if(io.MouseWheel != 0)
    {
        float Offset = io.MouseWheel * 0.1f * this->MouseSpeedWheel * this->Distance;
        this->Distance -= Offset;
        if(Distance < 0.1f) Distance = 0.1f;
        ShouldRecalculate=true;
    }


  
    if(ShouldRecalculate)
    {
        Recalculate();
        return true;
    }
    return false;

}

}