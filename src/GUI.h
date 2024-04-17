#pragma once
#include <stdint.h>
#include "imgui.h"
#include "ImGuizmo.h"
#include <set>
#include <vector>
#include <string>

namespace gpupt
{
struct application;
class gui
{
public:
    application *App;
    gui(application *app);
    void InstanceGUI(int InstanceInx);
    bool InstancesGUI();
    bool InstancesMultipleGUI();
    bool MaterialGUI(int MaterialInx);
    bool MaterialsGUI();
    bool TexturePickerGUI(std::string Name, int &TextureInx);
    bool ShapeGUI(int ShapeInx);
    bool ShapesGUI();
    bool CameraGUI(int CameraInx);
    bool CamerasGUI();
    void TextureGUI(int TextureInx);
    void TexturesGUI();
    bool EnvironmentGUI(int EnvironmentInx);
    bool EnvironmentsGUI();
    bool TracingGUI();
    void GUI();

    void CalculateWindowSizes();
    uint32_t GuiWidth = 100;

    // GUI
    int SelectedMaterial = -1;
    int SelectedShape = -1;
    std::vector<bool> SelectedInstances;
    std::set<int> SelectedInstanceIndices;
    int SelectedTexture = -1;
    int SelectedEnvironment = -1;
    int SelectedCamera = -1;

    
    ImGuizmo::OPERATION CurrentGizmoOperation = (ImGuizmo::TRANSLATE);
    ImGuizmo::MODE CurrentGizmoMode = (ImGuizmo::WORLD);    
};
}