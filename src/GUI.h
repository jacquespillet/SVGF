#pragma once
#include <stdint.h>

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
    bool MaterialGUI(int MaterialInx);
    bool MaterialsGUI();
    bool ShapeGUI(int ShapeInx);
    bool ShapesGUI();
    bool CameraGUI(int CameraInx);
    bool CamerasGUI();
    void TextureGUI(int TextureInx);
    void TexturesGUI();
    void EnvironmentsGUI();
    bool TracingGUI();
    void GUI();

    void CalculateWindowSizes();
    uint32_t GuiWidth = 100;

    // GUI
    int SelectedMaterial = -1;
    int SelectedShape = -1;
    int SelectedInstance = -1;
    int SelectedTexture = -1;
    int SelectedCamera = -1;
};
}