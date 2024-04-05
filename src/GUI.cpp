#include "GUI.h"
#include <imgui.h>

#include "App.h"

#include "BVH.h"
#include <nfd.h>
#include "IO.h"
#include "TextureGL.h"
#include "BufferCu.cuh"
#include "Window.h"
#include "GLTFLoader.h"

#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/matrix_transform.hpp>
namespace gpupt
{
    
std::string ExtractFilename(const std::string& filePath) {
    size_t LastSlash = filePath.find_last_of("/\\");  // Find the last occurrence of a slash or backslash
    if (LastSlash != std::string::npos) {
        return filePath.substr(LastSlash + 1);  // Extract the filename
    } else {
        // If no slash or backslash is found, the whole path is the filename
        return filePath;
    }
}

void DecomposeMatrixToComponents(glm::mat4 &matrix, float* translation, float* rotation, float* scale)
{
    scale[0] = glm::length(glm::vec3(glm::column(matrix, 0)));
    scale[1] = glm::length(glm::vec3(glm::column(matrix, 1)));
    scale[2] = glm::length(glm::vec3(glm::column(matrix, 2)));;


    rotation[0] = glm::degrees(atan2f(matrix[1][2], matrix[2][2]));
    rotation[1] = glm::degrees(atan2f(-matrix[0][2], sqrtf(matrix[1][2] * matrix[1][2] + matrix[2][2] * matrix[2][2])));
    rotation[2] = glm::degrees(atan2f(matrix[0][1], matrix[0][0]));

    translation[0] = matrix[3][0];
    translation[1] = matrix[3][1];
    translation[2] = matrix[3][2];
}

void RecomposeMatrixFromComponents(const float* translation, const float* rotation, const float* scale, glm::mat4 &matrix)
{
    glm::mat4 rot[3];
    glm::vec3 directionUnary[3] = {
        glm::vec3(1,0,0),
        glm::vec3(0,1,0),
        glm::vec3(0,0,1)
    };
    for (int i = 0; i < 3; i++)
    {
        rot[i] = glm::rotate(glm::radians(rotation[i]), directionUnary[i]);
    }

    matrix = rot[0] * rot[1] * rot[2];

    float validScale[3];
    for (int i = 0; i < 3; i++)
    {
        if (fabsf(scale[i]) < FLT_EPSILON)
        {
            validScale[i] = 0.001f;
        }
        else
        {
            validScale[i] = scale[i];
        }
    }

    matrix[0][0] *= validScale[0];
    matrix[0][1] *= validScale[0];
    matrix[0][2] *= validScale[0];

    matrix[1][0] *= validScale[1];
    matrix[1][1] *= validScale[1];
    matrix[1][2] *= validScale[1];

    matrix[2][0] *= validScale[2];
    matrix[2][1] *= validScale[2];
    matrix[2][2] *= validScale[2];

    matrix[3][0] = translation[0];
    matrix[3][1] = translation[1];
    matrix[3][2] = translation[2];

}

gui::gui(application *App) : App(App){}

void gui::InstanceGUI(int InstanceInx)
{
    bool TransformChanged = false;
    // XForm
    ImGui::Text("Transform");

    glm::vec3 Scale;
    glm::vec3 Rotation;
    glm::vec3 Translation;
    DecomposeMatrixToComponents(App->Scene->Instances[InstanceInx].ModelMatrix, &Translation[0], &Rotation[0], &Scale[0]);

    TransformChanged |= ImGui::DragFloat3("Position", &Translation[0], 0.1);
    TransformChanged |= ImGui::DragFloat3("Rotation", &Rotation[0], 1);

    static bool UniformScale = false;
    ImGui::Checkbox("Uniform Scale", &UniformScale);
    if(UniformScale)
    {
        float ScaleUniform = (Scale.x + Scale.y + Scale.z) / 3.0f;
        TransformChanged |= ImGui::DragFloat("Scale", &ScaleUniform, 0.1);
        Scale = glm::vec3(ScaleUniform);
    }
    else
        TransformChanged |= ImGui::DragFloat3("Scale", &Scale[0], 0.1);
    

    if(TransformChanged)
    {
        RecomposeMatrixFromComponents(&Translation[0], &Rotation[0], &Scale[0], App->Scene->Instances[InstanceInx].ModelMatrix);
        App->BVH->UpdateTLAS(InstanceInx);
        App->ResetRender=true;
    }

    ImGui::Separator();

    // Shape selection
    ImGui::Text("Shape : "); ImGui::SameLine(); ImGui::Text(App->Scene->ShapeNames[App->Scene->Instances[InstanceInx].Shape].c_str());
    if(ImGui::Button("Change Shape"))
    {
        ImGui::OpenPopup("Instance_Shape_Selection");
    }

    if (ImGui::BeginPopup("Instance_Shape_Selection"))
    {
        static int SelectedShape = App->Scene->Instances[InstanceInx].Shape;
        for (int i = 0; i < App->Scene->Shapes.size(); i++)
        {
            if (ImGui::Selectable(App->Scene->ShapeNames[i].c_str(), SelectedShape == i))
            {
                SelectedShape = i;
                App->BVH->UpdateShape(InstanceInx, SelectedShape);
                App->Scene->Instances[InstanceInx].Shape = SelectedShape;
                App->ResetRender=true;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
    
    ImGui::Separator();

    // Material selection
    ImGui::Text("Material : "); ImGui::SameLine(); ImGui::Text(App->Scene->MaterialNames[App->Scene->Instances[InstanceInx].Material].c_str());
    if(ImGui::Button("Change Material"))
    {
        ImGui::OpenPopup("Instance_Material_Selection");
    }

    if (ImGui::BeginPopup("Instance_Material_Selection"))
    {
        static int SelectedMaterial = App->Scene->Instances[InstanceInx].Material;
        for (int i = 0; i < App->Scene->Materials.size(); i++)
        {
            if (ImGui::Selectable(App->Scene->MaterialNames[i].c_str(), SelectedMaterial == i))
            {
                int PreviousMaterial = App->Scene->Instances[InstanceInx].Material;

                SelectedMaterial = i;
                App->Scene->Instances[InstanceInx].Material = SelectedMaterial;
                App->BVH->UpdateMaterial(InstanceInx, SelectedMaterial);
                App->ResetRender=true;                        

                // If the new material is emissive, or the old material was emissive, we rebuild the lights
                if(glm::length(App->Scene->Materials[SelectedMaterial].Emission) > 1e-3f || glm::length(App->Scene->Materials[PreviousMaterial].Emission) > 1e-3f)
                {
                    App->Lights = GetLights(App->Scene, App->Params);
                    App->LightsBuffer->updateData(&App->Lights, sizeof(lights));
                }

                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
    
    ImGui::Separator();

    if(MaterialGUI(App->Scene->Instances[InstanceInx].Material))
    {
        App->ResetRender=true;
    }
}

bool gui::InstancesGUI()
{
    bool Changed = false;
    static int SelectedInstance = -1;
    for (int i = 0; i < App->Scene->Instances.size(); i++)
    {
        if (ImGui::Selectable(App->Scene->InstanceNames[i].c_str(), SelectedInstance == i))
            SelectedInstance = i;
    }
    
    if(ImGui::Button("Add"))
    {
        ImGui::OpenPopup("Add_Instance_Popup");
    }
    if(ImGui::BeginPopupModal("Add_Instance_Popup"))
    {
        static int SelectedShape = -1;
        static int SelectedMaterial = -1;
        
        static char Name[256];
        ImGui::InputText("Instance Name : ", Name, 256);

        ImGui::Text("Select Shape");
        for (int i = 0; i < App->Scene->Shapes.size(); i++)
        {
            ImGui::PushID(i);
            if (ImGui::Selectable(App->Scene->ShapeNames[i].c_str(), SelectedShape == i, ImGuiSelectableFlags_DontClosePopups))
                SelectedShape = i;
            ImGui::PopID();;
        }            

        ImGui::Separator();
        ImGui::Text("Select Material");
        for (int i = 0; i < App->Scene->Materials.size(); i++)
        {
            ImGui::PushID(App->Scene->Shapes.size() + i);
            if (ImGui::Selectable(App->Scene->MaterialNames[i].c_str(), SelectedMaterial == i, ImGuiSelectableFlags_DontClosePopups))
                SelectedMaterial = i;
            ImGui::PopID();
        }      

        if(ImGui::Button("Confirm"))
        {
            App->Scene->Instances.emplace_back();
            instance &NewInstance = App->Scene->Instances.back(); 
            NewInstance.Shape = SelectedShape;
            NewInstance.Material = SelectedMaterial;
            App->Scene->InstanceNames.push_back(std::string(Name));      
            App->BVH->AddInstance(App->Scene->Instances.size()-1);
            App->ResetRender = true;
            ImGui::CloseCurrentPopup();
        }     
        ImGui::EndPopup(); 
    }


    ImGui::Separator();

    if(SelectedInstance != -1)
    {
        InstanceGUI(SelectedInstance);
    }

    

    return Changed;
}



bool gui::MaterialGUI(int MaterialInx)
{
    bool Changed = false;

    ImGui::Text(App->Scene->MaterialNames[MaterialInx].c_str());
    material &Mat = App->Scene->Materials[MaterialInx];

    Changed |= ImGui::Combo("Type", &Mat.MaterialType, "Matte\0PBR\0\0");

    glm::vec3 PrevEmission = Mat.Emission;
    Changed |= ImGui::DragFloat3("Emission", &Mat.Emission[0], 2, 0, 100);
    Changed |= ImGui::ColorEdit3("Colour", &Mat.Colour[0]);

    if(Mat.MaterialType != MATERIAL_TYPE_MATTE)
    {
        Changed |= ImGui::DragFloat("Roughness", &Mat.Roughness, 0.01f, 0, 1);
        Changed |= ImGui::DragFloat("Metallic", &Mat.Metallic, 0.01f, 0, 1);
    }
    
    ImGui::Separator();
    ImGui::Text("Textures : ");

    // Colour
    std::string ColourTextureName = Mat.ColourTexture >=0 ? App->Scene->TextureNames[Mat.ColourTexture] : "Empty";
    ImGui::Text("Colour"); ImGui::SameLine(); ImGui::Text(ColourTextureName.c_str()); ImGui::SameLine(); if(ImGui::Button("Choose"))
    {
        ImGui::OpenPopup("ColourTexturePicker");
    }

    if(ImGui::BeginPopup("ColourTexturePicker"))
    {
        static int SelectedIndex = -1;
        for (int i = 0; i < App->Scene->Textures.size(); i++)
        {
            if (ImGui::Selectable(App->Scene->TextureNames[i].c_str(), SelectedIndex == i, ImGuiSelectableFlags_DontClosePopups))
                SelectedIndex = i;

        }

        if(ImGui::Button("Choose") && SelectedIndex >=0)
        {
            Mat.ColourTexture = SelectedIndex;
            Changed |= true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Roughness
    std::string RoughnessTextureName = Mat.RoughnessTexture >=0 ? App->Scene->TextureNames[Mat.RoughnessTexture] : "Empty";
    ImGui::Text("Roughness"); ImGui::SameLine(); ImGui::Text(RoughnessTextureName.c_str()); ImGui::SameLine(); if(ImGui::Button("Choose"))
    {
        ImGui::OpenPopup("RoughnessTexturePicker");
    }

    if(ImGui::BeginPopup("RoughnessTexturePicker"))
    {
        static int SelectedIndex = -1;
        for (int i = 0; i < App->Scene->Textures.size(); i++)
        {
            if (ImGui::Selectable(App->Scene->TextureNames[i].c_str(), SelectedIndex == i, ImGuiSelectableFlags_DontClosePopups))
                SelectedIndex = i;

        }

        if(ImGui::Button("Choose") && SelectedIndex >=0)
        {
            Mat.RoughnessTexture = SelectedIndex;
            Changed |= true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Normal
    std::string NormalTextureName = Mat.NormalTexture >=0 ? App->Scene->TextureNames[Mat.NormalTexture] : "Empty";
    ImGui::Text("Normal"); ImGui::SameLine(); ImGui::Text(NormalTextureName.c_str()); ImGui::SameLine(); if(ImGui::Button("Choose"))
    {
        ImGui::OpenPopup("NormalTexturePicker");
    }

    if(ImGui::BeginPopup("NormalTexturePicker"))
    {
        static int SelectedIndex = -1;
        for (int i = 0; i < App->Scene->Textures.size(); i++)
        {
            if (ImGui::Selectable(App->Scene->TextureNames[i].c_str(), SelectedIndex == i, ImGuiSelectableFlags_DontClosePopups))
                SelectedIndex = i;

        }

        if(ImGui::Button("Choose") && SelectedIndex >=0)
        {
            Mat.NormalTexture = SelectedIndex;
            Changed |= true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Emission
    std::string EmissionTextureName = Mat.EmissionTexture >=0 ? App->Scene->TextureNames[Mat.EmissionTexture] : "Empty";
    ImGui::Text("Emission"); ImGui::SameLine(); ImGui::Text(EmissionTextureName.c_str()); ImGui::SameLine(); if(ImGui::Button("Choose"))
    {
        ImGui::OpenPopup("EmissionTexturePicker");
    }

    if(ImGui::BeginPopup("EmissionTexturePicker"))
    {
        static int SelectedIndex = -1;
        for (int i = 0; i < App->Scene->Textures.size(); i++)
        {
            if (ImGui::Selectable(App->Scene->TextureNames[i].c_str(), SelectedIndex == i, ImGuiSelectableFlags_DontClosePopups))
                SelectedIndex = i;

        }

        if(ImGui::Button("Choose") && SelectedIndex >=0)
        {
            Mat.EmissionTexture = SelectedIndex;
            Changed |= true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if((glm::length(PrevEmission) <= 1e-3f && glm::length(Mat.Emission) > 1e-3f) || (glm::length(PrevEmission) > 1e-3f && glm::length(Mat.Emission)<= 1e-3f))
    {
        App->Lights = GetLights(App->Scene, App->Params);
        App->LightsBuffer->updateData(&App->Lights, sizeof(lights));
        App->ResetRender=true;
    }

    if(Changed)
        App->UploadMaterial(MaterialInx);
    return Changed;
}


bool gui::MaterialsGUI()
{
    bool Changed = false;

    for (int i = 0; i < App->Scene->Materials.size(); i++)
    {
        if (ImGui::Selectable(App->Scene->MaterialNames[i].c_str(), SelectedMaterial == i))
            SelectedMaterial = i;
    }
    ImGui::Separator();

    if(ImGui::Button("Add"))
    {
        ImGui::OpenPopup("Add_Material_Popup");
    }        

    if(ImGui::BeginPopup("Add_Material_Popup"))
    {
        static char Name[256];
        ImGui::InputText("Material Name : ", Name, 256);


        if(ImGui::Button("Create"))
        {
            App->Scene->Materials.emplace_back();
            material Mat = App->Scene->Materials.back(); 
            App->Scene->MaterialNames.push_back(Name);
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if(SelectedMaterial != -1)
    {
        Changed |= MaterialGUI(SelectedMaterial);
    }

    return Changed;
}

bool gui::ShapeGUI(int ShapeInx)
{
    bool Changed = false;

    ImGui::Text(App->Scene->ShapeNames[ShapeInx].c_str());

    ImGui::Separator();

    ImGui::Text("Triangles Count : "); ImGui::SameLine(); ImGui::Text( std::to_string(App->Scene->Shapes[ShapeInx].Triangles.size()).c_str());
    ImGui::Text("Positions Count : "); ImGui::SameLine(); ImGui::Text( std::to_string(App->Scene->Shapes[ShapeInx].Positions.size()).c_str());
    ImGui::Text("Normals Count : "); ImGui::SameLine(); ImGui::Text( std::to_string(App->Scene->Shapes[ShapeInx].Normals.size()).c_str());
    ImGui::Text("TexCoords Count : "); ImGui::SameLine(); ImGui::Text( std::to_string(App->Scene->Shapes[ShapeInx].TexCoords.size()).c_str());
    ImGui::Text("Colours Count : "); ImGui::SameLine(); ImGui::Text( std::to_string(App->Scene->Shapes[ShapeInx].Colours.size()).c_str());
    ImGui::Text("Tangents Count : "); ImGui::SameLine(); ImGui::Text( std::to_string(App->Scene->Shapes[ShapeInx].Tangents.size()).c_str());

    return Changed;
}

bool gui::ShapesGUI()
{
    bool Changed = false;

    for (int i = 0; i < App->Scene->Shapes.size(); i++)
    {
        if (ImGui::Selectable(App->Scene->ShapeNames[i].c_str(), SelectedShape == i))
            SelectedShape = i;
    }

    if(ImGui::Button("Add"))
    {
        nfdchar_t *AssetPath = NULL;
        nfdresult_t Result = NFD_OpenDialog( NULL, NULL, &AssetPath );
        if ( Result == NFD_OKAY ) 
        {
            int PreviousShapesInx = App->Scene->Shapes.size();
            LoadGLTF(AssetPath, App->Scene, false);
            for(int i=PreviousShapesInx; i<App->Scene->Shapes.size(); i++)
            {
                App->BVH->AddShape(i);
            }
        }            
    }

    ImGui::Separator();

    if(SelectedShape != -1)
    {
        Changed |= ShapeGUI(SelectedShape);
    }


    return Changed;
}

bool gui::CameraGUI(int CameraInx)
{
    bool Changed = false;

    ImGui::Text(App->Scene->CameraNames[CameraInx].c_str());

    Changed |= ImGui::DragFloat("Lens", &App->Scene->Cameras[CameraInx].Lens, 0.005f, 0.0001f, 0.5f);
    Changed |= ImGui::DragFloat("Film", &App->Scene->Cameras[CameraInx].Film, 0.005f, 0.0001f, 0.5f);
    Changed |= ImGui::DragFloat("Focus", &App->Scene->Cameras[CameraInx].Focus, 0.1f);
    Changed |= ImGui::DragFloat("Aperture", &App->Scene->Cameras[CameraInx].Aperture, 0.005f, 0.0001f, 0.5f);
    
    return Changed;
}

bool gui::CamerasGUI()
{
    bool Changed = false;

    for (int i = 0; i < App->Scene->Cameras.size(); i++)
    {
        if (ImGui::Selectable(App->Scene->CameraNames[i].c_str(), SelectedCamera == i))
            SelectedCamera = i;
    }
    ImGui::Separator();

    if(SelectedCamera != -1)
    {
        Changed |= CameraGUI(SelectedCamera);
    }
    
    if(ImGui::Button("Add"))
    {
        // TODO
    }

    return Changed;        
}

void gui::TextureGUI(int TextureInx)
{
    ImGui::Text(App->Scene->TextureNames[TextureInx].c_str());
    ImGui::Text("Width : "); ImGui::SameLine(); ImGui::Text(std::to_string(App->Scene->Textures[TextureInx].Width).c_str());
    ImGui::Text("Height : "); ImGui::SameLine(); ImGui::Text(std::to_string(App->Scene->Textures[TextureInx].Height).c_str());
    ImGui::Text("Channels : "); ImGui::SameLine(); ImGui::Text(std::to_string(App->Scene->Textures[TextureInx].Height).c_str());
}

void gui::TexturesGUI()
{
    bool Changed = false;

    for (int i = 0; i < App->Scene->Textures.size(); i++)
    {
        if (ImGui::Selectable(App->Scene->TextureNames[i].c_str(), SelectedTexture == i))
            SelectedTexture = i;
    }

    if(ImGui::Button("Add"))
    {
        nfdchar_t *ImagePath = NULL;
        nfdresult_t Result = NFD_OpenDialog( NULL, NULL, &ImagePath );
        if ( Result == NFD_OKAY ) 
        {
            App->Scene->Textures.emplace_back();
            texture &Texture = App->Scene->Textures.back(); 
            Texture.SetFromFile(ImagePath, App->Scene->TextureWidth, App->Scene->TextureHeight);
            App->Scene->TextureNames.push_back(ExtractFilename(ImagePath));
            App->Scene->ReloadTextureArray();
        }             
    }
    ImGui::Separator();

    if(SelectedTexture != -1)
    {
        TextureGUI(SelectedTexture);
    }

}

bool gui::TracingGUI()
{
    bool Changed = false;
    

    ImGui::Text("Samples : ");
    std::string SamplesStr = std::to_string(App->Params.CurrentSample) + "/" + std::to_string(App->Params.TotalSamples);
    ImGui::Text(SamplesStr.c_str());
    ImGui::DragInt("Total Samples", &App->Params.TotalSamples);

    Changed |= ImGui::SliderInt("Batches", &App->Params.Batch, 0, 32);
    Changed |= ImGui::SliderInt("Bounces", &App->Params.Bounces, 0, 32);
    Changed |= ImGui::DragFloat("Clamp", &App->Params.Clamp, 0.1f, 0.0f, 32.0f);    


    return Changed;
}


void gui::GUI()
{
    App->CalculateWindowSizes(); 


    ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(GuiWidth, App->Window->Height), ImGuiCond_Always);
    ImGui::Begin("_", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
    GuiWidth = ImGui::GetWindowSize().x;
    ImGui::PushID(0);
    if (ImGui::BeginTabBar("", ImGuiTabBarFlags_None))
    {
        if (ImGui::BeginTabItem("Instances"))
        {
            if(InstancesGUI())
            {
                App->ResetRender=true;
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Materials"))
        {
            if(MaterialsGUI())
            {
                App->ResetRender = true;
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Shapes"))
        {
            ShapesGUI();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Cameras"))
        {
            if(CamerasGUI())
            {
                App->ResetRender=true;
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Textures"))
        {
            TexturesGUI();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Tracing Params"))
        {
            if(TracingGUI())
            {
                App->ResetRender = true;
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }     
    ImGui::PopID();   

    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(GuiWidth,0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(App->RenderWindowWidth, App->RenderWindowHeight), ImGuiCond_Always);
    ImGui::Begin("__", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDecoration);
    App->Controller.Locked = !ImGui::IsWindowFocused();
    int RenderWindowWidth = ImGui::GetWindowSize().x;
    int RenderWindowHeight = ImGui::GetWindowSize().y;

    ImGui::Image((ImTextureID)App->TonemapTexture->TextureID, ImVec2(App->Window->Width, App->Window->Height));
    ImGui::End();
}
}