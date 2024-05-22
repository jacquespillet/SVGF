#include "GUI.h"
#include <imgui.h>

#include "App.h"

#include "BVH.h"
#include <nfd.h>
#include "IO.h"
#include "TextureGL.h"
#include "Buffer.h"
#include "Window.h"
#include "AssetLoader.h"
#include "Framebuffer.h"

#include <glm/gtc/matrix_access.hpp>
#include <glm/trigonometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
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
        rot[i] = glm::rotate(glm::mat4(1.0f), glm::radians(rotation[i]), directionUnary[i]);
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

bool gui::InstancesMultipleGUI()
{
    ImGui::Text("Multiple Instances");

    // if(ImGui::CollapsingHeader("Transform"))
    // {
    //     // XForm
    //     glm::vec3 Scale;
    //     glm::vec3 Rotation;
    //     glm::vec3 Translation;
    //     DecomposeMatrixToComponents(App->Scene->Instances[InstanceInx].Transform, &Translation[0], &Rotation[0], &Scale[0]);

    //     if (ImGui::IsKeyPressed(90))
    //         CurrentGizmoOperation = ImGuizmo::TRANSLATE;
    //     if (ImGui::IsKeyPressed(69))
    //         CurrentGizmoOperation = ImGuizmo::ROTATE;
    //     if (ImGui::IsKeyPressed(82))
    //         CurrentGizmoOperation = ImGuizmo::SCALE;
    //     ImGui::Text("Gizmo Operation : ");
    //     if (ImGui::RadioButton("Translate", CurrentGizmoOperation == ImGuizmo::TRANSLATE))
    //         CurrentGizmoOperation = ImGuizmo::TRANSLATE;
    //     ImGui::SameLine();
    //     if (ImGui::RadioButton("Rotate", CurrentGizmoOperation == ImGuizmo::ROTATE))
    //         CurrentGizmoOperation = ImGuizmo::ROTATE;
    //     ImGui::SameLine();
    //     if (ImGui::RadioButton("Scale", CurrentGizmoOperation == ImGuizmo::SCALE))
    //         CurrentGizmoOperation = ImGuizmo::SCALE;

    //     if (CurrentGizmoOperation != ImGuizmo::SCALE)
    //     {
    //         ImGui::Text("Gizmo Space : ");
    //         if (ImGui::RadioButton("Local", CurrentGizmoMode == ImGuizmo::LOCAL))
    //             CurrentGizmoMode = ImGuizmo::LOCAL;
    //         ImGui::SameLine();
    //         if (ImGui::RadioButton("World", CurrentGizmoMode == ImGuizmo::WORLD))
    //             CurrentGizmoMode = ImGuizmo::WORLD;
    //     }        

    //     TransformChanged |= ImGui::DragFloat3("Position", &Translation[0], 0.1);
    //     TransformChanged |= ImGui::DragFloat3("Rotation", &Rotation[0], 1);

    //     static bool UniformScale = false;
    //     ImGui::Checkbox("Uniform Scale", &UniformScale);
    //     if(UniformScale)
    //     {
    //         float ScaleUniform = (Scale.x + Scale.y + Scale.z) / 3.0f;
    //         TransformChanged |= ImGui::DragFloat("Scale", &ScaleUniform, 0.1);
    //         Scale = glm::vec3(ScaleUniform);
    //     }
    //     else
    //         TransformChanged |= ImGui::DragFloat3("Scale", &Scale[0], 0.1);
        

    //     if(TransformChanged)
    //     {
    //         RecomposeMatrixFromComponents(&Translation[0], &Rotation[0], &Scale[0], App->Scene->Instances[InstanceInx].Transform);
    //         App->Scene->BVH->UpdateTLAS(InstanceInx);
    //         App->ResetRender=true;
    //     }
    // }
    
    return false;
}

void gui::InstanceGUI(int InstanceInx)
{
    if(InstanceInx >= App->Scene->Instances.size()) return;

    bool TransformChanged = false;

    if(ImGui::Button("Delete"))
    {
        App->Scene->RemoveInstance(InstanceInx);
        SelectedInstances[InstanceInx] = false;
        SelectedInstanceIndices.erase(InstanceInx);
        App->ResetRender=true;
        return;
    }

    if(ImGui::Button("Duplicate"))
    {
        instance Instance = App->Scene->Instances[InstanceInx];
        std::string Name = App->Scene->InstanceNames[InstanceInx];
        App->Scene->Instances.push_back(Instance);
        App->Scene->InstanceNames.push_back(Name + "_Duplicated");
        App->Scene->BVH->AddInstance(App->Scene->Instances.size()-1);
    }

    if(ImGui::CollapsingHeader("Transform"))
    {
        // XForm
        glm::vec3 Scale;
        glm::vec3 Rotation;
        glm::vec3 Translation;
        DecomposeMatrixToComponents(App->Scene->Instances[InstanceInx].Transform, &Translation[0], &Rotation[0], &Scale[0]);

        ImGui::Text("Gizmo Operation : ");
        if (ImGui::RadioButton("Translate", CurrentGizmoOperation == ImGuizmo::TRANSLATE))
            CurrentGizmoOperation = ImGuizmo::TRANSLATE;
        if (ImGui::RadioButton("Rotate", CurrentGizmoOperation == ImGuizmo::ROTATE))
            CurrentGizmoOperation = ImGuizmo::ROTATE;
        if (ImGui::RadioButton("Scale", CurrentGizmoOperation == ImGuizmo::SCALE))
            CurrentGizmoOperation = ImGuizmo::SCALE;

        if (CurrentGizmoOperation != ImGuizmo::SCALE)
        {
            ImGui::Text("Gizmo Space : ");
            if (ImGui::RadioButton("Local", CurrentGizmoMode == ImGuizmo::LOCAL))
                CurrentGizmoMode = ImGuizmo::LOCAL;
            ImGui::SameLine();
            if (ImGui::RadioButton("World", CurrentGizmoMode == ImGuizmo::WORLD))
                CurrentGizmoMode = ImGuizmo::WORLD;
        }        

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
            RecomposeMatrixFromComponents(&Translation[0], &Rotation[0], &Scale[0], App->Scene->Instances[InstanceInx].Transform);
            App->Scene->BVH->UpdateTLAS(InstanceInx);

            if(glm::length(App->Scene->Materials[App->Scene->Instances[InstanceInx].Material].Emission) > 1e-3f)
            {
                App->Scene->Lights->Build(App->Scene.get());
            }

            App->ResetRender=true;

        }
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
                App->Scene->BVH->UpdateShape(InstanceInx, SelectedShape);
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
        static int SelectedInstanceMaterial = App->Scene->Instances[InstanceInx].Material;
        for (int i = 0; i < App->Scene->Materials.size(); i++)
        {
            if (ImGui::Selectable(App->Scene->MaterialNames[i].c_str(), SelectedInstanceMaterial == i))
            {
                int PreviousMaterial = App->Scene->Instances[InstanceInx].Material;

                SelectedInstanceMaterial = i;
                App->Scene->Instances[InstanceInx].Material = SelectedInstanceMaterial;
                App->Scene->BVH->UpdateMaterial(InstanceInx, SelectedInstanceMaterial);
                App->ResetRender=true;                        

                // If the new material is emissive, or the old material was emissive, we rebuild the lights
                if(glm::length(App->Scene->Materials[SelectedInstanceMaterial].Emission) > 1e-3f || glm::length(App->Scene->Materials[PreviousMaterial].Emission) > 1e-3f)
                {
                    App->Scene->UpdateLights();
                }

                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }

    
    ImGui::Separator();

    if(ImGui::CollapsingHeader("Material"))
    {
        if(MaterialGUI(App->Scene->Instances[InstanceInx].Material))
        {
            App->ResetRender=true;
        }
    }
}

bool gui::InstancesGUI()
{
    if(ImGui::Button("Reset Project"))
    {
        App->Scene->Clear();
        App->ResetRender=true;
    }
    ImGui::Separator();
    
    bool Changed = false;
    for (int i = 0; i < App->Scene->Instances.size(); i++)
    {
        
        if (ImGui::Selectable(App->Scene->InstanceNames[i].c_str(), SelectedInstances[i]))
        {
            if (ImGui::GetIO().KeyCtrl)
            {
                if(SelectedInstances[i]) 
                {
                    SelectedInstances[i]=false;
                    SelectedInstanceIndices.erase(i);
                }
                else 
                {
                    SelectedInstances[i] = true;
                    SelectedInstanceIndices.insert(i);
                }
            }
            else
            {
                SelectedInstanceIndices.clear();
                SelectedInstanceIndices.insert(i);
                for (size_t j = 0; j < SelectedInstances.size(); j++)
                {
                    SelectedInstances[j]=false;
                }
                SelectedInstances[i] = true;
            }
        }
    }
    
    if(ImGui::Button("Add"))
    {
        ImGui::OpenPopup("Add_Instance_Popup");
    }
    if(ImGui::Button("Clear"))
    {
        App->Scene->ClearInstances();
        App->ResetRender=true;
    }
    if(ImGui::BeginPopup("Add_Instance_Popup"))
    {
        static int SelectedShape = -1;
        static int SelectedMaterial = -1;
        
        static char Name[256] = "New Instance";
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
        static bool CreateMaterial = true;
        ImGui::Checkbox("New Material", &CreateMaterial);
        if(!CreateMaterial)
        {
            ImGui::Text("Select Material");
            for (int i = 0; i < App->Scene->Materials.size(); i++)
            {
                ImGui::PushID(App->Scene->Shapes.size() + i);
                if (ImGui::Selectable(App->Scene->MaterialNames[i].c_str(), SelectedMaterial == i, ImGuiSelectableFlags_DontClosePopups))
                    SelectedMaterial = i;
                ImGui::PopID();
            }
        }

        int MaterialInx = CreateMaterial ? App->Scene->Materials.size() : SelectedMaterial;
        if(ImGui::Button("Confirm"))
        {
            if(SelectedShape != -1 && MaterialInx != -1)
            {
                if(CreateMaterial)
                {
                    App->Scene->Materials.emplace_back();
                    material &NewMaterial = App->Scene->Materials.back(); 
                    NewMaterial.Colour = {0.725f, 0.71f, 0.68f};            
                    App->Scene->MaterialNames.push_back("New Material");
                    App->Scene->MaterialBuffer->Reallocate(App->Scene->Materials.data(), App->Scene->Materials.size() * sizeof(material));
                }

                App->Scene->Instances.emplace_back();
                instance &NewInstance = App->Scene->Instances.back(); 
                NewInstance.Shape = SelectedShape;
                NewInstance.Material = MaterialInx;
                App->Scene->InstanceNames.push_back(std::string(Name));      
                App->Scene->BVH->AddInstance(App->Scene->Instances.size()-1);
                App->ResetRender = true;
                App->Scene->CheckNames();
                
                ImGui::CloseCurrentPopup();
            }
        }     
        ImGui::EndPopup(); 
    }


    ImGui::Separator();

    if(SelectedInstanceIndices.size() > 1)
    {
        InstancesMultipleGUI();
    }
    else if(SelectedInstanceIndices.size() == 1)
    {
        InstanceGUI(*SelectedInstanceIndices.begin());
    }

    

    return Changed;
}


bool gui::TexturePickerGUI(std::string Name, int &TextureInx, std::vector<std::string> &TextureNames)
{
    bool Changed = false;
    std::string TextureName = TextureInx >=0 ? TextureNames[TextureInx] : "Empty";
    ImGui::Text(Name.c_str()); ImGui::SameLine(); 
    ImGui::Text(TextureName.c_str()); ImGui::SameLine(); 
    
    ImGui::PushID(Name.c_str());
    if(ImGui::Button("Choose"))
    {
        ImGui::PopID();
        ImGui::OpenPopup((Name + "TexturePicker").c_str());
    }
    else 
        ImGui::PopID();

    if(ImGui::BeginPopup((Name + "TexturePicker").c_str()))
    {
        if (ImGui::Selectable("None", TextureInx == -1, ImGuiSelectableFlags_DontClosePopups))
        {
            TextureInx=-1;
        }
        for (int i = 0; i < TextureNames.size(); i++)
        {
            if (ImGui::Selectable(TextureNames[i].c_str(), TextureInx == i, ImGuiSelectableFlags_DontClosePopups))
                TextureInx = i;
        }

        if(ImGui::Button("Choose"))
        {
            Changed |= true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    return Changed;
}

bool gui::MaterialGUI(int MaterialInx)
{
    if(MaterialInx >= App->Scene->Materials.size()) return false;
    
    bool Changed = false;

    ImGui::Text(App->Scene->MaterialNames[MaterialInx].c_str());
    material &Mat = App->Scene->Materials[MaterialInx];

    int MaterialTypeInt = (int)Mat.MaterialType;
    Changed |= ImGui::Combo("Type", &MaterialTypeInt, "Matte\0PBR\0Volumetric\0Glass\0Subsurface\0\0");
    Mat.MaterialType = (float)MaterialTypeInt;

    glm::vec3 PrevEmission = Mat.Emission;
    Changed |= ImGui::DragFloat3("Emission", &Mat.Emission[0], 2, 0, 100);
    Changed |= ImGui::ColorEdit3("Colour", &Mat.Colour[0]);
    Changed |= ImGui::DragFloat("Opacity", &Mat.Opacity, 0.01f, 0, 1);

    if(Mat.MaterialType == MATERIAL_TYPE_VOLUMETRIC || Mat.MaterialType == MATERIAL_TYPE_GLASS)
    {
        ImGui::Separator();
        Changed |= ImGui::DragFloat("TrDepth", &Mat.TransmissionDepth, 0.001f, 0, 1);
        Changed |= ImGui::DragFloat("Anisotropy", &Mat.Anisotropy, 0.01f, -1, 1);
        Changed |= ImGui::ColorEdit3("Scattering", &Mat.ScatteringColour[0]);
    }

    if(Mat.MaterialType != MATERIAL_TYPE_MATTE)
    {
        Changed |= ImGui::DragFloat("Roughness", &Mat.Roughness, 0.01f, 0, 1);
        Changed |= ImGui::DragFloat("Metallic", &Mat.Metallic, 0.01f, 0, 1);
    }
    
    ImGui::Separator();
    ImGui::Text("Textures : ");

    Changed |= TexturePickerGUI("Colour", Mat.ColourTexture, App->Scene->TextureNames);
    Changed |= TexturePickerGUI("Roughness", Mat.RoughnessTexture, App->Scene->TextureNames);
    Changed |= TexturePickerGUI("Normal", Mat.NormalTexture, App->Scene->TextureNames);
    Changed |= TexturePickerGUI("Emission", Mat.EmissionTexture, App->Scene->TextureNames);

    // // Colour
  
    // // Roughness
    // std::string RoughnessTextureName = Mat.RoughnessTexture >=0 ? App->Scene->TextureNames[Mat.RoughnessTexture] : "Empty";
    // ImGui::Text("Roughness"); ImGui::SameLine(); ImGui::Text(RoughnessTextureName.c_str()); ImGui::SameLine(); if(ImGui::Button("Choose"))
    // {
    //     ImGui::OpenPopup("RoughnessTexturePicker");
    // }

    // if(ImGui::BeginPopup("RoughnessTexturePicker"))
    // {
    //     static int SelectedIndex = -1;
    //     for (int i = 0; i < App->Scene->Textures.size(); i++)
    //     {
    //         if (ImGui::Selectable(App->Scene->TextureNames[i].c_str(), SelectedIndex == i, ImGuiSelectableFlags_DontClosePopups))
    //             SelectedIndex = i;

    //     }

    //     if(ImGui::Button("Choose") && SelectedIndex >=0)
    //     {
    //         Mat.RoughnessTexture = SelectedIndex;
    //         Changed |= true;
    //         ImGui::CloseCurrentPopup();
    //     }
    //     ImGui::EndPopup();
    // }

    // // Normal
    // std::string NormalTextureName = Mat.NormalTexture >=0 ? App->Scene->TextureNames[Mat.NormalTexture] : "Empty";
    // ImGui::Text("Normal"); ImGui::SameLine(); ImGui::Text(NormalTextureName.c_str()); ImGui::SameLine(); if(ImGui::Button("Choose"))
    // {
    //     ImGui::OpenPopup("NormalTexturePicker");
    // }

    // if(ImGui::BeginPopup("NormalTexturePicker"))
    // {
    //     static int SelectedIndex = -1;
    //     for (int i = 0; i < App->Scene->Textures.size(); i++)
    //     {
    //         if (ImGui::Selectable(App->Scene->TextureNames[i].c_str(), SelectedIndex == i, ImGuiSelectableFlags_DontClosePopups))
    //             SelectedIndex = i;

    //     }

    //     if(ImGui::Button("Choose") && SelectedIndex >=0)
    //     {
    //         Mat.NormalTexture = SelectedIndex;
    //         Changed |= true;
    //         ImGui::CloseCurrentPopup();
    //     }
    //     ImGui::EndPopup();
    // }

    // // Emission
    // std::string EmissionTextureName = Mat.EmissionTexture >=0 ? App->Scene->TextureNames[Mat.EmissionTexture] : "Empty";
    // ImGui::Text("Emission"); ImGui::SameLine(); ImGui::Text(EmissionTextureName.c_str()); ImGui::SameLine(); if(ImGui::Button("Choose"))
    // {
    //     ImGui::OpenPopup("EmissionTexturePicker");
    // }

    // if(ImGui::BeginPopup("EmissionTexturePicker"))
    // {
    //     static int SelectedIndex = -1;
    //     for (int i = 0; i < App->Scene->Textures.size(); i++)
    //     {
    //         if (ImGui::Selectable(App->Scene->TextureNames[i].c_str(), SelectedIndex == i, ImGuiSelectableFlags_DontClosePopups))
    //             SelectedIndex = i;

    //     }

    //     if(ImGui::Button("Choose") && SelectedIndex >=0)
    //     {
    //         Mat.EmissionTexture = SelectedIndex;
    //         Changed |= true;
    //         ImGui::CloseCurrentPopup();
    //     }
    //     ImGui::EndPopup();
    // }

    if((glm::length(PrevEmission) <= 1e-3f && glm::length(Mat.Emission) > 1e-3f) || (glm::length(PrevEmission) > 1e-3f && glm::length(Mat.Emission)<= 1e-3f))
    {
        App->Scene->UpdateLights();
        Changed |= true;
    }

    if(Changed)
        App->Scene->UploadMaterial(MaterialInx);
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
            App->Scene->MaterialNames.push_back(Name);
            App->Scene->MaterialBuffer->Reallocate(App->Scene->Materials.data(), App->Scene->Materials.size() * sizeof(material));
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if(SelectedMaterial != -1)
    {
        if(ImGui::Button("Delete"))
        {
            App->Scene->Materials.erase(App->Scene->Materials.begin() + SelectedMaterial);
            App->Scene->MaterialNames.erase(App->Scene->MaterialNames.begin() + SelectedMaterial);
            App->Scene->MaterialBuffer->Reallocate(App->Scene->Materials.data(), App->Scene->Materials.size() * sizeof(material));
            ImGui::CloseCurrentPopup();
        }
        if(ImGui::Button("Duplicate"))
        {
            App->Scene->Materials.push_back(App->Scene->Materials[SelectedMaterial]);
            App->Scene->MaterialNames.push_back(App->Scene->MaterialNames[SelectedMaterial] + "_Duplicate");
            App->Scene->MaterialBuffer->Reallocate(App->Scene->Materials.data(), App->Scene->Materials.size() * sizeof(material));
            ImGui::CloseCurrentPopup();
        }
        Changed |= MaterialGUI(SelectedMaterial);
    }

    return Changed;
}

bool gui::ShapeGUI(int ShapeInx)
{
    if(ShapeInx >= App->Scene->Shapes.size()) return false;

    bool Changed = false;

    ImGui::Text(App->Scene->ShapeNames[ShapeInx].c_str());

    ImGui::Separator();

    ImGui::Text("Triangles Count : "); ImGui::SameLine(); ImGui::Text( std::to_string(App->Scene->Shapes[ShapeInx].Triangles.size()).c_str());

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
            ImGui::OpenPopup("Instance_Shape_Selection");
    }

    if (ImGui::BeginPopup("Instance_Shape_Selection"))
    {
        static bool LoadInstances = true;
        static bool LoadMaterials = true;
        static bool LoadTextures = true;
        static float GlobalScale = 1.0f;
        ImGui::Checkbox("Load Instances", &LoadInstances); ImGui::SameLine(); ImGui::Checkbox("Load Materials", &LoadMaterials); ImGui::SameLine(); ImGui::Checkbox("Load Textures", &LoadTextures);
        ImGui::DragFloat("Global Scale", &GlobalScale, 0.01f, 0, 1);
        if(ImGui::Button("Add"))
        {
            nfdpathset_t ShapesPaths;
            nfdresult_t Result = NFD_OpenDialogMultiple(NULL, NULL, &ShapesPaths);
            if ( Result == NFD_OKAY ) 
            {
                for (size_t i = 0; i < NFD_PathSet_GetCount(&ShapesPaths); ++i )
                {
                    nfdchar_t *Path = NFD_PathSet_GetPath(&ShapesPaths, i);
                    LoadAsset(Path, App->Scene.get(), LoadInstances, LoadMaterials, LoadTextures, GlobalScale);
                }
                NFD_PathSet_Free(&ShapesPaths);            
            }   
        }
        ImGui::EndPopup();
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
    if(CameraInx >= App->Scene->Cameras.size()) return false;

    bool Changed = false;

    ImGui::Text(App->Scene->CameraNames[CameraInx].c_str());

    Changed |= ImGui::DragFloat("Aperture", &App->Scene->Cameras[CameraInx].FOV, 0.5f, 10.0f, 180.0f);
    
    if(ImGui::Button("Duplicate"))
    {
        App->Scene->Cameras.push_back(App->Scene->Cameras[CameraInx]);
        App->Scene->CameraNames.push_back(App->Scene->CameraNames[CameraInx] + "_Duplicated");
        App->Scene->Cameras.back().Controlled=false;
        App->Scene->CamerasBuffer = std::make_shared<buffer>(App->Scene->Cameras.size() * sizeof(camera), App->Scene->Cameras.data());
        App->Scene->EnvironmentsBuffer = std::make_shared<buffer>(App->Scene->Environments.size() * sizeof(environment), App->Scene->Environments.data());
    }

    if(ImGui::Button("Make Current"))
    {
        App->Params.CurrentCamera = CameraInx;
        Changed = true;
    }

    if(ImGui::Checkbox("Controlled", (bool*)&App->Scene->Cameras[CameraInx].Controlled))
    {
        if(App->Scene->Cameras[CameraInx].Controlled)
        {
            for(int i=0; i<App->Scene->Cameras.size(); i++)
            {
                if(i != CameraInx)
                {
                    App->Scene->Cameras[i].Controlled=0;
                }
            }
        }
    }
    
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
        App->Scene->Cameras.emplace_back();
        camera &Camera = App->Scene->Cameras.back();
        Camera.FOV = 60.0f;
        Camera.Aspect = (float)App->RenderWindowWidth / (float)App->RenderWindowWidth;
        Camera.Controlled = 1;  
        App->Scene->CameraNames.push_back("New Camera");
        App->Scene->PreProcess();
                    
    }

    return Changed;        
}

void gui::TextureGUI(int TextureInx, std::vector<texture> &Textures, std::vector<std::string> &TextureNames)
{
    if(TextureInx >= Textures.size()) return;

    ImGui::Text(TextureNames[TextureInx].c_str());
    ImGui::Text("Width : "); ImGui::SameLine(); ImGui::Text(std::to_string(Textures[TextureInx].Width).c_str());
    ImGui::Text("Height : "); ImGui::SameLine(); ImGui::Text(std::to_string(Textures[TextureInx].Height).c_str());
    ImGui::Text("Channels : "); ImGui::SameLine(); ImGui::Text(std::to_string(Textures[TextureInx].Height).c_str());
}

void gui::TexturesGUI()
{
    bool Changed = false;
    
    if (ImGui::BeginTabBar("", ImGuiTabBarFlags_None))
    {
        if (ImGui::BeginTabItem("Textures"))
        {

            for (int i = 0; i < App->Scene->Textures.size(); i++)
            {
                if (ImGui::Selectable(App->Scene->TextureNames[i].c_str(), SelectedTexture == i))
                    SelectedTexture = i;
            }

            if(ImGui::Button("Add"))
            {
                nfdpathset_t ImagePaths;
                nfdresult_t Result = NFD_OpenDialogMultiple(NULL, NULL, &ImagePaths);
                if ( Result == NFD_OKAY ) 
                {
                    for (size_t i = 0; i < NFD_PathSet_GetCount(&ImagePaths); ++i )
                    {
                        nfdchar_t *Path = NFD_PathSet_GetPath(&ImagePaths, i);
                        App->Scene->Textures.emplace_back();
                        texture &Texture = App->Scene->Textures.back(); 
                        Texture.SetFromFile(Path, App->Scene->TextureWidth, App->Scene->TextureHeight);
                        App->Scene->TextureNames.push_back(ExtractFilename(Path));
                    }
                    App->Scene->ReloadTextureArray();
                    NFD_PathSet_Free(&ImagePaths);            
                }             
            }
            ImGui::Separator();

            if(SelectedTexture != -1)
            {
                TextureGUI(SelectedTexture, App->Scene->Textures, App->Scene->TextureNames);
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Environment Textures"))
        {

            for (int i = 0; i < App->Scene->EnvTextures.size(); i++)
            {
                if (ImGui::Selectable(App->Scene->EnvTextureNames[i].c_str(), SelectedEnvTexture == i))
                    SelectedEnvTexture = i;
            }

            if(ImGui::Button("Add"))
            {
                nfdpathset_t ImagePaths;
                nfdresult_t Result = NFD_OpenDialogMultiple(NULL, NULL, &ImagePaths);
                if ( Result == NFD_OKAY ) 
                {
                    for (size_t i = 0; i < NFD_PathSet_GetCount(&ImagePaths); ++i )
                    {
                        nfdchar_t *Path = NFD_PathSet_GetPath(&ImagePaths, i);
                        App->Scene->EnvTextures.emplace_back();
                        texture &Texture = App->Scene->EnvTextures.back(); 
                        Texture.SetFromFile(Path, App->Scene->EnvTextureWidth, App->Scene->EnvTextureHeight);
                        App->Scene->EnvTextureNames.push_back(ExtractFilename(Path));
                    }
                    App->Scene->ReloadTextureArray();
                    NFD_PathSet_Free(&ImagePaths);            
                }             
            }
            ImGui::Separator();

            if(SelectedEnvTexture != -1)
            {
                TextureGUI(SelectedEnvTexture, App->Scene->EnvTextures, App->Scene->EnvTextureNames);
            }
            ImGui::EndTabItem();
        }        
        ImGui::EndTabBar();
    }

}

bool gui::EnvironmentGUI(int EnvironmentInx)
{
    if(EnvironmentInx >= App->Scene->Environments.size()) return false;

    bool Changed=false;
    environment &Env = App->Scene->Environments[EnvironmentInx];
    
    glm::vec3 PrevEmission = Env.Emission;

    glm::vec3 Scale;
    glm::vec3 Rotation;
    glm::vec3 Translation;
    DecomposeMatrixToComponents(Env.Transform, &Translation[0], &Rotation[0], &Scale[0]);

    Changed |= ImGui::DragFloat3("Rotation", &Rotation[0], 1);

    static bool UniformEmission = true;
    ImGui::Checkbox("Uniform Emission", &UniformEmission);

    if(UniformEmission)
    {
        float Scale = (Env.Emission.x + Env.Emission.y + Env.Emission.z) / 3.0f;
        Changed |= ImGui::DragFloat("Emission", &Scale, 0.5f, 0, 100000);
        Env.Emission = glm::vec3(Scale, Scale, Scale);
    }
    else
    {
        Changed |= ImGui::DragFloat3("Emission", &Scale[0], 0.5f, 0, 100000);
    }

    if(Changed)
    {
        RecomposeMatrixFromComponents(&Translation[0], &Rotation[0], &Scale[0], Env.Transform);
    }

    bool TextureChanged = TexturePickerGUI("Texture", Env.EmissionTexture, App->Scene->EnvTextureNames);
    

    if((glm::length(PrevEmission) <= 1e-3f && glm::length(Env.Emission) > 1e-3f) || (glm::length(PrevEmission) > 1e-3f && glm::length(Env.Emission)<= 1e-3f) || TextureChanged)
    {
        App->Scene->UpdateLights();
        Changed = true;
    }

    return Changed;
}

bool gui::EnvironmentsGUI()
{
    bool Changed = false;

    for (int i = 0; i < App->Scene->Environments.size(); i++)
    {
        if (ImGui::Selectable(App->Scene->EnvironmentNames[i].c_str(), SelectedEnvironment == i))
            SelectedEnvironment = i;
    }

    if(ImGui::Button("Add"))
    {
        App->Scene->Environments.emplace_back();
        App->Scene->EnvironmentNames.push_back("Sky");
        environment &Env = App->Scene->Environments.back();
        Env.Emission = {1,1,1};
        Env.EmissionTexture = -1;
        Env.Transform = glm::mat4(1);
        App->Scene->Lights->Build(App->Scene.get());
        App->Scene->EnvironmentsBuffer = std::make_shared<buffer>(App->Scene->Environments.size() * sizeof(environment), App->Scene->Environments.data());
        App->ResetRender=true;
    }

    ImGui::Separator();

    if(SelectedEnvironment != -1)
    {
        Changed |= EnvironmentGUI(SelectedEnvironment);
    }

    return Changed;
}

bool gui::TracingGUI()
{
    bool Changed = false;
    

    ImGui::Text("Samples : ");
    std::string SamplesStr = std::to_string(App->Params.CurrentSample) + "/" + std::to_string(App->Params.TotalSamples);
    ImGui::Text(SamplesStr.c_str());
    ImGui::DragInt("Total Samples", &App->Params.TotalSamples);

    ImGui::SliderInt("Batches", &App->Params.Batch, 0, 32);
    Changed |= ImGui::SliderInt("Bounces", &App->Params.Bounces, 0, 32);
    Changed |= ImGui::DragFloat("Clamp", &App->Params.Clamp, 0.1f, 0.0f, 32.0f);    
    
    ImGui::Checkbox("SVGF", &App->DoSVGF);
    if(App->DoSVGF)
        Changed |= ImGui::Combo("Debug Output", (int*)&App->SVGFDebutOutput, "Final Output\0Raw Output\0Normal\0Motion\0Position\0Barycentric Coords\0Temporal Filter\0A-Trous Wavelet Filter\0\0");
    
    
    if(ImGui::DragInt("Resolution", &App->RenderResolution, 10, 128, 3840))
    {
        App->CalculateWindowSizes();
    }

    Changed |= ImGui::Combo("Type", &App->Params.SamplingMode, "BSDF\0Light\0Bsdf + Light\0MIS\0\0");

    if(ImGui::Button("Save"))
    {
        nfdchar_t *SavePath = 0;
        nfdresult_t Result = NFD_SaveDialog(NULL, NULL, &SavePath);

        if(Result == NFD_OKAY)
        {
            App->Scene->ToFile(SavePath);
            this->LoadedFile = SavePath;
        }
    }

    if(ImGui::Button("Load"))
    {
        nfdchar_t *LoadPath = 0;
        nfdresult_t Result = NFD_OpenDialog(NULL, NULL, &LoadPath);
        if(Result == NFD_OKAY)
        {
            App->Scene->Clear();
            App->Scene->FromFile(LoadPath);
            App->Scene->PreProcess();
            App->ResetRender=true;
            this->LoadedFile = LoadPath;
        }
    }

    if(ImGui::Button("Save Render"))
    {
        nfdchar_t *SavePath = 0;
        nfdresult_t Result = NFD_SaveDialog(NULL, NULL, &SavePath);

        if(Result == NFD_OKAY)
        {
            App->SaveRender(SavePath);
        }
    }

    return Changed;
}


void gui::GUI()
{
    if (ImGui::IsKeyPressed(ImGuiKey_T))
        CurrentGizmoOperation = ImGuizmo::TRANSLATE;
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        CurrentGizmoOperation = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed(ImGuiKey_S))
        CurrentGizmoOperation = ImGuizmo::SCALE;    
    
    
    if(ImGui::IsKeyPressed(ImGuiKey_S) && ImGui::GetIO().KeyCtrl)
    {
        if(LoadedFile != "")
            App->Scene->ToFile(LoadedFile);
        else
        {
            nfdchar_t *SavePath = 0;
            nfdresult_t Result = NFD_SaveDialog(NULL, NULL, &SavePath);

            if(Result == NFD_OKAY)
            {
                App->Scene->ToFile(SavePath);
                this->LoadedFile = SavePath;
            }
        }
    }
    
    this->SelectedInstances.resize(App->Scene->Instances.size(), false);
    
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
        if (ImGui::BeginTabItem("Environments"))
        {
            if(EnvironmentsGUI())
            {
                App->Scene->EnvironmentsBuffer->updateData(App->Scene->Environments.data(), App->Scene->Environments.size() * sizeof(environment));
                App->ResetRender=true;
            }
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
    App->Controller.Locked = !ImGui::IsWindowFocused() || ImGuizmo::IsUsing();
    int RenderWindowWidth = ImGui::GetWindowSize().x;
    int RenderWindowHeight = ImGui::GetWindowSize().y;

    

    ImGui::Image((ImTextureID)App->RenderTexture->TextureID, ImVec2(RenderWindowWidth, RenderWindowHeight),ImVec2(0, 1), ImVec2(1, 0));
    // ImGui::Image((ImTextureID)App->Framebuffer[0]->GetTexture(1), ImVec2(RenderWindowWidth, RenderWindowHeight), ImVec2(0, 1), ImVec2(1, 0));
    if(SelectedInstanceIndices.size()==1)
    {
        int SelectedInstance = *SelectedInstanceIndices.begin();
        ImGuiIO &io = ImGui::GetIO();

        ImGuizmo::SetRect(GuiWidth, 0, App->RenderWindowWidth, App->RenderWindowHeight);
        instance &Instance = App->Scene->Instances[SelectedInstance];
        camera &Camera = App->Scene->Cameras[int(App->Params.CurrentCamera)];
        glm::mat4 ViewMatrix = glm::inverse(Camera.Frame);
        
        glm::mat4 ModelMatrix = App->Scene->Instances[SelectedInstance].Transform;

        ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList()); 
        
        glm::mat4 CorrectedTransform = glm::translate(ModelMatrix, App->Scene->Shapes[Instance.Shape].Centroid);
        if(ImGuizmo::Manipulate(glm::value_ptr(ViewMatrix), glm::value_ptr(Camera.ProjectionMatrix), CurrentGizmoOperation, CurrentGizmoMode, glm::value_ptr(CorrectedTransform), NULL, NULL))
        {
            App->Scene->Instances[SelectedInstance].Transform = glm::translate(CorrectedTransform, -App->Scene->Shapes[Instance.Shape].Centroid);
            App->Scene->BVH->UpdateTLAS(SelectedInstance);
            
            if(glm::length(App->Scene->Materials[App->Scene->Instances[SelectedInstance].Material].Emission) > 1e-3f)
            {
                App->Scene->Lights->Build(App->Scene.get());
            }
                        
            App->ResetRender=true;
        }
    }
    ImGui::End();

}
}