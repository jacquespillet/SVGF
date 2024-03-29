#include "App.h"
#include <GL/glew.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>


#include "Window.h"
#include "ShaderGL.h"
#include "TextureGL.h"
#include "BufferCu.cuh"
#include "CudaUtil.h"
#include "PathTrace.cu"
#include "Scene.h"


namespace gpupt
{
std::shared_ptr<application> application::Singleton = {};

application *application::Get()
{
    if(Singleton==nullptr){
        Singleton = std::make_shared<application>();
    }

    return Singleton.get();
}

void application::InitImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(Window->Handle, true);
    ImGui_ImplOpenGL3_Init("#version 460");
}

void application::InitGpuObjects()
{
#if API==API_GL
    PathTracingShader = std::make_shared<shaderGL>("resources/shaders/PathTrace.glsl");
    RenderTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);
#elif API==API_CU
    RenderTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);
    RenderBuffer = std::make_shared<bufferCu>(Window->Width * Window->Height * 4 * sizeof(float));
    RenderTextureMapping = CreateMapping(RenderTexture);    
#endif
}
    
void application::Init()
{
    Window = std::make_shared<window>(800, 600);
    InitImGui();
    Scene = CreateCornellBox();
    InitGpuObjects();
}

void application::StartFrame()
{
    glViewport(0, 0, Window->Width, Window->Height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void application::EndFrame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    Window->Present();
}

void application::Run()
{
    while(!Window->ShouldClose())
    {
        Window->PollEvents();
        StartFrame();

#if API==API_GL
        PathTracingShader->Use();
        PathTracingShader->SetTexture(0, RenderTexture->TextureID, GL_READ_WRITE);
        PathTracingShader->Dispatch(Window->Width / 16 + 1, Window->Height / 16 +1, 1);
#elif API==API_CU
        dim3 blockSize(16, 16);
        dim3 gridSize((Window->Width / blockSize.x)+1, (Window->Height / blockSize.y) + 1);
        TraceKernel<<<gridSize, blockSize>>>((glm::vec4*)RenderBuffer->Data, Window->Width, Window->Height);
        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer->Data, Window->Width * Window->Height * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
#endif
        
        ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(Window->Width, Window->Height), ImGuiCond_Always);
        ImGui::Begin("RenderWindow", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);
        ImGui::Image((ImTextureID)RenderTexture->TextureID, ImVec2(Window->Width, Window->Height));
        ImGui::End();

        EndFrame();
    }
}

void application::Cleanup()
{
       
}

}