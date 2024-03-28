#include "App.h"
#include <GL/glew.h>

#include "Window.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

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
    
void application::Init()
{
    Window = std::make_shared<window>(800, 600);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(Window->Handle, true);
    ImGui_ImplOpenGL3_Init("#version 460");    

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

        ImGui::ShowDemoWindow();

        EndFrame();
    }
}

void application::Cleanup()
{
    
}

}