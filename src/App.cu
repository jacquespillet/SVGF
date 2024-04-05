#include "App.h"
#include <GL/glew.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>


#include "Window.h"
#include "ShaderGL.h"
#include "TextureGL.h"
#include "BufferCu.cuh"
#include "BufferGL.h"
#include "CudaUtil.h"
#include "Scene.h"
#include "BVH.h"
#if API==API_CU
#include "PathTrace.cu"
#include "TextureArrayCu.cuh"
#endif


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

glm::uvec2 application::GetSize()
{
    return glm::uvec2(
        Singleton->Window->Width,
        Singleton->Window->Height
    );
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
    TonemapShader = std::make_shared<shaderGL>("resources/shaders/Tonemap.glsl");
    RenderTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);
    TonemapTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);    
    TracingParamsBuffer = std::make_shared<uniformBufferGL>(sizeof(tracingParameters), &Params);
    MaterialBuffer = std::make_shared<bufferGL>(sizeof(material) * Scene->Materials.size(), Scene->Materials.data());
#elif API==API_CU
    TonemapTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);
    RenderBuffer = std::make_shared<bufferCu>(Window->Width * Window->Height * 4 * sizeof(float));
    TonemapBuffer = std::make_shared<bufferCu>(Window->Width * Window->Height * 4 * sizeof(float));
    RenderTextureMapping = CreateMapping(TonemapTexture);    
    TracingParamsBuffer = std::make_shared<bufferCu>(sizeof(tracingParameters), &Params);
    MaterialBuffer = std::make_shared<bufferCu>(sizeof(material) * Scene->Materials.size(), Scene->Materials.data());
#endif
}
    
void application::Init()
{
    Window = std::make_shared<window>(800, 600);
    InitImGui();
    Scene = CreateCornellBox();
    BVH = CreateBVH(Scene); 
    
    Params =  GetTracingParameters();  

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

void application::Trace()
{
    if(ResetRender) 
    {
        Params.CurrentSample=0;
    }

    if(Params.CurrentSample < Params.TotalSamples)
    {
        TracingParamsBuffer->updateData(&Params, sizeof(tracingParameters));
        Scene->CamerasBuffer->updateData(0 * sizeof(camera), Scene->Cameras.data(), Scene->Cameras.size() * sizeof(camera));

    #if API==API_GL
        PathTracingShader->Use();
        PathTracingShader->SetTexture(0, RenderTexture->TextureID, GL_READ_WRITE);
        PathTracingShader->SetSSBO(BVH->TrianglesBuffer, 1);
        PathTracingShader->SetSSBO(BVH->TrianglesExBuffer, 2);
        PathTracingShader->SetSSBO(BVH->BVHBuffer, 3);
        PathTracingShader->SetSSBO(BVH->IndicesBuffer, 4);
        PathTracingShader->SetSSBO(BVH->IndexDataBuffer, 5);
        PathTracingShader->SetSSBO(BVH->TLASInstancesBuffer, 6);
        PathTracingShader->SetSSBO(BVH->TLASNodeBuffer, 7);        
        PathTracingShader->SetSSBO(Scene->CamerasBuffer, 8);
        PathTracingShader->SetSSBO(MaterialBuffer, 12);
        PathTracingShader->SetUBO(TracingParamsBuffer, 9);
        PathTracingShader->SetTextureArray(Scene->TexArray, 13, "SceneTextures");
        PathTracingShader->Dispatch(Window->Width / 16 + 1, Window->Height / 16 +1, 1);
#elif API==API_CU
        dim3 blockSize(16, 16);
        dim3 gridSize((Window->Width / blockSize.x)+1, (Window->Height / blockSize.y) + 1);
        TraceKernel<<<gridSize, blockSize>>>((glm::vec4*)RenderBuffer->Data, Window->Width, Window->Height,
                                            (triangle*)BVH->TrianglesBuffer->Data, (triangleExtraData*) BVH->TrianglesExBuffer->Data, (bvhNode*) BVH->BVHBuffer->Data, (u32*) BVH->IndicesBuffer->Data, (indexData*) BVH->IndexDataBuffer->Data, (bvhInstance*)BVH->TLASInstancesBuffer->Data, (tlasNode*) BVH->TLASNodeBuffer->Data,
                                            (camera*)Scene->CamerasBuffer->Data, (tracingParameters*)TracingParamsBuffer->Data, (material*)MaterialBuffer->Data, Scene->TexArray->TexObject);
        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer->Data, Window->Width * Window->Height * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
#endif
        Params.CurrentSample+= Params.Batch;
    }

#if API==API_GL
    TonemapShader->Use();
    TonemapShader->SetTexture(0, RenderTexture->TextureID, GL_READ_WRITE);
    TonemapShader->SetTexture(1, TonemapTexture->TextureID, GL_READ_WRITE);
    TonemapShader->Dispatch(Window->Width / 16 + 1, Window->Height / 16 + 1, 1);
#elif API==API_CU
    dim3 blockSize(16, 16);
    dim3 gridSize((Window->Width / blockSize.x)+1, (Window->Height / blockSize.y) + 1);
    TonemapKernel<<<gridSize, blockSize>>>((glm::vec4*)RenderBuffer->Data, (glm::vec4*)TonemapBuffer->Data, Window->Width, Window->Height);
    cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, TonemapBuffer->Data, Window->Width * Window->Height * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
#endif

}

void application::Run()
{
    while(!Window->ShouldClose())
    {
        Window->PollEvents();
        StartFrame();
        ResetRender=false;

        ResetRender |= Controller.Update();
        Scene->Cameras[0].Frame = Controller.ModelMatrix;


        Trace();
        
        ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(Window->Width, Window->Height), ImGuiCond_Always);
        ImGui::Begin("RenderWindow", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);
        ImGui::Image((ImTextureID)TonemapTexture->TextureID, ImVec2(Window->Width, Window->Height));
        ImGui::End();

        EndFrame();
    }
}

void application::Cleanup()
{
       
}

}