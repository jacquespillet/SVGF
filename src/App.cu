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
#include "GUI.h"
#if API==API_CU
#include "PathTrace.cu"
#include "TextureArrayCu.cuh"
#endif

namespace gpupt
{

void OnResizeWindow(window &Window, glm::ivec2 NewSize);

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
        Singleton->RenderWidth,
        Singleton->RenderHeight
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
    LightsBuffer = std::make_shared<bufferGL>(sizeof(lights), &Lights);
#elif API==API_CU
    TonemapTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);
    RenderBuffer = std::make_shared<bufferCu>(Window->Width * Window->Height * 4 * sizeof(float));
    TonemapBuffer = std::make_shared<bufferCu>(Window->Width * Window->Height * 4 * sizeof(float));
    RenderTextureMapping = CreateMapping(TonemapTexture);    
    TracingParamsBuffer = std::make_shared<bufferCu>(sizeof(tracingParameters), &Params);
    MaterialBuffer = std::make_shared<bufferCu>(sizeof(material) * Scene->Materials.size(), Scene->Materials.data());
    LightsBuffer = std::make_shared<bufferCu>(sizeof(lights), &Lights);
#endif
}
    
void application::Init()
{
    this->GUI = std::make_shared<gui>(this);

    GUI->GuiWidth = 200;
    RenderResolution = 600;
    

    Window = std::make_shared<window>(800, 600);
    Window->OnResize = OnResizeWindow;

    InitImGui();
    Scene = CreateCornellBox();
    BVH = CreateBVH(Scene); 
    
    Params =  GetTracingParameters();  
    Lights = GetLights(Scene, Params);

    InitGpuObjects();
    Inited=true;
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
        PathTracingShader->SetSSBO(LightsBuffer, 10);
        PathTracingShader->SetTextureArray(Scene->TexArray, 13, "SceneTextures");
        PathTracingShader->Dispatch(RenderWidth / 16 + 1, RenderHeight / 16 +1, 1);
#elif API==API_CU
        dim3 blockSize(16, 16);
        dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);
        TraceKernel<<<gridSize, blockSize>>>((glm::vec4*)RenderBuffer->Data, RenderWidth, RenderHeight,
                                            (triangle*)BVH->TrianglesBuffer->Data, (triangleExtraData*) BVH->TrianglesExBuffer->Data, (bvhNode*) BVH->BVHBuffer->Data, (uint32_t*) BVH->IndicesBuffer->Data, (indexData*) BVH->IndexDataBuffer->Data, (bvhInstance*)BVH->TLASInstancesBuffer->Data, (tlasNode*) BVH->TLASNodeBuffer->Data,
                                            (camera*)Scene->CamerasBuffer->Data, (tracingParameters*)TracingParamsBuffer->Data, (material*)MaterialBuffer->Data, Scene->TexArray->TexObject, (lights*)LightsBuffer->Data);
        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
#endif
        Params.CurrentSample+= Params.Batch;
    }

#if API==API_GL
    TonemapShader->Use();
    TonemapShader->SetTexture(0, RenderTexture->TextureID, GL_READ_WRITE);
    TonemapShader->SetTexture(1, TonemapTexture->TextureID, GL_READ_WRITE);
    TonemapShader->Dispatch(RenderWidth / 16 + 1, RenderHeight / 16 + 1, 1);
#elif API==API_CU
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);
    TonemapKernel<<<gridSize, blockSize>>>((glm::vec4*)RenderBuffer->Data, (glm::vec4*)TonemapBuffer->Data, RenderWidth, RenderHeight);
    cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, TonemapBuffer->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
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


        GUI->GUI();
        Trace();
        

        EndFrame();
    }  
}

void application::Cleanup()
{
       
}

void application::UploadMaterial(int MaterialInx)
{
    MaterialBuffer->updateData((size_t)MaterialInx * sizeof(material), (void*)&Scene->Materials[MaterialInx], sizeof(material));
}


void application::ResizeRenderTextures()
{
    if(!Inited) return;
#if API==API_GL
    RenderTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);
    TonemapTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);    
#elif API==API_CU
    cudaDeviceSynchronize();
    TonemapTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);
    RenderBuffer = std::make_shared<bufferCu>(RenderWidth * RenderHeight * 4 * sizeof(float));
    TonemapBuffer = std::make_shared<bufferCu>(RenderWidth * RenderHeight * 4 * sizeof(float));
    RenderTextureMapping = CreateMapping(TonemapTexture);
#endif

    Params.CurrentSample=0;
    Scene->Cameras[0].Aspect = (float)RenderWidth / (float)RenderHeight;

    ResetRender=true;
}

void application::CalculateWindowSizes()
{
    if(!Inited) return;
    
    // GUIWindow size
    RenderWindowWidth = Window->Width - GUI->GuiWidth;
    RenderWindowHeight = Window->Height;

    // Aspect ratio of the GUI window
    RenderAspectRatio = (float)RenderWindowWidth / (float)RenderWindowHeight;

    // Set the render width accordingly
    uint32_t NewRenderWidth, NewRenderHeight;

    if(RenderAspectRatio > 1)
    {
        NewRenderWidth = RenderResolution * RenderAspectRatio;
        NewRenderHeight = RenderResolution;
    }
    else
    {
        NewRenderWidth = RenderResolution;
        NewRenderHeight = RenderResolution / RenderAspectRatio;
    }

    if(Inited && Scene->Cameras[0].Aspect != RenderAspectRatio)
    {
        Scene->Cameras[0].Aspect = RenderAspectRatio;
        ResetRender = true;
    }

    if(NewRenderWidth != RenderWidth || NewRenderHeight != RenderHeight)
    {
        RenderWidth = NewRenderWidth;
        RenderHeight = NewRenderHeight;
        ResizeRenderTextures();
    }
}

void application::OnResize(uint32_t NewWidth, uint32_t NewHeight)
{
    // std::cout << "ON RESIZE " << NewWidth << std::endl;
    Window->Width = NewWidth;
    Window->Height = NewHeight;
}

void OnResizeWindow(window &Window, glm::ivec2 NewSize)
{
	application::Get()->OnResize(NewSize.x, NewSize.y);
}

}