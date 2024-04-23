#include "App.h"
#include <GL/glew.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>


#include "Window.h"
#include "ShaderGL.h"
#include "TextureGL.h"
#include "Buffer.h"
#include "CudaUtil.h"
#include "Scene.h"
#include "BVH.h"
#include "GUI.h"
#include "Buffer.h"
#if API==API_CU
#include "PathTrace.cu"
#include "TextureArrayCu.cuh"
#endif
#include "GLTexToCuBuffer.cu"
#include <ImGuizmo.h>


#include <iostream>
#define CUDA_CHECK_ERROR(err) \
    do { \
        cudaError_t error = err; \
        if (error != cudaSuccess) { \
            std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            assert(false); \
        } \
    } while (0)

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
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(Window->Handle, true);
    ImGui_ImplOpenGL3_Init("#version 460");
}

void application::InitGpuObjects()
{
#if API==API_GL
    PathTracingShader = std::make_shared<shaderGL>("resources/shaders/PathTrace.glsl");
    TonemapShader = std::make_shared<shaderGL>("resources/shaders/Tonemap.glsl");

    TonemapTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);    
    RenderTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);
    DenoisedTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);    

    DenoiseMapping = CreateMapping(DenoisedTexture, true);    
    RenderMapping = CreateMapping(RenderTexture);    

    TracingParamsBuffer = std::make_shared<uniformBufferGL>(sizeof(tracingParameters), &Params);    
#elif API==API_CU  
    TonemapTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);
    RenderBuffer = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec4));
    TonemapBuffer = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec4));
    RenderTextureMapping = CreateMapping(TonemapTexture);

    TracingParamsBuffer = std::make_shared<buffer>(sizeof(tracingParameters), &Params);
#endif
}

void application::CreateOIDNFilter()
{
    cudaStreamCreate(&Stream);
    Device = oidn::newCUDADevice(0, Stream);
    Device.commit();

    Filter = Device.newFilter("RT");
    Filter.set("hdr", true);        
    Filter.set("cleanAux", true);           
    Filter.set("quality", OIDN_QUALITY_BALANCED);        
    Filter.commit();
}


    
void application::Init()
{
    this->GUI = std::make_shared<gui>(this);

    GUI->GuiWidth = 200;
    RenderResolution = 600;
    

    Window = std::make_shared<window>(800, 600);
    Window->OnResize = OnResizeWindow;

    InitImGui();

    // 
    Scene = CreateCornellBox();
    
    Scene->Clear();
    Scene->FromFile("C:\\Users\\jacqu\\Documents\\Boulot\\MIS");
    
    Scene->PreProcess();
    
    Params =  GetTracingParameters();
    Params.SamplingMode = SAMPLING_MODE_MIS;

    InitGpuObjects();
    CreateOIDNFilter();
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
    ImGuizmo::BeginFrame();
}

void application::EndFrame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    Window->Present();
}

void application::Denoise()
{
#if API==API_GL 
    GLTexToCuBuffer(RenderMapping->CudaBuffer, RenderMapping->TexObj, RenderWidth, RenderHeight);
    Filter.execute();
    cudaMemcpyToArray(DenoiseMapping->CudaTextureArray, 0, 0, DenoisedBufferData, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
#else
    Filter.execute();
#endif
    Denoised = true;
}

void application::Trace()
{
    const char* errorMessage;
    if (Device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;

    if(ResetRender) 
    {
        Params.CurrentSample=0;
    }

    if(Params.CurrentSample < Params.TotalSamples)
    {
        Denoised=false;

        TracingParamsBuffer->updateData(&Params, sizeof(tracingParameters));
        Scene->CamerasBuffer->updateData(0 * sizeof(camera), Scene->Cameras.data(), Scene->Cameras.size() * sizeof(camera));

    #if API==API_GL
        PathTracingShader->Use();
        PathTracingShader->SetTexture(0, RenderTexture->TextureID, GL_READ_WRITE);
        PathTracingShader->SetSSBO(Scene->BVH->TrianglesBuffer, 1);
        PathTracingShader->SetSSBO(Scene->BVH->BVHBuffer, 3);
        PathTracingShader->SetSSBO(Scene->BVH->IndicesBuffer, 4);
        PathTracingShader->SetSSBO(Scene->BVH->IndexDataBuffer, 5);
        PathTracingShader->SetSSBO(Scene->BVH->TLASInstancesBuffer, 6);
        PathTracingShader->SetSSBO(Scene->BVH->TLASNodeBuffer, 7);        
        PathTracingShader->SetSSBO(Scene->CamerasBuffer, 8);
        PathTracingShader->SetUBO(TracingParamsBuffer, 9);
        PathTracingShader->SetSSBO(Scene->Lights->LightsBuffer, 10);
        PathTracingShader->SetSSBO(Scene->EnvironmentsBuffer, 11);
        PathTracingShader->SetSSBO(Scene->MaterialBuffer, 12);
        PathTracingShader->SetTextureArray(Scene->TexArray, 13, "SceneTextures");
        PathTracingShader->SetTextureArray(Scene->EnvTexArray, 14, "EnvTextures");
        PathTracingShader->SetSSBO(Scene->Lights->LightsCDFBuffer, 15);
        PathTracingShader->SetInt("EnvironmentsCount", Scene->Environments.size());
        PathTracingShader->SetInt("LightsCount", Scene->Lights->Lights.size());
        PathTracingShader->SetInt("EnvTexturesWidth", Scene->EnvTextureWidth);
        PathTracingShader->SetInt("EnvTexturesHeight", Scene->EnvTextureHeight);
  
        PathTracingShader->Dispatch(RenderWidth / 16 + 1, RenderHeight / 16 +1, 1);
#elif API==API_CU
        dim3 blockSize(16, 16);
        dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);
        TraceKernel<<<gridSize, blockSize>>>((glm::vec4*)RenderBuffer->Data, RenderWidth, RenderHeight,
                                            (triangle*)Scene->BVH->TrianglesBuffer->Data, (bvhNode*) Scene->BVH->BVHBuffer->Data, (uint32_t*) Scene->BVH->IndicesBuffer->Data, (indexData*) Scene->BVH->IndexDataBuffer->Data, (instance*)Scene->BVH->TLASInstancesBuffer->Data, (tlasNode*) Scene->BVH->TLASNodeBuffer->Data,
                                            (camera*)Scene->CamerasBuffer->Data, (tracingParameters*)TracingParamsBuffer->Data, (material*)Scene->MaterialBuffer->Data, Scene->TexArray->TexObject, Scene->TextureWidth, Scene->TextureHeight, (light*)Scene->Lights->LightsBuffer->Data, (float*)Scene->Lights->LightsCDFBuffer->Data, (int)Scene->Lights->Lights.size(), 
                                            (environment*)Scene->EnvironmentsBuffer->Data, (int)Scene->Environments.size(), Scene->EnvTexArray->TexObject, Scene->EnvTextureWidth, Scene->EnvTextureHeight);
#endif
        Params.CurrentSample+= Params.Batch;
    }

    if(DoDenoise && !Denoised)
    {
        Denoise();
    }

#if API==API_GL
    TonemapShader->Use();
    TonemapShader->SetTexture(0, Denoised ? DenoisedTexture->TextureID : RenderTexture->TextureID, GL_READ_WRITE);
    TonemapShader->SetTexture(1, TonemapTexture->TextureID, GL_READ_WRITE);
    TonemapShader->Dispatch(RenderWidth / 16 + 1, RenderHeight / 16 + 1, 1);
#elif API==API_CU
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);
    TonemapKernel<<<gridSize, blockSize>>>(Denoised ? (glm::vec4*)DenoisedBuffer->Data : (glm::vec4*)RenderBuffer->Data, (glm::vec4*)TonemapBuffer->Data, RenderWidth, RenderHeight);
    cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, TonemapBuffer->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
    CUDA_CHECK_ERROR(cudaGetLastError());
#endif
}

void application::Run()
{
    uint64_t Frame=0;
    while(!Window->ShouldClose())
    {
        if(Frame % 10==0)
        {
            Timer.Start();
        }

        Window->PollEvents();
        StartFrame();
        ResetRender=false;

        
        if(Scene->Cameras[int(Params.CurrentCamera)].Controlled)
        {
            ResetRender |= Controller.Update();        
            Scene->Cameras[int(Params.CurrentCamera)].Frame = Controller.ModelMatrix;
        }


        GUI->GUI();
        Trace();
        

        EndFrame();

        if(Frame % 10==0)
        {
            double Time = Timer.Stop();
            std::cout << "frame time " <<  Time << std::endl;
        }
        Frame++;
    }  
}

void application::Cleanup()
{
#if API==API_GL
    cudaFree(DenoisedBufferData);
#endif
}



void application::ResizeRenderTextures()
{
    if(!Inited) return;
#if API==API_GL
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    RenderTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);
    TonemapTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);    
    DenoisedTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);    
    
    RenderMapping = CreateMapping(RenderTexture);    
    DenoiseMapping = CreateMapping(DenoisedTexture, true);    

    cudaFree(DenoisedBufferData);
    cudaMalloc((void**)&DenoisedBufferData, RenderWidth * RenderHeight * sizeof(glm::vec4));
    
    // Allocate cuda buffer for denoise
    Filter.setImage("color",  RenderMapping->CudaBuffer,   oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
    Filter.setImage("output", DenoisedBufferData, oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
    Filter.commit();
#elif API==API_CU
    cudaDeviceSynchronize();
    TonemapTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);
    RenderBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(float));
    TonemapBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(float));
    RenderTextureMapping = CreateMapping(TonemapTexture);

    DenoisedBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(float));
    Filter.setImage("color",  RenderBuffer->Data,   oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
    Filter.setImage("output", DenoisedBuffer->Data, oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
    Filter.commit();
#endif


    Params.CurrentSample=0;
    Scene->Cameras[int(Params.CurrentCamera)].Aspect = (float)RenderWidth / (float)RenderHeight;

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

    if(Inited && Scene->Cameras[int(Params.CurrentCamera)].Aspect != RenderAspectRatio)
    {
        Scene->Cameras[int(Params.CurrentCamera)].Aspect = RenderAspectRatio;
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