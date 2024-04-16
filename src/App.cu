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
#include "GLTexToCuBuffer.cu"


#include <iostream>
#define CUDA_CHECK_ERROR(err) \
    do { \
        cudaError_t error = err; \
        if (error != cudaSuccess) { \
            std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
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
    MaterialBuffer = std::make_shared<bufferGL>(sizeof(material) * Scene->Materials.size(), Scene->Materials.data());
    LightsBuffer = std::make_shared<bufferGL>(sizeof(light) * Lights.Lights.size(), Lights.Lights.data());
    LightsCDFBuffer = std::make_shared<bufferGL>(sizeof(float) * Lights.LightsCDF.size(), Lights.LightsCDF.data());
    

#elif API==API_CU
    TonemapTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);
    RenderBuffer = std::make_shared<bufferCu>(Window->Width * Window->Height * sizeof(glm::vec4));
    TonemapBuffer = std::make_shared<bufferCu>(Window->Width * Window->Height * sizeof(glm::vec4));
    RenderTextureMapping = CreateMapping(TonemapTexture);

    TracingParamsBuffer = std::make_shared<bufferCu>(sizeof(tracingParameters), &Params);
    MaterialBuffer = std::make_shared<bufferCu>(sizeof(material) * Scene->Materials.size(), Scene->Materials.data());
    LightsBuffer = std::make_shared<bufferCu>(sizeof(light) * Lights.Lights.size(), Lights.Lights.data());
    LightsCDFBuffer = std::make_shared<bufferCu>(sizeof(float) * Lights.LightsCDF.size(), Lights.LightsCDF.data());
#endif
}

void application::CreateOIDNFilter()
{
    cudaStreamCreate(&Stream);
    Device = oidn::newCUDADevice(0, Stream);
    Device.commit();
    DenoisedBuffer = std::make_shared<bufferCu>(RenderWidth * RenderHeight * 4 * sizeof(float));

    Filter = Device.newFilter("RT");
#if API==API_CU
    Filter.setImage("color",  RenderBuffer->Data,   oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
    Filter.setImage("output", DenoisedBuffer->Data, oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
#else
    Filter.setImage("color",  RenderMapping->CudaBuffer,   oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
    Filter.setImage("output", DenoisedBuffer->Data, oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
#endif
    Filter.set("hdr", true);        
    Filter.set("cleanAux", true);           
    Filter.set("quality", OIDN_QUALITY_BALANCED);        
    Filter.commit();
}
void application::UpdateLights()
{
    Lights = GetLights(Scene, Params);
    if(Lights.Lights.size()>0)
    {
    // TODO: Only recreate the buffer if it's bigger.
    #if API==API_CU
        LightsBuffer = std::make_shared<bufferCu>(sizeof(light) * Lights.Lights.size(), Lights.Lights.data());
        LightsCDFBuffer = std::make_shared<bufferCu>(sizeof(float) * Lights.LightsCDF.size(), Lights.LightsCDF.data());
    #else
        LightsBuffer = std::make_shared<bufferGL>(sizeof(light) * Lights.Lights.size(), Lights.Lights.data());
        LightsCDFBuffer = std::make_shared<bufferGL>(sizeof(float) * Lights.LightsCDF.size(), Lights.LightsCDF.data());
    #endif
    }
    ResetRender=true;
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
    cudaMemcpyToArray(DenoiseMapping->CudaTextureArray, 0, 0, DenoisedBuffer->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
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
        PathTracingShader->SetSSBO(BVH->TrianglesBuffer, 1);
        PathTracingShader->SetSSBO(BVH->BVHBuffer, 3);
        PathTracingShader->SetSSBO(BVH->IndicesBuffer, 4);
        PathTracingShader->SetSSBO(BVH->IndexDataBuffer, 5);
        PathTracingShader->SetSSBO(BVH->TLASInstancesBuffer, 6);
        PathTracingShader->SetSSBO(BVH->TLASNodeBuffer, 7);        
        PathTracingShader->SetSSBO(Scene->CamerasBuffer, 8);
        PathTracingShader->SetUBO(TracingParamsBuffer, 9);
        PathTracingShader->SetSSBO(LightsBuffer, 10);
        PathTracingShader->SetSSBO(Scene->EnvironmentsBuffer, 11);
        PathTracingShader->SetSSBO(MaterialBuffer, 12);
        PathTracingShader->SetTextureArray(Scene->TexArray, 13, "SceneTextures");
        PathTracingShader->SetTextureArray(Scene->EnvTexArray, 14, "EnvTextures");
        PathTracingShader->SetSSBO(LightsCDFBuffer, 15);
        PathTracingShader->SetInt("EnvironmentsCount", Scene->Environments.size());
        PathTracingShader->SetInt("LightsCount", Lights.Lights.size());
        PathTracingShader->SetInt("EnvTexturesWidth", Scene->EnvTextureWidth);
        PathTracingShader->SetInt("EnvTexturesHeight", Scene->EnvTextureHeight);
  
        PathTracingShader->Dispatch(RenderWidth / 16 + 1, RenderHeight / 16 +1, 1);
#elif API==API_CU
        dim3 blockSize(16, 16);
        dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);
        TraceKernel<<<gridSize, blockSize>>>((glm::vec4*)RenderBuffer->Data, RenderWidth, RenderHeight,
                                            (triangle*)BVH->TrianglesBuffer->Data, (bvhNode*) BVH->BVHBuffer->Data, (uint32_t*) BVH->IndicesBuffer->Data, (indexData*) BVH->IndexDataBuffer->Data, (bvhInstance*)BVH->TLASInstancesBuffer->Data, (tlasNode*) BVH->TLASNodeBuffer->Data,
                                            (camera*)Scene->CamerasBuffer->Data, (tracingParameters*)TracingParamsBuffer->Data, (material*)MaterialBuffer->Data, Scene->TexArray->TexObject, Scene->TextureWidth, Scene->TextureHeight, (light*)LightsBuffer->Data, (float*)LightsCDFBuffer->Data, (int)Lights.Lights.size(), 
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
       
}

void application::UploadMaterial(int MaterialInx)
{
    MaterialBuffer->updateData((size_t)MaterialInx * sizeof(material), (void*)&Scene->Materials[MaterialInx], sizeof(material));
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

    DenoisedBuffer = std::make_shared<bufferCu>(RenderWidth * RenderHeight * 4 * sizeof(float));
    // Allocate cuda buffer for denoise
    Filter.setImage("color",  RenderMapping->CudaBuffer,   oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
    Filter.setImage("output", DenoisedBuffer->Data, oidn::Format::Float3, RenderWidth, RenderHeight, 0, sizeof(glm::vec4), sizeof(glm::vec4) * RenderWidth);
    Filter.commit();
#elif API==API_CU
    cudaDeviceSynchronize();
    TonemapTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);
    RenderBuffer = std::make_shared<bufferCu>(RenderWidth * RenderHeight * 4 * sizeof(float));
    TonemapBuffer = std::make_shared<bufferCu>(RenderWidth * RenderHeight * 4 * sizeof(float));
    RenderTextureMapping = CreateMapping(TonemapTexture);

    DenoisedBuffer = std::make_shared<bufferCu>(RenderWidth * RenderHeight * 4 * sizeof(float));
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