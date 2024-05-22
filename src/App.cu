#include "App.h"

#include <glad/gl.h>

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
#include "PathTrace.cuh"
#include "Filter.cuh"
#include "TextureArrayCu.cuh"
#include "GLTexToCuBuffer.cu"
#include "ImageLoader.h"
#include "Framebuffer.h"
#include "VertexBuffer.h"

#include <ImGuizmo.h>
#include <algorithm>

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
    TonemapTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);
    RenderTexture = std::make_shared<textureGL>(Window->Width, Window->Height, 4);
    RenderBuffer[0] = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec4));
    RenderBuffer[1] = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec4));
    FilterBuffer = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec4));
    RenderTextureMapping = CreateMapping(RenderTexture);
    MomentsBuffer = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec4));
    HistoryLengthBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * sizeof(uint8_t));


    TracingParamsBuffer = std::make_shared<buffer>(sizeof(tracingParameters), &Params);
}

void application::CreateOIDNFilter()
{
    cudaStreamCreate(&Stream);
    Device = oidn::newCUDADevice(0, Stream);
    Device.commit();
    CUDA_CHECK_ERROR(cudaGetLastError());

    Filter = Device.newFilter("RT");
    Filter.set("hdr", true);        
    Filter.set("cleanAux", true);           
    Filter.set("quality", OIDN_QUALITY_BALANCED);        
    Filter.commit();
    CUDA_CHECK_ERROR(cudaGetLastError());
}


    
void application::Init()
{
    Time=0;

    this->GUI = std::make_shared<gui>(this);

    GUI->GuiWidth = 200;
    RenderResolution = 600;
    

    Window = std::make_shared<window>(800, 600);
    Window->OnResize = OnResizeWindow;

    InitImGui();

    CUDA_CHECK_ERROR(cudaGetLastError());

    // 
    Scene = std::make_shared<scene>();
    CUDA_CHECK_ERROR(cudaGetLastError());
    Scene->PreProcess();
    CUDA_CHECK_ERROR(cudaGetLastError());

    
    Params =  GetTracingParameters();

    InitGpuObjects();
    CreateOIDNFilter();
    CUDA_CHECK_ERROR(cudaGetLastError());
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

    Scene->Cameras[Params.CurrentCamera].PreviousFrame = Scene->Cameras[Params.CurrentCamera].Frame;

    PingPongInx = 1 - PingPongInx;
}

void application::Tonemap()
{
//     bool DoClear = (Scene->Instances.size()==0 || Scene->Lights->Lights.size()==0);
// #if API==API_GL
//     TonemapShader->Use();
//     TonemapShader->SetTexture(0, Denoised ? DenoisedTexture->TextureID : RenderTexture->TextureID, GL_READ_WRITE);
//     TonemapShader->SetTexture(1, TonemapTexture->TextureID, GL_READ_WRITE);
//     TonemapShader->SetInt("DoClear", (int)DoClear);
//     TonemapShader->Dispatch(RenderWidth / 16 + 1, RenderHeight / 16 + 1, 1);
// #elif API==API_CU
//     dim3 blockSize(16, 16);
//     dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);

//     glm::vec4 *Buffer = Denoised ? (glm::vec4*)DenoisedBuffer->Data : (glm::vec4*)RenderBuffer->Data;

//     TonemapKernel<<<gridSize, blockSize>>>(Buffer, (glm::vec4*)TonemapBuffer->Data, RenderWidth, RenderHeight, DoClear);
//     if(!DoSVGF) cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, TonemapBuffer->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);

// #endif
}

void application::SaveRender(std::string ImagePath)
{
    std::vector<uint8_t> DataU8;
    TonemapTexture->Download(DataU8);
    ImageToFile(ImagePath, DataU8, RenderWidth, RenderHeight, 4);
}

void application::Render()
{
    Scene->CamerasBuffer->updateData(0 * sizeof(camera), Scene->Cameras.data(), Scene->Cameras.size() * sizeof(camera));
    TracingParamsBuffer->updateData(&Params, sizeof(tracingParameters));


    // Rasterize scene
    // if(CameraMoved)
    {
        Framebuffer[PingPongInx]->Bind();
        glViewport(0,0, RenderWidth, RenderHeight);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST); 
        glDepthMask(GL_TRUE);

        for(int i=0; i<Scene->Instances.size(); i++)
        {
            GBufferShader->Use();

            glm::mat4 MVP = Scene->Cameras[Params.CurrentCamera].ProjectionMatrix * glm::inverse(Scene->Cameras[Params.CurrentCamera].Frame) * Scene->Instances[i].Transform;
            glm::mat4 PrevMVP = Scene->Cameras[Params.CurrentCamera].ProjectionMatrix * glm::inverse(Scene->Cameras[Params.CurrentCamera].PreviousFrame) * Scene->Instances[i].Transform;
            glm::vec3 CameraPosition = Scene->Cameras[Params.CurrentCamera].Frame * glm::vec4(0,0,0,1);

            GBufferShader->SetMat4("ModelMatrix", Scene->Instances[i].Transform);
            GBufferShader->SetMat4("NormalMatrix", Scene->Instances[i].NormalTransform);
            GBufferShader->SetMat4("MVP", MVP);
            GBufferShader->SetMat4("PreviousMVP", PrevMVP);
            GBufferShader->SetVec3("CameraPosition", CameraPosition);

            GBufferShader->SetInt("MaterialIndex", Scene->Instances[i].Material);
            GBufferShader->SetInt("InstanceIndex", i);
            GBufferShader->SetInt("Width", RenderWidth);
            GBufferShader->SetInt("Height", RenderHeight);
            GBufferShader->SetSSBO(Scene->MaterialBufferGL, 0);

            Scene->VertexBuffer->Draw(Scene->Instances[i].Shape);

        }
        Framebuffer[PingPongInx]->Unbind();
    }

    // Path Trace
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);
    
    // Outputs into RenderBuffer[PingPongInx]
    pathtracing::TraceKernel<<<gridSize, blockSize>>>(
                                        (glm::vec4*)RenderBuffer[PingPongInx]->Data, 
                                        {Framebuffer[PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[PingPongInx]->CudaMappings[3]->TexObj},
                                        RenderWidth, RenderHeight,
                                        (triangle*)Scene->BVH->TrianglesBuffer->Data, (bvhNode*) Scene->BVH->BVHBuffer->Data, (uint32_t*) Scene->BVH->IndicesBuffer->Data, (indexData*) Scene->BVH->IndexDataBuffer->Data, (instance*)Scene->BVH->TLASInstancesBuffer->Data, (tlasNode*) Scene->BVH->TLASNodeBuffer->Data,
                                        (camera*)Scene->CamerasBuffer->Data, (tracingParameters*)TracingParamsBuffer->Data, (material*)Scene->MaterialBuffer->Data, Scene->TexArray->TexObject, Scene->TextureWidth, Scene->TextureHeight, (light*)Scene->Lights->LightsBuffer->Data, (float*)Scene->Lights->LightsCDFBuffer->Data, (int)Scene->Lights->Lights.size(), 
                                        (environment*)Scene->EnvironmentsBuffer->Data, (int)Scene->Environments.size(), Scene->EnvTexArray->TexObject, Scene->EnvTextureWidth, Scene->EnvTextureHeight, Time);



    // filter::TemporalFilter<<<gridSize, blockSize>>>((glm::vec4*)RenderBuffer[1 - PingPongInx]->Data, (glm::vec4*)RenderBuffer[PingPongInx]->Data, 
    //                                                 {Framebuffer[PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[PingPongInx]->CudaMappings[3]->TexObj}, 
    //                                                 {Framebuffer[1 - PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[1 - PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[1 - PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[1 - PingPongInx]->CudaMappings[3]->TexObj}, 
    //                                                 (uint32_t*)HistoryLengthBuffer->Data,  (glm::vec4*)MomentsBuffer->Data,
    //                                                 RenderWidth, RenderHeight);

    // // Copies RenderBuffer into FilterBuffer
    // cudaMemcpy((void*)FilterBuffer->Data, RenderBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * 4 * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    

    // //Outputs into RenderBuffer
    // filter::FilterKernel<<<gridSize, blockSize>>>((glm::vec4*)FilterBuffer->Data, (glm::vec4*)MomentsBuffer->Data, Framebuffer[PingPongInx]->CudaMappings[3]->TexObj, Framebuffer[PingPongInx]->CudaMappings[1]->TexObj,
    //      (uint32_t*)HistoryLengthBuffer->Data, (glm::vec4*)RenderBuffer[PingPongInx]->Data, RenderWidth, RenderHeight, 1);



    cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
    // cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, FilterBuffer->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
    Params.CurrentSample += Params.Batch;


    

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

        Time += 0.001f;

        Window->PollEvents();
        StartFrame();
        ResetRender=false;
        
        
        if(Scene->Cameras.size() > 0 &&  Scene->Cameras[int(Params.CurrentCamera)].Controlled)
        {
            CameraMoved = Controller.Update() || Frame==0;
            ResetRender |= CameraMoved; 
            Scene->Cameras[int(Params.CurrentCamera)].Frame = Controller.ModelMatrix;
        }


        GUI->GUI();
        CUDA_CHECK_ERROR(cudaGetLastError());
        
        Render();
        CUDA_CHECK_ERROR(cudaGetLastError());
        

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



void application::ResizeRenderTextures()
{
    if(!Inited) return;

    std::vector<framebufferDescriptor> Desc = 
    {
        {GL_RGBA32F, GL_RGBA, GL_FLOAT, sizeof(glm::vec4)}, //Position
        {GL_RGBA32F, GL_RGBA, GL_FLOAT, sizeof(glm::vec4)}, //Normal 
        {GL_RGBA32F, GL_RGBA, GL_FLOAT, sizeof(glm::vec4)}, //Barycentric coordinates
        {GL_RGBA32F, GL_RGBA, GL_FLOAT, sizeof(glm::vec4)}, //Motion Vectors
    };
    Framebuffer[0] = std::make_shared<framebuffer>(RenderWidth, RenderHeight, Desc);
    Framebuffer[1] = std::make_shared<framebuffer>(RenderWidth, RenderHeight, Desc);
    GBufferShader = std::make_shared<shaderGL>("resources/shaders/GBuffer.vert", "resources/shaders/GBuffer.frag");

    
    cudaDeviceSynchronize();
    TonemapTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);
    RenderTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, 4);
    RenderBuffer[0] = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(float));
    RenderBuffer[1] = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(float));
    RenderTextureMapping = CreateMapping(RenderTexture);
    MomentsBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(float));
    FilterBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(float));
    HistoryLengthBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * sizeof(uint32_t));


    Params.CurrentSample=0;
    Scene->Cameras[int(Params.CurrentCamera)].SetAspect((float)RenderWidth / (float)RenderHeight);
    
    ResetRender=true;
}

void application::CalculateWindowSizes()
{
    if(!Inited) return;
    if(Scene->Cameras.size() == 0) return;
    
    CUDA_CHECK_ERROR(cudaGetLastError());

    // GUIWindow size
    RenderWindowWidth = Window->Width - GUI->GuiWidth;
    RenderWindowHeight = Window->Height;

    // Aspect ratio of the GUI window
    RenderAspectRatio = (float)RenderWindowWidth / (float)RenderWindowHeight;

    // Set the render width accordingly
    uint32_t NewRenderWidth, NewRenderHeight;

    CUDA_CHECK_ERROR(cudaGetLastError());
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

    CUDA_CHECK_ERROR(cudaGetLastError());
    if(Inited && Scene->Cameras[int(Params.CurrentCamera)].Aspect != RenderAspectRatio)
    {
        Scene->Cameras[int(Params.CurrentCamera)].SetAspect(RenderAspectRatio);
        ResetRender = true;
    }

    if(NewRenderWidth != RenderWidth || NewRenderHeight != RenderHeight)
    {
        RenderWidth = NewRenderWidth;
        RenderHeight = NewRenderHeight;
        ResizeRenderTextures();
    
    }
    CUDA_CHECK_ERROR(cudaGetLastError());
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