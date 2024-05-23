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
    FilterBuffer[0] = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec4));
    FilterBuffer[1] = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec4));
    RenderTextureMapping = CreateMapping(RenderTexture);
    MomentsBuffer[0] = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec2));
    MomentsBuffer[1] = std::make_shared<buffer>(Window->Width * Window->Height * sizeof(glm::vec2));
    HistoryLengthBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * sizeof(uint8_t));


    TracingParamsBuffer = std::make_shared<buffer>(sizeof(tracingParameters), &Params);
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


void application::Rasterize()
{
    DebugRasterize = (SVGFDebugOutput==SVGFDebugOutputEnum::Normal ||  SVGFDebugOutput==SVGFDebugOutputEnum::Motion ||  SVGFDebugOutput==SVGFDebugOutputEnum::Position ||  SVGFDebugOutput==SVGFDebugOutputEnum::BarycentricCoords);
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
        GBufferShader->SetInt("Debug", int(DebugRasterize));
        GBufferShader->SetSSBO(Scene->MaterialBufferGL, 0);

        Scene->VertexBuffer->Draw(Scene->Instances[i].Shape);

    }
    Framebuffer[PingPongInx]->Unbind();
}
void application::Trace()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    pathtracing::TraceKernel<<<gridSize, blockSize>>>(
        (glm::vec4*)RenderBuffer[PingPongInx]->Data, 
        {Framebuffer[PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[PingPongInx]->CudaMappings[3]->TexObj},
        RenderWidth, RenderHeight,
        (triangle*)Scene->BVH->TrianglesBuffer->Data, (bvhNode*) Scene->BVH->BVHBuffer->Data, (uint32_t*) Scene->BVH->IndicesBuffer->Data, (indexData*) Scene->BVH->IndexDataBuffer->Data, (instance*)Scene->BVH->TLASInstancesBuffer->Data, (tlasNode*) Scene->BVH->TLASNodeBuffer->Data,
        (camera*)Scene->CamerasBuffer->Data, (tracingParameters*)TracingParamsBuffer->Data, (material*)Scene->MaterialBuffer->Data, Scene->TexArray->TexObject, Scene->TextureWidth, Scene->TextureHeight, (light*)Scene->Lights->LightsBuffer->Data, (float*)Scene->Lights->LightsCDFBuffer->Data, (int)Scene->Lights->Lights.size(), 
        (environment*)Scene->EnvironmentsBuffer->Data, (int)Scene->Environments.size(), Scene->EnvTexArray->TexObject, Scene->EnvTextureWidth, Scene->EnvTextureHeight, Time);
}
void application::TemporalFilter()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    filter::TemporalFilter<<<gridSize, blockSize>>>((glm::vec4*)RenderBuffer[1 - PingPongInx]->Data, (glm::vec4*)RenderBuffer[PingPongInx]->Data, 
                                                    {Framebuffer[PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[PingPongInx]->CudaMappings[3]->TexObj}, 
                                                    {Framebuffer[1 - PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[1 - PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[1 - PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[1 - PingPongInx]->CudaMappings[3]->TexObj}, 
                                                    (uint32_t*)HistoryLengthBuffer->Data,  (glm::vec2*)MomentsBuffer[PingPongInx]->Data, (glm::vec2*)MomentsBuffer[1 - PingPongInx]->Data,
                                                    RenderWidth, RenderHeight, DepthThreshold, NormalThreshold, HistoryLength);
}

void application::FilterMoments()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    cudaMemcpy(FilterBuffer[0]->Data, RenderBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    //  int _Width, int _Height, float PhiColour, float PhiNormal)
    filter::FilterMoments<<<gridSize, blockSize>>>((glm::vec4*) FilterBuffer[0]->Data, (glm::vec4*)RenderBuffer[PingPongInx]->Data, (glm::vec2*)MomentsBuffer[0]->Data,
                                                    Framebuffer[PingPongInx]->CudaMappings[(int)rasterizeOutputs::Motion]->TexObj,
                                                    Framebuffer[PingPongInx]->CudaMappings[(int)rasterizeOutputs::Normal]->TexObj,
                                                    (uint32_t*)HistoryLengthBuffer->Data,
                                                    RenderWidth, RenderHeight, PhiColour, PhiNormal);
}

void application::WaveletFilter()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    
    cudaMemcpy(FilterBuffer[0]->Data, RenderBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

    int PingPong=0;
    for(int i=0; i<SpatialFilterSteps; i++)
    {
        glm::vec4 *Input = (glm::vec4 *)FilterBuffer[PingPong]->Data;
        glm::vec4 *Output = (glm::vec4 *)FilterBuffer[1 - PingPong]->Data;

        int StepSize = 1 << i;

        filter::FilterKernel<<<gridSize, blockSize>>>(Input, Framebuffer[PingPongInx]->CudaMappings[3]->TexObj, Framebuffer[PingPongInx]->CudaMappings[1]->TexObj,
            (uint32_t*)HistoryLengthBuffer->Data, Output, RenderWidth, RenderHeight, StepSize, PhiColour, PhiNormal);

        PingPong = 1 - PingPong;

        if(i==0)
        {
            cudaMemcpy(RenderBuffer[PingPongInx]->Data, Output, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        }
    }

    if(SpatialFilterSteps%2 != 0)
    {
        cudaMemcpy(FilterBuffer[0]->Data, FilterBuffer[1]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

void application::TAA()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    

    filter::TAAFilterKernel<<<gridSize, blockSize>>>((glm::vec4*)FilterBuffer[0]->Data, (glm::vec4*)FilterBuffer[1]->Data, RenderWidth, RenderHeight);
}

void application::Tonemap()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    

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


    if(SVGFDebugOutput == SVGFDebugOutputEnum::FinalOutput)
    {
        Rasterize(); // Outputs to CurrentFramebuffer
        Trace();     // Read CurrentFrmaebuffer, Writes to RenderBuffer[PingPongInx]
        TemporalFilter(); //Reads RenderBuffer[PingPongInx], Writes to RenderBuffer[PingPongInx]
        FilterMoments(); // Reads RenderBuffer[PingPongInx], writes to RenderBuffer[PingPongInx]
        WaveletFilter(); //Reads from RenderBuffer[PingPongInx], Writes to FilterBuffer[0]
        TAA(); //Reads from FilterBuffer[0], WRites to FilterBuffer[1]

        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, FilterBuffer[1]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
        
        OutputTexture = RenderTexture->TextureID;
        DebugTint = glm::vec4(1);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::RawOutput)
    {
        Rasterize();
        Trace();
        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
        OutputTexture = RenderTexture->TextureID;
        DebugTint = glm::vec4(1);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::Normal)
    {
        Rasterize();
        OutputTexture = Framebuffer[PingPongInx]->GetTexture((int)rasterizeOutputs::Normal);
        DebugTint = glm::vec4(1,1,1,0);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::Motion)
    {
        Rasterize();
        OutputTexture = Framebuffer[PingPongInx]->GetTexture((int)rasterizeOutputs::Motion);
        DebugTint = glm::vec4(1,1,0,0);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::Position)
    {
        Rasterize();
        OutputTexture = Framebuffer[PingPongInx]->GetTexture((int)rasterizeOutputs::Position);
        DebugTint = glm::vec4(1,1,1,0);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::BarycentricCoords)
    {
        Rasterize();
        OutputTexture = Framebuffer[PingPongInx]->GetTexture((int)rasterizeOutputs::UV);
        DebugTint = glm::vec4(1,1,0,0);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::TemporalFilter)
    {
        Rasterize();
        Trace();
        TemporalFilter();
        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
        OutputTexture = RenderTexture->TextureID;
        DebugTint = glm::vec4(1,1,1,1);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::ATrousWaveletFilter)
    {
        Rasterize();
        Trace();
        TemporalFilter();
        WaveletFilter();
        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, FilterBuffer[0]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
        OutputTexture = RenderTexture->TextureID;        
        DebugTint = glm::vec4(1,1,1,1);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::Moments)
    {
        // Rasterize();
        // Trace();
        // TemporalFilter();
        // WaveletFilter();
        // cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, MomentsBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
        // OutputTexture = RenderTexture->TextureID;        
        // DebugTint = glm::vec4(1,1,0,1);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::Depth)
    {
        Rasterize();
        Trace();
        TemporalFilter();
        WaveletFilter();
        OutputTexture = Framebuffer[PingPongInx]->GetTexture((int)rasterizeOutputs::Motion);
        DebugTint = glm::vec4(0,0,1,0);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::Variance)
    {
        // Rasterize();
        // Trace();
        // TemporalFilter();
        // WaveletFilter();
        // cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, MomentsBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
        // OutputTexture = RenderTexture->TextureID;        
        // DebugTint = glm::vec4(0,0,1,0);
    }
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
    MomentsBuffer[0] = std::make_shared<buffer>(RenderWidth * RenderHeight * 2 * sizeof(float));
    MomentsBuffer[1] = std::make_shared<buffer>(RenderWidth * RenderHeight * 2 * sizeof(float));
    FilterBuffer[0] = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(float));
    FilterBuffer[1] = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(float));
    HistoryLengthBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * sizeof(uint32_t));


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