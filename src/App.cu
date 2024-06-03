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
#include "ImageLoader.h"
#include "Framebuffer.h"
#include "VertexBuffer.h"

#include <ImGuizmo.h>
#include <algorithm>
#include <fstream>
#include <iostream>

#if USE_OPTIX
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
void LogCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    std::cout << "[" << level << "][" << (tag ? tag : "no tag") << "]: " << (message ? message : "no message") << "\n";
}
#endif

#define CUDA_CHECK_ERROR(err) \
    do { \
        cudaError_t error = err; \
        if (error != cudaSuccess) { \
            std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            assert(false); \
        } \
    } while (0)

#define OPTIX_CHECK( x ) \
    do { \
        OptixResult result = x; \
        if ( result != OPTIX_SUCCESS ) { \
            std::cerr << "OptiX call " #x " failed with code " << result << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            assert(false); \
            exit(1); \
        } \
    } while(0)    
#define OPTIX_CHECK_LOG(call)                                                \
    {                                                                        \
        OptixResult res = call;                                              \
        if (res != OPTIX_SUCCESS) {                                          \
            std::cerr << "OptiX call (" << #call << ") failed: "             \
                      << optixGetErrorString(res) << " (code " << res << ")" \
                      << "\nLog:\n" << log                                   \
                      << std::endl;                                          \
            assert(false);                                       \
        }                                                                    \
    }



void checkOpenGLError(const std::string& location) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::string error;
        switch (err) {
            case GL_INVALID_ENUM:
                error = "GL_INVALID_ENUM";
                break;
            case GL_INVALID_VALUE:
                error = "GL_INVALID_VALUE";
                break;
            case GL_INVALID_OPERATION:
                error = "GL_INVALID_OPERATION";
                break;
            case GL_STACK_OVERFLOW:
                error = "GL_STACK_OVERFLOW";
                break;
            case GL_STACK_UNDERFLOW:
                error = "GL_STACK_UNDERFLOW";
                break;
            case GL_OUT_OF_MEMORY:
                error = "GL_OUT_OF_MEMORY";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                error = "GL_INVALID_FRAMEBUFFER_OPERATION";
                break;
            default:
                error = "UNKNOWN_ERROR";
                break;
        }
        std::cerr << "OpenGL Error: " << error << " at " << location << std::endl;
    }
}    

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
    TracingParamsBuffer = std::make_shared<buffer>(sizeof(tracingParameters), &Params);
}

#if USE_OPTIX
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RayGenSbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissSbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupSbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    float data; // example payload
};

std::string readPTXFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Failed to open PTX file " << filename << std::endl;
        return "";
    }
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return std::string(buffer.begin(), buffer.end());
}

void application::CreateSBT()
{
    std::string raygen_ptx = readPTXFile("resources/ptx/raygen.ptx");
    std::string closesthit_ptx = readPTXFile("resources/ptx/closesthit.ptx");
    std::string miss_ptx = readPTXFile("resources/ptx/miss.ptx");    

    OptixModule module_raygen;
    OptixModule module_closesthit;
    OptixModule module_miss;    

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
    
    // Create and configure pipeline and program groups
    OptixPipelineCompileOptions PipelineCompileOptions = {};
    PipelineCompileOptions.usesMotionBlur = false;
    PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    PipelineCompileOptions.numPayloadValues = 5;
    PipelineCompileOptions.numAttributeValues = 2;
    PipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    PipelineCompileOptions.pipelineLaunchParamsVariableName = "KernelParams";

    char log[8192];
    size_t sizeof_log = sizeof(log);


    OPTIX_CHECK_LOG(optixModuleCreate(
        OptixContext, 
        &module_compile_options, 
        &PipelineCompileOptions, 
        raygen_ptx.c_str(), 
        raygen_ptx.size(), 
        log, 
        &sizeof_log, 
        &module_raygen));

    OPTIX_CHECK_LOG(optixModuleCreate(
        OptixContext, 
        &module_compile_options, 
        &PipelineCompileOptions, 
        closesthit_ptx.c_str(), 
        closesthit_ptx.size(), 
        log, 
        &sizeof_log, 
        &module_closesthit));

    OPTIX_CHECK_LOG(optixModuleCreate(
        OptixContext, 
        &module_compile_options, 
        &PipelineCompileOptions, 
        miss_ptx.c_str(), 
        miss_ptx.size(), 
        log, 
        &sizeof_log, 
        &module_miss));



    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    raygen_prog_group_desc.raygen.module = module_raygen;
    OptixProgramGroup raygen_prog_group;
    OPTIX_CHECK(optixProgramGroupCreate(OptixContext, &raygen_prog_group_desc, 1, &program_group_options, nullptr, nullptr, &raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    miss_prog_group_desc.miss.module = module_miss;
    OptixProgramGroup miss_prog_group;
    OPTIX_CHECK(optixProgramGroupCreate(OptixContext, &miss_prog_group_desc, 1, &program_group_options, nullptr, nullptr, &miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleCH = module_closesthit;
    OptixProgramGroup hitgroup_prog_group;
    OPTIX_CHECK(optixProgramGroupCreate(OptixContext, &hitgroup_prog_group_desc, 1, &program_group_options, nullptr, nullptr, &hitgroup_prog_group));

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;

    std::vector<OptixProgramGroup> ProgramGroups = {
        raygen_prog_group,
        miss_prog_group,
        hitgroup_prog_group
    };
    // OPTIX_CHECK(optixPipelineCreate(OptixContext, &PipelineCompileOptions, &pipeline_link_options, &raygen_prog_group, 1, nullptr, nullptr, &pipeline));
    // OPTIX_CHECK(optixPipelineCreate(OptixContext, &PipelineCompileOptions, &pipeline_link_options, &miss_prog_group, 1, nullptr, nullptr, &pipeline));
    // OPTIX_CHECK(optixPipelineCreate(OptixContext, &PipelineCompileOptions, &pipeline_link_options, &hitgroup_prog_group, 1, nullptr, nullptr, &pipeline));
    OPTIX_CHECK(optixPipelineCreate(OptixContext, &PipelineCompileOptions, &pipeline_link_options, ProgramGroups.data(), 3, nullptr, nullptr, &pipeline));


    RayGenSbtRecord raygen_record;
    MissSbtRecord miss_record;
    HitGroupSbtRecord hitgroup_record;

    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &raygen_record));
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &miss_record));
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hitgroup_record));

    // Allocate device memory for SBT
    CUdeviceptr d_raygen_record;
    cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RayGenSbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice);

    CUdeviceptr d_miss_record;
    cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissSbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_miss_record), &miss_record, sizeof(MissSbtRecord), cudaMemcpyHostToDevice);

    CUdeviceptr d_hitgroup_record;
    cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), sizeof(HitGroupSbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), &hitgroup_record, sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice);

    
    SBT.raygenRecord = d_raygen_record;
    SBT.missRecordBase = d_miss_record;
    SBT.missRecordStrideInBytes = sizeof(MissSbtRecord);
    SBT.missRecordCount = 1;
    SBT.hitgroupRecordBase = d_hitgroup_record;
    SBT.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    SBT.hitgroupRecordCount = 1;    

    KernelParamsBuffer = std::make_shared<buffer>(sizeof(commonCu::kernelParams));
}
#endif
void application::Init()
{
    Time=0;

#if USE_OPTIX
    cudaFree(0);
    OPTIX_CHECK(optixInit());
    CUcontext cuCtx = 0; // Zero means take the current context
    
    OptixDeviceContextOptions Options = {};
    Options.logCallbackFunction = &LogCallback;
    Options.logCallbackLevel = 4;  // Set to a higher level for more verbose logging
    Options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &Options, &OptixContext));    
    OPTIX_CHECK(optixDeviceContextSetCacheEnabled(OptixContext, 0));

    CreateSBT();
#endif

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
#if !USE_OPTIX
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    pathtracing::TraceKernel<<<gridSize, blockSize>>>(
        (commonCu::half4*)RenderBuffer[PingPongInx]->Data, 
        {Framebuffer[PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[PingPongInx]->CudaMappings[3]->TexObj},
        RenderWidth, RenderHeight,
        (triangle*)Scene->BVH->TrianglesBuffer->Data, (bvhNode*) Scene->BVH->BVHBuffer->Data, (uint32_t*) Scene->BVH->IndicesBuffer->Data, (indexData*) Scene->BVH->IndexDataBuffer->Data, (instance*)Scene->BVH->TLASInstancesBuffer->Data, (tlasNode*) Scene->BVH->TLASNodeBuffer->Data,
        (camera*)Scene->CamerasBuffer->Data, (tracingParameters*)TracingParamsBuffer->Data, (material*)Scene->MaterialBuffer->Data, Scene->TexArray->TexObject, Scene->TextureWidth, Scene->TextureHeight, (light*)Scene->Lights->LightsBuffer->Data, (float*)Scene->Lights->LightsCDFBuffer->Data, (int)Scene->Lights->Lights.size(), 
        (environment*)Scene->EnvironmentsBuffer->Data, (int)Scene->Environments.size(), Scene->EnvTexArray->TexObject, Scene->EnvTextureWidth, Scene->EnvTextureHeight, Time);
#else

    commonCu::kernelParams KernelParams;
    KernelParams.Height = RenderHeight;
    KernelParams.Width = RenderWidth;
    KernelParams.OutputBuffer = (CUdeviceptr)RenderBuffer[PingPongInx]->Data;
    KernelParams.Handle = Scene->BVH->OptixAS->InstanceASHandle;

    KernelParams.TriangleBuffer = (triangle*)Scene->BVH->TrianglesBuffer->Data;
    KernelParams.IndicesBuffer = (uint32_t*) Scene->BVH->IndicesBuffer->Data;
    KernelParams.Cameras = (camera*) Scene->CamerasBuffer->Data;
    KernelParams.Parameters = (tracingParameters*)TracingParamsBuffer->Data;
    KernelParams.Materials = (material*)Scene->MaterialBuffer->Data;
    KernelParams.Instances = (instance*)Scene->BVH->TLASInstancesBuffer->Data;
    KernelParams.SceneTextures = Scene->TexArray->TexObject;
    KernelParams.EnvTextures = Scene->EnvTexArray->TexObject;
    KernelParams.EnvironmentsCount = Scene->Environments.size();
    KernelParams.TexturesWidth = Scene->TextureWidth;
    KernelParams.TexturesHeight = Scene->TextureHeight;
    KernelParams.LightsCount = Scene->Lights->Lights.size();
    KernelParams.Lights = (light*)Scene->Lights->LightsBuffer->Data;
    KernelParams.LightsCDF = (float*)Scene->Lights->LightsCDFBuffer->Data;
    KernelParams.Environments = (environment*)Scene->EnvironmentsBuffer->Data;
    KernelParams.EnvTexturesWidth = Scene->EnvTextureWidth;
    KernelParams.EnvTexturesHeight = Scene->EnvTextureHeight;
    KernelParams.Time = Time;
    KernelParams.IndexDataBuffer = (indexData*) Scene->BVH->IndexDataBuffer->Data;
    KernelParams.ShapeASHandles = (OptixTraversableHandle*) Scene->BVH->OptixAS->ShapeASHandlesBuffer->Data;
    KernelParams.CurrentFramebuffer = {Framebuffer[PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[PingPongInx]->CudaMappings[3]->TexObj};

    // cudaMemcpyToSymbol(commonCu::KernelParams, &KernelParams, sizeof(commonCu::kernelParams));
    KernelParamsBuffer->updateData(&KernelParams, sizeof(commonCu::kernelParams));


    OPTIX_CHECK(optixLaunch(pipeline, 0, (CUdeviceptr)KernelParamsBuffer->Data, sizeof(commonCu::kernelParams), &SBT, RenderWidth, RenderHeight, 1));


    // cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer[0]->Data, RenderWidth * RenderHeight * sizeof(filter::half4), cudaMemcpyDeviceToDevice);
    // CUDA_CHECK_ERROR(cudaGetLastError());
    
    // OutputTexture = RenderTexture->TextureID;   
#endif
}
void application::TemporalFilter()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    filter::TemporalFilter<<<gridSize, blockSize>>>((filter::half4*)RenderBuffer[1 - PingPongInx]->Data, (filter::half4*)RenderBuffer[PingPongInx]->Data, 
                                                    {Framebuffer[PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[PingPongInx]->CudaMappings[3]->TexObj}, 
                                                    {Framebuffer[1 - PingPongInx]->CudaMappings[0]->TexObj, Framebuffer[1 - PingPongInx]->CudaMappings[1]->TexObj, Framebuffer[1 - PingPongInx]->CudaMappings[2]->TexObj, Framebuffer[1 - PingPongInx]->CudaMappings[3]->TexObj}, 
                                                    (uint8_t*)HistoryLengthBuffer->Data,  (filter::half2*)MomentsBuffer[PingPongInx]->Data, (filter::half2*)MomentsBuffer[1 - PingPongInx]->Data,
                                                    RenderWidth, RenderHeight, DepthThreshold, NormalThreshold, HistoryLength);
}

void application::FilterMoments()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    filter::FilterMoments<<<gridSize, blockSize>>>((filter::half4*)RenderBuffer[PingPongInx]->Data, (filter::half4*) FilterBuffer[0]->Data, (filter::half2*)MomentsBuffer[0]->Data,
                                                    Framebuffer[PingPongInx]->CudaMappings[(int)rasterizeOutputs::Motion]->TexObj,
                                                    Framebuffer[PingPongInx]->CudaMappings[(int)rasterizeOutputs::Normal]->TexObj,
                                                    (uint8_t*)HistoryLengthBuffer->Data,
                                                    RenderWidth, RenderHeight, PhiColour, PhiNormal);
}

void application::WaveletFilter()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    
    int PingPong=0;
    for(int i=0; i<SpatialFilterSteps; i++)
    {
        filter::half4 *Input = (filter::half4 *)FilterBuffer[PingPong]->Data;
        filter::half4 *Output = (filter::half4 *)FilterBuffer[1 - PingPong]->Data;

        int StepSize = 1 << i;

        filter::FilterKernel<<<gridSize, blockSize>>>(Input, Framebuffer[PingPongInx]->CudaMappings[(int)rasterizeOutputs::Motion]->TexObj, Framebuffer[PingPongInx]->CudaMappings[(int)rasterizeOutputs::Normal]->TexObj,
            (uint8_t*)HistoryLengthBuffer->Data, Output, (filter::half4*) RenderBuffer[PingPongInx]->Data, RenderWidth, RenderHeight, StepSize, PhiColour, PhiNormal, i);

        PingPong = 1 - PingPong;
    }

    if(SpatialFilterSteps%2 != 0)
    {
        cudaMemcpy(FilterBuffer[0]->Data, FilterBuffer[1]->Data, RenderWidth * RenderHeight * sizeof(filter::half4), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
}

void application::TAA()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    

    filter::TAAFilterKernel<<<gridSize, blockSize>>>((filter::half4*)FilterBuffer[0]->Data, (filter::half4*)FilterBuffer[1]->Data, RenderWidth, RenderHeight);
}

void application::Tonemap()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((RenderWidth / blockSize.x)+1, (RenderHeight / blockSize.y) + 1);    
    

}

void application::SaveRender(std::string ImagePath)
{
    // std::vector<uint8_t> DataU8;
    // TonemapTexture->Download(DataU8);
    // ImageToFile(ImagePath, DataU8, RenderWidth, RenderHeight, 4);
}

void application::Render()
{
    Scene->CamerasBuffer->updateData(0 * sizeof(camera), Scene->Cameras.data(), Scene->Cameras.size() * sizeof(camera));
    TracingParamsBuffer->updateData(&Params, sizeof(tracingParameters));

#if 1
    if(SVGFDebugOutput == SVGFDebugOutputEnum::FinalOutput)
    {
        CUDA_CHECK_ERROR(cudaGetLastError());
        Rasterize(); // Outputs to CurrentFramebuffer
        CUDA_CHECK_ERROR(cudaGetLastError());
        Trace();     // Read CurrentFrmaebuffer, Writes to RenderBuffer[PingPongInx]
        CUDA_CHECK_ERROR(cudaGetLastError());
        TemporalFilter(); //Reads RenderBuffer[PingPongInx], Writes to RenderBuffer[PingPongInx]
        CUDA_CHECK_ERROR(cudaGetLastError());
        FilterMoments(); // Reads RenderBuffer[PingPongInx], writes to FilterBuffer[0]
        CUDA_CHECK_ERROR(cudaGetLastError());
        WaveletFilter(); //Reads from FilterBuffer[0], Writes to FilterBuffer[0]
        CUDA_CHECK_ERROR(cudaGetLastError());
        TAA(); //Reads from FilterBuffer[0], WRites to FilterBuffer[1]
        CUDA_CHECK_ERROR(cudaGetLastError());

        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, FilterBuffer[1]->Data, RenderWidth * RenderHeight * sizeof(filter::half4), cudaMemcpyDeviceToDevice);
        CUDA_CHECK_ERROR(cudaGetLastError());
        
        OutputTexture = RenderTexture->TextureID;
        DebugTint = glm::vec4(1);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::RawOutput)
    {
        Rasterize();
        CUDA_CHECK_ERROR(cudaGetLastError());
        Trace();
        CUDA_CHECK_ERROR(cudaGetLastError());
        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * sizeof(filter::half4), cudaMemcpyDeviceToDevice);
        OutputTexture = RenderTexture->TextureID;
        DebugTint = glm::vec4(1);
        CUDA_CHECK_ERROR(cudaGetLastError());
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
        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer[PingPongInx]->Data, RenderWidth * RenderHeight * sizeof(filter::half4), cudaMemcpyDeviceToDevice);
        OutputTexture = RenderTexture->TextureID;
        DebugTint = glm::vec4(1,1,1,1);
    }
    else if(SVGFDebugOutput == SVGFDebugOutputEnum::ATrousWaveletFilter)
    {
        Rasterize();
        Trace();
        TemporalFilter();
        WaveletFilter();
        cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, FilterBuffer[0]->Data, RenderWidth * RenderHeight * sizeof(filter::half4), cudaMemcpyDeviceToDevice);
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
#else
    commonCu::kernelParams KernelParams;
    KernelParams.Height = RenderHeight;
    KernelParams.Width = RenderWidth;
    KernelParams.OutputBuffer = (CUdeviceptr)RenderBuffer[0]->Data;
    KernelParams.Handle = Scene->BVH->OptixAS->InstanceASHandle;

    KernelParams.TriangleBuffer = (triangle*)Scene->BVH->TrianglesBuffer->Data;
    KernelParams.IndicesBuffer = (uint32_t*) Scene->BVH->IndicesBuffer->Data;
    KernelParams.Cameras = (camera*) Scene->CamerasBuffer->Data;
    KernelParams.Parameters = (tracingParameters*)TracingParamsBuffer->Data;
    KernelParams.Materials = (material*)Scene->MaterialBuffer->Data;
    KernelParams.Instances = (instance*)Scene->BVH->TLASInstancesBuffer->Data;
    KernelParams.SceneTextures = Scene->TexArray->TexObject;
    KernelParams.EnvTextures = Scene->EnvTexArray->TexObject;
    KernelParams.EnvironmentsCount = Scene->Environments.size();
    KernelParams.TexturesWidth = Scene->TextureWidth;
    KernelParams.TexturesHeight = Scene->TextureHeight;
    KernelParams.LightsCount = Scene->Lights->Lights.size();
    KernelParams.Lights = (light*)Scene->Lights->LightsBuffer->Data;
    KernelParams.LightsCDF = (float*)Scene->Lights->LightsCDFBuffer->Data;
    KernelParams.Environments = (environment*)Scene->EnvironmentsBuffer->Data;
    KernelParams.EnvTexturesWidth = Scene->EnvTextureWidth;
    KernelParams.EnvTexturesHeight = Scene->EnvTextureHeight;
    KernelParams.Time = Time;
    KernelParams.IndexDataBuffer = (indexData*) Scene->BVH->IndexDataBuffer->Data;
    KernelParams.ShapeASHandles = (OptixTraversableHandle*) Scene->BVH->OptixAS->ShapeASHandlesBuffer->Data;

    // cudaMemcpyToSymbol(commonCu::KernelParams, &KernelParams, sizeof(commonCu::kernelParams));
    KernelParamsBuffer->updateData(&KernelParams, sizeof(commonCu::kernelParams));


    OPTIX_CHECK(optixLaunch(pipeline, 0, (CUdeviceptr)KernelParamsBuffer->Data, sizeof(commonCu::kernelParams), &SBT, RenderWidth, RenderHeight, 1));


    cudaMemcpyToArray(RenderTextureMapping->CudaTextureArray, 0, 0, RenderBuffer[0]->Data, RenderWidth * RenderHeight * sizeof(filter::half4), cudaMemcpyDeviceToDevice);
    CUDA_CHECK_ERROR(cudaGetLastError());
    
    OutputTexture = RenderTexture->TextureID;    
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
        checkOpenGLError("");
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
        {GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, 4 * sizeof(uint16_t)}, //Normal
        {GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, 4 * sizeof(uint16_t)}, //Barycentric coordinates
        {GL_RGBA32F, GL_RGBA, GL_FLOAT, sizeof(glm::vec4)}, //Motion Vectors and depth
    };
    Framebuffer[0] = std::make_shared<framebuffer>(RenderWidth, RenderHeight, Desc);
    Framebuffer[1] = std::make_shared<framebuffer>(RenderWidth, RenderHeight, Desc);
    GBufferShader = std::make_shared<shaderGL>("resources/shaders/GBuffer.vert", "resources/shaders/GBuffer.frag");

    
    cudaDeviceSynchronize();
    // TODO: Make that uint8
    // TonemapTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, textureGL::channels::RGBA, textureGL::types::Uint8);

    RenderTexture = std::make_shared<textureGL>(RenderWidth, RenderHeight, textureGL::channels::RGBA, textureGL::types::Half);
    RenderBuffer[0] = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(filter::half4));
    RenderBuffer[1] = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(filter::half4));

    RenderTextureMapping = CreateMapping(RenderTexture);
    MomentsBuffer[0] = std::make_shared<buffer>(RenderWidth * RenderHeight * 2 * sizeof(filter::half2));
    MomentsBuffer[1] = std::make_shared<buffer>(RenderWidth * RenderHeight * 2 * sizeof(filter::half2));
    
    FilterBuffer[0] = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(filter::half4));
    FilterBuffer[1] = std::make_shared<buffer>(RenderWidth * RenderHeight * 4 * sizeof(filter::half4));
    
    HistoryLengthBuffer = std::make_shared<buffer>(RenderWidth * RenderHeight * sizeof(uint8_t));

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