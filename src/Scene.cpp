#include "Scene.h"

#include "Buffer.h"
#include "App.h"
#include "GLTFLoader.h"
#include "AssimpLoader.h"
#include "ImageLoader.h"
#include "TextureArrayGL.h"
#include "TextureArrayCu.cuh"
#include "VertexBuffer.h"
#include "BVH.h"
#include <unordered_map>


#include "fstream"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <deprecated/stb_image_resize.h>

#if USE_OPTIX
#include <optix_stubs.h>
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

namespace gpupt
{
template<typename T>
void SerializeVector(std::ofstream &Stream, const std::vector<T>& Vec) {
    size_t VecSize = Vec.size();
    Stream.write((char*)&VecSize, sizeof(size_t));
    Stream.write((char*)Vec.data(), Vec.size() * sizeof(T));
}

template<typename T>
void DeserializeVector(std::ifstream &Stream, std::vector<T>& Vec) {
    size_t VecSize;
    Stream.read((char*)&VecSize, sizeof(size_t));
    Vec.resize(VecSize);
    Stream.read((char*)Vec.data(), Vec.size() * sizeof(T));
}

void SerializeStrVector(std::ofstream &Stream, const std::vector<std::string>& Vec) {
    size_t VecSize = Vec.size();
    Stream.write((char*)&VecSize, sizeof(size_t));
    for(int i=0; i<VecSize; i++)
    {
        size_t StrSize = Vec[i].size();
        Stream.write((char*)&StrSize, sizeof(size_t));
        Stream.write((char*)Vec[i].data(), Vec[i].size());
    }
}

void DeserializeStrVector(std::ifstream &Stream, std::vector<std::string>& Vec) {
    size_t VecSize = Vec.size();
    Stream.read((char*)&VecSize, sizeof(size_t));
    Vec.resize(VecSize);
    for(int i=0; i<VecSize; i++)
    {
        size_t StrSize;
        Stream.read((char*)&StrSize, sizeof(size_t));
        Vec[i].resize(StrSize);
        Stream.read((char*)Vec[i].data(), Vec[i].size());
    }
}


void EnsureUnicity(std::vector<std::string> &Names, std::string DefaultName)
{
    std::unordered_map<std::string, int> Counts;

    for (std::string& Name : Names) {
        if(Name == "") Name = DefaultName;
        Counts[Name]++;
    }

    for (std::string& Name : Names) {
        if (Counts[Name] > 1) {
            Name = Name + "_" + std::to_string(Counts[Name]--);
        }
    }
}

void camera::CalculateProj()
{
    ProjectionMatrix = glm::perspective(glm::radians(FOV), Aspect, 0.001f, 1000.0f);
}

void camera::SetAspect(float _Aspect)
{
    this->Aspect = _Aspect;
    CalculateProj();
}

void shape::CalculateTangents()
{
    std::vector<glm::vec4> tan1(this->PositionsTmp.size(), glm::vec4(0));
    std::vector<glm::vec4> tan2(this->PositionsTmp.size(), glm::vec4(0));
    if (this->TangentsTmp.size() != this->PositionsTmp.size()) this->TangentsTmp.resize(this->PositionsTmp.size());
    if(this->TexCoordsTmp.size() != this->PositionsTmp.size()) return;

    for(uint64_t i=0; i<this->IndicesTmp.size(); i++) {
        glm::vec3 v1 = this->PositionsTmp[this->IndicesTmp[i].x];
        glm::vec3 v2 = this->PositionsTmp[this->IndicesTmp[i].y];
        glm::vec3 v3 = this->PositionsTmp[this->IndicesTmp[i].z];

        glm::vec2 w1 = this->TexCoordsTmp[this->IndicesTmp[i].x];
        glm::vec2 w2 = this->TexCoordsTmp[this->IndicesTmp[i].y];
        glm::vec2 w3 = this->TexCoordsTmp[this->IndicesTmp[i].z];

        float x1 = v2.x - v1.x;
        float x2 = v3.x - v1.x;
        float y1 = v2.y - v1.y;
        float y2 = v3.y - v1.y;
        float z1 = v2.z - v1.z;
        float z2 = v3.z - v1.z;

        float s1 = w2.x - w1.x;
        float s2 = w3.x - w1.x;
        float t1 = w2.y - w1.y;
        float t2 = w3.y - w1.y;

        float r = 1.0F / (s1 * t2 - s2 * t1);
        glm::vec4 sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r, 0);
        glm::vec4 tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r, 0);

        tan1[this->IndicesTmp[i].x] += sdir;
        tan1[this->IndicesTmp[i].y] += sdir;
        tan1[this->IndicesTmp[i].z] += sdir;
        
        tan2[this->IndicesTmp[i].x] += tdir;
        tan2[this->IndicesTmp[i].y] += tdir;
        tan2[this->IndicesTmp[i].z] += tdir;

    }

    for(uint64_t i=0; i<this->PositionsTmp.size(); i++) { 
        glm::vec3 n = this->NormalsTmp[i];
        glm::vec3 t = glm::vec3(tan1[i]);

        this->TangentsTmp[i] = glm::vec4(glm::normalize((t - n * glm::dot(n, t))), 1);
        
        this->TangentsTmp[i].w = (glm::dot(glm::cross(n, t), glm::vec3(tan2[i])) < 0.0F) ? -1.0F : 1.0F;
    }
}

void shape::PreProcess()
{
    if(this->NormalsTmp.size() == 0)
    {
        this->NormalsTmp.resize(this->PositionsTmp.size());
        for (size_t j = 0; j < this->IndicesTmp.size(); j++)
        {
            glm::ivec3 Tri = this->IndicesTmp[j];
            glm::vec3 v0 = this->PositionsTmp[Tri.x];
            glm::vec3 v1 = this->PositionsTmp[Tri.y];
            glm::vec3 v2 = this->PositionsTmp[Tri.z];

            glm::vec3 Normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
            this->NormalsTmp[Tri.x] = Normal;
            this->NormalsTmp[Tri.y] = Normal;
            this->NormalsTmp[Tri.z] = Normal;
        }
    }
    if(this->TangentsTmp.size() ==0)
    {
        this->CalculateTangents();            
    }
    if(this->TexCoordsTmp.size() != this->PositionsTmp.size()) this->TexCoordsTmp.resize(this->PositionsTmp.size());
    

    uint32_t AddedTriangles=0;
    Triangles.resize(IndicesTmp.size());
    for(size_t j=0; j<IndicesTmp.size(); j++)
    {
        uint32_t i0 = IndicesTmp[j].x;
        uint32_t i1 = IndicesTmp[j].y;
        uint32_t i2 = IndicesTmp[j].z;
        glm::vec3 v0 = glm::vec3(PositionsTmp[i0]);
        glm::vec3 v1 = glm::vec3(PositionsTmp[i1]);
        glm::vec3 v2 = glm::vec3(PositionsTmp[i2]);

        glm::vec3 n0 = glm::vec3(NormalsTmp[i0]);
        glm::vec3 n1 = glm::vec3(NormalsTmp[i1]);
        glm::vec3 n2 = glm::vec3(NormalsTmp[i2]);
        
        Triangles[AddedTriangles].PositionUvX0=glm::vec4(v0, TexCoordsTmp[i0].x);
        Triangles[AddedTriangles].PositionUvX1=glm::vec4(v1, TexCoordsTmp[i1].x);
        Triangles[AddedTriangles].PositionUvX2=glm::vec4(v2, TexCoordsTmp[i2].x);
        
        Triangles[AddedTriangles].NormalUvY0=glm::vec4(n0, TexCoordsTmp[i0].y);
        Triangles[AddedTriangles].NormalUvY1=glm::vec4(n1, TexCoordsTmp[i1].y);
        Triangles[AddedTriangles].NormalUvY2=glm::vec4(n2, TexCoordsTmp[i2].y);

        Triangles[AddedTriangles].Tangent0 = TangentsTmp[i0];
        Triangles[AddedTriangles].Tangent1 = TangentsTmp[i1];
        Triangles[AddedTriangles].Tangent2 = TangentsTmp[i2];
        

        AddedTriangles++;
    }

    double InverseSize = 1.0 / double(this->PositionsTmp.size()); 
    glm::dvec3 Centroid;
    for(size_t j=0; j < this->PositionsTmp.size(); j++)
    {
        Centroid += glm::dvec3(this->PositionsTmp[j]) * InverseSize;
    }
    this->Centroid = glm::vec3(Centroid);    

    // PositionsTmp.resize(0);
    // NormalsTmp.resize(0);
    // TexCoordsTmp.resize(0);
    // TangentsTmp.resize(0);
    // IndicesTmp.resize(0);

    BVH = std::make_shared<blas>(this);
#if USE_OPTIX
    // Create OptiX
    CUdeviceptr DevVertices;
    cudaMalloc((void**)&DevVertices, PositionsTmp.size() * sizeof(glm::vec3));
    cudaMemcpy((void*)DevVertices, PositionsTmp.data(), PositionsTmp.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    CUdeviceptr DevIndices;
    cudaMalloc((void**)&DevIndices, IndicesTmp.size() * sizeof(glm::ivec3));
    cudaMemcpy((void*)DevIndices, IndicesTmp.data(), IndicesTmp.size() * sizeof(glm::ivec3), cudaMemcpyHostToDevice);

    uint32_t Flags = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput BuildInput = {};
    BuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    BuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    BuildInput.triangleArray.numVertices = PositionsTmp.size();
    BuildInput.triangleArray.vertexBuffers = &DevVertices;
    BuildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    BuildInput.triangleArray.numIndexTriplets = IndicesTmp.size();
    BuildInput.triangleArray.indexBuffer = DevIndices;
    BuildInput.triangleArray.numSbtRecords = 1;
    BuildInput.triangleArray.flags = &Flags;
    

    OptixAccelBuildOptions ASOptions = {};
    ASOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    ASOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ASBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(application::Get()->OptixContext, &ASOptions, &BuildInput, 1, &ASBufferSizes));

    ASBuffer = std::make_shared<buffer>(ASBufferSizes.outputSizeInBytes);
    CUdeviceptr TempBuffer;
    cudaMalloc((void**)&TempBuffer, ASBufferSizes.tempSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        application::Get()->OptixContext,
        0,
        &ASOptions,
        &BuildInput,
        1,
        TempBuffer,
        ASBufferSizes.tempSizeInBytes,
        (CUdeviceptr)ASBuffer->Data,
        ASBufferSizes.outputSizeInBytes,
        &ASHandle,
        nullptr,
        0
    ));

    cudaFree((void*)TempBuffer);    
#endif
}

void shape::ToFile(std::ofstream &Stream)
{
    SerializeVector(Stream, PositionsTmp);
    SerializeVector(Stream, NormalsTmp);
    SerializeVector(Stream, TexCoordsTmp);
    SerializeVector(Stream, TangentsTmp);
    SerializeVector(Stream, IndicesTmp);
    SerializeVector(Stream, Triangles);
    Stream.write((char*)&Centroid, sizeof(glm::vec3));
}

void shape::FromFile(std::ifstream &Stream)
{
    DeserializeVector(Stream, PositionsTmp);
    DeserializeVector(Stream, NormalsTmp);
    DeserializeVector(Stream, TexCoordsTmp);
    DeserializeVector(Stream, TangentsTmp);
    DeserializeVector(Stream, IndicesTmp);
    DeserializeVector(Stream, Triangles);

    PositionsTmp.resize(Triangles.size() * 3);
    NormalsTmp.resize(Triangles.size() * 3);
    TexCoordsTmp.resize(Triangles.size() * 3);
    TangentsTmp.resize(Triangles.size() * 3);
    IndicesTmp.resize(Triangles.size());

    uint32_t Inx=0;
    for(uint32_t i=0; i<Triangles.size(); i++)
    {
        triangle &Tri = Triangles[i];
        {
            glm::vec4 PosUVX = Tri.PositionUvX0;
            glm::vec4 NormUVY = Tri.NormalUvY0;
            PositionsTmp[Inx] = PosUVX;
            NormalsTmp[Inx] = NormUVY;
            TexCoordsTmp[Inx].x = PosUVX.w;
            TexCoordsTmp[Inx].y = NormUVY.w;
            TangentsTmp[Inx] = Tri.Tangent0;
            IndicesTmp[i].z = Inx;
            Inx++;
        }
        {
            glm::vec4 PosUVX = Tri.PositionUvX1;
            glm::vec4 NormUVY = Tri.NormalUvY1;
            PositionsTmp[Inx] = PosUVX;
            NormalsTmp[Inx] = NormUVY;
            TexCoordsTmp[Inx].x = PosUVX.w;
            TexCoordsTmp[Inx].y = NormUVY.w;
            TangentsTmp[Inx] = Tri.Tangent1;
            IndicesTmp[i].x = Inx;
            Inx++;
        }
        {
            glm::vec4 PosUVX = Tri.PositionUvX2;
            glm::vec4 NormUVY = Tri.NormalUvY2;
            PositionsTmp[Inx] = PosUVX;
            NormalsTmp[Inx] = NormUVY;
            TexCoordsTmp[Inx].x = PosUVX.w;
            TexCoordsTmp[Inx].y = NormUVY.w;
            TangentsTmp[Inx] = Tri.Tangent2; 
            IndicesTmp[i].y = Inx;
            Inx++;
        }
    }

    Stream.read((char*)&Centroid, sizeof(glm::vec3));
}

void scene::CalculateInstanceTransform(int Inx)
{
    instance &Instance = Instances[Inx];

    Instance.InverseTransform = glm::inverse(Instance.Transform);
    Instance.NormalTransform = glm::inverseTranspose(Instance.Transform);
    blas *BVH = Shapes[Instance.Shape].BVH.get();
    glm::vec3 Min = BVH->BVHNodes[0].AABBMin;
    glm::vec3 Max = BVH->BVHNodes[0].AABBMax;
    Instance.Bounds = {};
    for (int i = 0; i < 8; i++)
    {
		Instance.Bounds.Grow( Instance.Transform *  glm::vec4( 
                                    i & 1 ? Max.x : Min.x,
                                    i & 2 ? Max.y : Min.y, 
                                    i & 4 ? Max.z : Min.z,
                                    1.0f ));
    }      
}

scene::scene()
{
    glm::uvec2 RenderSize = application::GetSize();


    this->Cameras.emplace_back();
    camera &Camera = this->Cameras.back();
    Camera.FOV = 60.0f;
    Camera.Aspect = (float)RenderSize.x / (float)RenderSize.y;
    Camera.Controlled = 1;  
    this->CameraNames.push_back("Main Camera");
    
    // LoadAssimp("resources/models/BaseShapes/Cube/Cube.obj", this, false, false, false, 1.0f);
    // LoadAssimp("resources/models/BaseShapes/Cone/Cone.obj", this, false, false, false, 1.0f);
    // LoadAssimp("resources/models/BaseShapes/Cylinder/Cylinder.obj", this, false, false, false, 1.0f);
    // LoadAssimp("resources/models/BaseShapes/Sphere/Sphere.obj", this, false, false, false, 1.0f);
    // LoadAssimp("resources/models/BaseShapes/Torus/Torus.obj", this, false, false, false, 1.0f);
    LoadAssimp("resources/models/BaseShapes/Plane/Plane.obj", this, false, false, false, 1.0f);

    this->Materials.emplace_back();
    material &BaseMaterial = this->Materials.back(); 
    BaseMaterial.Colour = {0.725f, 0.71f, 0.68f};
    this->MaterialNames.push_back("Base");

    
    // {
    //     this->Instances.emplace_back();
    //     instance &CubeInstance = this->Instances.back();
    //     CubeInstance.Shape = (int)0;
    //     CubeInstance.Material = (int)this->Materials.size()-1;
    //     CubeInstance.Transform = glm::mat4(1);
    //     this->InstanceNames.push_back("Cube");
    // }
    
    {
        this->Instances.emplace_back();
        instance &FloorInstance = this->Instances.back();
        FloorInstance.Shape = (int)this->Shapes.size()-1;
        FloorInstance.Material = (int)this->Materials.size()-1;
        FloorInstance.Transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.0f, 0.0f)) * glm::scale(glm::mat4(1.0f), glm::vec3(4.0f, 4.0f, 4.0f));
        this->InstanceNames.push_back("Floor");
    }
    
    this->Materials.emplace_back();
    material &LightMaterial = this->Materials.back();
    LightMaterial.Emission = {40, 40, 40};    
    this->Instances.emplace_back();
    instance &LightInstance = this->Instances.back(); 
    LightInstance.Shape = (int)this->Shapes.size()-1;
    LightInstance.Material = (int)this->Materials.size()-1;
    LightInstance.Transform = glm::translate(glm::mat4(1.0f), glm::vec3(0, 2, 0));
    this->InstanceNames.push_back("Light");
    this->MaterialNames.push_back("Light");
    
}

void scene::CheckNames()
{
    EnsureUnicity(this->ShapeNames, "Shape");
    EnsureUnicity(this->InstanceNames, "Instance");
    EnsureUnicity(this->TextureNames, "Texture");
    EnsureUnicity(this->EnvironmentNames, "Environment");
    EnsureUnicity(this->CameraNames, "Camera");
    EnsureUnicity(this->EnvTextureNames, "EnvTexture");
    EnsureUnicity(this->MaterialNames, "Material");
}

void scene::UpdateLights()
{
    this->Lights->Build(this);
}

void scene::UploadMaterial(int MaterialInx)
{
    this->MaterialBuffer->updateData((size_t)MaterialInx * sizeof(material), (void*)&this->Materials[MaterialInx], sizeof(material));
    this->MaterialBufferGL->updateData((size_t)MaterialInx * sizeof(material), (void*)&this->Materials[MaterialInx], sizeof(material));
}

void scene::RemoveInstance(int InstanceInx)
{
    InstanceNames.erase(InstanceNames.begin() + InstanceInx);
    BVH->RemoveInstance(InstanceInx);
    Lights->RemoveInstance(this, InstanceInx);
}

void scene::PreProcess()
{
    this->ReloadTextureArray();

    // Ensure name unicity
    CheckNames();

    for(size_t i=0; i<Instances.size(); i++)
    {
        Instances[i].Index = i;
        CalculateInstanceTransform(i);
    }


    BVH = CreateBVH(this); 
    VertexBuffer = std::make_shared<vertexBuffer>(this);
    Lights = std::make_shared<lights>();
    Lights->Build(this);
    this->CamerasBuffer = std::make_shared<buffer>(this->Cameras.size() * sizeof(camera), this->Cameras.data());
    this->EnvironmentsBuffer = std::make_shared<buffer>(this->Environments.size() * sizeof(environment), this->Environments.data());
    this->MaterialBuffer = std::make_shared<buffer>(sizeof(material) * Materials.size(), Materials.data());
    this->MaterialBufferGL = std::make_shared<bufferGL>(sizeof(material) * Materials.size(), Materials.data());
}


void scene::ReloadTextureArray()
{
    assert(8192 % TextureWidth==0);
    assert(8192 % TextureHeight==0);
    assert(8192 % EnvTextureWidth==0);
    assert(8192 % EnvTextureHeight==0);

    CUDA_CHECK_ERROR(cudaGetLastError());

    TexArray = std::make_shared<textureArrayCu>();
    EnvTexArray = std::make_shared<textureArrayCu>();
    CUDA_CHECK_ERROR(cudaGetLastError());

    TexArray->CreateTextureArray(TextureWidth, TextureHeight, Textures.size());
    for (size_t i = 0; i < Textures.size(); i++)
    {
        TexArray->LoadTextureLayer(i, Textures[i].Pixels, TextureWidth, TextureHeight);
        
    }

    EnvTexArray->CreateTextureArray(EnvTextureWidth, EnvTextureHeight, EnvTextures.size(), true);
    for (size_t i = 0; i < EnvTextures.size(); i++)
    {
        EnvTexArray->LoadTextureLayer(i, EnvTextures[i].PixelsF, EnvTextureWidth, EnvTextureHeight);
    }
    CUDA_CHECK_ERROR(cudaGetLastError());
}



void scene::ToFile(std::string FileName)
{
    std::ofstream OutStream;
    OutStream.open(FileName, std::ios_base::binary);
    if(!OutStream.is_open())
    {
        std::cout << "Could not open file" << FileName << std::endl;
        return;
    }

    SerializeVector(OutStream, Cameras);
    SerializeVector(OutStream, Materials);
    SerializeVector(OutStream, Instances);
    SerializeVector(OutStream, Environments);

    size_t ShapesSize = Shapes.size();
    OutStream.write((char*)&ShapesSize, sizeof(size_t));
    for(int i=0; i<Shapes.size(); i++)
    {
        Shapes[i].ToFile(OutStream);
    }
    size_t EnvTexturesSize = EnvTextures.size();
    OutStream.write((char*)&EnvTexturesSize, sizeof(size_t));
    for(int i=0; i<EnvTextures.size(); i++)
    {
        EnvTextures[i].ToFile(OutStream);
    }
    size_t TexturesSize = Textures.size();
    OutStream.write((char*)&TexturesSize, sizeof(size_t));
    for(int i=0; i<Textures.size(); i++)
    {
        Textures[i].ToFile(OutStream);
    }    
    
    SerializeStrVector(OutStream, CameraNames);
    SerializeStrVector(OutStream, InstanceNames);
    SerializeStrVector(OutStream, ShapeNames);
    SerializeStrVector(OutStream, MaterialNames);
    SerializeStrVector(OutStream, TextureNames);
    SerializeStrVector(OutStream, EnvTextureNames);
    SerializeStrVector(OutStream, EnvironmentNames);

    OutStream.write((char*)&TextureWidth, sizeof(int));
    OutStream.write((char*)&TextureHeight, sizeof(int));
    OutStream.write((char*)&EnvTextureWidth, sizeof(int));
    OutStream.write((char*)&EnvTextureHeight, sizeof(int));
}

void scene::FromFile(std::string FileName)
{
    std::ifstream InStream;
    InStream.open(FileName, std::ios_base::binary);
    if(!InStream.is_open())
    {
        std::cout << "Could not open file" << FileName << std::endl;
        return;
    }

    struct oldCamStruct
    {
        glm::mat4 Frame;
        
        float Lens;
        float Film;
        float Aspect;
        float Focus;
        
        glm::vec3 Padding0;
        float Aperture;
        
        int Orthographic;
        int Controlled;
        glm::ivec2 Padding;
    };
    std::vector<oldCamStruct> _Cameras;

    DeserializeVector(InStream, _Cameras);
    for(int i=0; i<_Cameras.size(); i++)
    {
        camera Camera = {};
        Camera.Frame = _Cameras[i].Frame;
        Camera.SetAspect(_Cameras[i].Aspect);
        Camera.Controlled = _Cameras[i].Controlled;
        Cameras.push_back(Camera);
    }
        
    DeserializeVector(InStream, Materials);
    DeserializeVector(InStream, Instances);
    DeserializeVector(InStream, Environments);
    
    
    size_t ShapesSize;
    InStream.read((char*)&ShapesSize, sizeof(size_t));
    Shapes.resize(ShapesSize);
    for(int i=0; i<Shapes.size(); i++)
    {
        Shapes[i].FromFile(InStream);
    }
    size_t EnvTexturesSize;
    InStream.read((char*)&EnvTexturesSize, sizeof(size_t));
    EnvTextures.resize(EnvTexturesSize);
    for(int i=0; i<EnvTextures.size(); i++)
    {
        EnvTextures[i].FromFile(InStream);
    }
    size_t TexturesSize;
    InStream.read((char*)&TexturesSize, sizeof(size_t));
    Textures.resize(TexturesSize);
    for(int i=0; i<Textures.size(); i++)
    {
        Textures[i].FromFile(InStream);
    }        
    
    DeserializeStrVector(InStream, CameraNames);
    DeserializeStrVector(InStream, InstanceNames);
    DeserializeStrVector(InStream, ShapeNames);
    DeserializeStrVector(InStream, MaterialNames);
    DeserializeStrVector(InStream, TextureNames);
    DeserializeStrVector(InStream, EnvTextureNames);
    DeserializeStrVector(InStream, EnvironmentNames);

    InStream.read((char*)&TextureWidth, sizeof(int));
    InStream.read((char*)&TextureHeight, sizeof(int));
    InStream.read((char*)&EnvTextureWidth, sizeof(int));
    InStream.read((char*)&EnvTextureHeight, sizeof(int));
    
    TextureWidth = TEX_WIDTH;
    TextureHeight = TEX_WIDTH;
    EnvTextureWidth = ENV_TEX_WIDTH;
    EnvTextureHeight = ENV_TEX_WIDTH/2;

    for(int i=0; i<Shapes.size(); i++)
    {
        Shapes[i].PreProcess();
        // Shapes[i].BVH = std::make_shared<blas>(&Shapes[i]);
    }
}

void scene::Clear()
{
    Cameras.clear();
    Instances.clear();
    Shapes.clear();
    Materials.clear();
    Textures.clear();
    EnvTextures.clear();
    Environments.clear();

    
    CameraNames.clear();
    InstanceNames.clear();
    ShapeNames.clear();
    MaterialNames.clear();
    TextureNames.clear();
    EnvTextureNames.clear();
    EnvironmentNames.clear();
}

void scene::ClearInstances()
{
    Instances.clear();
    InstanceNames.clear();
    Lights->Build(this);
}

glm::vec4 texture::Sample(glm::ivec2 Coords)
{
    glm::vec4 Res;
    Res.x = (float)this->Pixels[(Coords.y * Width + Coords.x) * 4 + 0] * 255.0f;
    Res.y = (float)this->Pixels[(Coords.y * Width + Coords.x) * 4 + 1] * 255.0f;
    Res.z = (float)this->Pixels[(Coords.y * Width + Coords.x) * 4 + 2] * 255.0f;
    Res.w = (float)this->Pixels[(Coords.y * Width + Coords.x) * 4 + 3] * 255.0f;
    return Res;
}

glm::vec4 texture::SampleF(glm::ivec2 Coords)
{
    glm::vec4 Res;
    Res.x = this->PixelsF[(Coords.y * Width + Coords.x) * 4 + 0];
    Res.y = this->PixelsF[(Coords.y * Width + Coords.x) * 4 + 1];
    Res.z = this->PixelsF[(Coords.y * Width + Coords.x) * 4 + 2];
    Res.w = this->PixelsF[(Coords.y * Width + Coords.x) * 4 + 3];
    return Res;
}

void texture::SetFromFile(const std::string &FileName, int Width, int Height)
{
    if(IsHDR(FileName))
    {
        int NumChannels=4;
        ImageFromFile(FileName, this->PixelsF, Width, Height, NumChannels);
        this->NumChannels = this->Pixels.size() / (Width * Height);
        this->Width = Width;
        this->Height = Height;
    }
    else
    {
        int NumChannels=4;
        ImageFromFile(FileName, this->Pixels, Width, Height, NumChannels);
        this->NumChannels = this->Pixels.size() / (Width * Height);
        this->Width = Width;
        this->Height = Height;
    }
}

void texture::ToFile(std::ofstream &Stream)
{
    SerializeVector(Stream, Pixels);
    SerializeVector(Stream, PixelsF);
    Stream.write((char*)&Width, sizeof(int));
    Stream.write((char*)&Height, sizeof(int));
    Stream.write((char*)&NumChannels, sizeof(int));
}

void texture::FromFile(std::ifstream &Stream)
{
    DeserializeVector(Stream, Pixels);
    DeserializeVector(Stream, PixelsF);
    Stream.read((char*)&Width, sizeof(int));
    Stream.read((char*)&Height, sizeof(int));
    Stream.read((char*)&NumChannels, sizeof(int));

    Resize(TEX_WIDTH);
}

void texture::Resize(int TargetSize)
{
    if(Pixels.size()!=0)
    {
        uint8_t* ResizedImage = new uint8_t[TargetSize * TargetSize * 4]; // Assuming RGBA format
        int result = stbir_resize_uint8(Pixels.data(), Width, Height, 0, ResizedImage, TargetSize, TargetSize, 0, 4);
        
        if (!result) {
            assert(false);
            delete[] ResizedImage;
            return;
        }

        // Resize the pixel data, and copy to it
        Pixels.resize(TargetSize * TargetSize * 4);
        memcpy(Pixels.data(), ResizedImage, Pixels.size());
        delete[] ResizedImage;            
    }
    if(PixelsF.size()!=0)
    {
        float* ResizedImage = new float[TargetSize * TargetSize * 4]; // Assuming RGBA format
        int result = stbir_resize_float(PixelsF.data(), Width, Height, 0, ResizedImage, TargetSize, TargetSize, 0, 4);
        
        if (!result) {
            assert(false);
            delete[] ResizedImage;
            return;
        } 

        // Resize the pixel data, and copy to it
        PixelsF.resize(TargetSize * TargetSize * 4);
        memcpy(PixelsF.data(), ResizedImage, PixelsF.size() * sizeof(float));
        delete[] ResizedImage;            
    }
    this->Width = TargetSize;
    this->Height = TargetSize;
}

void texture::SetFromPixels(const std::vector<uint8_t> &PixelData, int Width, int Height)
{

}


}