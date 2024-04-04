#include "GLTFLoader.h"
#include "Scene.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

namespace gpupt
{

void LoadGeometry(tinygltf::Model &GLTFModel, std::shared_ptr<scene> Scene, std::vector<std::vector<uint32_t>> &InstanceMapping)
{
    uint32_t GIndexBase=0;
    InstanceMapping.resize(GLTFModel.meshes.size());
    
    for(int MeshIndex=0; MeshIndex<GLTFModel.meshes.size(); MeshIndex++)
    {
        tinygltf::Mesh gltfMesh = GLTFModel.meshes[MeshIndex];
        std::vector<shape> &Shapes = Scene->Shapes;
        std::vector<std::string> &ShapeNames = Scene->ShapeNames;

        uint32_t MeshBaseIndex = Shapes.size();

        Shapes.resize(MeshBaseIndex + gltfMesh.primitives.size());
        ShapeNames.resize(MeshBaseIndex + gltfMesh.primitives.size());
        InstanceMapping[MeshIndex].resize(gltfMesh.primitives.size());

        for(int j=0; j<gltfMesh.primitives.size(); j++)
        {
            int Inx = MeshBaseIndex + j;

            tinygltf::Primitive GLTFPrimitive = gltfMesh.primitives[j];
            
            ShapeNames[Inx] = gltfMesh.name;
            InstanceMapping[MeshIndex][j] = Inx;
            shape &Shape = Scene->Shapes[Inx];

            if(GLTFPrimitive.mode != TINYGLTF_MODE_TRIANGLES)
                continue;
            
            //Get the index of each needed attribute
            int IndicesIndex = GLTFPrimitive.indices;
            int PositionIndex = -1;
            int NormalIndex = -1;
            int TangentIndex = -1;
            int UVIndex=-1;
            if(GLTFPrimitive.attributes.count("POSITION") >0)
                PositionIndex = GLTFPrimitive.attributes["POSITION"];
            if(GLTFPrimitive.attributes.count("NORMAL") >0)
                NormalIndex = GLTFPrimitive.attributes["NORMAL"];

            //Positions
            tinygltf::Accessor PositionAccessor = GLTFModel.accessors[PositionIndex];
            tinygltf::BufferView PositionBufferView = GLTFModel.bufferViews[PositionAccessor.bufferView];
            const tinygltf::Buffer &PositionBuffer = GLTFModel.buffers[PositionBufferView.buffer];
            const uint8_t *PositionBufferAddress = PositionBuffer.data.data();
            //3 * float
            int PositionStride = tinygltf::GetComponentSizeInBytes(PositionAccessor.componentType) * tinygltf::GetNumComponentsInType(PositionAccessor.type);
            if(PositionBufferView.byteStride > 0) PositionStride = (int)PositionBufferView.byteStride;

            //Normals
            tinygltf::Accessor NormalAccessor;
            tinygltf::BufferView NormalBufferView;
            const uint8_t *NormalBufferAddress=0;
            int NormalStride=0;
            if(NormalIndex >= 0)
            {
                NormalAccessor = GLTFModel.accessors[NormalIndex];
                NormalBufferView = GLTFModel.bufferViews[NormalAccessor.bufferView];
                const tinygltf::Buffer &normalBuffer = GLTFModel.buffers[NormalBufferView.buffer];
                NormalBufferAddress = normalBuffer.data.data();
                //3 * float
                NormalStride = tinygltf::GetComponentSizeInBytes(NormalAccessor.componentType) * tinygltf::GetNumComponentsInType(NormalAccessor.type);
                if(NormalBufferView.byteStride > 0) NormalStride =(int) NormalBufferView.byteStride;
            }

            //Indices
            tinygltf::Accessor IndicesAccessor = GLTFModel.accessors[IndicesIndex];
            tinygltf::BufferView IndicesBufferView = GLTFModel.bufferViews[IndicesAccessor.bufferView];
            const tinygltf::Buffer &IndicesBuffer = GLTFModel.buffers[IndicesBufferView.buffer];
            const uint8_t *IndicesBufferAddress = IndicesBuffer.data.data();
            int IndicesStride = tinygltf::GetComponentSizeInBytes(IndicesAccessor.componentType) * tinygltf::GetNumComponentsInType(IndicesAccessor.type); 

            Shape.Positions.resize(PositionAccessor.count);
            Shape.Normals.resize(PositionAccessor.count);
            for (size_t k = 0; k < PositionAccessor.count; k++)
            {
                glm::vec3 Position;
                {
                    const uint8_t *Address = PositionBufferAddress + PositionBufferView.byteOffset + PositionAccessor.byteOffset + (k * PositionStride);
                    memcpy(&Position, Address, 12);
                }

                glm::vec3 Normal;
                if(NormalIndex>=0)
                {
                    const uint8_t *Address = NormalBufferAddress + NormalBufferView.byteOffset + NormalAccessor.byteOffset + (k * NormalStride);
                    memcpy(&Normal, Address, 12);
                }

                Shape.Positions[k] = glm::vec3(Position.x, Position.y, Position.z);
                Shape.Normals[k] = glm::vec3(Normal.x, Normal.y, Normal.z);
            }


            //Fill indices buffer
            Shape.Triangles.resize(IndicesAccessor.count/3);
            const uint8_t *baseAddress = IndicesBufferAddress + IndicesBufferView.byteOffset + IndicesAccessor.byteOffset;
            if(IndicesStride == 1)
            {
                std::vector<uint8_t> Quarter;
                Quarter.resize(IndicesAccessor.count);
                memcpy(Quarter.data(), baseAddress, (IndicesAccessor.count) * IndicesStride);
                for(size_t i=0, j=0; i<IndicesAccessor.count; i+=3, j++)
                {
                    Shape.Triangles[j].x = Quarter[i+0];
                    Shape.Triangles[j].y = Quarter[i+1];
                    Shape.Triangles[j].z = Quarter[i+2];
                }
            }
            else if(IndicesStride == 2)
            {
                std::vector<uint16_t> Half;
                Half.resize(IndicesAccessor.count);
                memcpy(Half.data(), baseAddress, (IndicesAccessor.count) * IndicesStride);
                for(size_t i=0, j=0; i<IndicesAccessor.count; i+=3, j++)
                {
                    Shape.Triangles[j].x = Half[i+0];
                    Shape.Triangles[j].y = Half[i+1];
                    Shape.Triangles[j].z = Half[i+2];
                }
            }
            else
            {
                std::vector<uint32_t> Uint;
                Uint.resize(IndicesAccessor.count);
                memcpy(Uint.data(), baseAddress, (IndicesAccessor.count) * IndicesStride);
                for(size_t i=0, j=0; i<IndicesAccessor.count; i+=3, j++)
                {
                    Shape.Triangles[j].x = Uint[i+0];
                    Shape.Triangles[j].y = Uint[i+1];
                    Shape.Triangles[j].z = Uint[i+2];
                }                
            }
        }
    }
}

void LoadMaterials(tinygltf::Model &GLTFModel, std::shared_ptr<scene> Scene)
{
    std::vector<material> &Materials = Scene->Materials;
    std::vector<std::string> &MaterialNames = Scene->MaterialNames;
    
    // AScene->mMaterials[i]->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), Materials[i].MaterialData.BaseColorTextureID);
    uint32_t BaseInx = Materials.size();
    Materials.resize(Materials.size() + GLTFModel.materials.size());
    MaterialNames.resize(MaterialNames.size() + GLTFModel.materials.size());
    for (size_t i = 0; i < GLTFModel.materials.size(); i++)
    {
        uint32_t Inx = BaseInx + i;

        const tinygltf::Material GLTFMaterial = GLTFModel.materials[i];
        const tinygltf::PbrMetallicRoughness PBR = GLTFMaterial.pbrMetallicRoughness;
        
        MaterialNames[Inx] = GLTFMaterial.name;
        

        // TODO: Handle Blend modes
        // TODO: Handle Double Sided
        
        material &Material = Materials[Inx];
        Material = {};
        Material.MaterialType = MATERIAL_TYPE_PBR;
        
        // TODO: 
        // Opacity
        // AlphaCutoff
        Material.Colour = glm::vec3(PBR.baseColorFactor[0], PBR.baseColorFactor[1], PBR.baseColorFactor[2]);
        Material.Roughness = std::max(0.01, PBR.roughnessFactor);
        Material.Metallic = PBR.metallicFactor;
        Material.Colour = glm::vec3(0.7);
        // Material.Emission = glm::vec3(GLTFMaterial.emissiveFactor[0], GLTFMaterial.emissiveFactor[1], GLTFMaterial.emissiveFactor[2]);

        // TODO: Textures
    }
}    


void TraverseNodes(tinygltf::Model &GLTFModel, uint32_t nodeIndex, glm::mat4 ParentTransform, std::shared_ptr<scene> &Scene, std::vector<std::vector<uint32_t>> &InstanceMapping)
{
    tinygltf::Node GLTFNode = GLTFModel.nodes[nodeIndex];

    std::string NodeName = GLTFNode.name;
    if(NodeName.compare("") == 0)
    {
        NodeName = "Node";
    }

    std::vector<instance> &Instances = Scene->Instances;
    std::vector<std::string> &InstanceNames = Scene->InstanceNames;
    InstanceNames.push_back(NodeName);

    glm::mat4 NodeTransform;
    if(GLTFNode.matrix.size() > 0)
    {
        NodeTransform[0][0] = (float)GLTFNode.matrix[0]; NodeTransform[0][1] = (float)GLTFNode.matrix[1]; NodeTransform[0][2] = (float)GLTFNode.matrix[2]; NodeTransform[0][3] = (float)GLTFNode.matrix[3];
        NodeTransform[1][0] = (float)GLTFNode.matrix[4]; NodeTransform[1][1] = (float)GLTFNode.matrix[5]; NodeTransform[1][2] = (float)GLTFNode.matrix[6]; NodeTransform[1][3] = (float)GLTFNode.matrix[7];
        NodeTransform[2][0] = (float)GLTFNode.matrix[8]; NodeTransform[2][1] = (float)GLTFNode.matrix[9]; NodeTransform[2][2] = (float)GLTFNode.matrix[10]; NodeTransform[2][3] = (float)GLTFNode.matrix[11];
        NodeTransform[3][0] = (float)GLTFNode.matrix[12]; NodeTransform[3][1] = (float)GLTFNode.matrix[13]; NodeTransform[3][2] = (float)GLTFNode.matrix[14]; NodeTransform[3][3] = (float)GLTFNode.matrix[15];
    }
    else
    {
            glm::mat4 translate(1);
            glm::mat4 rotation(1);
            glm::mat4 scale(1);
            if(GLTFNode.translation.size()>0)
            {
                translate[3][0] = (float)GLTFNode.translation[0];
                translate[3][1] = (float)GLTFNode.translation[1];
                translate[3][2] = (float)GLTFNode.translation[2];
            }
            if(GLTFNode.rotation.size() > 0)
            {
                glm::quat Quat((float)GLTFNode.rotation[3], (float)GLTFNode.rotation[0], (float)GLTFNode.rotation[1], (float)GLTFNode.rotation[2]);
                rotation = glm::toMat4(Quat);

            }
            if(GLTFNode.scale.size() > 0)
            {
                scale[0][0] = (float)GLTFNode.scale[0];
                scale[1][1] = (float)GLTFNode.scale[1];
                scale[2][2] = (float)GLTFNode.scale[2];
            }
            NodeTransform = scale * rotation * translate;
    }

    glm::mat4 Transform = ParentTransform * NodeTransform;

    //Leaf node
    if(GLTFNode.children.size() == 0 && GLTFNode.mesh != -1)
    {
        tinygltf::Mesh GLTFMesh = GLTFModel.meshes[GLTFNode.mesh];
        for(int i=0; i<GLTFMesh.primitives.size(); i++)
        {
            Scene->InstanceNames.push_back(GLTFNode.name);

            Scene->Instances.emplace_back();
            instance &Instance = Scene->Instances.back();
            Instance.ModelMatrix = Transform;
            Instance.Shape = InstanceMapping[GLTFNode.mesh][i];
            Instance.Material = Scene->Materials.size() + GLTFMesh.primitives[i].material; 
        }   
    }

    for (size_t i = 0; i < GLTFNode.children.size(); i++)
    {
        TraverseNodes(GLTFModel, GLTFNode.children[i], Transform, Scene, InstanceMapping);
    }
}

void LoadInstances(tinygltf::Model &GLTFModel, std::shared_ptr<scene> Scene, std::vector<std::vector<uint32_t>> &InstanceMapping)
{
    // glm::mat4 Scale = glm::scale(glm::mat4(1), glm::vec3(25));
    // glm::mat4 Translate = glm::translate(glm::mat4(1), glm::vec3(0, 0.4, 0));
    // glm::mat4 RootTransform =  Translate * Scale;
    glm::mat4 RootTransform(0.5f);
    const tinygltf::Scene GLTFScene = GLTFModel.scenes[GLTFModel.defaultScene];
    for (size_t i = 0; i < GLTFScene.nodes.size(); i++)
    {
        TraverseNodes(GLTFModel, GLTFScene.nodes[i], RootTransform, Scene, InstanceMapping);
    }
}

void LoadGLTF(std::string FileName, std::shared_ptr<scene> Scene)
{
    tinygltf::Model GLTFModel;
    tinygltf::TinyGLTF ModelLoader;

    std::string Error, Warning;

    std::string Extension = FileName.substr(FileName.find_last_of(".") + 1);
    bool OK = false;
    if(Extension == "gltf")
    {
        OK = ModelLoader.LoadASCIIFromFile(&GLTFModel, &Error, &Warning, FileName);
    }
    else if(Extension == "glb")
    {
        OK = ModelLoader.LoadBinaryFromFile(&GLTFModel, &Error, &Warning, FileName);
    }
    else OK=false;
        
    if(!OK) 
    {
        printf("Could not load model %s \n",FileName);
        return;
    }

    std::vector<std::vector<uint32_t>> InstanceMapping;
    
    LoadGeometry(GLTFModel, Scene, InstanceMapping);
    LoadInstances(GLTFModel, Scene, InstanceMapping);
    LoadMaterials(GLTFModel, Scene);
}    
}