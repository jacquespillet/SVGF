#pragma once

#include "Scene.h"
#include <memory>
#include <vector>

namespace gpupt
{

class bufferGL;
class bufferCu;

struct aabb
{
    glm::vec3 Min =glm::vec3(1e30f);
    float pad0;
    glm::vec3 Max =glm::vec3(-1e30f);
    float pad1;
    float Area();
    void Grow(glm::vec3 Position);
    void Grow(aabb &AABB);
};

struct ray
{
    glm::vec3 Origin;
    glm::vec3 Direction;
    glm::vec3 InverseDirection;
    ray(glm::vec3 O, glm::vec3 D, glm::vec3 ID) : Origin(O), Direction(D), InverseDirection(ID){}
    ray() = default;
};

struct triangle
{
    glm::vec3 v0;
    float padding0;
    
    glm::vec3 v1;
    float padding1;
    
    glm::vec3 v2;
    float padding2; 
    
    glm::vec3 Centroid;
    float padding3; 
};

struct triangleExtraData
{
    glm::vec3 Normal0; 
    float padding0;

    glm::vec3 Normal1; 
    float padding1;
    
    glm::vec3 Normal2; 
    float padding2;
    
    glm::vec2 UV0, UV1, UV2; 
    glm::vec2 padding3;

    glm::vec4 Colour0;
    glm::vec4 Colour1;  
    glm::vec4 Colour2;
    
    glm::vec4 Tangent0;
    glm::vec4 Tangent1;  
    glm::vec4 Tangent2;
};

struct bvhNode
{
    glm::vec3 AABBMin;
    float padding0;
    glm::vec3 AABBMax;
    float padding1;
    uint32_t LeftChildOrFirst;
    uint32_t TriangleCount;
    glm::uvec2 padding2;
    bool IsLeaf();
};


struct bin
{
    aabb Bounds;
    uint32_t TrianglesCount=0;
};

struct rayPayload
{
    float Distance;
    float U, V;
    glm::vec3 Emission;
    uint32_t InstanceIndex;
    uint32_t MaterialIndex;
    uint32_t PrimitiveIndex;
    uint32_t RandomState;
    uint8_t Depth;
};


struct mesh;

struct bvh
{
    bvh(mesh *Mesh);
    void Build();
    void Refit();

    void Subdivide(uint32_t NodeIndex);
    void UpdateNodeBounds(uint32_t NodeIndex);
    float FindBestSplitPlane(bvhNode &Node, int &Axis, float &SplitPosition);

    float EvaluateSAH(bvhNode &Node, int Axis, float Position);
    float CalculateNodeCost(bvhNode &Node);

    mesh *Mesh;
    
    std::vector<uint32_t> TriangleIndices;

    std::vector<bvhNode> BVHNodes;
    uint32_t NodesUsed=1;
    uint32_t RootNodeIndex=0;
};

struct vertex
{
    glm::vec4 Position;
    glm::vec4 Normal;
    glm::vec4 Tangent;
    glm::vec4 MatInx;
};

struct mesh
{
    mesh(const shape &Shape);
    bvh *BVH;
    std::vector<triangle> Triangles;
    std::vector<triangleExtraData> TrianglesExtraData;
};



struct tlasNode
{
    glm::vec3 AABBMin;
    uint32_t LeftRight;
    glm::vec3 AABBMax;
    uint32_t BLAS;
    bool IsLeaf() {return LeftRight==0;}
};

struct bvhInstance
{
    bvhInstance(std::vector<mesh*> *Meshes, uint32_t MeshIndex, glm::mat4 Transform, uint32_t Index) : MeshIndex(MeshIndex), Index(Index)
    {
        SetTransform(Transform, Meshes);
    }
    void SetTransform(glm::mat4 &Transform, std::vector<mesh*> *Meshes);

    //Store the mesh index in the scene instead, and a pointer to the mesh array to access the bvh.
    glm::mat4 InverseTransform;
    glm::mat4 Transform;
    glm::mat4 NormalTransform;
    aabb Bounds;

    uint32_t MeshIndex;
    uint32_t Index=0;
    uint32_t pad;
};

struct tlas
{
    tlas(std::vector<bvhInstance>* Instances);
    tlas();
    void Build();

    int FindBestMatch(std::vector<int>& List, int N, int A);

    //Instances
    std::vector<bvhInstance>* BLAS;
    
    std::vector<tlasNode> Nodes;

    uint32_t NodesUsed=0;
};



struct indexData
{
    uint32_t triangleDataStartInx;
    uint32_t IndicesDataStartInx;
    uint32_t BVHNodeDataStartInx;
    uint32_t TriangleCount;
};

struct sceneBVH
{
    tlas TLAS;

    std::vector<mesh*> Meshes;
    std::vector<bvhInstance> Instances;
    std::vector<indexData> IndexData;
    std::vector<triangle> AllTriangles;
    std::vector<triangleExtraData> AllTrianglesEx;
    std::vector<uint32_t> AllTriangleIndices;
    std::vector<bvhNode> AllBVHNodes;

#if API == API_GL
    std::shared_ptr<bufferGL> TrianglesBuffer;
    std::shared_ptr<bufferGL> TrianglesExBuffer;
    std::shared_ptr<bufferGL> BVHBuffer;
    std::shared_ptr<bufferGL> IndicesBuffer;
    std::shared_ptr<bufferGL> IndexDataBuffer;
    std::shared_ptr<bufferGL> TLASInstancesBuffer;
    std::shared_ptr<bufferGL> TLASNodeBuffer;
#elif API == API_CU
    std::shared_ptr<bufferCu> TrianglesBuffer;
    std::shared_ptr<bufferCu> TrianglesExBuffer;
    std::shared_ptr<bufferCu> BVHBuffer;
    std::shared_ptr<bufferCu> IndicesBuffer;
    std::shared_ptr<bufferCu> IndexDataBuffer;
    std::shared_ptr<bufferCu> TLASInstancesBuffer;
    std::shared_ptr<bufferCu> TLASNodeBuffer;
#endif

    void Destroy();
    std::shared_ptr<scene> Scene;
};

std::shared_ptr<sceneBVH> CreateBVH(std::shared_ptr<scene> Scene);

}
