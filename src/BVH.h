#pragma once

#include "Scene.h"
#include <memory>
#include <vector>

namespace gpupt
{

class buffer;




struct bvhNode
{
    glm::vec3 AABBMin;
    float LeftChildOrFirst;
    glm::vec3 AABBMax;
    float TriangleCount;
    bool IsLeaf();
};


struct bin
{
    aabb Bounds;
    uint32_t TrianglesCount=0;
};


struct shape;

struct blas
{
    blas(shape *Shape);
    void Build();
    void Refit();

    void Subdivide(uint32_t NodeIndex);
    void UpdateNodeBounds(uint32_t NodeIndex);
    float FindBestSplitPlane(bvhNode &Node, int &Axis, float &SplitPosition);

    float EvaluateSAH(bvhNode &Node, int Axis, float Position);
    float CalculateNodeCost(bvhNode &Node);

    shape *Shape;
    
    std::vector<uint32_t> TriangleIndices;

    std::vector<bvhNode> BVHNodes;
    uint32_t NodesUsed=1;
    uint32_t RootNodeIndex=0;
};





struct tlasNode
{
    glm::vec3 AABBMin;
    uint32_t LeftRight;
    glm::vec3 AABBMax;
    uint32_t BLAS;
    bool IsLeaf() {return LeftRight==0;}
};


struct tlas
{
    tlas(std::vector<instance>* Instances);
    tlas();
    void Build();

    int FindBestMatch(std::vector<int>& List, int N, int A);

    //Instances
    std::vector<instance>* BLAS;
    
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

#if USE_OPTIX
struct optixAS
{
    std::vector<OptixInstance> OptixInstances;
    OptixTraversableHandle InstanceASHandle;
    std::shared_ptr<buffer> InstanceASBuffer;
    std::shared_ptr<buffer> ShapeASHandlesBuffer;
    optixAS(scene *Scene);
    scene *Scene;
    void Build();
};
#endif

struct sceneBVH
{
    tlas TLAS;
#if USE_OPTIX
    std::shared_ptr<optixAS> OptixAS;
#endif

    std::vector<indexData> IndexData;
    std::vector<triangle> AllTriangles;
    std::vector<uint32_t> AllTriangleIndices;
    std::vector<bvhNode> AllBVHNodes;

    std::shared_ptr<buffer> TrianglesBuffer;
    std::shared_ptr<buffer> BVHBuffer;
    std::shared_ptr<buffer> IndicesBuffer;
    std::shared_ptr<buffer> IndexDataBuffer;
    std::shared_ptr<buffer> TLASInstancesBuffer;
    std::shared_ptr<buffer> TLASNodeBuffer;

    void UpdateShape(uint32_t InstanceInx, uint32_t ShapeInx);
    void UpdateMaterial(uint32_t InstanceInx, uint32_t MaterialInx);
    void UpdateTLAS(uint32_t InstanceInx);
    void AddInstance(uint32_t InstanceInx);
    void RemoveInstance(uint32_t InstanceInx);
    void AddShape(uint32_t ShapeInx);

    int SelectedInstance = -1;
    ~sceneBVH();
    void Destroy();
    scene* Scene;
};

std::shared_ptr<sceneBVH> CreateBVH(scene* Scene);

}
