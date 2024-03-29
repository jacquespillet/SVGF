#include "BVH.h"
#include <glm/ext.hpp>


#include "BufferGL.h"
#include "BufferCu.cuh"

#define BINS 8

namespace gpupt
{

float aabb::Area()
{
    glm::vec3 e = Max - Min; // box extent
    return e.x * e.y + e.y * e.z + e.z * e.x;
}

void aabb::Grow(glm::vec3 Position)
{
    Min = glm::min(Min, Position);
    Max = glm::max(Max, Position);
}

void aabb::Grow(aabb &AABB)
{
    if(AABB.Min.x != 1e30f)
    {
        Grow(AABB.Min);
        Grow(AABB.Max);
    }
}
////////////////////////////////////////////////////////////////////////////////////////

bool bvhNode::IsLeaf()
{
    return TriangleCount > 0;
}

////////////////////////////////////////////////////////////////////////////////////////

mesh::mesh(const shape &Shape)
{
    uint32_t AddedTriangles=0;
    Triangles.resize(Shape.Triangles.size());
    TrianglesExtraData.resize(Shape.Triangles.size());
    for(size_t j=0; j<Shape.Triangles.size(); j++)
    {
        uint32_t i0 = Shape.Triangles[j].x;
        uint32_t i1 = Shape.Triangles[j].y;
        uint32_t i2 = Shape.Triangles[j].z;
        glm::vec4 v0 = glm::vec4(Shape.Positions[i0], 0);
        glm::vec4 v1 = glm::vec4(Shape.Positions[i1], 0);
        glm::vec4 v2 = glm::vec4(Shape.Positions[i2], 0);

        glm::vec4 n0 = glm::vec4(Shape.Normals[i0], 0);
        glm::vec4 n1 = glm::vec4(Shape.Normals[i1], 0);
        glm::vec4 n2 = glm::vec4(Shape.Normals[i2], 0);
        
        Triangles[AddedTriangles].v0=v0;
        Triangles[AddedTriangles].v1=v1;
        Triangles[AddedTriangles].v2=v2;
        
        TrianglesExtraData[AddedTriangles].Normal0=n0;
        TrianglesExtraData[AddedTriangles].Normal1=n1;
        TrianglesExtraData[AddedTriangles].Normal2=n2;

        TrianglesExtraData[AddedTriangles].UV0 = Shape.TexCoords[i0];
        TrianglesExtraData[AddedTriangles].UV1 = Shape.TexCoords[i1];
        TrianglesExtraData[AddedTriangles].UV2 = Shape.TexCoords[i2];
        
        TrianglesExtraData[AddedTriangles].Colour0 = Shape.Colours[i0];
        TrianglesExtraData[AddedTriangles].Colour1 = Shape.Colours[i1];
        TrianglesExtraData[AddedTriangles].Colour2 = Shape.Colours[i2];
        
        TrianglesExtraData[AddedTriangles].Tangent0 = Shape.Tangents[i0];
        TrianglesExtraData[AddedTriangles].Tangent1 = Shape.Tangents[i1];
        TrianglesExtraData[AddedTriangles].Tangent2 = Shape.Tangents[i2];
        

        AddedTriangles++;
    }

    BVH = new bvh(this);
}

////////////////////////////////////////////////////////////////////////////////////////

bvh::bvh(mesh *_Mesh)
{
    this->Mesh = _Mesh;
    Build();
}

void bvh::Build()
{
    BVHNodes.resize(Mesh->Triangles.size() * 2 - 1);
    TriangleIndices.resize(Mesh->Triangles.size());

    // Calculate the centroid of each triangle
    for(size_t i=0; i<Mesh->Triangles.size(); i++)
    {
        Mesh->Triangles[i].Centroid = (Mesh->Triangles[i].v0 + Mesh->Triangles[i].v1 + Mesh->Triangles[i].v2) * 0.33333f;
        TriangleIndices[i] = (uint32_t)i;
    }

    // Create root node that encompasses the whole object
    bvhNode &Root = BVHNodes[RootNodeIndex];
    Root.LeftChildOrFirst = 0;
    Root.TriangleCount = (uint32_t)Mesh->Triangles.size();
    UpdateNodeBounds(RootNodeIndex);
    
    // Subdivide the node recursively
    Subdivide(RootNodeIndex);
}


float bvh::EvaluateSAH(bvhNode &Node, int Axis, float Position)
{
	aabb leftBox, rightBox;
	int leftCount = 0, rightCount = 0;
	for (uint32_t i = 0; i < Node.TriangleCount; i++)
	{
		triangle& Triangle = Mesh->Triangles[TriangleIndices[Node.LeftChildOrFirst + i]];
		if (Triangle.Centroid[Axis] < Position)
		{
			leftCount++;
			leftBox.Grow( Triangle.v0 );
			leftBox.Grow( Triangle.v1 );
			leftBox.Grow( Triangle.v2 );
		}
		else
		{
			rightCount++;
			rightBox.Grow( Triangle.v0 );
			rightBox.Grow( Triangle.v1 );
			rightBox.Grow( Triangle.v2 );
		}
	}
	float cost = leftCount * leftBox.Area() + rightCount * rightBox.Area();
	return cost > 0 ? cost : 1e30f;    
}


float bvh::FindBestSplitPlane(bvhNode &Node, int &Axis, float &SplitPosition)
{
    float BestCost = 1e30f;
    for(int CurrentAxis=0; CurrentAxis<3; CurrentAxis++)
    {
        float BoundsMin = 1e30f;
        float BoundsMax = -1e30f;
        for(uint32_t i=0; i<Node.TriangleCount; i++)
        {
            triangle &Triangle = Mesh->Triangles[TriangleIndices[Node.LeftChildOrFirst + i]];
            BoundsMin = std::min(BoundsMin, Triangle.Centroid[CurrentAxis]);
            BoundsMax = std::max(BoundsMax, Triangle.Centroid[CurrentAxis]);
        }
        if(BoundsMin == BoundsMax) continue;
        
        
        bin Bins[BINS];
        float Scale = BINS / (BoundsMax - BoundsMin);
        for(uint32_t i=0; i<Node.TriangleCount; i++)
        {
            triangle &Triangle = Mesh->Triangles[TriangleIndices[Node.LeftChildOrFirst + i]];
            int BinIndex = std::min(BINS - 1, (int)((Triangle.Centroid[CurrentAxis] - BoundsMin) * Scale));
            Bins[BinIndex].TrianglesCount++;
            Bins[BinIndex].Bounds.Grow(Triangle.v0);
            Bins[BinIndex].Bounds.Grow(Triangle.v1);
            Bins[BinIndex].Bounds.Grow(Triangle.v2);
        }

        float LeftArea[BINS-1], RightArea[BINS-1];
        int LeftCount[BINS-1], RightCount[BINS-1];
        
        aabb LeftBox, RightBox;
        int LeftSum=0, RightSum=0;

        for(int i=0; i<BINS-1; i++)
        {
            //Info from the left to the right
            LeftSum += Bins[i].TrianglesCount;
            LeftCount[i] = LeftSum; //Number of primitives to the right of this plane
            LeftBox.Grow(Bins[i].Bounds);
            LeftArea[i] = LeftBox.Area(); //Area to the right of this plane
            
            //Info from the right to the left
            RightSum += Bins[BINS-1-i].TrianglesCount;
            RightCount[BINS-2-i] = RightSum; //Number of primitives to the left of this plane
            RightBox.Grow(Bins[BINS-1-i].Bounds);
            RightArea[BINS-2-i] = RightBox.Area(); //Area to the left of this plane
        }

        Scale = (BoundsMax - BoundsMin) / BINS;
        for(int i=0; i<BINS-1; i++)
        {
            float PlaneCost = LeftCount[i] * LeftArea[i] + RightCount[i] * RightArea[i];
            if(PlaneCost < BestCost)
            {
                Axis = CurrentAxis;
                SplitPosition = BoundsMin + Scale * (i+1);
                BestCost = PlaneCost;
            }
        }
    }
    return BestCost;

}

void bvh::Subdivide(uint32_t NodeIndex)
{
    bvhNode &Node = BVHNodes[NodeIndex];

    // Get the split details
    int Axis=-1;
    float SplitPosition = 0;
    float SplitCost = FindBestSplitPlane(Node, Axis, SplitPosition);
    float NoSplitCost = CalculateNodeCost(Node);
    if(SplitCost >= NoSplitCost) return;

    // Do split the triangles
    int i=Node.LeftChildOrFirst;
    int j = i + Node.TriangleCount -1;
    while(i <= j)
    {
        if(Mesh->Triangles[TriangleIndices[i]].Centroid[Axis] < SplitPosition)
        {
            i++;
        }
        else
        {
            std::swap(TriangleIndices[i], TriangleIndices[j--]);
        }
    }

    uint32_t LeftCount = i - Node.LeftChildOrFirst;
    if(LeftCount==0 || LeftCount == Node.TriangleCount) return;

    // Build the 2 nodes
    int LeftChildIndex = NodesUsed++;
    int RightChildIndex = NodesUsed++;
    
    BVHNodes[LeftChildIndex].LeftChildOrFirst = Node.LeftChildOrFirst;
    BVHNodes[LeftChildIndex].TriangleCount = LeftCount;
    BVHNodes[RightChildIndex].LeftChildOrFirst = i;
    BVHNodes[RightChildIndex].TriangleCount = Node.TriangleCount - LeftCount;
    Node.LeftChildOrFirst = LeftChildIndex;
    Node.TriangleCount=0;

    // Set the 2 node bounds
    UpdateNodeBounds(LeftChildIndex);
    UpdateNodeBounds(RightChildIndex);

    // subdivide each node
    Subdivide(LeftChildIndex);
    Subdivide(RightChildIndex);
}


float bvh::CalculateNodeCost(bvhNode &Node)
{
    glm::vec3 e = Node.AABBMax - Node.AABBMin;
    float ParentArea = e.x * e.y + e.x * e.z + e.y * e.z;
    float NodeCost = Node.TriangleCount * ParentArea;
    return NodeCost;
}


void bvh::UpdateNodeBounds(uint32_t NodeIndex)
{
    // Calculate the bounds of the given node
    bvhNode &Node = BVHNodes[NodeIndex];
    Node.AABBMin = glm::vec3(1e30f);
    Node.AABBMax = glm::vec3(-1e30f);
    for(uint32_t First=Node.LeftChildOrFirst, i=0; i<Node.TriangleCount; i++)
    {
        uint32_t TriangleIndex = TriangleIndices[First + i];
        triangle &Triangle = Mesh->Triangles[TriangleIndex];
        Node.AABBMin = glm::min(Node.AABBMin, Triangle.v0);
        Node.AABBMin = glm::min(Node.AABBMin, Triangle.v1);
        Node.AABBMin = glm::min(Node.AABBMin, Triangle.v2);
        Node.AABBMax = glm::max(Node.AABBMax, Triangle.v0);
        Node.AABBMax = glm::max(Node.AABBMax, Triangle.v1);
        Node.AABBMax = glm::max(Node.AABBMax, Triangle.v2);
    }
}

////////////////////////////////////////////////////////////////////////////////////////

void bvhInstance::SetTransform(glm::mat4 &Transform, std::vector<mesh*> *Meshes)
{
    bvh *BVH = Meshes->at(MeshIndex)->BVH;
    this->InverseTransform = glm::inverse(Transform);
    this->Transform = Transform;
    this->NormalTransform = glm::inverseTranspose(Transform);
    
    glm::vec3 Min = BVH->BVHNodes[0].AABBMin;
    glm::vec3 Max = BVH->BVHNodes[0].AABBMax;
    Bounds = {};
    for (int i = 0; i < 8; i++)
    {
		Bounds.Grow( Transform *  glm::vec4( 
                                    i & 1 ? Max.x : Min.x,
                                    i & 2 ? Max.y : Min.y, 
                                    i & 4 ? Max.z : Min.z,
                                    1.0f ));
    }    
}

////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////

tlas::tlas()
{}

tlas::tlas(std::vector<bvhInstance>* BVHList)
{
    BLAS = BVHList;
    NodesUsed=2;
}

int tlas::FindBestMatch(std::vector<int>& List, int N, int A)
{
    float Smallest = 1e30f;
    int BestB = -1;
    for(int B=0; B< N; B++)
    {
        if(B != A)
        {
            glm::vec3 BMax = glm::max(Nodes[List[A]].AABBMax, Nodes[List[B]].AABBMax);
            glm::vec3 BMin = glm::min(Nodes[List[A]].AABBMin, Nodes[List[B]].AABBMin);
            glm::vec3 Diff = BMax - BMin;
            float Area = Diff.x * Diff.y + Diff.y * Diff.z + Diff.x * Diff.z;
            if(Area < Smallest) 
            {
                Smallest = Area;
                BestB = B;
            }
        }
    }

    return BestB;
}

void tlas::Build()
{
    Nodes.resize(BLAS->size() * 2);
    
    std::vector<int> NodeIndex(BLAS->size());
    int NodeIndices = (int)BLAS->size();
    NodesUsed=1;
    for(uint32_t i=0; i<BLAS->size(); i++)
    {
        NodeIndex[i] = NodesUsed;
        Nodes[NodesUsed].AABBMin = (*BLAS)[i].Bounds.Min;
        Nodes[NodesUsed].AABBMax = (*BLAS)[i].Bounds.Max;
        Nodes[NodesUsed].BLAS = i;
        Nodes[NodesUsed].LeftRight = 0; //Makes it a leaf.
        NodesUsed++;
    }


    int A = 0;
    int B= FindBestMatch(NodeIndex, NodeIndices, A); //Best match for A
    while(NodeIndices >1)
    {
        int C = FindBestMatch(NodeIndex, NodeIndices, B); //Best match for B
        if(A == C) //There is no better match --> Create a parent for them
        {
            int NodeIndexA = NodeIndex[A];
            int NodeIndexB = NodeIndex[B];
            tlasNode &NodeA = Nodes[NodeIndexA];
            tlasNode &NodeB = Nodes[NodeIndexB];
            
            tlasNode &NewNode = Nodes[NodesUsed];
            NewNode.LeftRight = NodeIndexA + (NodeIndexB << 16);
            NewNode.AABBMin = glm::min(NodeA.AABBMin, NodeB.AABBMin);
            NewNode.AABBMax = glm::max(NodeA.AABBMax, NodeB.AABBMax);
            
            NodeIndex[A] = NodesUsed++;
            NodeIndex[B] = NodeIndex[NodeIndices-1];
            B = FindBestMatch(NodeIndex, --NodeIndices, A);
        }
        else
        {
            A = B;
            B = C;
        }
    }

    Nodes[0] = Nodes[NodeIndex[A]];




}


std::shared_ptr<sceneBVH> CreateBVH(std::shared_ptr<scene> Scene)
{
    // Build the low level bvh of each mesh
    std::shared_ptr<sceneBVH> Result = std::make_shared<sceneBVH>();
    for(size_t i=0; i<Scene->Shapes.size(); i++)
    {
        Result->Meshes.push_back(new mesh(Scene->Shapes[i]));
    }

    //Build the array of instances
    for(size_t i=0; i<Scene->Instances.size(); i++)
    {
        Result->Instances.push_back(
            bvhInstance(&Result->Meshes, Scene->Instances[i].Shape,
                        Scene->Instances[i].GetModelMatrix(), (uint32_t)i)
        );
    }
    
    // Build the top level data structure
    Result->TLAS = tlas(&Result->Instances);
    Result->TLAS.Build();

    //Build big buffers with all the shape datas inside
    uint64_t TotalTriangleCount=0;
    uint64_t TotalIndicesCount=0;
    uint64_t TotalBVHNodes=0;
    for(int i=0; i<Result->Meshes.size(); i++)
    {
        TotalTriangleCount += Result->Meshes[i]->Triangles.size();
        TotalIndicesCount += Result->Meshes[i]->BVH->TriangleIndices.size();
        TotalBVHNodes += Result->Meshes[i]->BVH->NodesUsed;
    }
    Result->AllTriangles = std::vector<triangle> (TotalTriangleCount);
    Result->AllTrianglesEx = std::vector<triangleExtraData> (TotalTriangleCount);
    Result->AllTriangleIndices = std::vector<uint32_t> (TotalIndicesCount);
    Result->AllBVHNodes = std::vector<bvhNode> (TotalBVHNodes);
    Result->IndexData.resize(Result->Meshes.size());


    // Fill the buffers
    uint32_t RunningTriangleCount=0;
    uint32_t RunningIndicesCount=0;
    uint32_t RunningBVHNodeCount=0;
    for(int i=0; i<Result->Meshes.size(); i++)
    {
        memcpy((void*)(Result->AllTriangles.data() + RunningTriangleCount), Result->Meshes[i]->Triangles.data(), Result->Meshes[i]->Triangles.size() * sizeof(triangle));
        memcpy((void*)(Result->AllTrianglesEx.data() + RunningTriangleCount), Result->Meshes[i]->TrianglesExtraData.data(), Result->Meshes[i]->TrianglesExtraData.size() * sizeof(triangleExtraData));
        memcpy((void*)(Result->AllTriangleIndices.data() + RunningIndicesCount), Result->Meshes[i]->BVH->TriangleIndices.data(), Result->Meshes[i]->BVH->TriangleIndices.size() * sizeof(uint32_t));
        memcpy((void*)(Result->AllBVHNodes.data() + RunningBVHNodeCount), Result->Meshes[i]->BVH->BVHNodes.data(), Result->Meshes[i]->BVH->NodesUsed * sizeof(bvhNode));

        Result->IndexData[i] = 
        {
            RunningTriangleCount,
            RunningIndicesCount,
            RunningBVHNodeCount,
            (uint32_t)Result->Meshes[i]->Triangles.size()
        };

        RunningTriangleCount += (uint32_t)Result->Meshes[i]->Triangles.size();
        RunningIndicesCount += (uint32_t)Result->Meshes[i]->BVH->TriangleIndices.size();
        RunningBVHNodeCount += (uint32_t)Result->Meshes[i]->BVH->NodesUsed;
    }

    // Upload to the gpu
#if API==API_GL
    // BLAS
    Result->TrianglesBuffer =std::make_shared<bufferGL>(Result->AllTriangles.size() * sizeof(triangle), Result->AllTriangles.data());
    Result->TrianglesExBuffer =std::make_shared<bufferGL>(Result->AllTrianglesEx.size() * sizeof(triangleExtraData), Result->AllTrianglesEx.data());
    Result->BVHBuffer =std::make_shared<bufferGL>(Result->AllBVHNodes.size() * sizeof(bvhNode), Result->AllBVHNodes.data());
    Result->IndicesBuffer =std::make_shared<bufferGL>(Result->AllTriangleIndices.size() * sizeof(uint32_t), Result->AllTriangleIndices.data());
    Result->IndexDataBuffer =std::make_shared<bufferGL>(Result->IndexData.size() * sizeof(indexData), Result->IndexData.data());

    // TLAS
    Result->TLASInstancesBuffer =std::make_shared<bufferGL>(Result->TLAS.BLAS->size() * sizeof(bvhInstance), Result->TLAS.BLAS->data());
    Result->TLASNodeBuffer =std::make_shared<bufferGL>(Result->TLAS.Nodes.size() * sizeof(tlasNode), Result->TLAS.Nodes.data());
#elif API==API_CU
    // BLAS
    Result->TrianglesBuffer =std::make_shared<bufferCu>(Result->AllTriangles.size() * sizeof(triangle), Result->AllTriangles.data());
    Result->TrianglesExBuffer =std::make_shared<bufferCu>(Result->AllTrianglesEx.size() * sizeof(triangleExtraData), Result->AllTrianglesEx.data());
    Result->BVHBuffer =std::make_shared<bufferCu>(Result->AllBVHNodes.size() * sizeof(bvhNode), Result->AllBVHNodes.data());
    Result->IndicesBuffer =std::make_shared<bufferCu>(Result->AllTriangleIndices.size() * sizeof(uint32_t), Result->AllTriangleIndices.data());
    Result->IndexDataBuffer =std::make_shared<bufferCu>(Result->IndexData.size() * sizeof(indexData), Result->IndexData.data());

    // TLAS
    Result->TLASInstancesBuffer =std::make_shared<bufferCu>(Result->TLAS.BLAS->size() * sizeof(bvhInstance), Result->TLAS.BLAS->data());
    Result->TLASNodeBuffer =std::make_shared<bufferCu>(Result->TLAS.Nodes.size() * sizeof(tlasNode), Result->TLAS.Nodes.data());
#endif
    Result->Scene = Scene;
    return Result;
}

void sceneBVH::Destroy()
{
    for (size_t i = 0; i < this->Meshes.size(); i++)
    {
        delete this->Meshes[i]->BVH;
        delete this->Meshes[i];
    }
}

}