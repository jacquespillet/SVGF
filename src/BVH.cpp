#include "BVH.h"
#include <glm/ext.hpp>


#include "Buffer.h"
#include <cuda_runtime_api.h>

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


////////////////////////////////////////////////////////////////////////////////////////

blas::blas(shape *_Shape)
{
    this->Shape = _Shape;
    Build();
}

void blas::Build()
{
    BVHNodes.resize(Shape->Triangles.size() * 2 - 1);
    TriangleIndices.resize(Shape->Triangles.size());

    // Calculate the centroid of each triangle
    for(size_t i=0; i<Shape->Triangles.size(); i++)
    {
        Shape->Triangles[i].Centroid = (glm::vec3(Shape->Triangles[i].PositionUvX0) + glm::vec3(Shape->Triangles[i].PositionUvX1) + glm::vec3(Shape->Triangles[i].PositionUvX2)) * 0.33333f;
        TriangleIndices[i] = (uint32_t)i;
    }

    // Create root node that encompasses the whole object
    bvhNode &Root = BVHNodes[RootNodeIndex];
    Root.LeftChildOrFirst = 0;
    Root.TriangleCount = (uint32_t)Shape->Triangles.size();
    UpdateNodeBounds(RootNodeIndex);
    
    // Subdivide the node recursively
    Subdivide(RootNodeIndex);
}


float blas::EvaluateSAH(bvhNode &Node, int Axis, float Position)
{
	aabb leftBox, rightBox;
	int leftCount = 0, rightCount = 0;
	for (uint32_t i = 0; i < Node.TriangleCount; i++)
	{
		triangle& Triangle = Shape->Triangles[TriangleIndices[Node.LeftChildOrFirst + i]];
		if (Triangle.Centroid[Axis] < Position)
		{
			leftCount++;
			leftBox.Grow( glm::vec3(Triangle.PositionUvX0) );
			leftBox.Grow( glm::vec3(Triangle.PositionUvX1) );
			leftBox.Grow( glm::vec3(Triangle.PositionUvX2) );
		}
		else
		{
			rightCount++;
			rightBox.Grow( glm::vec3(Triangle.PositionUvX0) );
			rightBox.Grow( glm::vec3(Triangle.PositionUvX1) );
			rightBox.Grow( glm::vec3(Triangle.PositionUvX2) );
		}
	}
	float cost = leftCount * leftBox.Area() + rightCount * rightBox.Area();
	return cost > 0 ? cost : 1e30f;    
}


float blas::FindBestSplitPlane(bvhNode &Node, int &Axis, float &SplitPosition)
{
    float BestCost = 1e30f;
    for(int CurrentAxis=0; CurrentAxis<3; CurrentAxis++)
    {
        float BoundsMin = 1e30f;
        float BoundsMax = -1e30f;
        for(uint32_t i=0; i<Node.TriangleCount; i++)
        {
            triangle &Triangle = Shape->Triangles[TriangleIndices[Node.LeftChildOrFirst + i]];
            BoundsMin = std::min(BoundsMin, Triangle.Centroid[CurrentAxis]);
            BoundsMax = std::max(BoundsMax, Triangle.Centroid[CurrentAxis]);
        }
        if(BoundsMin == BoundsMax) continue;
        
        
        bin Bins[BINS];
        float Scale = BINS / (BoundsMax - BoundsMin);
        for(uint32_t i=0; i<Node.TriangleCount; i++)
        {
            triangle &Triangle = Shape->Triangles[TriangleIndices[Node.LeftChildOrFirst + i]];
            int BinIndex = std::min(BINS - 1, (int)((Triangle.Centroid[CurrentAxis] - BoundsMin) * Scale));
            Bins[BinIndex].TrianglesCount++;
            Bins[BinIndex].Bounds.Grow(glm::vec3(Triangle.PositionUvX0));
            Bins[BinIndex].Bounds.Grow(glm::vec3(Triangle.PositionUvX1));
            Bins[BinIndex].Bounds.Grow(glm::vec3(Triangle.PositionUvX2));
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

void blas::Subdivide(uint32_t NodeIndex)
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
        if(Shape->Triangles[TriangleIndices[i]].Centroid[Axis] < SplitPosition)
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


float blas::CalculateNodeCost(bvhNode &Node)
{
    glm::vec3 e = Node.AABBMax - Node.AABBMin;
    float ParentArea = e.x * e.y + e.x * e.z + e.y * e.z;
    float NodeCost = Node.TriangleCount * ParentArea;
    return NodeCost;
}


void blas::UpdateNodeBounds(uint32_t NodeIndex)
{
    // Calculate the bounds of the given node
    bvhNode &Node = BVHNodes[NodeIndex];
    Node.AABBMin = glm::vec3(1e30f);
    Node.AABBMax = glm::vec3(-1e30f);
    for(uint32_t First=Node.LeftChildOrFirst, i=0; i<Node.TriangleCount; i++)
    {
        uint32_t TriangleIndex = TriangleIndices[First + i];
        triangle &Triangle = Shape->Triangles[TriangleIndex];
        Node.AABBMin = glm::min(Node.AABBMin, glm::vec3(Triangle.PositionUvX0));
        Node.AABBMin = glm::min(Node.AABBMin, glm::vec3(Triangle.PositionUvX1));
        Node.AABBMin = glm::min(Node.AABBMin, glm::vec3(Triangle.PositionUvX2));
        Node.AABBMax = glm::max(Node.AABBMax, glm::vec3(Triangle.PositionUvX0));
        Node.AABBMax = glm::max(Node.AABBMax, glm::vec3(Triangle.PositionUvX1));
        Node.AABBMax = glm::max(Node.AABBMax, glm::vec3(Triangle.PositionUvX2));
    }
}

////////////////////////////////////////////////////////////////////////////////////////


tlas::tlas()
{}

tlas::tlas(std::vector<instance>* BVHList)
{
    BLAS = BVHList;
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
    if(BLAS->size()==0) return;

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


std::shared_ptr<sceneBVH> CreateBVH(scene* Scene)
{
    std::shared_ptr<sceneBVH> Result = std::make_shared<sceneBVH>();
    
    //Build big buffers with all the shape datas inside
    uint64_t TotalTriangleCount=0;
    uint64_t TotalIndicesCount=0;
    uint64_t TotalBVHNodes=0;
    for(int i=0; i<Scene->Shapes.size(); i++)
    {
        TotalTriangleCount += Scene->Shapes[i].Triangles.size();
        TotalIndicesCount += Scene->Shapes[i].BVH->TriangleIndices.size();
        TotalBVHNodes += Scene->Shapes[i].BVH->NodesUsed;
    }
    Result->AllTriangles = std::vector<triangle> (TotalTriangleCount);
    Result->AllTriangleIndices = std::vector<uint32_t> (TotalIndicesCount);
    Result->AllBVHNodes = std::vector<bvhNode> (TotalBVHNodes);
    Result->IndexData.resize(Scene->Shapes.size());


    // Fill the buffers
    uint32_t RunningTriangleCount=0;
    uint32_t RunningIndicesCount=0;
    uint32_t RunningBVHNodeCount=0;
    for(int i=0; i<Scene->Shapes.size(); i++)
    {
        memcpy((void*)(Result->AllTriangles.data() + RunningTriangleCount), Scene->Shapes[i].Triangles.data(), Scene->Shapes[i].Triangles.size() * sizeof(triangle));
        memcpy((void*)(Result->AllTriangleIndices.data() + RunningIndicesCount), Scene->Shapes[i].BVH->TriangleIndices.data(), Scene->Shapes[i].BVH->TriangleIndices.size() * sizeof(uint32_t));
        memcpy((void*)(Result->AllBVHNodes.data() + RunningBVHNodeCount), Scene->Shapes[i].BVH->BVHNodes.data(), Scene->Shapes[i].BVH->NodesUsed * sizeof(bvhNode));

        Result->IndexData[i] = 
        {
            RunningTriangleCount,
            RunningIndicesCount,
            RunningBVHNodeCount,
            (uint32_t)Scene->Shapes[i].Triangles.size()
        };

        RunningTriangleCount += (uint32_t)Scene->Shapes[i].Triangles.size();
        RunningIndicesCount += (uint32_t)Scene->Shapes[i].BVH->TriangleIndices.size();
        RunningBVHNodeCount += (uint32_t)Scene->Shapes[i].BVH->NodesUsed;
    }
    // BLAS
    Result->TrianglesBuffer =std::make_shared<buffer>(Result->AllTriangles.size() * sizeof(triangle), Result->AllTriangles.data());
    Result->BVHBuffer =std::make_shared<buffer>(Result->AllBVHNodes.size() * sizeof(bvhNode), Result->AllBVHNodes.data());
    Result->IndicesBuffer =std::make_shared<buffer>(Result->AllTriangleIndices.size() * sizeof(uint32_t), Result->AllTriangleIndices.data());
    Result->IndexDataBuffer =std::make_shared<buffer>(Result->IndexData.size() * sizeof(indexData), Result->IndexData.data());
    
    // Build the top level data structure
    Result->TLAS = tlas(&Scene->Instances);
    Result->TLAS.Build();


    // Upload to the gpu
    Result->TLASInstancesBuffer =std::make_shared<buffer>(Result->TLAS.BLAS->size() * sizeof(instance), Result->TLAS.BLAS->data());
    Result->TLASNodeBuffer =std::make_shared<buffer>(Result->TLAS.Nodes.size() * sizeof(tlasNode), Result->TLAS.Nodes.data());

    Result->Scene = Scene;
    return Result;
}


void sceneBVH::UpdateShape(uint32_t InstanceInx, uint32_t ShapeInx)
{
    Scene->CalculateInstanceTransform(InstanceInx);
    Scene->Instances[InstanceInx].Shape = ShapeInx;
    TLAS.Build();
    this->TLASInstancesBuffer->updateData(this->TLAS.BLAS->data(), this->TLAS.BLAS->size() * sizeof(instance));
    this->TLASNodeBuffer->updateData(this->TLAS.Nodes.data(), this->TLAS.Nodes.size() * sizeof(tlasNode));
}

void sceneBVH::UpdateMaterial(uint32_t InstanceInx, uint32_t MaterialInx)
{
    Scene->Instances[InstanceInx].Material = MaterialInx;
    this->TLASInstancesBuffer->updateData(this->TLAS.BLAS->data(), this->TLAS.BLAS->size() * sizeof(instance));
}

void sceneBVH::UpdateTLAS(uint32_t InstanceInx)
{
    Scene->CalculateInstanceTransform(InstanceInx);
    TLAS.Build();
    this->TLASInstancesBuffer->updateData(this->TLAS.BLAS->data(), this->TLAS.BLAS->size() * sizeof(instance));
    this->TLASNodeBuffer->updateData(this->TLAS.Nodes.data(), this->TLAS.Nodes.size() * sizeof(tlasNode));
}

void sceneBVH::AddInstance(uint32_t InstanceInx)
{
    Scene->CalculateInstanceTransform(InstanceInx);
    for(int i=0; i<Scene->Instances.size(); i++)
    {
        Scene->Instances[i].Index = i;   
    }
    TLAS.Build();
    this->TLASInstancesBuffer =std::make_shared<buffer>(this->TLAS.BLAS->size() * sizeof(instance), this->TLAS.BLAS->data());
    this->TLASNodeBuffer =std::make_shared<buffer>(this->TLAS.Nodes.size() * sizeof(tlasNode), this->TLAS.Nodes.data());
}

void sceneBVH::RemoveInstance(uint32_t InstanceInx)
{
    Scene->Instances.erase(Scene->Instances.begin() + InstanceInx);
    for(int i=0; i<Scene->Instances.size(); i++)
    {
        Scene->Instances[i].Index = i;
    }
    TLAS.Build();   
    this->TLASInstancesBuffer =std::make_shared<buffer>(this->TLAS.BLAS->size() * sizeof(instance), this->TLAS.BLAS->data());
    this->TLASNodeBuffer =std::make_shared<buffer>(this->TLAS.Nodes.size() * sizeof(tlasNode), this->TLAS.Nodes.data());
}



void sceneBVH::AddShape(uint32_t ShapeInx)
{
    shape &Shape = this->Scene->Shapes[ShapeInx];
    uint32_t RunningTriangleCount = AllTriangles.size();
    uint32_t RunningIndicesCount = AllTriangleIndices.size();
    uint32_t RunningBVHNodeCount = AllBVHNodes.size();
    
    AllTriangles.resize(AllTriangles.size() + Shape.Triangles.size());
    AllTriangleIndices.resize(AllTriangleIndices.size() + Shape.BVH->TriangleIndices.size());
    AllBVHNodes.resize(AllBVHNodes.size() + Shape.BVH->NodesUsed);
    IndexData.resize(Scene->Shapes.size());


    memcpy((void*)(AllTriangles.data() + RunningTriangleCount), Scene->Shapes[ShapeInx].Triangles.data(), Scene->Shapes[ShapeInx].Triangles.size() * sizeof(triangle));
    memcpy((void*)(AllTriangleIndices.data() + RunningIndicesCount), Scene->Shapes[ShapeInx].BVH->TriangleIndices.data(), Scene->Shapes[ShapeInx].BVH->TriangleIndices.size() * sizeof(uint32_t));
    memcpy((void*)(AllBVHNodes.data() + RunningBVHNodeCount), Scene->Shapes[ShapeInx].BVH->BVHNodes.data(), Scene->Shapes[ShapeInx].BVH->NodesUsed * sizeof(bvhNode));

    IndexData[ShapeInx] = 
    {
        RunningTriangleCount,
        RunningIndicesCount,
        RunningBVHNodeCount,
        (uint32_t)Scene->Shapes[ShapeInx].Triangles.size()
    };


    // BLAS
    TrianglesBuffer =std::make_shared<buffer>(AllTriangles.size() * sizeof(triangle), AllTriangles.data());
    BVHBuffer =std::make_shared<buffer>(AllBVHNodes.size() * sizeof(bvhNode), AllBVHNodes.data());
    IndicesBuffer =std::make_shared<buffer>(AllTriangleIndices.size() * sizeof(uint32_t), AllTriangleIndices.data());
    IndexDataBuffer =std::make_shared<buffer>(IndexData.size() * sizeof(indexData), IndexData.data());
}


sceneBVH::~sceneBVH()
{
    this->Destroy();
}

void sceneBVH::Destroy()
{
    for (size_t i = 0; i < this->Scene->Shapes.size(); i++)
    {
        delete this->Scene->Shapes[i].BVH;
    }
}

}