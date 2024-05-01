#include "Tracing.h"
#include "Scene.h"
#include "ImageLoader.h"
#include "Buffer.h"
#include <iostream>

#define PI_F 3.141592653589

namespace gpupt
{

inline float TriangleArea(glm::vec3 P0, glm::vec3 P1, glm::vec3 P2)
{
    // We take the 2 vectors going from A to C, and from A to B.
    // Calculate the cross product N of these 2 vectors
    // The magnitude of this vector corresponds to the area of the parallelogram formed by AB and AC.
    // So the area of the triangle is half of the area of this parallelogram.
    return glm::length(glm::cross(P1 - P0, P2 - P0)) / 2;
}

float MaxElem(const glm::vec4 &A)
{
    return std::max(A.x, std::max(A.y, A.z));
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
void lights::RemoveInstance(scene *Scene, int InstanceInx)
{
    bool ShouldRebuild = false;
    for(int i=0; i<Lights.size(); i++)
    {
        if(Lights[i].Instance == InstanceInx)
        {
            Build(Scene);
            ShouldRebuild=true;
        }
        else if(Lights[i].Instance > InstanceInx)
        {
            Lights[i].Instance--;
            ShouldRebuild=true;
        }
    }
    if(ShouldRebuild)
    {
        RecreateBuffers();
    }
}

light &lights::AddLight()
{
    Lights.emplace_back();
    return Lights.back();
}

void lights::RecreateBuffers()
{
    if(this->Lights.size()>0)
    {
        this->LightsBuffer = std::make_shared<buffer>(sizeof(light) * Lights.size(), Lights.data());
        this->LightsCDFBuffer = std::make_shared<buffer>(sizeof(float) * LightsCDF.size(), LightsCDF.data());
    }
}

int UpperBound(int CDFStart, int CDFCount, float X, std::vector<float> &LightsCDF)
{
    int Mid;
    int Low = CDFStart;
    int High = CDFStart + CDFCount;
 
    while (Low < High) {
        Mid = Low + (High - Low) / 2;
        if (X >= LightsCDF[Mid]) {
            Low = Mid + 1;
        }
        else {
            High = Mid;
        }
    }
   
    // if X is greater than arr[n-1]
    if(Low < CDFStart + CDFCount && LightsCDF[Low] <= X) {
       Low++;
    }
 
    // Return the upper_bound index
    return Low;
}
 

void lights::Build(scene *Scene)
{
    LightsCDF.clear();
    Lights.clear();
    for (size_t i = 0; i < Scene->Instances.size(); i++)
    {

        //Check if the object is emitting
        const instance &Instance = Scene->Instances[i];
        const material &Material = Scene->Materials[Instance.Material];
        if(Material.Emission == glm::vec3{0,0,0}) continue;

        //Check if the object contains geometry
        const shape &Shape = Scene->Shapes[Instance.Shape];
        if(Shape.Triangles.empty()) continue;

        //Initialize the light
        light &Light = AddLight();
        Light.Instance = i;
        Light.Environment = InvalidID;

        //Calculate the cumulative distribution function for the primitive,
        //Which is essentially the cumulated area of the shape.
        if(!Shape.Triangles.empty())
        {
            Light.CDFCount = Shape.Triangles.size();
            Light.CDFStart = LightsCDF.size();
            LightsCDF.resize(LightsCDF.size() + Light.CDFCount);
            for(size_t j=0; j<Light.CDFCount; j++)
            {
                glm::mat4 InstanceTransform = Instance.Transform;
                const glm::vec3 &Pos0 = glm::vec3(InstanceTransform * glm::vec4(glm::vec3(Shape.Triangles[j].PositionUvX0), 1));
                const glm::vec3 &Pos1 = glm::vec3(InstanceTransform * glm::vec4(glm::vec3(Shape.Triangles[j].PositionUvX1), 1));
                const glm::vec3 &Pos2 = glm::vec3(InstanceTransform * glm::vec4(glm::vec3(Shape.Triangles[j].PositionUvX2), 1));

                LightsCDF[Light.CDFStart + j] = TriangleArea(Pos0, Pos1, Pos2);
                if(j != 0) LightsCDF[Light.CDFStart + j] += LightsCDF[Light.CDFStart + j-1]; 
            }
        }
    }

    for(size_t i=0; i<Scene->Environments.size(); i++)
    {
        const environment &Environment = Scene->Environments[i];
        if(Environment.Emission == glm::vec3{0,0,0}) continue;

        light &Light = AddLight();
        Light.Instance = InvalidID;
        Light.Environment = (int)i;
        if(Environment.EmissionTexture != InvalidID)
        {
            texture& Texture = Scene->EnvTextures[Environment.EmissionTexture];
            Light.CDFCount = Texture.Width * Texture.Height;
            Light.CDFStart = LightsCDF.size();
            LightsCDF.resize(LightsCDF.size() + Light.CDFCount);
            
            for (size_t i=0; i<Light.CDFCount; i++) {
                glm::ivec2 IJ((int)i % Texture.Width, (int)i / Texture.Width);
                float Theta    = (IJ.y + 0.5f) * PI_F / Texture.Height;
                glm::vec4 Value = Texture.SampleF(IJ);
                LightsCDF[Light.CDFStart + i] = MaxElem(Value) * sin(Theta);
                if (i != 0) LightsCDF[Light.CDFStart + i] += LightsCDF[Light.CDFStart + i - 1];
            }
        }

    }    

    RecreateBuffers();
}

}