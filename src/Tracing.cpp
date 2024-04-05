#include "Tracing.h"
#include "Scene.h"

namespace gpupt
{

light &AddLight(lights &Lights)
{
    return Lights.Lights[Lights.LightsCount++];
}

inline float TriangleArea(glm::vec3 P0, glm::vec3 P1, glm::vec3 P2)
{
    // We take the 2 vectors going from A to C, and from A to B.
    // Calculate the cross product N of these 2 vectors
    // The magnitude of this vector corresponds to the area of the parallelogram formed by AB and AC.
    // So the area of the triangle is half of the area of this parallelogram.
    return glm::length(glm::cross(P1 - P0, P2 - P0)) / 2;
}

lights GetLights(std::shared_ptr<scene> Scene, tracingParameters &Parameters)
{
    lights Lights = {};

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
        light &Light = AddLight(Lights);
        Light.Instance = i;

        //Calculate the cumulative distribution function for the primitive,
        //Which is essentially the cumulated area of the shape.
        if(!Shape.Triangles.empty())
        {
            Light.CDFCount = Shape.Triangles.size();
            for(size_t j=0; j<Light.CDFCount; j++)
            {
                const glm::ivec3 &Tri = Shape.Triangles[j];
                Light.CDF[j] = TriangleArea(Shape.Positions[Tri.x], Shape.Positions[Tri.y], Shape.Positions[Tri.z]);
                if(j != 0) Light.CDF[j] += Light.CDF[j-1]; 
            }
        }
    }
    return Lights;
}     
}