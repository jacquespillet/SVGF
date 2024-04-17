#include "Tracing.h"
#include "Scene.h"
#include "ImageLoader.h"

#define PI_F 3.141592653589

namespace gpupt
{

light &AddLight(std::shared_ptr<lights> Lights)
{
    Lights->Lights.emplace_back();
    return Lights->Lights.back();
}

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


int UpperBound(std::vector<float> &LightsCDF, int CDFStart, int CDFCount, int X)
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
 

int SampleDiscrete(lights &Lights, std::shared_ptr<scene> Scene, int LightInx, float R)
{
    //Remap R from 0 to the size of the distribution
    int CDFStart = Lights.Lights[LightInx].CDFStart;
    int CDFCount = Lights.Lights[LightInx].CDFCount;

    float LastValue = Lights.LightsCDF[CDFStart + CDFCount-1];

    R = glm::clamp(R * LastValue, 0.0f, LastValue - 0.00001f);
    // Returns the first element in the array that's greater than R.#
    int Inx= UpperBound(Lights.LightsCDF, CDFStart, CDFCount, R);
    return glm::clamp(Inx, 0, CDFCount-1);
}

float RandF()
{
    return float(rand()) / float(RAND_MAX);
}

void Test_SampleEnvMap(lights &Lights, std::shared_ptr<scene> Scene)
{
    for(int i=0; i<Scene->EnvTextures[0].PixelsF.size(); i+=4)
    {
        Scene->EnvTextures[0].PixelsF[i+0] *= glm::clamp(Scene->EnvTextures[0].PixelsF[i+0], 0.0f, 0.5f);
        Scene->EnvTextures[0].PixelsF[i+1] *= glm::clamp(Scene->EnvTextures[0].PixelsF[i+1], 0.0f, 0.5f);
        Scene->EnvTextures[0].PixelsF[i+2] *= glm::clamp(Scene->EnvTextures[0].PixelsF[i+2], 0.0f, 0.5f);
        
    }
    for(int i=0; i<10000000; i++)
    {
        int SampleInx = SampleDiscrete(Lights, Scene, 0, RandF());
        glm::vec2 UV = glm::vec2((SampleInx % Scene->EnvTextureWidth) ,
            (SampleInx / Scene->EnvTextureWidth));
        
        Scene->EnvTextures[0].PixelsF[(UV.y * Scene->EnvTextureWidth + UV.x) * 4+0] += 10.0f;
        Scene->EnvTextures[0].PixelsF[(UV.y * Scene->EnvTextureWidth + UV.x) * 4+4] = 1.0f;



        glm::vec3 dir = glm::normalize(glm::vec3(cos(UV.x * 2 * PI_F) * sin(UV.y * PI_F), 
                    cos(UV.y * PI_F),
                    sin(UV.x * 2 * PI_F) * sin(UV.y * PI_F)));            
    }    
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

std::shared_ptr<lights> GetLights(scene* Scene)
{
    std::shared_ptr<lights> Lights = std::make_shared<lights>();

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
        Light.Environment = InvalidID;

        //Calculate the cumulative distribution function for the primitive,
        //Which is essentially the cumulated area of the shape.
        if(!Shape.Triangles.empty())
        {
            Light.CDFCount = Shape.Triangles.size();
            Light.CDFStart = Lights->LightsCDF.size();
            Lights->LightsCDF.resize(Lights->LightsCDF.size() + Light.CDFCount);
            for(size_t j=0; j<Light.CDFCount; j++)
            {
                const glm::ivec3 &Tri = Shape.Triangles[j];
                Lights->LightsCDF[Light.CDFStart + j] = TriangleArea(Shape.Positions[Tri.x], Shape.Positions[Tri.y], Shape.Positions[Tri.z]);
                if(j != 0) Lights->LightsCDF[Light.CDFStart + j] += Lights->LightsCDF[Light.CDFStart + j-1]; 
            }
        }
    }

    for(size_t i=0; i<Scene->Environments.size(); i++)
    {
        const environment &Environment = Scene->Environments[i];
        if(Environment.Emission == glm::vec3{0,0,0}) continue;

        light &Light = AddLight(Lights);
        Light.Instance = InvalidID;
        Light.Environment = (int)i;
        if(Environment.EmissionTexture != InvalidID)
        {
            texture& Texture = Scene->EnvTextures[Environment.EmissionTexture];
            Light.CDFCount = Texture.Width * Texture.Height;
            Light.CDFStart = Lights->LightsCDF.size();
            Lights->LightsCDF.resize(Lights->LightsCDF.size() + Light.CDFCount);
            
            for (size_t i=0; i<Light.CDFCount; i++) {
                glm::ivec2 IJ((int)i % Texture.Width, (int)i / Texture.Width);
                float Theta    = (IJ.y + 0.5f) * PI_F / Texture.Height;
                glm::vec4 Value = Texture.SampleF(IJ);
                Lights->LightsCDF[Light.CDFStart + i] = MaxElem(Value) * sin(Theta);
                if (i != 0) Lights->LightsCDF[Light.CDFStart + i] += Lights->LightsCDF[Light.CDFStart + i - 1];
            }
        }

    }    
#if 0
    Test_SampleEnvMap(Lights, Scene);
    ImageToFile("Test.hdr", Scene->EnvTextures[0].PixelsF, Scene->EnvTextureWidth, Scene->EnvTextureHeight, 4);
#endif
    return Lights;
}     
}