#version 460 core

in vec3 FragPosition;     // Fragment position in world space
in vec3 FragNormal;      // Normal in world space
in vec3 BarycentricCoord;      // Normal in world space
flat in uint FragPrimitiveIndex;
in vec4 FragCurrentScreenPos;
in vec4 FragPrevScreenPos;

layout (location = 0) out vec4 OutPosition;
layout (location = 1) out vec4 OutNormal;
layout (location = 2) out vec4 OutUV;
layout (location = 3) out vec4 OutMotionVectors;

struct material
{
    vec3 Emission;
    float Roughness;
    
    vec3 Colour;
    float Metallic;
    
    float Padding;
    float Anisotropy;
    float MaterialType;
    float Opacity;

    vec3 ScatteringColour;
    float TransmissionDepth;

    int EmissionTexture;
    int ColourTexture;
    int RoughnessTexture;
    int NormalTexture;
};

layout(std430, binding = 0) buffer MaterialsBuffer {
  material Materials[];
};

uniform int InstanceIndex;
uniform int MaterialIndex;
uniform int Width;
uniform int Height;
uniform int Debug;
uniform vec3 CameraPosition;

void main()
{
    OutUV = vec4(BarycentricCoord, 1.0f); 
    OutNormal = vec4(normalize(FragNormal), 1.0f);
    OutPosition = vec4(FragPosition, 1.0f);
    vec4 PrevScreenPos = FragPrevScreenPos / FragPrevScreenPos.w;
    vec4 CurrentScreenPos = FragCurrentScreenPos / FragCurrentScreenPos.w;
    vec2 MotionVector = (vec2(PrevScreenPos) - vec2(CurrentScreenPos)) * (0.5 * vec2(float(Width), float(Height)));
    float Depth = distance(CameraPosition, OutPosition.xyz);
    
    OutMotionVectors = vec4(MotionVector, 0, 1);

    if(Debug==0)
    {
      OutUV.w = float(InstanceIndex);
      OutNormal.w = float(MaterialIndex);
      OutPosition.w = float(FragPrimitiveIndex);
      
      OutMotionVectors.z = Depth;
    }

}
