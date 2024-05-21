#version 460 core

layout(location = 0) in vec3 VertexPosition;    // Vertex position attribute
layout(location = 1) in vec3 VertexNormal;    // Vertex position attribute
layout(location = 2) in vec2 VertexUV;    // Vertex position attribute
layout(location = 3) in uint PrimitiveIndex;    // Vertex position attribute

uniform mat4 MVP;    // Model-view-projection matrix
uniform mat4 PrevMVP;    // Model-view-projection matrix
uniform mat4 NormalMatrix;
uniform mat4 ModelMatrix;

out vec3 FragPosition;     // Fragment position in world space
out vec3 FragNormal;      // Normal in world space
out vec2 FragUV;
out vec3 BarycentricCoord; 
flat out vec2 MotionVector;
flat out uint FragPrimitiveIndex;

void main()
{
    vec4 CurrentScreenPos = MVP * vec4(VertexPosition, 1.0);
    gl_Position = CurrentScreenPos;  // Transform vertex position with MVP matrix
    FragPosition = vec3(ModelMatrix * vec4(VertexPosition, 1.0));  // Transform vertex position to world space
    FragNormal = vec3(NormalMatrix * vec4(VertexNormal, 0.0));
    FragUV = VertexUV;
    FragPrimitiveIndex = PrimitiveIndex;

    vec4 PrevScreenPos = PrevMVP * vec4(VertexPosition, 1.0);
    PrevScreenPos /= PrevScreenPos.w;
    
    CurrentScreenPos /= CurrentScreenPos.w;
    MotionVector = vec2(CurrentScreenPos) - vec2(PrevScreenPos);
    

    if (gl_VertexID % 3 == 1)
        BarycentricCoord = vec3(1.0, 0.0, 0.0);  // Vertex A
    else if (gl_VertexID % 3 == 2)
        BarycentricCoord = vec3(0.0, 1.0, 0.0);  // Vertex B
    else
        BarycentricCoord = vec3(0.0, 0.0, 1.0);  // Vertex C    
}
