#include "Scene.h"
#include <glm/ext.hpp>

namespace gpupt
{


glm::mat4 instance::GetModelMatrix() const
{
    // Create transformation matrices for translation, rotation, and scale
    glm::mat4 TranslationMatrix = glm::translate(glm::mat4(1.0f), this->Position);
    glm::mat4 RotationMatrix = glm::mat4_cast(glm::quat(glm::radians(this->Rotation)));
    glm::mat4 ScaleMatrix = glm::scale(glm::mat4(1.0f), this->Scale);

    // Combine the matrices to get the final model matrix (order matters)
    glm::mat4 ModelMatrix = TranslationMatrix * RotationMatrix * ScaleMatrix;

    return ModelMatrix;    
}


void CalculateTangents(shape &Shape)
{
    std::vector<glm::vec4> tan1(Shape.Positions.size(), glm::vec4(0));
    std::vector<glm::vec4> tan2(Shape.Positions.size(), glm::vec4(0));
    if (Shape.Tangents.size() != Shape.Positions.size()) Shape.Tangents.resize(Shape.Positions.size());
    if(Shape.TexCoords.size() != Shape.Positions.size()) return;

    for(uint64_t i=0; i<Shape.Triangles.size(); i++) {
        glm::vec3 v1 = Shape.Positions[Shape.Triangles[i].x];
        glm::vec3 v2 = Shape.Positions[Shape.Triangles[i].y];
        glm::vec3 v3 = Shape.Positions[Shape.Triangles[i].z];

        glm::vec2 w1 = Shape.TexCoords[Shape.Triangles[i].x];
        glm::vec2 w2 = Shape.TexCoords[Shape.Triangles[i].y];
        glm::vec2 w3 = Shape.TexCoords[Shape.Triangles[i].z];

        float x1 = v2.x - v1.x;
        float x2 = v3.x - v1.x;
        float y1 = v2.y - v1.y;
        float y2 = v3.y - v1.y;
        float z1 = v2.z - v1.z;
        float z2 = v3.z - v1.z;

        float s1 = w2.x - w1.x;
        float s2 = w3.x - w1.x;
        float t1 = w2.y - w1.y;
        float t2 = w3.y - w1.y;

        float r = 1.0F / (s1 * t2 - s2 * t1);
        glm::vec4 sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r, 0);
        glm::vec4 tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r, 0);

        tan1[Shape.Triangles[i].x] += sdir;
        tan1[Shape.Triangles[i].y] += sdir;
        tan1[Shape.Triangles[i].z] += sdir;
        
        tan2[Shape.Triangles[i].x] += tdir;
        tan2[Shape.Triangles[i].y] += tdir;
        tan2[Shape.Triangles[i].z] += tdir;

    }

    for(uint64_t i=0; i<Shape.Positions.size(); i++) { 
        glm::vec3 n = Shape.Normals[i];
        glm::vec3 t = glm::vec3(tan1[i]);

        Shape.Tangents[i] = glm::vec4(glm::normalize((t - n * glm::dot(n, t))), 1);
        
        Shape.Tangents[i].w = (glm::dot(glm::cross(n, t), glm::vec3(tan2[i])) < 0.0F) ? -1.0F : 1.0F;
    }
}


}