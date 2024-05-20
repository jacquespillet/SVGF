#pragma once
#include <glad/gl.h>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <vector>

namespace gpupt
{

struct scene;

struct vertex
{
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 UV;
    uint32_t PrimitiveIndex;
};

class vertexBuffer
{
public:
    vertexBuffer(scene *Scene);
    ~vertexBuffer();

    void Draw(uint32_t ShapeIndex);

    std::vector<uint32_t> Offsets;

    GLuint VAO;
    GLuint VBO;
    GLuint EBO;

    uint32_t Count=0;
};

}