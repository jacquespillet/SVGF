#pragma once
#include <string>
#include <memory>
#include <glad/gl.h>
#include <glm/mat4x4.hpp>

namespace gpupt
{

class bufferGL;
class uniformBufferGL;
class textureArrayGL;

class shaderGL {
public:
    shaderGL() = default;
    shaderGL(const char* computePath);
    shaderGL(const char* VertexPath, const char *FragmentPath);
    void Destroy();
    void Use();
    void SetInt(const std::string& name, int value);
    void SetVec3(const std::string &Name, glm::vec3 &Matrix);
    void SetMat4(const std::string &Name, glm::mat4 &Vector);
    void SetTexture(int ImageUnit, GLuint TextureID, GLenum Access);
    void SetTexture(int ImageUnit, GLuint TextureID) const;
    void SetSSBO(std::shared_ptr<bufferGL> Buffer, int BindingPoint);
    void SetUBO(std::shared_ptr<uniformBufferGL> Buffer, int BindingPoint);
    void SetTextureArray(std::shared_ptr<textureArrayGL> Texture, int Unit, std::string Name);
    void Dispatch(uint32_t X, uint32_t Y, uint32_t Z);
    void Barrier();
    ~shaderGL();
private:   
    GLuint ID;
    std::string ReadFile(const char* FilePath) const;
    GLuint CompileShader(GLenum Type, const char* SourceCode) const;
    GLuint LinkShader(GLuint ComputeShader) const;
};        
}