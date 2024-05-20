#include "ShaderGL.h"
#include <fstream>
#include <iostream>
#include <sstream>

#include "Buffer.h"
#include "TextureArrayGL.h"
#include <glm/gtc/type_ptr.hpp>

namespace gpupt
{
shaderGL::shaderGL(const char* computePath) {
    std::string VShaderCode = ReadFile(computePath);
    GLuint Shader = CompileShader(GL_COMPUTE_SHADER, VShaderCode.c_str());
    ID = LinkShader(Shader);
    glDeleteShader(Shader);
}

shaderGL::shaderGL(const char* VertexPath, const char *FragmentPath) {
    std::string VShaderCode = ReadFile(VertexPath);
    GLuint VertexShader = CompileShader(GL_VERTEX_SHADER, VShaderCode.c_str());
    std::string FShaderCode = ReadFile(FragmentPath);
    GLuint FragmentShader = CompileShader(GL_FRAGMENT_SHADER, FShaderCode.c_str());
    
    if (!VertexShader || !FragmentShader) {
        std::cerr << "Failed to create shader program." << std::endl;
        exit(0);
    }

    ID = glCreateProgram();
    glAttachShader(ID, VertexShader);
    glAttachShader(ID, FragmentShader);
    glLinkProgram(ID);

    GLint success;
    glGetProgramiv(ID, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(ID, 512, NULL, infoLog);
        std::cerr << "Shader program linking error:\n" << infoLog << std::endl;
        glDeleteProgram(ID);
        exit(0);
    }

    glDeleteShader(VertexShader);
    glDeleteShader(FragmentShader);
}

shaderGL::~shaderGL()
{
    this->Destroy();
}

void shaderGL::Destroy()
{
    glDeleteProgram(this->ID);
}

void shaderGL::Use() {
    glUseProgram(ID);
}

// Utility functions to bind values to the shader
void shaderGL::SetInt(const std::string& name, int value) {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void shaderGL::SetMat4(const std::string& name, glm::mat4 &Matrix) {
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(Matrix));
}

void shaderGL::SetTexture(int ImageUnit, GLuint TextureID, GLenum Access) {
    glBindImageTexture(ImageUnit, TextureID, 0, GL_FALSE, 0, Access, GL_RGBA32F);
}

void shaderGL::SetTexture(int ImageUnit, GLuint TextureID) const {    
    glBindTextureUnit(ImageUnit, TextureID);
}

void shaderGL::SetSSBO(std::shared_ptr<bufferGL> Buffer, int BindingPoint)
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, Buffer->BufferID);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BindingPoint, Buffer->BufferID);
}

void shaderGL::SetUBO(std::shared_ptr<uniformBufferGL> Buffer, int BindingPoint)
{
    glBindBufferBase(GL_UNIFORM_BUFFER, BindingPoint, Buffer->BufferID);
}

void shaderGL::SetTextureArray(std::shared_ptr<textureArrayGL> Texture, int Unit, std::string Name)
{
    glActiveTexture(GL_TEXTURE0 + Unit);
    glBindTexture(GL_TEXTURE_2D_ARRAY, Texture->TextureID);
    glUniform1i(glGetUniformLocation(ID, Name.c_str()), Unit);
}

void shaderGL::Dispatch(uint32_t X, uint32_t Y, uint32_t Z)
{
    this->Use();
    glDispatchCompute(X, Y, Z);
}

void shaderGL::Barrier()
{
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

std::string shaderGL::ReadFile(const char* FilePath) const {
    std::string IncludeIdentifier = "#include ";
    static bool RecursiveCall=false;

    std::string FullSource = "";
    std::ifstream File(FilePath);

    if(!File.is_open())
    {
        std::cout << "Could not open shader File " << FilePath << std::endl;
        return FullSource;
    }

    std::string LineBuffer;
    while(std::getline(File, LineBuffer))
    {
        if(LineBuffer.find(IncludeIdentifier) != LineBuffer.npos)
        {
            LineBuffer.erase(0, IncludeIdentifier.size());
            size_t found = std::string(FilePath).find_last_of("/\\");
            std::string PathWithoutFileName = std::string(FilePath).substr(0, found + 1);                
            LineBuffer.insert(0, PathWithoutFileName);
            RecursiveCall = true;
            FullSource += ReadFile(LineBuffer.c_str());
            continue;
        }

        FullSource += LineBuffer + "\n";
    }

    if(RecursiveCall)
    {
        FullSource += "\0";
    }

    File.close();

    return FullSource;       
}


// Utility function to compile a shader
GLuint shaderGL::CompileShader(GLenum Type, const char* SourceCode) const {
    GLuint Shader = glCreateShader(Type);
    glShaderSource(Shader, 1, &SourceCode, nullptr);
    glCompileShader(Shader);

    // Check for compilation errors
    int Success;
    char InfoLog[512];
    glGetShaderiv(Shader, GL_COMPILE_STATUS, &Success);
    if (!Success) {
        glGetShaderInfoLog(Shader, 512, nullptr, InfoLog);
        std::cout << "Shader compilation error: " << InfoLog << std::endl;
        std::string Src(SourceCode);
        int LineNumber = 1;
        std::istringstream iss(Src);
        std::string line;        
        while (std::getline(iss, line)) {
            std::cout << "Line " << LineNumber << ": " << line << std::endl;
            LineNumber++;
        }
        exit(0);
    }

    return Shader;
}

// Utility function to link shaders into a program
GLuint shaderGL::LinkShader(GLuint ComputeShader) const {
    GLuint Program = glCreateProgram();
    glAttachShader(Program, ComputeShader);
    glLinkProgram(Program);

    // Check for linking errors
    int Success;
    char InfoLog[512];
    glGetProgramiv(Program, GL_LINK_STATUS, &Success);
    if (!Success) {
        glGetProgramInfoLog(Program, 512, nullptr, InfoLog);
        std::cout << "Shader linking error: " << InfoLog << std::endl;
        exit(0);
    }

    return Program;
}
}