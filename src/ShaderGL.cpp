#include "ShaderGL.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace gpupt
{
shaderGL::shaderGL(const char* computePath) {
    std::string VShaderCode = ReadFile(computePath);
    GLuint Shader = CompileShader(GL_COMPUTE_SHADER, VShaderCode.c_str());
    ID = LinkShader(Shader);
    glDeleteShader(Shader);
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

void shaderGL::SetTexture(int ImageUnit, GLuint TextureID, GLenum Access) {
    glBindImageTexture(ImageUnit, TextureID, 0, GL_FALSE, 0, Access, GL_RGBA32F);
}

void shaderGL::SetTexture(int ImageUnit, GLuint TextureID) const {    
    glBindTextureUnit(ImageUnit, TextureID);
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