project(gpuPathTracing CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD  17)
set(CUDA_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(API_GL 0)
set(API_CU 1)
set(API ${API_CU})

add_definitions(-DAPI_GL=${API_GL})
add_definitions(-DAPI_CU=${API_CU})
add_definitions(-DAPI=${API})

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)


set(IMGUI_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/submodules/dearimgui ${CMAKE_CURRENT_SOURCE_DIR}/submodules/dearimgui/backends ${CMAKE_CURRENT_SOURCE_DIR}/submodules/ImGuizmo/)
set(IMGUI_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/submodules/dearimgui/imgui_demo.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/submodules/dearimgui/backends/imgui_impl_glfw.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/submodules/dearimgui/imgui.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/submodules/dearimgui/imgui_draw.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/submodules/dearimgui/imgui_widgets.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/submodules/dearimgui/imgui_tables.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/submodules/dearimgui/backends/imgui_impl_opengl3.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/submodules/ImGuizmo/ImGuizmo.cpp
)
include_directories(${IMGUI_INC_DIR})

# GLM
set(GLM_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/submodules/glm)
include_directories(${GLM_INC_DIR})

# TinyGLTF
set(TINYGLTF_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/submodules/tinygltf)
include_directories(${TINYGLTF_INC_DIR})

# STB
set(STB_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/submodules/stb)
include_directories(${STB_INC_DIR})

# FileDialog
set(FILEDIALOG_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/submodules/fileDialog/src/include)
set(FILEDIALOG_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/submodules/fileDialog/src/nfd_common.c)
set(FILEDIALOG_SOURCE ${FILEDIALOG_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/submodules/fileDialog/src/nfd_win.cpp)
include_directories(${FILEDIALOG_INC_DIR})


# Glfw
set(GLFW_INSTALL FALSE)
add_subdirectory(submodules/glfw)
include_directories(submodules/glfw/include)

# # Glad
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/submodules/glad/cmake" glad_cmake)
glad_add_library(glad_gl_core_33 STATIC API gl:core=3.3)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/gladsources/glad_gl_core_33/include")
set(GLAD_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/gladsources/glad_gl_core_33/src/gl.c)

# ASSIMP
set(ASSIMP_INSTALL_PDB OFF)
add_subdirectory(submodules/assimp)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/submodules/assimp/include")
include_directories(submodules/assimp/include)

# # OIDN
SET(OIDN_DEVICE_CPU OFF)
SET(OIDN_DEVICE_CUDA ON)
add_subdirectory(submodules/oidn)

find_package(CUDA REQUIRED)
enable_language(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -std=c++11)

set(CUDA_LIB_DIR "$ENV{CUDA_PATH}/lib/x64")
link_directories(${CUDA_LIB_DIR})

set (sourceFiles
        src/Main.cpp
        src/Window.cpp
        src/App.cu
        src/ShaderGL.cpp
        src/TextureGL.cpp
        src/Buffer.cpp
        src/CameraController.cpp
        src/GLTFLoader.cpp
        src/AssimpLoader.cpp
        src/AssetLoader.cpp
        src/ImageLoader.cpp
        src/TextureArrayGL.cpp
        src/TextureArrayCu.cu
        src/Framebuffer.cpp
        src/VertexBuffer.cpp
        src/Scene.cpp
        src/Tracing.cpp
        src/GUI.cpp
        src/BVH.cpp
        ${IMGUI_SOURCE}
        ${FILEDIALOG_SOURCE}
        ${GLAD_SOURCE}
)

cuda_add_executable(PathTracer ${sourceFiles})
target_link_libraries(PathTracer glfw opengl32 OpenImageDenoise assimp)
target_link_libraries(PathTracer ${CUDA_LIBRARIES})

set_property(TARGET PathTracer PROPERTY CXX_STANDARD 17)
set_property(TARGET PathTracer PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET PathTracer PROPERTY CUDA_STANDARD 17)

install(TARGETS PathTracer RUNTIME)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION bin/resources)