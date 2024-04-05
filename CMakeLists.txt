project(gpuPathTracing)

set(API_GL 0)
set(API_CU 1)
set(API ${API_GL})

add_definitions(-DAPI_GL=${API_GL})
add_definitions(-DAPI_CU=${API_CU})
add_definitions(-DAPI=${API})


set(IMGUI_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/dearimgui)
set(IMGUI_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/dearimgui/imgui_demo.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/vendor/dearimgui/imgui_impl_glfw.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/vendor/dearimgui/imgui.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/vendor/dearimgui/imgui_draw.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/vendor/dearimgui/imgui_widgets.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/vendor/dearimgui/ImGuizmo.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/vendor/dearimgui/imgui_impl_opengl3.cpp
                )
include_directories(${IMGUI_INC_DIR})


# Glfw
set(GLFW_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw/build/install/include)
set(GLFW_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw/build/install/lib)
include_directories(${GLFW_INC_DIR})
link_directories(${GLFW_LIB_DIR})

# # Glew
set(GLEW_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glew/include)
set(GLEW_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glew/lib/Release/x64)
set(GLEW_DLL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glew/bin/Release/x64)
include_directories(${GLEW_INC_DIR})
link_directories(${GLEW_LIB_DIR})

# GLM
set(GLM_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glm)
include_directories(${GLM_INC_DIR})

# TinyGLTF
set(TINYGLTF_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/tinygltf)
include_directories(${TINYGLTF_INC_DIR})

# STB
set(STB_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/stb)
include_directories(${STB_INC_DIR})

find_package(CUDA REQUIRED)
enable_language(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

include_directories("gpuRT/include")
link_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\lib\\x64")

set (sourceFiles
        src/Main.cpp
        src/Window.cpp
        src/App.cu
        src/ShaderGL.cpp
        src/TextureGL.cpp
        src/BufferCu.cu
        src/BufferGL.cpp
        src/CameraController.cpp
        src/GLTFLoader.cpp
        src/ImageLoader.cpp
        src/TextureArrayGL.cpp
        src/TextureArrayCu.cu
        src/Scene.cpp
        src/Tracing.cpp
        src/BVH.cpp
        ${IMGUI_SOURCE}
)

cuda_add_executable(PathTracer ${sourceFiles})
target_link_libraries(PathTracer glfw3 glew32 opengl32)
target_link_libraries(PathTracer ${CUDA_LIBRARIES})

install(TARGETS PathTracer RUNTIME)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION bin/resources)
install(FILES ${GLEW_DLL_DIR}/glew32.dll DESTINATION bin)