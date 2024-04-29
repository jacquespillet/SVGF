project(gpuPathTracing)
set(CMAKE_CXX_STANDARD 17)

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

# FileDialog
set(FILEDIALOG_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/fileDialog/include)
set(FILEDIALOG_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/fileDialog/src/nfd_common.c)
set(FILEDIALOG_SOURCE ${FILEDIALOG_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/vendor/fileDialog/src/nfd_win.cpp)
include_directories(${FILEDIALOG_INC_DIR})
link_directories(${FILEDIALOG_LIB_DIR})

# # OIDN
set(OIDN_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/oidn/include)
set(OIDN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/oidn/lib)
set(OIDN_DLL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/oidn/bin)
include_directories(${OIDN_INC_DIR})
link_directories(${OIDN_LIB_DIR})

# ASSIMP
set(ASSIMP_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/assimp/include)
set(ASSIMP_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/assimp/lib)
set(ASSIMP_DLL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/assimp/bin)
include_directories(${ASSIMP_INC_DIR})
link_directories(${ASSIMP_LIB_DIR})


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
        src/Buffer.cpp
        src/CameraController.cpp
        src/GLTFLoader.cpp
        src/AssimpLoader.cpp
        src/AssetLoader.cpp
        src/ImageLoader.cpp
        src/TextureArrayGL.cpp
        src/TextureArrayCu.cu
        src/Scene.cpp
        src/Tracing.cpp
        src/GUI.cpp
        src/BVH.cpp
        ${IMGUI_SOURCE}
        ${FILEDIALOG_SOURCE}
)

cuda_add_executable(PathTracer ${sourceFiles})
target_link_libraries(PathTracer glfw3 glew32 opengl32 OpenImageDenoise assimp-vc143-mt)
target_link_libraries(PathTracer ${CUDA_LIBRARIES})

install(TARGETS PathTracer RUNTIME)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION bin/resources)
install(FILES ${OIDN_DLL_DIR}/OpenImageDenoise.dll DESTINATION bin)
install(FILES ${OIDN_DLL_DIR}/OpenImageDenoise_core.dll DESTINATION bin)
install(FILES ${OIDN_DLL_DIR}/OpenImageDenoise_device_cuda.dll DESTINATION bin)
install(FILES ${GLEW_DLL_DIR}/glew32.dll DESTINATION bin)
install(FILES ${ASSIMP_DLL_DIR}/assimp-vc143-mt.dll DESTINATION bin)