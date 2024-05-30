# FindOptiX.cmake
find_path(OptiX_INCLUDE_DIR
  NAMES optix.h
  HINTS ${OptiX_ROOT_DIR}
  PATH_SUFFIXES include
)

find_library(OptiX_LIBRARY
  NAMES optix
  HINTS ${OptiX_ROOT_DIR}
  PATH_SUFFIXES lib64 lib
)

if(OptiX_INCLUDE_DIR AND OptiX_LIBRARY)
  set(OptiX_FOUND TRUE)
  set(OptiX_LIBRARIES ${OptiX_LIBRARY})
  set(OptiX_INCLUDE_DIRS ${OptiX_INCLUDE_DIR})
else()
  set(OptiX_FOUND FALSE)
endif()

mark_as_advanced(OptiX_INCLUDE_DIR OptiX_LIBRARY)
