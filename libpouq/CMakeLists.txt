if(APPLE)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/Cellar/libomp/21.1.1/lib/libomp.dylib")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
endif()

find_package(OpenMP REQUIRED COMPONENTS C)

file(GLOB_RECURSE SOURCES "*.c")

add_library(libpouq ${SOURCES})
target_link_libraries(libpouq PRIVATE OpenMP::OpenMP_C m)
