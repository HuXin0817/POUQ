cmake_minimum_required(VERSION 3.10)

project(LibPOUQ)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(deps)
add_subdirectory(python)

add_library(pouq
        STATIC
        src/quantizer.cpp
        src/clusterer.cpp
        src/optimizer.cpp
)

target_link_libraries(pouq PRIVATE xsimd TBB::tbb nlopt)