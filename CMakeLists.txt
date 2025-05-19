cmake_minimum_required(VERSION 3.14)

project(POUQ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif ()

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Detected macOS, setting OpenMP_ROOT to Homebrew path")
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")

    set(pybind11_DIR "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/pybind11/share/cmake/pybind11")
endif ()

find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(pouq_bindings python/binding.cpp)
target_link_libraries(pouq_bindings PRIVATE OpenMP::OpenMP_CXX)
