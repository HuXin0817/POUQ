cmake_minimum_required(VERSION 3.12)
project(POUQ)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(libpouq)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE libpouq)
