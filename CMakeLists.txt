cmake_minimum_required(VERSION 3.12)
project(POUQ)

set(CMAKE_C_STANDARD 17)

add_subdirectory(libpouq)

add_executable(example example.c)
target_link_libraries(example PRIVATE libpouq)
