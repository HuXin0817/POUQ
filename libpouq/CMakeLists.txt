IF (APPLE)
    IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET(OpenMP_CXX_FLAGS "-fopenmp=libomp")
        SET(OpenMP_CXX_LIB_NAMES "omp")
        SET(OpenMP_omp_LIBRARY omp)
    ENDIF ()
ENDIF ()

IF (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
ELSEIF (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
ENDIF ()

FIND_PACKAGE(OpenMP REQUIRED)

ADD_LIBRARY(libpouq ./simd/distance.cpp
        ./simd/loss.cpp
        ./util.cpp
        ./optimize.cpp
        ./train.cpp
        ./segment.cpp)
TARGET_LINK_LIBRARIES(libpouq PRIVATE
        OpenMP::OpenMP_CXX
)

