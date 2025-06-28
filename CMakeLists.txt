CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
PROJECT(POUQ)

SET(CMAKE_CXX_STANDARD 17)

FIND_PACKAGE(OpenMP REQUIRED)

OPTION(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

IF (BUILD_PYTHON_BINDINGS)
    FIND_PACKAGE(Python COMPONENTS Interpreter Development REQUIRED)
    FIND_PACKAGE(pybind11 REQUIRED)

    pybind11_add_module(binding python/binding.cpp)

    TARGET_COMPILE_FEATURES(binding PRIVATE cxx_std_17)
    TARGET_COMPILE_OPTIONS(binding PRIVATE
            -O3
            -march=native
            -ffast-math
            -DNDEBUG
            -flto
    )

    SET_TARGET_PROPERTIES(binding PROPERTIES
            CXX_VISIBILITY_PRESET "hidden"
            VISIBILITY_INLINES_HIDDEN YES
            INTERPROCEDURAL_OPTIMIZATION TRUE
    )

    TARGET_LINK_LIBRARIES(binding PRIVATE OpenMP::OpenMP_CXX)

    TARGET_INCLUDE_DIRECTORIES(binding PRIVATE libpouq)

    SET_TARGET_PROPERTIES(binding PROPERTIES
            CXX_VISIBILITY_PRESET "hidden"
            VISIBILITY_INLINES_HIDDEN YES
    )

    ADD_CUSTOM_TARGET(install_python_package
            COMMAND ${Python_EXECUTABLE} -m pip install ${CMAKE_SOURCE_DIR}/python
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python
            COMMENT "Installing Python package via setup.py"
            DEPENDS binding
    )
ENDIF ()

ADD_EXECUTABLE(example example/example.cpp)

TARGET_LINK_LIBRARIES(example PRIVATE
        OpenMP::OpenMP_CXX
)
