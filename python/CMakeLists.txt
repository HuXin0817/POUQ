pybind11_add_module(pypouq binding.cpp)
target_link_libraries(pypouq PRIVATE pouq xsimd)