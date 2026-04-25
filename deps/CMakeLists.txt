set(TBB_TEST OFF CACHE BOOL "Disable TBB tests" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "Disable TBB examples" FORCE)

add_subdirectory(tbb)
add_subdirectory(xsimd)
add_subdirectory(nlopt)
add_subdirectory(pybind11)