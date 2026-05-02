add_library(pouq
        STATIC
        quantizer.cpp
        clusterer.cpp
        optimizer.cpp
)

target_link_libraries(pouq PRIVATE xsimd TBB::tbb nlopt)