add_executable(pouq_demo
        quantizer_test.cpp
        baseline/scalar_quantizer.cpp
)

target_link_libraries(pouq_demo
        PRIVATE
        pouq
        xsimd
        TBB::tbb
        nlopt
)