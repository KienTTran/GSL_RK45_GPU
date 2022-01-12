cmake_minimum_required(VERSION 3.19)
project(rk45_cuda LANGUAGES CXX CUDA)

set(BUILD_CLUSTER false CACHE BOOL "Enable for builds on the cluster")

add_executable(rk45_cuda
        src/main.cpp
        src/cuda/gpu_functions.cu
        src/cuda/gpu_sample.cu
        src/gpu_functions.h
        src/cpu_functions.cpp
        src/cpu_functions.h)
include_directories("E:\\SDK\\CUDA\\11.5\\Toolkit\\include")

if (BUILD_CLUSTER)
    add_compile_definitions( GSL_ENABLE )
    include_directories("/usr/local/cuda/")
    add_executable(rk45_cuda
            src/gsl_functions.cpp)
endif()

set_target_properties(
        rk45_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(rk45_cuda PROPERTIES CUDA_ARCHITECTURES "52")

if (BUILD_CLUSTER)
    target_link_libraries(rk45_cuda gsl gslcblas)
endif()
