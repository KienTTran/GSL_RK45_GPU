cmake_minimum_required(VERSION 3.18)
project(rk45_cuda LANGUAGES CXX CUDA)

set(BUILD_CLUSTER false CACHE BOOL "Enable for builds on the cluster")

if (BUILD_CLUSTER)
    add_compile_definitions( ON_CLUSTER )
    include_directories("/usr/local/cuda/")
    add_executable(rk45_cuda
            src/cuda/gpu_func_flu.cu
            src/cuda/gpu_func_pen.cu
            src/cuda/gpu_func_test.cu
            src/cuda/gpu_reduce.cu
            src/cuda/gpu_rk45.cu
            src/cuda/gpu_sample.cu
            src/cuda/gpu_rk45.h
            src/cpu_rk45.cpp
            src/cpu_rk45.h
            src/flu_func.cpp
            src/gsl_functions.cpp
            src/cpu_parameters.cpp
            src/cpu_parameters.h
            src/gpu_parameters.cpp
            src/gpu_parameters.h
            src/flu_default_params.h
            src/main.cpp)
else()
    include_directories("E:\\SDK\\CUDA\\11.5\\Toolkit\\include\\")
    add_executable(rk45_cuda
            src/cuda/gpu_func_flu.cu
            src/cuda/gpu_func_pen.cu
            src/cuda/gpu_func_test.cu
            src/cuda/gpu_rk45.cu
            src/cuda/gpu_sample.cu
            src/cuda/gpu_reduce.cu
            src/cuda/gpu_rk45.h
            src/cpu_rk45.cpp
            src/cpu_rk45.h
            src/flu_func.cpp
            src/cpu_parameters.cpp
            src/cpu_parameters.h
            src/gpu_parameters.cpp
            src/gpu_parameters.h
            src/flu_default_params.h
            src/main.cpp)
endif()

set_target_properties(
        rk45_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

if (BUILD_CLUSTER)
    set_target_properties(rk45_cuda PROPERTIES CUDA_ARCHITECTURES 37)
else()
    set_target_properties(rk45_cuda PROPERTIES CUDA_ARCHITECTURES 52)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_52,code=sm_52" )
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo" )
endif()

if (BUILD_CLUSTER)
    target_link_libraries(rk45_cuda gsl gslcblas)
endif()
