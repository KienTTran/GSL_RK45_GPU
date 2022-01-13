cmake_minimum_required(VERSION 3.18)
project(rk45_cuda LANGUAGES CXX CUDA)

set(BUILD_CLUSTER false CACHE BOOL "Enable for builds on the cluster")

if (BUILD_CLUSTER)
    add_compile_definitions( ON_CLUSTER )
    include_directories("/usr/local/cuda/")
    add_executable(rk45_cuda
            src/cuda/gpu_functions.cu
            src/cuda/gpu_sample.cu
            src/cuda/gpu_functions.h
            src/cpu_functions.cpp
            src/gsl_functions.cpp
            src/cpu_functions.h
            src/cpu_parameters.cpp
            src/cpu_parameters.h
            src/flu_default_params.h
            src/main.cpp)
else()
    include_directories("E:\\SDK\\CUDA\\11.5\\Toolkit\\include")
    add_executable(rk45_cuda
            src/cuda/flu_gpu_func.cu
            src/cuda/gpu_functions.cu
            src/cuda/gpu_sample.cu
            src/cuda/gpu_functions.h
            src/cpu_functions.cpp
            src/cpu_functions.h
            src/flu_func.cpp
            src/flu_cpu_prms.cpp
            src/flu_cpu_prms.h
            src/flu_gpu_prms.cpp
            src/flu_gpu_prms.h
            src/flu_readdata.cpp
            src/flu_readdata.h
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
endif()

if (BUILD_CLUSTER)
    target_link_libraries(rk45_cuda gsl gslcblas)
endif()
