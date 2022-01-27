cmake_minimum_required(VERSION 3.18)
project(rk45_cuda LANGUAGES CXX CUDA)

#
# On WSL compile with -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc
# Add -DCMAKE_CUDA_FLAGS="-default-stream per-thread" to use streams
#
set(BUILD_CLUSTER false CACHE BOOL "Enable for builds on the cluster")

if (WIN32)
    find_package(GSL REQUIRED)
    #CMake projects should use: "-DCMAKE_TOOLCHAIN_FILE=D:/BoniLab/vcpkg/scripts/buildsystems/vcpkg.cmake"
    include_directories("E:\\SDK\\CUDA\\11.5\\Toolkit\\include\\")
    add_executable(rk45_cuda
            src/cuda/gpu_func_flu.cu
            src/cuda/gpu_func_pen.cu
            src/cuda/gpu_func_test.cu
            src/cuda/gpu_rk45_global_kernels.cu
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
            src/gsl_functions.cpp
            src/main.cpp)
else()
    include_directories("/usr/local/cuda/include/")
    add_executable(rk45_cuda
            src/cuda/gpu_func_flu.cu
            src/cuda/gpu_func_pen.cu
            src/cuda/gpu_func_test.cu
            src/cuda/gpu_rk45_global_kernels.cu
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
            src/gsl_functions.cpp
            src/main.cpp)
endif()

set_target_properties(
        rk45_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#SET(CMAKE_CXX_FLAGS  "-O3")

if (BUILD_CLUSTER)
    set_target_properties(rk45_cuda PROPERTIES CUDA_ARCHITECTURES 37)
else()
    set_target_properties(rk45_cuda PROPERTIES CUDA_ARCHITECTURES 52)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_52,code=sm_52" )
    set(CUDA_NVCC_FLAGS "-lineinfo -O3 -default-stream per-thread --use_fast_math" )
endif()

if (UNIX)
    target_link_libraries(rk45_cuda gsl gslcblas)
    target_compile_options(rk45_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --use_fast_math
            --relocatable-device-code=true
            >)
else()
    target_link_libraries(rk45_cuda PRIVATE GSL::gsl GSL::gslcblas)
endif()
