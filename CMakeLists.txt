cmake_minimum_required(VERSION 3.18)
project(gpu_flu LANGUAGES CXX CUDA)

set(BUILD_CLUSTER false CACHE BOOL "Enable for builds on the cluster")

if(UNIX)
    include_directories("/usr/local/cuda/include/")
    #CMake projects should use: -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc
    #CMake projects should use: -DCMAKE_TOOLCHAIN_FILE="/mnt/d/BoniLab/vcpkg/scripts/buildsystems/vcpkg.cmake"
    set(CMAKE_TOOLCHAIN_FILE "/mnt/d/BoniLab/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
else(WIN32)
    #CMake projects should use: -DCMAKE_TOOLCHAIN_FILE="D:/BoniLab/vcpkg/scripts/buildsystems/vcpkg.cmake"
    set(CMAKE_TOOLCHAIN_FILE "D:/BoniLab/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    include_directories("E:\\SDK\\CUDA\\11.5\\Toolkit\\include\\")
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

find_package(GSL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gpu_flu
        src/main.cpp
        src/flu_default_params.h
        src/gpu/gpu_parameters.cu
        src/gpu/gpu_parameters.cuh
        src/gpu/gpu_func_flu.cu
        src/gpu/gpu_flu.cu
        src/gpu/gpu_flu.cuh
        src/csv/csv_data.cpp
        src/csv/csv_data.h
        src/cpu/cpu_functions.cpp
        src/cpu/gsl_functions.cpp
        src/cpu/cpu_functions.h
        src/gpu/gpu_reduce.cu
        src/gpu/gpu_ode.cu
        src/gpu/gpu_mcmc.cu
        src/gpu/flu_parameters.cu
        src/gpu/flu_parameters.cuh
        src/gpu/gpu_ode.cuh
        src/gpu/gpu_mcmc.cuh
        src/gpu/gpu_reduce.cuh
        src/gpu/gpu_ode_stream.cu
        src/gpu/gpu_ode_stream.cuh
        src/gpu/gpu_flu_stream.cu
        src/gpu/gpu_flu_stream.cuh
        )

set_target_properties(
        gpu_flu
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

if(BUILD_CLUSTER)
else()
    #    target_compile_options(gpu_flu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    ##            --generate-line-info
    #            --use_fast_math
    #            -ftz=false
    #            --relocatable-device-code=true
    #            --default-stream per-thread
    #            -O3
    #            -G
    #            -src-in-ptx
    ##            -res-usage
    #            -maxrregcount=255
    #            --ptxas-options=-v
    #            >)
    target_compile_options(gpu_flu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            #            --generate-line-info
            --use_fast_math
            --relocatable-device-code=true
            --default-stream per-thread
            -maxrregcount=255
            >)
endif()

if (BUILD_CLUSTER)
    set_target_properties(gpu_flu PROPERTIES CUDA_ARCHITECTURES 37)
else()
    set_target_properties(gpu_flu PROPERTIES CUDA_ARCHITECTURES 52)
endif()

target_link_libraries(gpu_flu PRIVATE GSL::gsl GSL::gslcblas) #vcpkg