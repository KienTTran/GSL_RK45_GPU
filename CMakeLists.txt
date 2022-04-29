cmake_minimum_required(VERSION 3.18)
project(rk45_cuda LANGUAGES CXX CUDA)

set(BUILD_CLUSTER false CACHE BOOL "Enable for builds on the cluster")

if(UNIX)
    include_directories("/usr/local/cuda/include/")
    include_directories("lib/stats/include/")
    #CMake projects should use: -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc
    #CMake projects should use: -DCMAKE_TOOLCHAIN_FILE="/mnt/d/BoniLab/vcpkg/scripts/buildsystems/vcpkg.cmake"
    set(CMAKE_TOOLCHAIN_FILE "/mnt/d/BoniLab/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
else(WIN32)
    include_directories("lib\\stats\\include\\")
    #CMake projects should use: -DCMAKE_TOOLCHAIN_FILE="D:/BoniLab/vcpkg/scripts/buildsystems/vcpkg.cmake"
    set(CMAKE_TOOLCHAIN_FILE "D:/BoniLab/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    include_directories("E:\\SDK\\CUDA\\11.5\\Toolkit\\include\\")
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

find_package(GSL REQUIRED)
find_package(DataFrame REQUIRED)
find_package(gcem CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rk45_cuda
        src/main.cpp
        src/flu_default_params.h
        src/gpu/gpu_parameters.cpp
        src/gpu/gpu_parameters.h
        src/gpu/gpu_func_test.cu
        src/gpu/gpu_rk45.cu
        src/gpu/gpu_rk45.h
        src/mcmc/mcmc.cpp
        src/mcmc/mcmc.h
        src/cpu/cpu_functions.cpp
        src/cpu/gsl_functions.cpp
        src/cpu/cpu_functions.h)

set_target_properties(
        rk45_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

if(BUILD_CLUSTER)
else()
    #    target_compile_options(rk45_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
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
    target_compile_options(rk45_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            #            --generate-line-info
            --use_fast_math
            --relocatable-device-code=true
            --default-stream per-thread
            -maxrregcount=255
            >)
endif()

if (BUILD_CLUSTER)
    set_target_properties(rk45_cuda PROPERTIES CUDA_ARCHITECTURES 37)
else()
    set_target_properties(rk45_cuda PROPERTIES CUDA_ARCHITECTURES 52)
endif()

target_link_libraries(rk45_cuda PRIVATE GSL::gsl GSL::gslcblas) #vcpkg
target_link_libraries(rk45_cuda PRIVATE DataFrame::DataFrame)
target_link_libraries(rk45_cuda PRIVATE gcem)