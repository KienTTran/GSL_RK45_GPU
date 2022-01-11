cmake_minimum_required(VERSION 3.19)
project(rk45_cuda LANGUAGES CXX CUDA)
add_executable(rk45_cuda src/main.cpp src/cuda/gpu_functions.cu src/cuda/gpu_sample.cu src/gpu_functions.h src/cpu_functions.cpp src/cpu_functions.h)
include_directories("E:\\SDK\\CUDA\\11.5\\Toolkit\\include")
#include_directories("/usr/local/cuda/")
set_target_properties(
        rk45_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(rk45_cuda PROPERTIES CUDA_ARCHITECTURES "52")
#set_target_properties(rk45_cuda PROPERTIES CUDA_CODE "52")
