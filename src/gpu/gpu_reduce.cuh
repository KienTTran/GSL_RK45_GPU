//
// Created by kient on 5/5/2022.
//

#ifndef GPU_FLU_GPU_REDUCE_CUH
#define GPU_FLU_GPU_REDUCE_CUH

#include <cuda_runtime.h>
#include "gpu_parameters.cuh"

__global__ void reduce_sum(double *input, double* output, int len);
__global__ void reduce_sum_padding(double *input, double* output, GPUParameters* gpu_params_d, int total_len);

#endif //GPU_FLU_GPU_REDUCE_CUH
