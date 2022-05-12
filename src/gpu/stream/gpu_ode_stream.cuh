//
// Created by kient on 5/10/2022.
//

#ifndef GPU_FLU_GPU_ODE_STREAM_CUH
#define GPU_FLU_GPU_ODE_STREAM_CUH
#include <cuda_runtime.h>
#include "../gpu_parameters.cuh"

__device__ void gpu_func_flu(double t, const double y[], double f[], double stf, int index, FluParameters* flu_params);
__global__ void solve_ode_n_stream(double *y_ode_input_d, double *y_ode_output_d, double *y_agg_input_d, double *y_agg_output_d, double* stf, GPUParameters *gpu_params, FluParameters* flu_params);

#endif //GPU_FLU_GPU_ODE_STREAM_CUH
