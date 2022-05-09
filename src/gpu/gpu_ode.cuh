//
// Created by kient on 5/5/2022.
//

#ifndef GPU_FLU_GPU_ODE_CUH
#define GPU_FLU_GPU_ODE_CUH

#include <cuda_runtime.h>

__device__ void gpu_func_flu(double t, const double y[], double f[], double stf, int index, FluParameters* flu_params);
__global__ void calculate_stf(double* stf_d[], GPUParameters* gpu_params, FluParameters* flu_params);
__device__ double pop_sum( double yy[] );
__global__ void solve_ode(double *y_ode_input_d[], double *y_ode_output_d[], double *y_agg_input_d[], double *y_agg_output_d[], double* stf[], GPUParameters *params, FluParameters* flu_params);

#endif //GPU_FLU_GPU_ODE_CUH
