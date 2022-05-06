//
// Created by kient on 5/5/2022.
//

#ifndef GPU_FLU_GPU_MCMC_CUH
#define GPU_FLU_GPU_MCMC_CUH

#include <cuda_runtime.h>

__global__ void mcmc_dnorm_padding(double *y_data_input_d[], double *y_ode_agg_d[], double y_mcmc_dnorm_d[], int ode_padding_size, GPUParameters *params);
__global__ void mcmc_update_parameters(FluParameters* current_flu_params[], FluParameters* new_flu_params[], int ode_number, curandState curand_state_d[]);

#endif //GPU_FLU_GPU_MCMC_CUH
