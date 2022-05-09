//
// Created by kient on 5/5/2022.
//

#ifndef GPU_FLU_GPU_MCMC_CUH
#define GPU_FLU_GPU_MCMC_CUH

#include <cuda_runtime.h>

__global__ void mcmc_dnorm_padding(double *y_data_input_d[], double *y_ode_agg_d[], double y_mcmc_dnorm_d[], int ode_padding_size, GPUParameters *params_d);
__global__ void mcmc_setup_states_for_random(curandState* curand_state_d);
__global__ void mcmc_compute_r(double y_mcmc_dnorm_d[], double r_d[], GPUParameters *params_d);
__global__ void mcmc_print_r(GPUParameters* gpu_params_d, double* r_d);
__global__ void mcmc_update_parameters(GPUParameters* gpu_params_d, FluParameters* flu_params_current_d, FluParameters* flu_params_new_d, curandState* curand_state_d);
__global__ void mcmc_check_acceptance(double r_denom_d[], double r_num_d[], GPUParameters *gpu_params_d, FluParameters* flu_params_current_d, FluParameters* flu_params_new_d,
                            curandState* curand_state_d);
#endif //GPU_FLU_GPU_MCMC_CUH
