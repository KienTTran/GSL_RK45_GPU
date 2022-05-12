#include "gpu/gpu_flu.cuh"
#include <iostream>

int main(int argc, char* argv[])
{
    GPUParameters* gpu_params = new GPUParameters();
    gpu_params->ode_output_day = NUMDAYSOUTPUT;
    gpu_params->ode_number = NUMODE;
    gpu_params->ode_dimension = DIM;
    gpu_params->agg_dimension = 6;
    gpu_params->display_number = 1;
    gpu_params->ode_t_target = gpu_params->ode_output_day;
    gpu_params->ode_t0 = 0.0;
    gpu_params->ode_step = 1.0;
    gpu_params->ode_h = 1e-6;
    gpu_params->mcmc_loop = MCMC_ITER;

    GPUFlu* gpu_flu = new GPUFlu();
    gpu_flu->set_gpu_parameters(gpu_params);
    gpu_flu->init();
    gpu_flu->run();
    delete gpu_flu;

    delete gpu_params;
    return 0;
}