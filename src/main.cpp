#include "cpu/cpu_functions.h"
#include "gpu/gpu_flu.cuh"
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char* argv[])
{
    GPUParameters* gpu_params = new GPUParameters();
    gpu_params->ode_output_day = NUMDAYSOUTPUT;
    gpu_params->ode_number = NUMODE;
    gpu_params->ode_dimension = DIM;
    gpu_params->agg_dimension = 6;
    gpu_params->display_number = 1;
    gpu_params->t_target = gpu_params->ode_output_day;
    gpu_params->t0 = 0.0;
    gpu_params->step = 1.0;
    gpu_params->h = 1e-6;
    gpu_params->mcmc_loop = MCMC_ITER;
    GPUFlu* gpu_flu = new GPUFlu();
    gpu_flu->set_gpu_parameters(gpu_params);
    gpu_flu->init();
    gpu_flu->run();

    delete gpu_flu;
    delete gpu_params;
    return 0;
}