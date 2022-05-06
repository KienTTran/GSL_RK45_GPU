#include "cpu/cpu_functions.h"
#include "gpu/gpu_flu.cuh"
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char* argv[])
{
    FluParameters** flu_params = new FluParameters*[NUMODE]();
    for(int ode_index = 0; ode_index < NUMODE; ode_index++){
        flu_params[ode_index] = new FluParameters();
        flu_params[ode_index]->init();
    }
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
    gpu_params->mcmc_loop = 1;
    gpu_params->init(flu_params);
    GPUFlu* gpu_flu = new GPUFlu();
    gpu_flu->set_flu_parameters(flu_params);
    gpu_flu->set_gpu_parameters(gpu_params);
    gpu_flu->run();

    delete gpu_flu;
    delete gpu_params;
    delete flu_params;
    return 0;
}