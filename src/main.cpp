#include "cpu/cpu_functions.h"
#include "gpu/gpu_ode_mcmc.h"
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char* argv[])
{
    GPUFlu* gpu_flu = new GPUFlu();
    GPUParameters* gpu_params = new GPUParameters();
    gpu_params->number_of_ode = 1;
    gpu_params->ode_dimension = DIM;
    gpu_params->agg_dimension = 6;
    gpu_params->display_number = 1;
    gpu_params->t_target = NUMDAYSOUTPUT;
    gpu_params->t0 = 0.0;
    gpu_params->step = 1.0;
    gpu_params->h = 1e-6;
    gpu_params->init();
    gpu_flu->set_parameters(gpu_params);
    gpu_flu->run();

    delete gpu_flu;
    return 0;

}