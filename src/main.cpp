#include "gpu/gpu_rk45.h"
#include "cpu/cpu_functions.h"
#include <iostream>
#include <thread>
#include <chrono>

int main(int argc, char* argv[])
{

    GPU_RK45* gpu_rk45 = new GPU_RK45();
    GPU_Parameters* gpu_params_test = new GPU_Parameters();
    gpu_params_test->number_of_ode = 1;
    gpu_params_test->dimension = DIM;
    gpu_params_test->display_number = 1;
    gpu_params_test->t_target = NUMDAYSOUTPUT;
    gpu_params_test->t0 = 0.0;
    gpu_params_test->step = 1.0;
    gpu_params_test->h = 1e-6;
    gpu_params_test->initTest(argc,argv);
    gpu_rk45->set_parameters(gpu_params_test);
    gpu_rk45->run();

    delete gpu_rk45;
    return 0;

}