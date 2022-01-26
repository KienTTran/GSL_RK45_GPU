#include "cuda/gpu_rk45.h"
#include "cpu_rk45.h"
#include <iostream>
#include <thread>
#include <chrono>
#include "cpu_parameters.h"
#include "cuda/gpu_rk45.h"

int main(int argc, char* argv[])
{
    int threads = 1;
    int display = 1;
#ifdef ON_CLUSTER
    std::cout << "Running GSL on CPU" << std::endl;
    rk45_gsl_simulate(threads,display);
#endif
    std::cout << "Running TEST on GPU" << std::endl;
    GPU_RK45* gpu_rk45 = new GPU_RK45();
    GPU_Parameters* gpu_params_test = new GPU_Parameters();
    gpu_params_test->number_of_ode = 1;
    gpu_params_test->dimension = DIM;
    gpu_params_test->display_number = display;
    gpu_params_test->t_target = NUMDAYSOUTPUT;
    gpu_params_test->t0 = 0.0;
    gpu_params_test->h = 1e-6;
    gpu_params_test->initTest(argc,argv);
    gpu_rk45->setParameters(gpu_params_test);
    gpu_rk45->run();

    delete gpu_rk45;
    return 0;

}
