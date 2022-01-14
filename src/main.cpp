#include "cuda/gpu_rk45.h"
#include "cpu_rk45.h"
#include <iostream>
#include <thread>
#include <chrono>
#include "cpu_parameters.h"
#include "cuda/gpu_rk45.h"

//int ode_function(double t, const double y[], double dydt[], void* params){
//    // 2 dim
//    const double m = 5.2;		// Mass of pendulum
//    const double g = -9.81;		// g
//    const double l = 2;		// Length of pendulum
//    const double A = 0.5;		// Amplitude of driving force
//    const double wd = 1;		// Angular frequency of driving force
//    const double b = 0.5;		// Damping coefficient
//
//    dydt[0] = y[1];
//    dydt[1] = -(g / l) * sin(y[0]) + (A * cos(wd * t) - b * y[1]) / (m * l * l);
//    return 0;
//}

int main(int argc, char* argv[])
{
    int threads = 1;
    int display = 1;
//    std::cout << "Enter number of ODE equations to run: " << std::endl;
//    std::cin >> threads;
//    std::cout << "Enter number of results to display randomly: " << std::endl;
//    std::cin >> display;
#ifdef ON_CLUSTER
    std::cout << "Running GSL on CPU" << std::endl;
    rk45_gsl_simulate(threads,display);
#endif
//    std::cout << "Running FLU on CPU" << std::endl;
//    CPU_RK45* cpu_rk45 = new CPU_RK45();
//    CPU_Parameters* cpu_params = new CPU_Parameters();
//    cpu_params->number_of_ode = 1024000;
//    cpu_params->dimension = 16;
//    cpu_params->initFlu(argc, argv);
//    cpu_params->display_number = display;
//    cpu_params->cpu_function = func;
//    cpu_params->t_target = NUMDAYSOUTPUT;
//    cpu_params->t0 = 0.0;
//    cpu_params->h = 1e-6;
//    cpu_rk45->setParameters(cpu_params);
//    cpu_rk45->run();

//    std::cout << std::endl;
//    std::cout << "Performing test 1 on GPU" << std::endl;
////    test_cuda_1();
//    std::cout << std::endl;
//    std::cout << "Performing test 2 on GPU" << std::endl;
////    test_cuda_2();
//    std::cout << std::endl;

//    std::cout << "Running PEN on GPU" << std::endl;
//    GPU_RK45* gpu_rk45 = new GPU_RK45();
//    GPU_Parameters* gpu_params_pen = new GPU_Parameters();
//    gpu_params_pen->number_of_ode = 1024;
//    gpu_params_pen->dimension = 2;
//    gpu_params_pen->initPen();
//    gpu_params_pen->display_number = display;
//    gpu_params_pen->t_target = 2.0;
//    gpu_params_pen->t0 = 0.0;
//    gpu_params_pen->h = 0.2;
//    gpu_rk45->setParameters(gpu_params_pen);
//    gpu_rk45->run();

//    std::cout << "Running FLU on GPU" << std::endl;
//    GPU_RK45* gpu_rk45_flu = new GPU_RK45();
//    GPU_Parameters* gpu_params_flu = new GPU_Parameters();
//    gpu_params_flu->number_of_ode = 1024;
//    gpu_params_flu->dimension = 16;
//    gpu_params_flu->initFlu(argc, argv);
//    gpu_params_flu->display_number = display;
//    gpu_params_flu->t_target = 1.0;
//    gpu_params_flu->t0 = 0.0;
//    gpu_params_flu->h = 1e-6;
//    gpu_rk45_flu->setParameters(gpu_params_flu);
//    gpu_rk45_flu->run();

    std::cout << "Running GSL on GPU" << std::endl;
    GPU_RK45* gpu_rk45 = new GPU_RK45();
    GPU_Parameters* gpu_params_test = new GPU_Parameters();
    gpu_params_test->number_of_ode = 1;
    gpu_params_test->dimension = 1;
    gpu_params_test->initTest();
    gpu_params_test->display_number = display;
    gpu_params_test->t_target = 1;
    gpu_params_test->t0 = 0.0;
    gpu_params_test->h = 0.2;
    gpu_rk45->setParameters(gpu_params_test);
    gpu_rk45->run();

//    flu_simulate(argc, argv);
//    flu_cpu_simulate();

    return 0;

}