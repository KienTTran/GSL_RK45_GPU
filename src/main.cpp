#include "gpu_functions.h"
#include "cpu_functions.h"
#include <iostream>
#include <thread>
#include <chrono>
int main()
{
    int threads = 100000;
    int display = 10;

//    std::cout << "Enter number of ODE equations to run: " << std::endl;
//    std::cin >> threads;
//    std::cout << "Enter number of results to display randomly: " << std::endl;
//    std::cin >> display;

#ifdef GSL_ENABLE
    std::cout << "Running GSL on CPU" << std::endl;
    rk45_gsl_simulate(threads,display);
#endif
    std::cout << "Running GSL re-implement on CPU" << std::endl;
    rk45_cpu_simulate(threads,display);
//    std::cout << "Sleep for 1s before running on GPU " << std::endl;
//    std::this_thread::sleep_for(std::chrono::milliseconds(5000) );
    std::cout << std::endl;
    std::cout << "Performing test 1 on GPU" << std::endl;
    test_cuda_1();
    std::cout << std::endl;
    std::cout << "Performing test 2 on GPU" << std::endl;
    test_cuda_2();
    std::cout << std::endl;
    std::cout << "Running GSL on GPU" << std::endl;
    rk45_gpu_simulate(threads,display);

    return 0;

}