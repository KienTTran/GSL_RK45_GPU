#include "gpu_functions.h"
#include "cpu_functions.h"
#include <iostream>


using namespace std::chrono;

int main()
{
    int threads;
    int display;

    std::cout << "Enter number of ODE equations to run: " << std::endl;
    std::cin >> threads;
    std::cout << "Enter number of results to display randomly: " << std::endl;
    std::cin >> display;
    std::cout << "Running on "<< threads << std::endl;

#ifdef GSL_ENABLE
    rk45_gsl_simulate(threads,display);
#endif
    rk45_gpu_simulate(threads,display);

    return 0;

}