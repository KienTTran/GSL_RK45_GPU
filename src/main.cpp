#include "gpu_functions.h"
#include "cpu_functions.h"
#include <iostream>


using namespace std::chrono;

int main()
{
    int threads;

    std::cout << "Enter number os equations: ";
    std::cin >> threads;
    std::cout << "Running on "<< threads << std::endl;

#ifdef GSL_ENABLE
    rk45_gsl_simulate(threads);
#endif
    rk45_gpu_simulate(threads);

    return 0;

}