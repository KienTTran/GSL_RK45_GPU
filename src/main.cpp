#include "gpu_functions.h"
#include "cpu_functions.h"
#include <chrono>

using namespace std::chrono;

int main()
{
    rk45_gsl_cpu_simulate();
    rk45_gsl_gpu_simulate();
    rk45_gsl_gpu_simulate_2();

    return 0;

}