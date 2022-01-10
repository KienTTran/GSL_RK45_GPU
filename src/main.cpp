#include "gpu_functions.h"
#include "cpu_functions.h"

using namespace std;

int main()
{

//    rk45_gsl_cpu_simulate();
    rk45_gsl_gpu_simulate();

    return 0;

}