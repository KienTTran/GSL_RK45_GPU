#include <cuda_runtime.h>
#include "gpu_rk45.h"

__host__ __device__
void gpu_func_pen(double t, const double y[], double dydt[], void *params){
    const double m = 5.2;		// Mass of pendulum
    const double g = -9.81;		// g
    const double l = 2;		// Length of pendulum
    const double A = 0.5;		// Amplitude of driving force
    const double wd = 1;		// Angular frequency of driving force
    const double b = 0.5;		// Damping coefficient

//    GPU_Parameters* gpu_params = (GPU_Parameters*) params;
//    printf("        [function]\n");

//        dydt[i] = y[i] - pow(t[i], 2) + 1;
    dydt[0] = y[1];
    dydt[1] = -(g / l) * sin(y[0]) + (A * cos(wd * t) - b * y[1]) / (m * l * l);
//    dydt[2] = dydt[1] * y[1] / y[0];
//    for (int i = 0; i < gpu_params->dimension; i ++)
//    {
//        printf("          dydt[%d] = %.10f\n",i,dydt[i]);
//    }
    return;
}