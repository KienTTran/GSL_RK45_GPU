#include <cuda_runtime.h>
#include "gpu_rk45.h"

__host__ __device__
void gpu_func_test(double t, const double y[], double dydt[], void *params){

    GPU_Parameters* params_gpu = (GPU_Parameters*) params;
    for(int i = 0; i < params_gpu->dimension; i++){
        dydt[i] = y[i];
    }
    return;
}