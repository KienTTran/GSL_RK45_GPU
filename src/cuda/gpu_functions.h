#ifndef GPU_FUNCTION_H
#define GPU_FUNCTION_H

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <random>
#include "../gpu_parameters.h"

__host__ __device__
void gpu_func(double t, const double y[], double f[], void *params);

class GPU_RK45{
public:
    explicit GPU_RK45();
    ~GPU_RK45();
    int rk45_gpu_simulate();
    void setParameters(GPU_Parameters* params);
    void predict(double t0, double t1, double h, double ** y0, GPU_Parameters* params_d);
    void run();
private:
    GPU_Parameters* params;
};

#endif
