#ifndef GPU_FUNCTION_H
#define GPU_FUNCTION_H

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <random>
#include <cooperative_groups.h>
#include "../gpu_parameters.h"

__host__ __device__ void gpu_func_flu(double t, const double y[], double dydt[], void *params);
__host__ __device__ void gpu_func_pen(double t, const double y[], double f[], void *params);
__device__ void gpu_func_test(double t, const double y[], double f[],
                              double* sum_foi, double* foi_on_susc_single_virus,
                              double* inflow_from_recovereds, double* foi_on_susc_all_viruses,
                              int index, void *params);

void testMax(int n, bool verbose);

class GPU_RK45{
public:
    explicit GPU_RK45();
    ~GPU_RK45();
    void run(int argc, char** argv);
private:
};

#endif
