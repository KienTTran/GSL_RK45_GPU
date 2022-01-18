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
__host__ __device__ void gpu_func_test(double t, const double y[], double f[], int index, void *params);

__device__ void reduce_max_n(double *data, double *out, unsigned int n);
__device__ void reduce_max_0(double *data, double *out, unsigned int n);
__device__ void reduce_sum_n(double *data, double *out, unsigned int n);
__device__ void reduce_sum_0(double *data, double *out, unsigned int n);
__device__ void test_reduce_sum_max(double* data, double* out, unsigned n);


class GPU_RK45{
public:
    explicit GPU_RK45();
    ~GPU_RK45();
    void setParameters(GPU_Parameters* params);
    void run();
private:
    GPU_Parameters* params;
};

#endif
