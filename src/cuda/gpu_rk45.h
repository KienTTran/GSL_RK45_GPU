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

const double ah[] = { 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0 };
const double b3[] = { 3.0/32.0, 9.0/32.0 };
const double b4[] = { 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0};
const double b5[] = { 8341.0/4104.0, -32832.0/4104.0, 29440.0/4104.0, -845.0/4104.0};
const double b6[] = { -6080.0/20520.0, 41040.0/20520.0, -28352.0/20520.0, 9295.0/20520.0, -5643.0/20520.0};

const double c1 = 902880.0/7618050.0;
const double c3 = 3953664.0/7618050.0;
const double c4 = 3855735.0/7618050.0;
const double c5 = -1371249.0/7618050.0;
const double c6 = 277020.0/7618050.0;

const double ec[] = { 0.0,
                             1.0 / 360.0,
                             0.0,
                             -128.0 / 4275.0,
                             -2197.0 / 75240.0,
                             1.0 / 50.0,
                             2.0 / 55.0
};

static double eps_abs = 1e-6;
static double eps_rel = 0.0;
static double a_y = 1.0;
static double a_dydt = 0.0;
static unsigned int ord = 5;
const double S = 0.9;

__host__ __device__ void gpu_func_flu(double t, const double y[], double dydt[], void *params);
__host__ __device__ void gpu_func_pen(double t, const double y[], double f[], void *params);
__device__ void gpu_func_test(double t, const double y[], double f[], int index, void *params);
__device__ void gpu_func_test(double t, const double y[], double f[], int index, int day, GPU_Parameters* gpu_params);
__device__ double seasonal_transmission_factor(GPU_Parameters* gpu_params, double t);

__global__
void calculate_y(double y[], double y_tmp[], double y_err[], double* h,  int step,
                     double k1[], double k2[], double k3[],
                     double k4[], double k5[], double k6[],
                     GPU_Parameters* params);
__global__
void calculate_r(double y[], double y_err[], double dydt_out[], double* h_0, double* h, int final_step, double r[], GPU_Parameters* params);

__global__
void gpu_func_test(double* t, const double y[], double f[], GPU_Parameters* gpu_params);
__global__
void reduce_max(double data[], double out[], unsigned int n);

void adjust_h(double r_max, double h_0, double* h, int final_step, int* adjustment_out);

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
