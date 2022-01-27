#ifndef GPU_FUNCTION_H
#define GPU_FUNCTION_H

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <random>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include "../gpu_parameters.h"

const float ah[] = { 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0 };
const float b3[] = { 3.0/32.0, 9.0/32.0 };
const float b4[] = { 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0};
const float b5[] = { 8341.0/4104.0, -32832.0/4104.0, 29440.0/4104.0, -845.0/4104.0};
const float b6[] = { -6080.0/20520.0, 41040.0/20520.0, -28352.0/20520.0, 9295.0/20520.0, -5643.0/20520.0};

const float c1 = 902880.0/7618050.0;
const float c3 = 3953664.0/7618050.0;
const float c4 = 3855735.0/7618050.0;
const float c5 = -1371249.0/7618050.0;
const float c6 = 277020.0/7618050.0;

const float ec[] = { 0.0,
                             1.0 / 360.0,
                             0.0,
                             -128.0 / 4275.0,
                             -2197.0 / 75240.0,
                             1.0 / 50.0,
                             2.0 / 55.0
};

static float eps_abs = 1e-6;
static float eps_rel = 0.0;
static float a_y = 1.0;
static float a_dydt = 0.0;
static unsigned int ord = 5;
const float S = 0.9;

__device__ void gpu_func_test(float t, const float y[], float f[], int index, int day, GPU_Parameters* gpu_params);
__device__ float seasonal_transmission_factor(GPU_Parameters* gpu_params, float t);
__device__ float pop_sum( float yy[] );

void *cpu_thread_display_output(void *params);

typedef struct {
    float* y;
    int total_size;
    int dimension;
    int stream_id;
} cpu_thread_params;

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
