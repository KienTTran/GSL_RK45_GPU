#ifndef GPU_FUNCTION_H
#define GPU_FUNCTION_H

#include "gpu_parameters.h"

__device__ void gpu_func_test(double t, const double y[], double f[], int index, GPU_Parameters* gpu_params);
__device__ double seasonal_transmission_factor(GPU_Parameters* gpu_params, double t);
__device__ double pop_sum( double yy[] );
__global__ void reduce_sum(double *data, double *out, int len);

void *cpu_thread_display_output(void *params);

typedef struct {
    double* y;
    int total_size;
    int dimension;
    int stream_id;
} cpu_thread_params;

class GPU_RK45{
public:
    explicit GPU_RK45();
    ~GPU_RK45();
    void set_parameters(GPU_Parameters* params);
    void run();
private:
    GPU_Parameters* params;
};

#endif
