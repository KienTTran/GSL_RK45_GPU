#ifndef GPU_FUNCTION_H
#define GPU_FUNCTION_H

#include "gpu_parameters.h"
__device__ void gpu_func_test(double t, const double y[], double f[], double stf, int index, GPUParameters* gpu_params);
__global__ void calculate_stf(double stf_d[], GPUParameters* params);
__device__ double pop_sum( double yy[] );
__global__ void solve_ode(double *y_ode_input_d[], double *y_ode_output_d[], double *y_ode_agg_d[], double stf[], GPUParameters *params);
__global__ void mcmc_dnorm(double *y_data_input_d[], double *y_ode_agg_d[], double y_mcmc_dnorm_d[], GPUParameters *params);
__global__ void reduce_sum(double *input, double *output, int len);
__global__ void reduce_sum_n(double *input, double* output, int ode_num, int total_len);


void *cpu_thread_display_output(void *params);

typedef struct {
    double* y;
    int total_size;
    int dimension;
    int stream_id;
} cpu_thread_params;

class GPUFlu{
public:
    explicit GPUFlu();
    ~GPUFlu();
    void set_parameters(GPUParameters* params);
    double rand_uniform(double range_from, double range_to);
    void run();
private:
    GPUParameters* params;
};

#endif
