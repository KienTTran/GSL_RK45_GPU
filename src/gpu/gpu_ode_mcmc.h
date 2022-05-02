#ifndef GPU_FUNCTION_H
#define GPU_FUNCTION_H

#include "gpu_parameters.h"
__device__ void gpu_func_test(double t, const double y[], double f[], int index, GPUParameters* gpu_params);
__device__ double seasonal_transmission_factor(GPUParameters* gpu_params, double t);
__device__ double pop_sum( double yy[] );
__global__ void reduce_sum(double *data, double *out, int len);
__global__ void solve_ode(double *y_ode_input_d[], double *y_ode_output_d[], double *y_ode_agg_d[], GPUParameters *params);
__global__ void mcmc_dnorm(double *y_data_input_d[], double *y_ode_agg_d[], double* y_mcmc_dnorm_d[], GPUParameters *params);
__global__ void mcmc_get_dnorm_outputs_1d(double* d_mcmc_dnorm_h1_b_h3_d[], double d_mcmc_dnorm_h1_1d_d[], double d_mcmc_dnorm_b_1d_d[], double d_mcmc_dnorm_h3_1d_d[], GPUParameters* params);

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
    void run();
private:
    GPUParameters* params;
};

#endif
