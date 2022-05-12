#ifndef GPU_FLU_H
#define GPU_FLU_H

#include "gpu_parameters.cuh"
#include "gpu_ode.cuh"
#include "gpu_mcmc.cuh"
#include "gpu_reduce.cuh"

class GPUFlu{
public:
    explicit GPUFlu();
    ~GPUFlu();
    void set_gpu_parameters(GPUParameters* gpu_params);
    void run();
    void init();
private:
    GPUParameters* gpu_params;
    FluParameters* flu_params;

    float all_ms; // elapsed time in milliseconds
    float transfer_h2d_ms;
    float compute_ms;
    float transfer_d2h_ms;
    float one_iter_ms;
    float one_mcmc_ms;
    float one_ode_ms;
    float one_stf_ms;
    float one_update_ms;

    cudaEvent_t start_event, stop_event;
    cudaEvent_t start_one_iter_event, stop_one_iter_event;
    cudaEvent_t start_one_ode_event, stop_one_ode_event;
    cudaEvent_t start_one_mcmc_event, stop_one_mcmc_event;
    cudaEvent_t start_one_update_event, stop_one_update_event;
    cudaEvent_t start_one_stf_event, stop_one_stf_event;
    cudaEvent_t start_event_all, stop_event_all;

    size_t ode_double_size;

    //temp pointers
    double **tmp_ptr = 0;

    /* stf_d - stf on device */
    double **stf_d = 0;
    size_t stf_d_size;

    /* y_ode_input_d - device */
    double **y_ode_input_d = 0;
    size_t y_ode_input_d_size;

    /* y_ode_output_d - device */
    double **y_ode_output_d = 0;
    size_t y_ode_output_d_size;

    /* y_data_input_d - device */
    double **y_data_input_d = 0;
    size_t y_data_input_d_size;

    /* y_agg_input_d - device */
    double **y_agg_input_d = 0;
    size_t y_agg_d_size;

    /* y_agg_output_d - device */
    double **y_agg_output_d = 0;

    //y_ode_output_h
    double **y_ode_output_h = 0;

    //y_output_agg_h
    double **y_output_agg_h = 0;

    /* dnorm 1 ode with padding - on host */
    int mcmc_dnorm_1_ode_padding_size;
    double *y_mcmc_dnorm_1_ode_h; /* 1 ODE no padding */
    int y_mcmc_dnorm_1_ode_h_size;
    double *y_mcmc_dnorm_1_ode_padding_h;/* 1 ODE with padding */
    int y_mcmc_dnorm_1_ode_padding_h_size;

    /* dnorm N ode with padding - on host */
    double *y_mcmc_dnorm_n_ode_padding_h;
    int y_mcmc_dnorm_n_ode_padding_h_size;

    /* dnorm N ode with padding - on device */
    double* y_mcmc_dnorm_n_ode_padding_d = 0;
    double* y_mcmc_dnorm_n_ode_padding_zero_d = 0;
    size_t y_mcmc_dnorm_n_ode_padding_d_size;

    /* gpu_params_d - on device */
    GPUParameters *gpu_params_d;

    /* flu_params_current_d - on device */
    FluParameters *flu_params_current_d;

    /* flu_params_new_d - on device */
    FluParameters *flu_params_new_d;

    /* curand_state_d - on device */
    curandState *curand_state_d;

    /* r_denom/r_num - on host */
    double *r_h = 0;

    /* r_denom - on device */
    double* r_denom_d = 0;

    /* r_num - on device */
    double* r_num_d = 0;

    int gpu_reduce_num_block;

};

#endif
