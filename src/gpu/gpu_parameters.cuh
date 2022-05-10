//
// Created by kient on 1/12/2022.
//

#ifndef RK45_CUDA_GPU_PARAMETERS_H
#define RK45_CUDA_GPU_PARAMETERS_H
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>
#include <random>
#include <iostream>
#include <chrono>
#include "flu_parameters.cuh"
#include "../flu_default_params.h"
#include "../csv/csv_data.h"

class GPUParameters {
public:
    __device__ __host__ explicit GPUParameters();
    ~GPUParameters();
    int num_blocks;
    int block_size;
    int mcmc_loop;
    int ode_output_day;
    int ode_number;
    int ode_dimension;
    int display_dimension;
    int agg_dimension;
    int data_dimension;
    CSVParameters data_params;
    int display_number;
    double t_target;
    double t0;
    double h;
    double step;
    double** y_ode_input;
    double** y_data_input;
    double** y_ode_output;
    double** y_agg;
    double** stf;
    void init(FluParameters* flu_params);
private:
};


#endif //RK45_CUDA_GPU_PARAMETERS_H
