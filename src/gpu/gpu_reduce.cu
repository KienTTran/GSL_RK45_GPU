//
// Created by kient on 5/1/2022.
//
#include "gpu_ode_mcmc.h"

static const int NUM_ELEMENTS = 512;

__global__
void reduce_sum(double *input, double* output, int len)
{
    __shared__ double s_data[NUM_ELEMENTS];
    int tid = threadIdx.x;
    int index = tid + blockIdx.x*blockDim.x;
    s_data[tid] = 0.0;
    if (index < len){
        s_data[tid] = input[index];
    }
    __syncthreads();

    for (int s = 2; s <= blockDim.x; s = s * 2){
        if ((tid%s) == 0){
            s_data[tid] += s_data[tid + s / 2];
        }
        __syncthreads();
    }

    if (tid == 0){
//        printf("sum = %.5f\n",s_data[tid]);
        output[blockIdx.x] = s_data[tid];
    }
}

