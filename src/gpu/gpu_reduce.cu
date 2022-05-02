//
// Created by kient on 5/1/2022.
//
#include "gpu_rk45.h"
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static const int NUM_ELEMENTS = 512;

//__global__
//void reduce_sum(double *input, double *output, int len) {
//    int index = threadIdx.x + blockIdx.x*blockDim.x;
//    const int gridSize = blockDim.x*gridDim.x;
//    int parallelsum = 0;
//    for (int i = index; i < arraySize; i += gridSize)
//        parallelsum += input[i];
//    __shared__ double data[NUM_ELEMENTS];
//    data[threadIdx.x] = parallelsum;
//    __syncthreads();
//    for (int size = blockDim.x/2; size>0; size/=2) {
//        if (threadIdx.x<size)
//            data[threadIdx.x] += data[threadIdx.x+size];
//        __syncthreads();
//    }
//    if (threadIdx.x == 0) {
//        output[blockIdx.x] = data[0];
//    }
//}

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
        printf("sum = %.5f\n",s_data[tid]);
        output[blockIdx.x] = s_data[tid];
    }
}

