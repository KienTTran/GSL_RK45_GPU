//
// Created by kient on 5/1/2022.
//
#include "gpu_flu.cuh"
#include "gpu_reduce.cuh"

__global__
void reduce_sum(double *input, double* output, int len)
{
    __shared__ double s_data[GPU_REDUCE_THREADS];
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
//        printf("sum1 = %.5f\n",s_data[tid]);
        output[blockIdx.x] = s_data[tid];
    }
}


__global__
void reduce_sum_padding(double *input, double* output, GPUParameters* gpu_params_d, int total_len)
{
    __shared__ double s_data[GPU_REDUCE_THREADS];
    int tid = threadIdx.x;
    int index = tid + blockIdx.x*blockDim.x;
    s_data[tid] = 0.0;

    if (index < total_len){
//        printf("tid %d index %d ODE[%d][%d] = %.1f\n",tid,index,ode_index,line_index,input[index]);
        s_data[tid] = input[index];
    }
    __syncthreads();

    for (int s = 2; s <= blockDim.x; s = s * 2){
        if ((tid%s) == 0){
//            if(s_data[tid] != 0.0 && s_data[tid + s / 2] != 0.0){
//                printf("s = %d tid = %d index = %d s_data[%d](%.1f) = s_data[%d](%.1f) + s_data[%d + %d / 2 = %d](%.1f) = %.1f\n",
//                       s,tid,index,tid,s_data[tid],tid,s_data[tid],tid,s,tid + s / 2,s_data[tid + s / 2],s_data[tid] + s_data[tid + s / 2]);
//            }
            s_data[tid] += s_data[tid + s / 2];
        }
        __syncthreads();
    }

    int ode_len = total_len / gpu_params_d->ode_number;
    int ode_index = index / ode_len;

    if (tid == 0){
        output[blockIdx.x + ode_index] = s_data[tid];
//        printf("sum1 = %.1f\n",output[blockIdx.x + ode_len]);
    }
}
