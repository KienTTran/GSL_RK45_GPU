#include "gpu_rk45.h"

#define reduce_dim DIM * sizeof(float)
#define blockSize 1

namespace cg = cooperative_groups;

__device__ float getMax(float x, float y) {
    return (x > y) ? x : y;
}

__device__ float getSum(float x, float y) {
    return x + y;
}

__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] = sdata[tid] >= sdata[tid + 32] ? sdata[tid] : sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] = sdata[tid] >= sdata[tid + 16] ? sdata[tid] : sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] = sdata[tid] >= sdata[tid + 8] ? sdata[tid] : sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] = sdata[tid] >= sdata[tid + 4] ? sdata[tid] : sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] = sdata[tid] >= sdata[tid + 2] ? sdata[tid] : sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] = sdata[tid] >= sdata[tid + 1] ? sdata[tid] : sdata[tid + 1];
}

__global__ void reduce_max(float data[], float out[], unsigned int n) {
    __shared__ float sdata[reduce_dim];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
//    printf("[reduce_max] data[%d] = %f\n",tid,data[tid]);
    while (i < n) {
//        printf("[reduce_max] before compare tid = %d i = %d sdata[tid=%d] = %f data[i=%d] = %f data[i=%d+blockSize=1] = %f\n",tid,i,tid,sdata[tid],i,data[i],i,data[i+blockSize]);
//        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        sdata[tid] = getMax(sdata[tid], getMax(data[i],data[i+blockSize]));
//        printf("[reduce_max] after compare tid = %d i = %d data[%d] = %f sdata[%d] = %f\n\n",tid,i,i,data[i],tid,sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = sdata[tid] >= sdata[tid + 256] ? sdata[tid] : sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = sdata[tid] >= sdata[tid + 128] ? sdata[tid] : sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sdata[tid] >= sdata[tid + 64] ? sdata[tid] : sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) {
//        printf("[reduce_max] tid == %d i = %d tid = %d max = %f\n",tid,i,tid,sdata[tid]);
        out[blockIdx.x] = sdata[tid];
    }
    return;
}

__device__ void reduce_max_device(float data[], float out[], unsigned int n) {
    __shared__ float sdata[reduce_dim];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
//    printf("[reduce_max] data[%d] = %f\n",tid,data[tid]);
    while (i < n) {
//        printf("[reduce_max] before compare tid = %d i = %d sdata[tid=%d] = %f data[i=%d] = %f data[i=%d+blockSize=1] = %f\n",tid,i,tid,sdata[tid],i,data[i],i,data[i+blockSize]);
//        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        sdata[tid] = getMax(sdata[tid], getMax(data[i],data[i+blockSize]));
//        printf("[reduce_max] after compare tid = %d i = %d data[%d] = %f sdata[%d] = %f\n\n",tid,i,i,data[i],tid,sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = sdata[tid] >= sdata[tid + 256] ? sdata[tid] : sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = sdata[tid] >= sdata[tid + 128] ? sdata[tid] : sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sdata[tid] >= sdata[tid + 64] ? sdata[tid] : sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) {
//        printf("[reduce_max] tid == %d i = %d tid = %d max = %f\n",tid,i,tid,sdata[tid]);
        out[blockIdx.x] = sdata[tid];
    }
    return;
}

__device__ void reduce_sum_device(float data[], float out[], unsigned int n) {
    __shared__ float sdata[128];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    printf("[reduce_sum_device] data[%d] = %f\n",tid,data[tid]);
    while (i < n) {
        printf("[reduce_sum_device] before compare tid = %d i = %d sdata[tid=%d] = %f data[i=%d] = %f data[i=%d+blockSize=1] = %f\n",tid,i,tid,sdata[tid],i,data[i],i,data[i+blockSize]);
        sdata[tid] += data[i] + data[i+blockSize];
//        sdata[tid] = getSum(sdata[tid], getSum(data[i],data[i+blockSize]));
        printf("[reduce_sum_device] after compare tid = %d i = %d data[%d] = %f sdata[%d] = %f\n\n",tid,i,i,data[i],tid,sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = sdata[tid] >= sdata[tid + 256] ? sdata[tid] : sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = sdata[tid] >= sdata[tid + 128] ? sdata[tid] : sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sdata[tid] >= sdata[tid + 64] ? sdata[tid] : sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) {
        printf("[reduce_sum_device] tid == %d i = %d tid = %d sum = %f\n",tid,i,tid,sdata[tid]);
        out[blockIdx.x] = sdata[tid];
    }
    return;
}

__device__ void reduce_sum_device_2(float data[], float out[], unsigned int n)
{
    __shared__ float s_data[128];
    int tid = threadIdx.x;
    int index = tid + blockIdx.x*blockDim.x;
    s_data[tid] = 0.0;
    if (index < n){
        s_data[tid] = data[index];
    }
    __syncthreads();

    for (int s = 2; s <= blockDim.x; s = s * 2){
        if ((tid%s) == 0){
            s_data[tid] += s_data[tid + s / 2];
        }
        __syncthreads();
    }

    if (tid == 0){
//        printf("[reduce_sum_device_2] tid == %d i = %d tid = %d sum = %f\n",tid,index,tid,s_data[tid]);
        out[blockIdx.x] = s_data[tid];
    }
}

__device__ void reduce_max_device_2(float data[], float out[], unsigned int n)
{
    __shared__ float s_data[128];
    int tid = threadIdx.x;
    int index = tid + blockIdx.x*blockDim.x;
    s_data[tid] = 0.0;
    if (index < n){
        s_data[tid] = data[index];
    }
    __syncthreads();

    for (int s = 2; s <= blockDim.x; s = s * 2){
        if ((tid%s) == 0){
            s_data[tid] = s_data[tid] > s_data[tid + s / 2] ? s_data[tid] : s_data[tid + s / 2];
        }
        __syncthreads();
    }

    if (tid == 0){
//        printf("[reduce_sum_device_2] tid == %d i = %d tid = %d max = %f\n",tid,index,tid,s_data[tid]);
        out[blockIdx.x] = s_data[tid];
    }
}
