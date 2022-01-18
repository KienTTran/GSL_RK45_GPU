#include "gpu_rk45.h"

#define reduce_dim DIM * sizeof(double)
#define blockSize 1

namespace cg = cooperative_groups;

__device__ double getMax(double x, double y) {
    return (x > y) ? x : y;
}

__device__ double getSum(double x, double y) {
    return x + y;
}

__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] = sdata[tid] >= sdata[tid + 32] ? sdata[tid] : sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] = sdata[tid] >= sdata[tid + 16] ? sdata[tid] : sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] = sdata[tid] >= sdata[tid + 8] ? sdata[tid] : sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] = sdata[tid] >= sdata[tid + 4] ? sdata[tid] : sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] = sdata[tid] >= sdata[tid + 2] ? sdata[tid] : sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] = sdata[tid] >= sdata[tid + 1] ? sdata[tid] : sdata[tid + 1];
}

__device__ void reduce_max_0_dynamic(double* sdata, double *data, double *out, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
//        printf("[reduce_max_6] before compare tid = %d i = %d sdata[tid=%d] = %f data[i=%d] = %f data[i=%d+blockSize=1] = %f\n",tid,i,tid,sdata[tid],i,data[i],i,data[i+blockSize]);
//        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        sdata[tid] = getMax(sdata[tid], getMax(data[i],data[i+blockSize]));
//        printf("[reduce_max_6] after compare tid = %d i = %d data[%d] = %f sdata[%d] = %f\n\n",tid,i,i,data[i],tid,sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = sdata[tid] >= sdata[tid + 256] ? sdata[tid] : sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = sdata[tid] >= sdata[tid + 128] ? sdata[tid] : sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sdata[tid] >= sdata[tid + 64] ? sdata[tid] : sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) {
//        printf("[reduce_max_n] tid == %d Index = %d i = %d tid = %d max = %f\n",n,tid,i,tid,sdata[tid]);
        out[blockIdx.x] = sdata[tid];
    }
    return;
}

__device__ void reduce_max_n(double *data, double *out, unsigned int n) {
    __shared__ double sdata[reduce_dim];
    unsigned int tid = threadIdx.x + n;
    unsigned int i = blockIdx.x*(blockSize*2) + tid - n;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
//        printf("[reduce_max_6] before compare tid = %d i = %d sdata[tid=%d] = %f data[i=%d] = %f data[i=%d+blockSize=1] = %f\n",tid,i,tid,sdata[tid],i,data[i],i,data[i+blockSize]);
//        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        sdata[tid] = getMax(sdata[tid], getMax(data[i],data[i+blockSize]));
//        printf("[reduce_max_6] after compare tid = %d i = %d data[%d] = %f sdata[%d] = %f\n\n",tid,i,i,data[i],tid,sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = sdata[tid] >= sdata[tid + 256] ? sdata[tid] : sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = sdata[tid] >= sdata[tid + 128] ? sdata[tid] : sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sdata[tid] >= sdata[tid + 64] ? sdata[tid] : sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == n) {
//        printf("[reduce_max_n] tid == %d Index = %d i = %d tid = %d max = %f\n",n,tid,i,tid,sdata[tid]);
        out[blockIdx.x] = sdata[tid];
    }
    return;
}

__device__ void reduce_max_0(double *data, double *out, unsigned int n) {
    __shared__ double sdata[reduce_dim];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
//        printf("[reduce_max_6] before compare tid = %d i = %d sdata[tid=%d] = %f data[i=%d] = %f data[i=%d+blockSize=1] = %f\n",tid,i,tid,sdata[tid],i,data[i],i,data[i+blockSize]);
//        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        sdata[tid] = getMax(sdata[tid], getMax(data[i],data[i+blockSize]));
//        printf("[reduce_max_6] after compare tid = %d i = %d data[%d] = %f sdata[%d] = %f\n\n",tid,i,i,data[i],tid,sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = sdata[tid] >= sdata[tid + 256] ? sdata[tid] : sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = sdata[tid] >= sdata[tid + 128] ? sdata[tid] : sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sdata[tid] >= sdata[tid + 64] ? sdata[tid] : sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) {
//        printf("[reduce_max_0] tid == %d Index = %d i = %d tid = %d max = %f\n",n,tid,i,tid,sdata[tid]);
        out[blockIdx.x] = sdata[tid];
    }
    return;
}

__device__ void reduce_sum_n(double *data, double *out, unsigned int n) {
    __shared__ double sdata[reduce_dim];
    unsigned int tid = threadIdx.x + n;
    unsigned int i = blockIdx.x*(blockSize*2) + tid - n;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
//        printf("[reduce_sum_6] before compare tid = %d i = %d sdata[tid=%d] = %f data[i=%d] = %f data[i=%d+blockSize=1] = %f\n",tid,i,tid,sdata[tid],i,data[i],i,data[i+blockSize]);
        sdata[tid] += data[i] + data[i+blockSize];
//        printf("[reduce_sum_6] after compare tid = %d i = %d data[%d] = %f sdata[%d] = %f\n\n",tid,i,i,data[i],tid,sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = sdata[tid] >= sdata[tid + 256] ? sdata[tid] : sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = sdata[tid] >= sdata[tid + 128] ? sdata[tid] : sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sdata[tid] >= sdata[tid + 64] ? sdata[tid] : sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == n) {
//        printf("[reduce_sum_n] tid == %d Index = %d i = %d tid = %d sum = %f\n",n,tid,i,tid,sdata[tid]);
        out[blockIdx.x] = sdata[tid];
    }
    return;
}

__device__ void reduce_sum_0(double *data, double *out, unsigned int n) {
    __shared__ double sdata[reduce_dim];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
//        printf("[reduce_sum_6] before compare tid = %d i = %d sdata[tid=%d] = %f data[i=%d] = %f data[i=%d+blockSize=1] = %f\n",tid,i,tid,sdata[tid],i,data[i],i,data[i+blockSize]);
        sdata[tid] += data[i] + data[i+blockSize];
//        printf("[reduce_sum_6] after compare tid = %d i = %d data[%d] = %f sdata[%d] = %f\n\n",tid,i,i,data[i],tid,sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = sdata[tid] >= sdata[tid + 256] ? sdata[tid] : sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = sdata[tid] >= sdata[tid + 128] ? sdata[tid] : sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sdata[tid] >= sdata[tid + 64] ? sdata[tid] : sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) {
//        printf("[reduce_sum_0] tid == %d Index = %d i = %d tid = %d sum = %f\n",n,tid,i,tid,sdata[tid]);
        out[blockIdx.x] = sdata[tid];
    }
    return;
}

__device__ void test_reduce_sum_max(double* data, double* out, unsigned n){

    /* TEST */
//    reduce_max_0(data,out,DIM);
//    printf("max_0 = %f\n",out[0]);
//    reduce_sum_0(data,out,DIM);
//    printf("sum_0 = %f\n",out[0]);
//    reduce_max_n(data,out,DIM);
//    printf("max_n = %f\n",out[0]);
//    reduce_sum_n(data,out,DIM);
//    printf("sum_n = %f\n",out[0]);
    return;
}