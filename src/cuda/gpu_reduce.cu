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

__global__ void reduce_max(double data[], double out[], unsigned int n) {
    __shared__ double sdata[reduce_dim];
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
