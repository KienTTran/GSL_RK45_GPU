#include <cstdio>
#include <cstdlib>
#include <complex>
#include "cuComplex.h"

#include <stdio.h>

#include <stdio.h>

//
// Nearly minimal CUDA example.
// Compile with:
//
// nvcc -o example example.cu
//

#define N0 10

__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N0) {
        b[i] = 2*a[i];
    }
}

int test_cuda_0() {
    //
    // Create int arrays on the CPU.
    // ('h' stands for "host".)
    //
    int ha[N0], hb[N0];

    //
    // Create corresponding int arrays on the GPU.
    // ('d' stands for "device".)
    //
    int *da, *db;
    cudaMalloc((void **)&da, N0*sizeof(int));
    cudaMalloc((void **)&db, N0*sizeof(int));

    //
    // Initialise the input data on the CPU.
    //
    for (int i = 0; i<N0; ++i) {
        ha[i] = i;
    }

    //
    // Copy input data to array on GPU.
    //
    cudaMemcpy(da, ha, N0*sizeof(int), cudaMemcpyHostToDevice);

    //
    // Launch GPU code with N threads, one per
    // array element.
    //
    add<<<N0, 1>>>(da, db);

    //
    // Copy output array from GPU back to CPU.
    //
    cudaMemcpy(hb, db, N0*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<10; ++i) {
        printf("%d\n", hb[i]);
    }

    //
    // Free up the arrays on the GPU.
    //
    cudaFree(da);
    cudaFree(db);

    return 0;
}

#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int test_cuda_1()
{
    int N1 = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N1*sizeof(float));
    cudaMallocManaged(&y, N1*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N1; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    std::cout << "[TEST] Adding " << N1 << " elements on GPU"<< std::endl;
    // Run kernel on 1M elements on the GPU
    add<<<1, 256>>>(N1, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N1; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "[TEST] Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}

#define N2  (2)
#define M  (4)

typedef std::complex<float> T;

__global__ void print_device_matrix (cuComplex** mat)
{
    printf ("[TEST] Matrix on device:\n");
    for (int i = 0; i < N2; i++) {
        for (int j = 0; j < M; j++) {
            printf ("(%f, %f)  ", cuCrealf (mat[i][j]), cuCimagf (mat[i][j]));
        }
        printf ("\n");
    }
}

int test_cuda_2()
{
    /* allocate host "matrix" */
    T **mat = (T**)malloc (N2 * sizeof (mat[0]));
    for (int i = 0; i < N2; i++) {
        mat[i] = (T *)malloc (M * sizeof (mat[0][0]));
    }

    /* fill in host "matrix" */
    for (int i = 0; i < N2; i++) {
        for (int j = 0; j < M; j++) {
            mat[i][j] = T (float(i)+1, float(j)+1);
        }
    }

    /* print host "matrix" */
    printf ("[TEST] matrix on host:\n");
    for (int i = 0; i < N2; i++) {
        for (int j = 0; j < M; j++) {
            printf ("(%f, %f)  ", real(mat[i][j]), imag(mat[i][j]));
        }
        printf ("\n");
    }

    /* allocate device "matrix" */
    T **tmp = (T**)malloc (N2 * sizeof (tmp[0]));
    for (int i = 0; i < N2; i++) {
        cudaMalloc ((void **)&tmp[i], M * sizeof (tmp[0][0]));
    }
    cuComplex **matD = 0;
    cudaMalloc ((void **)&matD, N2 * sizeof (matD[0]));

    /* copy "matrix" from host to device */
    cudaMemcpy (matD, tmp, N2 * sizeof (matD[0]), cudaMemcpyHostToDevice);
    for (int i = 0; i < N2; i++) {
        cudaMemcpy (tmp[i], mat[i], M * sizeof (matD[0][0]), cudaMemcpyHostToDevice);
    }
    free (tmp);

    /* print device "matrix" */
    print_device_matrix<<<1,1>>> (matD);

    /* free host "matrix" */
    for (int i = 0; i < N2; i++) {
        free (mat[i]);
    }
    free (mat);

    /* free device "matrix" */
    tmp = (T**)malloc (N2 * sizeof (tmp[0]));
    cudaMemcpy (tmp, matD, N2 * sizeof (matD[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N2; i++) {
        cudaFree (tmp[i]);
    }
    free (tmp);
    cudaFree (matD);

    return EXIT_SUCCESS;
}