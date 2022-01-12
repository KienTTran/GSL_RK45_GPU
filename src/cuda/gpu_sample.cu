#include <cstdio>
#include <cstdlib>
#include <complex>
#include "cuComplex.h"

#include <stdio.h>

//
// Nearly minimal CUDA example.
// Compile with:
//
// nvcc -o example example.cu
//


#include <iostream>
#include <math.h>

#define N 100

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int test_cuda_1()
{
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

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
    printf ("matrix on device:\n");
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
    printf ("matrix on host:\n");
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