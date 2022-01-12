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

#define N 10

__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}

int test_cuda_1() {
    //
    // Create int arrays on the CPU.
    // ('h' stands for "host".)
    //
    int ha[N], hb[N];

    //
    // Create corresponding int arrays on the GPU.
    // ('d' stands for "device".)
    //
    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    //
    // Initialise the input data on the CPU.
    //
    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }

    //
    // Copy input data to array on GPU.
    //
    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

    //
    // Launch GPU code with N threads, one per
    // array element.
    //
    add<<<N, 1>>>(da, db);

    //
    // Copy output array from GPU back to CPU.
    //
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        printf("%d\n", hb[i]);
    }

    //
    // Free up the arrays on the GPU.
    //
    cudaFree(da);
    cudaFree(db);

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