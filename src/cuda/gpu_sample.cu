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

//
// A function marked __global__
// runs on the GPU but can be called from
// the CPU.
//
// This function multiplies the elements of an array
// of ints by 2.
//
// The entire computation can be thought of as running
// with one thread per array element with blockIdx.x
// identifying the thread.
//
// The comparison i<N is because often it isn't convenient
// to have an exact 1-1 correspondence between threads
// and array elements. Not strictly necessary here.
//
// Note how we're mixing GPU and CPU code in the same source
// file. An alternative way to use CUDA is to keep
// C/C++ code separate from CUDA code and dynamically
// compile and load the CUDA code at runtime, a little
// like how you compile and load OpenGL shaders from
// C/C++ code.
//

#define N 1

__global__
void add(int *a, int *b) {
    int i = threadIdx.x;
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