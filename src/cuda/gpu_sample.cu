#include <cstdlib>
#include <complex>
#include "cuComplex.h"
#include <stdio.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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
    gpuErrchk(cudaMalloc((void **)&da, N0*sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&db, N0*sizeof(int)));

    //
    // Initialise the input data on the CPU.
    //
    for (int i = 0; i<N0; ++i) {
        ha[i] = i;
    }

    //
    // Copy input data to array on GPU.
    //
    gpuErrchk(cudaMemcpy(da, ha, N0*sizeof(int), cudaMemcpyHostToDevice));

    //
    // Launch GPU code with N threads, one per
    // array element.
    //
    add<<<N0, 1>>>(da, db);

    //
    // Copy output array from GPU back to CPU.
    //
    gpuErrchk(cudaMemcpy(hb, db, N0*sizeof(int), cudaMemcpyDeviceToHost));

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
void add(int n, double *x, double *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int test_cuda_1()
{
    int N1 = 1<<20;
    double *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    gpuErrchk(cudaMallocManaged(&x, N1*sizeof(double)));
    gpuErrchk(cudaMallocManaged(&y, N1*sizeof(double)));

    // initialize x and y arrays on the host
    for (int i = 0; i < N1; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    std::cout << "[TEST] Adding " << N1 << " elements on GPU"<< std::endl;
    // Run kernel on 1M elements on the GPU
    add<<<1, 256>>>(N1, x, y);

    // Wait for GPU to finish before accessing on host
    gpuErrchk(cudaDeviceSynchronize());

    // Check for errors (all values should be 3.0f)
    double maxError = 0.0f;
    for (int i = 0; i < N1; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "[TEST] Max error: " << maxError << std::endl;

    // Free memory
    gpuErrchk(cudaFree(x));
    gpuErrchk(cudaFree(y));

    return 0;
}

#define N2  (2)
#define M  (4)

typedef std::complex<double> T;

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
            mat[i][j] = T (double(i)+1, double(j)+1);
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
        gpuErrchk(cudaMalloc ((void **)&tmp[i], M * sizeof (tmp[0][0])));
    }
    cuComplex **matD = 0;
    gpuErrchk(cudaMalloc ((void **)&matD, N2 * sizeof (matD[0])));

    /* copy "matrix" from host to device */
    gpuErrchk(cudaMemcpy (matD, tmp, N2 * sizeof (matD[0]), cudaMemcpyHostToDevice));
    for (int i = 0; i < N2; i++) {
        gpuErrchk(cudaMemcpy (tmp[i], mat[i], M * sizeof (matD[0][0]), cudaMemcpyHostToDevice));
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
    gpuErrchk(cudaMemcpy (tmp, matD, N2 * sizeof (matD[0]), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N2; i++) {
        cudaFree (tmp[i]);
    }
    free (tmp);
    gpuErrchk(cudaFree (matD));

    return EXIT_SUCCESS;
}