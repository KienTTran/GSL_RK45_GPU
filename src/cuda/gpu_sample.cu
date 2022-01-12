#include <cstdio>
#include <cstdlib>
#include <complex>
#include "cuComplex.h"

#define N  (2)
#define M  (4)

typedef std::complex<float> T;

__global__ void print_device_matrix (cuComplex** mat)
{
    printf ("matrix on device:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf ("(%f, %f)  ", cuCrealf (mat[i][j]), cuCimagf (mat[i][j]));
        }
        printf ("\n");
    }
}

int test_cuda()
{
    /* allocate host "matrix" */
    T **mat = (T**)malloc (N * sizeof (mat[0]));
    for (int i = 0; i < N; i++) {
        mat[i] = (T *)malloc (M * sizeof (mat[0][0]));
    }

    /* fill in host "matrix" */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mat[i][j] = T (float(i)+1, float(j)+1);
        }
    }

    /* print host "matrix" */
    printf ("matrix on host:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf ("(%f, %f)  ", real(mat[i][j]), imag(mat[i][j]));
        }
        printf ("\n");
    }

    /* allocate device "matrix" */
    T **tmp = (T**)malloc (N * sizeof (tmp[0]));
    for (int i = 0; i < N; i++) {
        cudaMalloc ((void **)&tmp[i], M * sizeof (tmp[0][0]));
    }
    cuComplex **matD = 0;
    cudaMalloc ((void **)&matD, N * sizeof (matD[0]));

    /* copy "matrix" from host to device */
    cudaMemcpy (matD, tmp, N * sizeof (matD[0]), cudaMemcpyHostToDevice);
    for (int i = 0; i < N; i++) {
        cudaMemcpy (tmp[i], mat[i], M * sizeof (matD[0][0]), cudaMemcpyHostToDevice);
    }
    free (tmp);

    /* print device "matrix" */
    print_device_matrix<<<1,1>>> (matD);

    /* free host "matrix" */
    for (int i = 0; i < N; i++) {
        free (mat[i]);
    }
    free (mat);

    /* free device "matrix" */
    tmp = (T**)malloc (N * sizeof (tmp[0]));
    cudaMemcpy (tmp, matD, N * sizeof (matD[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        cudaFree (tmp[i]);
    }
    free (tmp);
    cudaFree (matD);

    return EXIT_SUCCESS;
}