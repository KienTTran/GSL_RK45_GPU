#include "gpu_functions.h"
#include "cpu_functions.h"

using namespace std;

int main()
{
//    int N = 1<<20;
//    float *x, *y;
//
//    // Allocate Unified Memory – accessible from CPU or GPU
//    cudaMallocManaged(&x, N*sizeof(float));
//    cudaMallocManaged(&y, N*sizeof(float));
//
//    // initialize x and y arrays on the host
//    for (int i = 0; i < N; i++) {
//        x[i] = 9.0f;
//        y[i] = 2.0f;
//    }
//
//    // Run kernel on 1M elements on the GPU
//    rk45_cuda(N, x, y);
//
//    // Wait for GPU to finish before accessing on host
//    cudaDeviceSynchronize();
//
//    // Check for errors (all values should be 3.0f)
//    float maxError = -1.0f;
//    for (int i = 0; i < N; i++){
////        printf("y[%d]: %.2f",i,y[i]);
//        maxError = fmax(maxError, fabs(y[i]-11.0f));
//    }
//    std::cout << "Max error: " << maxError << std::endl;
//
//    // Free memory
//    cudaFree(x);
//    cudaFree(y);

//    rk45_gsl_cpu_simulate();

//    double k = 5.0;
//    double *k_p = &k;
//    double k2 = *k_p;
//    std::cout << k << std::endl;
//    std::cout << &k << std::endl;
//    std::cout << k_p << std::endl;
//    std::cout << k2 << std::endl;
//    std::cout << &k2 << std::endl;

    rk45_gsl_gpu_simulate();

    return 0;

}