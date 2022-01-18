#include "gpu_rk45.h"
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

GPU_RK45::GPU_RK45(){
}

GPU_RK45::~GPU_RK45(){
}


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__device__
void gpu_adjust_h(double* y_d, double* y_err_d, double* dydt_out_d, double* max_in_d, double* max_out_d,
                   double &h, double h_0, int &adjustment_out, int final_step,
                  const int index, GPU_Parameters* params_d){
    /* adaptive adjustment */
    /* Available control object constructors.
     *
     * The standard control object is a four parameter heuristic
     * defined as follows:
     *    D0 = eps_abs + eps_rel * (a_y |y| + a_dydt h |y'|)
     *    D1 = |yerr|
     *    q  = consistency order of method (q=4 for 4(5) embedded RK)
     *    S  = safety factor (0.9 say)
     *
     *                      /  (D0/D1)^(1/(q+1))  D0 >= D1
     *    h_NEW = S h_OLD * |
     *                      \  (D0/D1)^(1/q)      D0 < D1
     *
     * This encompasses all the standard error scaling methods.
     *
     * The y method is the standard method with a_y=1, a_dydt=0.
     * The yp method is the standard method with a_y=0, a_dydt=1.
     */
    static double eps_abs = 1e-6;
    static double eps_rel = 0.0;
    static double a_y = 1.0;
    static double a_dydt = 0.0;
    static unsigned int ord = 5;
    const double S = 0.9;
    double h_old;

    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int index = index_gpu; index < params_d->dimension; index += stride)
    {
        if(final_step){
            h_old = h_0;
        }
        else{
            h_old = h;
        }

        //finding r_max
        double D0 = eps_rel * (a_y * fabs(y_d[index]) + a_dydt * fabs((h_old) * dydt_out_d[index])) + eps_abs;
        double r = fabs(y_err_d[index]) / fabs(D0);

//        if(index == 0 || index == params_d->dimension - 1) {
//            printf("  [gpu_adjust_h] Index = %d D0 = %.20f r = %.20f\n", index, D0, r);
//            printf("    IN y_d[%d] = %.20f\n",index,y_d[index]);
//            printf("    IN y_err_d[%d] = %.20f\n",index,y_err_d[index]);
//            printf("    IN dydt_out_d[%d] = %.20f\n",index,dydt_out_d[index]);
//            printf("    eps_rel[%d] = %.20f\n",index,eps_rel);
//            printf("    a_y[%d] = %.20f\n",index,a_y);
//            printf("    h_old[%d] = %.20f\n",index,h_old);
//            printf("    fabs((h_old) * dydt_out_d[%d])) = %.20f\n",index,fabs((h_old) * dydt_out_d[index]));
//            printf("    eps_abs[%d] = %.20f\n",index,eps_abs);
//            printf("    D0[%d] = %.20f\n",index,D0);
//            printf("    r[%d] = %.20f\n",index,r);
//        }

        max_in_d[index] = r;
        __syncthreads();

//        if(index == 0 || index == params_d->dimension - 1) {
//            printf("    Index = %d max_in_d[%d] =  %.20f\n", index, index, max_in_d[index]);
//        }

//        reduce_max_0(max_in_d,max_out_d,DIM);
//        __syncthreads();

        double r_max = max_out_d[0];
        r_max = 0.4;
        __syncthreads();

//        if(index == 0 || index == params_d->dimension - 1) {
//            printf("    Index = %d r_max =  %.20f\n", index, r_max);
//        }

        if (r_max > 1.1) {
            /* decrease step, no more than factor of 5, but a fraction S more
               than scaling suggests (for better accuracy) */
            double r = S / pow(r_max, 1.0 / ord);

            if (r < 0.2)
                r = 0.2;
            h = r * (h_old);

//            if(index == 0 || index == params_d->dimension - 1) {
//                printf("    Index = %d decrease by %.20f, h_old is %.20f new h is %.20f\n", index, r, h_old, h);
//            }
            adjustment_out = -1;
        } else if (r_max < 0.5) {
            /* increase step, no more than factor of 5 */
            double r = S / pow(r_max, 1.0 / (ord + 1.0));

            if (r > 5.0)
                r = 5.0;

            if (r < 1.0)  /* don't allow any decrease caused by S<1 */
                r = 1.0;

            h = r * (h_old);

//            if(index == 0 || index == params_d->dimension - 1) {
//                printf("    Index = %d increase by %.20f, h_old is %.20f new h is %.20f\n", index, r, h_old, h);
//            }
            adjustment_out = 1;
        } else {
            /* no change */
//            if(index == 0 || index == params_d->dimension - 1) {
//                printf("    Index = %d no change\n", index);
//            }
            adjustment_out = 0;
        }
//        if(index == 0 || index == params_d->dimension - 1) {
//            printf("  [gpu_adjust_h] Index = %d end\n", index);
//        }
    }
    return;
}

__device__
void gpu_step_apply(double t, double &h, double* y_d,
                    double* y_tmp_d, double* y_err_d, double* dydt_in, double* dydt_out_d,
                    double* k1_d, double* k2_d, double* k3_d, double* k4_d, double* k5_d, double* k6_d,
                    const int index, GPU_Parameters* params_d){
    static const double ah[] = { 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0 };
    static const double b3[] = { 3.0/32.0, 9.0/32.0 };
    static const double b4[] = { 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0};
    static const double b5[] = { 8341.0/4104.0, -32832.0/4104.0, 29440.0/4104.0, -845.0/4104.0};
    static const double b6[] = { -6080.0/20520.0, 41040.0/20520.0, -28352.0/20520.0, 9295.0/20520.0, -5643.0/20520.0};

    static const double c1 = 902880.0/7618050.0;
    static const double c3 = 3953664.0/7618050.0;
    static const double c4 = 3855735.0/7618050.0;
    static const double c5 = -1371249.0/7618050.0;
    static const double c6 = 277020.0/7618050.0;

    static const double ec[] = { 0.0,
                                 1.0 / 360.0,
                                 0.0,
                                 -128.0 / 4275.0,
                                 -2197.0 / 75240.0,
                                 1.0 / 50.0,
                                 2.0 / 55.0
    };

    y_tmp_d[index] = 0.0;
    y_err_d[index] = 0.0;
    dydt_out_d[index] = 0.0;
    k1_d[index] = 0.0;
    k2_d[index] = 0.0;
    k3_d[index] = 0.0;
    k4_d[index] = 0.0;
    k5_d[index] = 0.0;
    k6_d[index] = 0.0;
//    if(index == 0 || index == params_d->dimension - 1) {
//        printf("  [gpu_step_apply] Index = %d t = %.20f h = %.20f start\n",index,t,h);
//        printf("    IN y_d[%d] = %.20f\n",index,y_d[index]);
//        printf("    IN y_err_d[%d] = %.20f\n",index,y_err_d[index]);
//        printf("    IN dydt_out_d[%d] = %.20f\n",index,dydt_out_d[index]);
//    }

    /* k1_d */
    if (dydt_in != NULL)
    {
        memcpy(k1_d, dydt_in, params_d->dimension * sizeof(double));
//        if(index == 0 || index == params_d->dimension - 1) {
//            printf("    dydt_in != NULL\n");
//            printf("    memcpy(k1, dydt_in, dim)\n");
//        }
    }
    else {
        gpu_func_test(t, y_d, k1_d, params_d);
//        if(index == 0 || index == params_d->dimension - 1) {
//            printf("    dydt_in == NULL\n");
//            printf("    gpu_func_test(t,y_d,k1_d,params_d)\n");
//        }
    }
//    if(index == 0 || index == params_d->dimension - 1) {
//        printf("    k1_d[%d] = %.20f\n", index, k1_d[index]);
//    }
    y_tmp_d[index] = y_d[index] +  ah[0] * h * k1_d[index];
    __syncthreads();
    /* k2_d */
    gpu_func_test(t + ah[0] * h, y_tmp_d, k2_d, params_d);
//    if(index == 0 || index == params_d->dimension - 1) {
//            printf("    k2_d[%d] = %.20f\n",index,k2_d[index]);
//    }
    y_tmp_d[index] = y_d[index] + h * (b3[0] * k1_d[index] + b3[1] * k2_d[index]);
    __syncthreads();
    /* k3_d */
    gpu_func_test(t + ah[1] * h, y_tmp_d, k3_d, params_d);
//    if(index == 0 || index == params_d->dimension - 1) {
//            printf("    k3_d[%d] = %.20f\n",index,k3_d[index]);
//    }
    y_tmp_d[index] = y_d[index] + h * (b4[0] * k1_d[index] + b4[1] * k2_d[index] + b4[2] * k3_d[index]);
    __syncthreads();
    /* k4_d */
    gpu_func_test(t + ah[2] * h, y_tmp_d, k4_d, params_d);
//    if(index == 0 || index == params_d->dimension - 1) {
//            printf("    k4_d[%d] = %.20f\n",index,k4_d[index]);
//    }
    y_tmp_d[index] = y_d[index] + h * (b5[0] * k1_d[index] + b5[1] * k2_d[index] + b5[2] * k3_d[index] + b5[3] * k4_d[index]);
    __syncthreads();
    /* k5_d */
    gpu_func_test(t + ah[3] * h, y_tmp_d, k5_d, params_d);
//    if(index == 0 || index == params_d->dimension - 1) {
//            printf("    k5_d[%d] = %.20f\n",index,k5_d[index]);
//    }
    y_tmp_d[index] = y_d[index] + h * (b6[0] * k1_d[index] + b6[1] * k2_d[index] + b6[2] * k3_d[index] + b6[3] * k4_d[index] + b6[4] * k5_d[index]);
    __syncthreads();
    /* k6_d */
    gpu_func_test(t + ah[4] * h, y_tmp_d, k6_d,params_d);
    /* final sum */
//    if(index == 0 || index == params_d->dimension - 1) {
//        printf("    k6_d[%d] = %.20f\n", index, k6_d[index]);
//    }
    const double d_i = c1 * k1_d[index] + c3 * k3_d[index] + c4 * k4_d[index] + c5 * k5_d[index] + c6 * k6_d[index];
    y_d[index] += h * d_i;
    __syncthreads();
    /* Derivatives at output */
    gpu_func_test(t + h, y_d, dydt_out_d,params_d);
    /* difference between 4th and 5th order */
    y_err_d[index] = h * (ec[1] * k1_d[index] + ec[3] * k3_d[index] + ec[4] * k4_d[index] + ec[5] * k5_d[index] + ec[6] * k6_d[index]);
    __syncthreads();
//    if(index == 0 || index == params_d->dimension - 1) {
//        printf("    OUT y_d[%d] = %.20f\n",index,y_d[index]);
//        printf("    OUT y_err_d[%d] = %.20f\n",index,y_err_d[index]);
//        printf("    OUT dydt_out_d[%d] = %.20f\n",index,dydt_out_d[index]);
//        printf("  [gpu_step_apply] Index = %d t = %.20f h = %.20f end\n",index,t,h);
//    }
    return;
}

__global__
void gpu_evolve_apply(double* t_start_d, double* t_target_d, double* h_d, double* y_d, GPU_Parameters* params_d){
    //<<<num_blocks, block_size>>>
    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    __shared__ double y_0_d[DIM];
    __shared__ double y_tmp_d[DIM];
    __shared__ double y_err_d[DIM];
    __shared__ double dydt_in_d[DIM];
    __shared__ double dydt_out_d[DIM];
    __shared__ double k1_d[DIM];
    __shared__ double k2_d[DIM];
    __shared__ double k3_d[DIM];
    __shared__ double k4_d[DIM];
    __shared__ double k5_d[DIM];
    __shared__ double k6_d[DIM];
    __shared__ double max_in_d[DIM];
    __shared__ double max_out_d[DIM];
    for(int index = index_gpu; index < params_d->dimension; index += stride){
        y_0_d[index] = 0.0;
        y_tmp_d[index] = 0.0;
        y_err_d[index] = 0.0;
        dydt_in_d[index] = 0.0;
        dydt_out_d[index] = 0.0;
        k1_d[index] = 0.0;
        k2_d[index] = 0.0;
        k3_d[index] = 0.0;
        k4_d[index] = 0.0;
        k5_d[index] = 0.0;
        k6_d[index] = 0.0;
        max_in_d[index] = 0.0;
        max_out_d[index] = 0.0;
        while(t_start_d[index] < t_target_d[index]){
//            if(index == 0 || index == params_d->dimension - 1) {
//                printf("[evolve apply] Index = %d t = %.20f h = %.20f start one day\n", index, t_start_d, h_d[index]);
//            }
            double t = t_start_d[index];
            double t1 = t + 1.0;
            double h = h_d[index];
            while(t < t1)
            {
                const double t_0 = t;
                double h_0 = h;
                double dt = t1 - t_0;
                int final_step = 0;
                int adjustment_out = 999;
                y_0_d[index] = y_d[index];
//                if(index == 0 || index == params_d->dimension - 1) {
//                    printf("[evolve apply] Index = %d t = %.20f t_0 = %.20f h = %.20f dt = %.20f start one iteration\n", index, t, t_0, h,dt);
//                }
//                if(index == 0 || index == params_d->dimension - 1) {
//                    printf("[evolve apply] Use_dydt_in\n");
//                }
                memcpy(y_0_d,y_d, params_d->dimension * sizeof(double));//
                gpu_func_test(t_0, y_d, dydt_in_d, params_d);
                while(true)
                {
                    if ((dt >= 0.0 && h_0 > dt) || (dt < 0.0 && h_0 < dt)) {
                        h_0 = dt;
                        final_step = 1;
                    } else {
                        final_step = 0;
                    }
                    gpu_step_apply(t_0, h_0, y_d,
                                   y_tmp_d, y_err_d, dydt_in_d, dydt_out_d,
                                   k1_d, k2_d, k3_d, k4_d, k5_d, k6_d,
                                   index, params_d);
                    if (final_step) {
                        t = t1;
                    } else {
                        t = t_0 + h_0;
                    }
                    double h_old = h_0;
                    gpu_adjust_h(y_d, y_err_d, dydt_out_d, max_in_d, max_out_d,
                                 h, h_0, adjustment_out, final_step,
                                 index, params_d);
                    //Extra step to get data from h
                    h_0 = h;
                    if (adjustment_out == -1)
                    {
                        double t_curr = (t);
                        double t_next = (t) + h_0;
                        if (fabs(h_0) < fabs(h_old) && t_next != t_curr) {
                            /* Step was decreased. Undo step, and try again with new h0. */
//                            if(index == 0 || index == params_d->dimension - 1){
//                                printf("[evolve apply] index = %d step decreased, y = y0\n", index);
//                            }
                            y_d[index] = y_0_d[index];
//                                memcpy(y_d,y_0_d, params_d->dimension * sizeof(double));
                        } else {
//                            if(index == 0 || index == params_d->dimension - 1){
//                                printf("[evolve apply] index = %d step decreased h_0 = h_old\n", index);
//                            }
                            h_0 = h_old; /* keep current step size */
                            break;
                        }
                    }
                    else{
//                        if(index == 0 || index == params_d->dimension - 1){
//                            printf("[evolve apply] index = %d step increased or no change\n", index);
//                        }
                        break;
                    }
                }
                h = h_0;  /* suggest step size for next time-step */
//                if(index == 0 || index == params_d->dimension - 1) {
//                    printf("[evolve apply] Index = %d t = %.20f t_0 = %.20f h = %.20f dt = %.20f end one iteration\n", index, t, t_0, h,dt);
//                        printf("    y_d[%d] = %.20f\n",index,y_d[index]);
//                        if(index == 0) printf("\n");
//                }
//                if(index == 0 || index == params_d->dimension - 1) {
//                    printf("[gpu_evolve_apply] Index = %d t = %.20f h = %.20f end\n", index, t, h);
//                }
//                /* Test */
//                t += h_d[index];
            }
//            if(index == 0 || index == params_d->dimension - 1) {
//                printf("[evolve apply] Index = %d t = %.20f h = %.20f end one day\n", index, t_start_d, h_d);
//            }
            t_start_d[index] += 1.0;
        }
    }
    return;
}

void GPU_RK45::run(int argc, char** argv){
    auto start = std::chrono::high_resolution_clock::now();
    GPU_Parameters* params = new GPU_Parameters;
    params->number_of_ode = 1;
    params->dimension = DIM;
    params->t_target_initial = NUMDAYSOUTPUT;
    params->t0_initial = 0.0;
    params->h_initial = 1e-6;
    params->display_number = 1;
//    params->initTestPen(argc,argv);
    params->initTestFlu(argc,argv);

    double *y_d;
    gpuErrchk(cudaMalloc ((void **)&y_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    double *t_0_d;
    gpuErrchk(cudaMalloc ((void **)&t_0_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (t_0_d, params->t0, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    double *t_target_d;
    gpuErrchk(cudaMalloc ((void **)&t_target_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (t_target_d, params->t_target, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    double *h_d;
    gpuErrchk(cudaMalloc ((void **)&h_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (h_d, params->h, params->dimension * sizeof (double), cudaMemcpyHostToDevice));

    GPU_Parameters* params_d;
    gpuErrchk(cudaMalloc((void **)&params_d, sizeof(GPU_Parameters)));
    gpuErrchk(cudaMemcpy(params_d, params, sizeof(GPU_Parameters), cudaMemcpyHostToDevice));

    int num_SMs;
    gpuErrchk(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
//    int numBlocks = 32*num_SMs; //multiple of 32
    int block_size = 128; //max is 1024
    int num_blocks = (params->dimension + block_size - 1) / block_size;
    params->block_size = block_size;
    params->num_blocks = num_blocks;
    printf("[GSL GPU] SMs = %d block_size = %d num_blocks = %d\n",num_SMs,block_size,num_blocks);
    dim3 dimBlock(block_size, block_size); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
    dim3 dimGrid(num_blocks, num_blocks); // 1*1 blocks in a grid

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %lld micro seconds which is %.20f seconds\n",duration.count(),(duration.count()/1e6));

    cudaFuncSetCacheConfig(gpu_evolve_apply, cudaFuncCachePreferShared);

    start = std::chrono::high_resolution_clock::now();
    gpu_evolve_apply<<<params->num_blocks, params->block_size>>>(t_0_d, t_target_d, h_d, y_d, params_d);
    gpuErrchk(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for compute %d ODE with %d parameters, step %.20f in %.20f days on GPU: %lld micro seconds which is %.20f seconds\n",params->number_of_ode,params->dimension,params->h_initial,params->t_target_initial,duration.count(),(duration.count()/1e6));

    gpuErrchk(cudaMemcpy (params->y, y_d, params->dimension * sizeof (double), cudaMemcpyDeviceToHost));
    printf("Display on Host\n");
    for(int i = 0; i < params->dimension; i++){
        printf("  y[%d] = %.20f\n",i,params->y[i]);
    }
    return;
}