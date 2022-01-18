#include "gpu_rk45.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__device__ double getMax2(double x, double y) {
    return (x > y) ? x : y;
}

GPU_RK45::GPU_RK45(){
    params = new GPU_Parameters();
}

GPU_RK45::~GPU_RK45(){
    params = nullptr;
}

void GPU_RK45::setParameters(GPU_Parameters* params_) {
    params = &(*params_);
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
void rk45_gpu_adjust_h(double y[], double y_err[], double dydt_out[],
                       double &h, double h_0, int &adjustment_out, int final_step,
                       double* r, double* D0, double* r_max,
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
    if(final_step){
        h_old = h_0;
    }
    else{
        h_old = h;
    }

//    printf("    [adjust h] index = %d begin\n",index);
//    printf("      y[%d] = %.20f\n",index,y[index]);
//    printf("      y_err[%d] = %.20f\n",index,y_err[index]);
//    printf("      dydt_out[%d] = %.20f\n",index,dydt_out[index]);

    r_max[index] = 2.2250738585072014e-308;
    D0[index] = eps_rel * (a_y * fabs(y[index]) + a_dydt * fabs((h_old) * dydt_out[index])) + eps_abs;
    r[index]  = fabs(y_err[index]) / fabs(D0[index]);

//    printf("      index = %d D0[%d] = %.20f\n",index,index,D0[index]);
//    printf("      index = %d r[%d] = %.20f\n",index,index,D0[index]);

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    for (int i = 0; i < params_d->dimension; i++)
    {
//        const double D0 = eps_rel * (a_y * fabs(y[i]) + a_dydt * fabs((h_old) * dydt_out[i])) + eps_abs;
//        const double r  = fabs(y_err[i]) / fabs(D0);
//        printf("      index = %d i = %d compare r[%d] = %.20f r_max = %.20f\n",index,i,i,r[i],r_max[index]);
        r_max[index] = max(r[i], r_max[index]);
    }
    block.sync();
//    printf("      r_max = %.20f\n",r_max[index]);

    if (r_max[index] > 1.1) {
        /* decrease step, no more than factor of 5, but a fraction S more
           than scaling suggests (for better accuracy) */
        double r = S / pow(r_max[index], 1.0 / ord);

        if (r < 0.2)
            r = 0.2;
        h = r * (h_old);

//        printf("      index = %d decrease by %.20f, h_old is %.20f new h is %.20f\n",index, r, h_old, h);
        adjustment_out = -1;
    } else if (r_max[index] < 0.5) {
        /* increase step, no more than factor of 5 */
        double r = S / pow(r_max[index], 1.0 / (ord + 1.0));

        if (r > 5.0)
            r = 5.0;

        if (r < 1.0)  /* don't allow any decrease caused by S<1 */
            r = 1.0;

        h = r * (h_old);

//        printf("      index = %d increase by %.20f, h_old is %.20f new h is %.20f\n",index, r, h_old, h);
        adjustment_out = 1;
    } else {
        /* no change */
//        printf("      index = %d no change\n",index);
        adjustment_out = 0;
    }
//    printf("    [adjust h] index = %d end\n",index);
    return;
}

__device__
void rk45_gpu_step_apply(double t, double h,
                         double y[], double y_tmp[], double y_err[], double dydt_in[], double dydt_out[],
                         double k1[], double k2[], double k3[], double k4[], double k5[], double k6[],
                         const int index, GPU_Parameters* params_d)
{
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

//    printf("    [step apply] index = %d start\n",index);
//    printf("      t = %.20f h = %.20f\n",t,h);

    y_tmp[index] = 0.0;
    y_err[index] = 0.0;
    dydt_out[index] = 0.0;
    k1[index] = 0.0;
    k2[index] = 0.0;
    k3[index] = 0.0;
    k4[index] = 0.0;
    k5[index] = 0.0;
    k6[index] = 0.0;

//    printf("      y[%d] = %.20f\n",index,y[index]);
//    printf("      y_tmp[%d] = %.20f\n",index,y_tmp[index]);
//    printf("      y_err[%d] = %.20f\n",index,y_err[index]);
//    printf("      dydt_out[%d] = %.20f\n",index,dydt_out[index]);

    /* k1_d */
    if (dydt_in != NULL)
    {
        k1[index] = dydt_in[index];
    }
    else {
        gpu_func_test(t, y, k1, index, params_d);
    }
//    printf("      k1[%d] = %.20f\n",index,k1[index]);
    y_tmp[index] = y[index] +  ah[0] * h * k1[index];
    /* k2 */
    gpu_func_test(t + ah[0] * h, y_tmp, k2, index, params_d);
//    printf("      k2[%d] = %.20f\n",index,k2[index]);
    y_tmp[index] = y[index] + h * (b3[0] * k1[index] + b3[1] * k2[index]);
    /* k3 */
    gpu_func_test(t + ah[1] * h, y_tmp, k3, index, params_d);
//    printf("      k3[%d] = %.20f\n",index,k3[index]);
    y_tmp[index] = y[index] + h * (b4[0] * k1[index] + b4[1] * k2[index] + b4[2] * k3[index]);
    /* k4 */
    gpu_func_test(t + ah[2] * h, y_tmp, k4, index, params_d);
//    printf("      k4[%d] = %.20f\n",index,k4[index]);
    y_tmp[index] = y[index] + h * (b5[0] * k1[index] + b5[1] * k2[index] + b5[2] * k3[index] + b5[3] * k4[index]);
    /* k5 */
    gpu_func_test(t + ah[3] * h, y_tmp, k5, index, params_d);
//    printf("      k5[%d] = %.20f\n",index,k5[index]);
    y_tmp[index] = y[index] + h * (b6[0] * k1[index] + b6[1] * k2[index] + b6[2] * k3[index] + b6[3] * k4[index] + b6[4] * k5[index]);
    /* k6 */
    gpu_func_test(t + ah[4] * h, y_tmp, k6, index, params_d);
    /* final sum */
//    printf("      k6[%d] = %.20f\n",index,k6[index]);
    const double d_i = c1 * k1[index] + c3 * k3[index] + c4 * k4[index] + c5 * k5[index] + c6 * k6[index];
    y[index] += h * d_i;
    /* Derivatives at output */
    gpu_func_test(t + h, y, dydt_out,index,params_d);
    /* difference between 4th and 5th order */
    y_err[index] = h * (ec[1] * k1[index] + ec[3] * k3[index] + ec[4] * k4[index] + ec[5] * k5[index] + ec[6] * k6[index]);
    //debug printout
//    printf("      y[%d] = %.20f\n",index,y[index]);
//    printf("      y_err[%d] = %.20f\n",index,y_err[index]);
//    printf("      dydt_out[%d] = %.20f\n",index,dydt_out[index]);
//    printf("    [step apply] index = %d end\n",index);
    return;
}

__global__
void rk45_gpu_evolve_apply(double t, double t_target, double t_delta, double h, double* y, GPU_Parameters* params_d){

    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ double r_max[DIM];
    __shared__ double D0[DIM];
    __shared__ double r[DIM];
    __shared__ double y_0[DIM];
    __shared__ double y_tmp[DIM];
    __shared__ double y_err[DIM];
    __shared__ double dydt_in[DIM];
    __shared__ double dydt_out[DIM];
    __shared__ double k1[DIM];
    __shared__ double k2[DIM];
    __shared__ double k3[DIM];
    __shared__ double k4[DIM];
    __shared__ double k5[DIM];
    __shared__ double k6[DIM];

    for(int index = index_gpu; index < params_d->dimension; index += stride){
        while(t < t_target){
            double device_t;
            double device_t1;
            double device_h;
            double device_h_0;
            int device_adjustment_out = 999;
            int device_final_step = 0;
            device_t1 = t_target;
            device_t = t;
            device_h = h;

//            printf("\n  Will run from %f to %f, step %.20f\n", t, device_t1, h);
//            printf("    t = %.20f t_1 = %.20f  h = %.20f\n",device_t,device_t1,device_h);

            while(device_t < device_t1)
            {
                const double device_t_0 = device_t;
                device_h_0 = device_h;
                double device_dt = device_t1 - device_t_0;

//                printf("  [evolve apply] index = %d start\n",index);
//                printf("    t = %.20f t_0 = %.20f  h = %.20f h_0 = %.20f dt = %.20f\n",device_t,device_t_0,device_h,device_h_0,device_dt);

                y_0[index] = y[index];
                device_final_step = 0;
                gpu_func_test(device_t_0, y, dydt_in, index, params_d);

                while(true)
                {
                    if ((device_dt >= 0.0 && device_h_0 > device_dt) || (device_dt < 0.0 && device_h_0 < device_dt)) {
                        device_h_0 = device_dt;
                        device_final_step = 1;
                    } else {
                        device_final_step = 0;
                    }

                    rk45_gpu_step_apply(device_t_0, device_h_0,
                                        y,y_tmp,y_err, dydt_in, dydt_out,
                                        k1, k2, k3, k4, k5, k6,
                                        index, params_d);

                    if (device_final_step) {
                        device_t = device_t1;
                    } else {
                        device_t = device_t_0 + device_h_0;
                    }

                    double h_old = device_h_0;

//                    printf("    before adjust t = %.20f t_0 = %.20f  h = %.20f h_0 = %.20f h_old = %.20f\n",device_t,device_t_0,device_h,device_h_0,h_old);

                    rk45_gpu_adjust_h(y, y_err, dydt_out,
                                      device_h, device_h_0, device_adjustment_out, device_final_step,
                                      r, D0, r_max,
                                      index, params_d);

                    //Extra step to get data from h
                    device_h_0 = device_h;

//                  printf("    after adjust t = %.20f t_0 = %.20f  h = %.20f h_0 = %.20f h_old = %.20f\n",device_t,device_t_0,device_h,device_h_0,h_old);

                    if (device_adjustment_out == -1)
                    {
                        double t_curr = (device_t);
                        double t_next = (device_t) + device_h_0;

                        if (fabs(device_h_0) < fabs(h_old) && t_next != t_curr) {
                            /* Step was decreased. Undo step, and try again with new h0. */
//                          printf("  [evolve apply] index = %d step decreased, y = y0\n",index);
                            y[index] = y_0[index];
                        } else {
//                            printf("  [evolve apply] index = %d step decreased h_0 = h_old\n",index);
                            device_h_0 = h_old; /* keep current step size */
                            break;
                        }
                    }
                    else{
//                        printf("  [evolve apply] index = %d step increased or no change\n",index);
                        break;
                    }
                }
                device_h = device_h_0;  /* suggest step size for next time-step */
                t = device_t;
                h = device_h;
//                printf("    index = %d t = %.20f t_0 = %.20f  h = %.20f h_0 = %.20f\n",index,device_t,device_t_0,device_h,device_h_0);
//                printf("    index = %d y[%d] = %.20f\n",index,index,y[index]);
//                if(device_final_step)
//                {
//                    if(index == 0 || index == params_d->dimension - 1) {
//                        if(index == 0)
//                        {
//                            printf("[output] index = %d t = %.20f t_0 = %.20f  h = %.20f h_0 = %.20f\n", index, device_t,
//                               device_t_0, device_h, device_h_0);
//                            printf("[output] index = %d y[%d] = %.20f\n", index, index, y[index]);
//                            printf("\n");
//                        }
//                    }
//                }
//                printf("  [evolve apply] index = %d end\n",index);
            }
//            if(index == 0 || index == params_d->dimension - 1)
//            {
//                printf("after evolve t = %.20f h = %.20f\n",t,h);
//                printf("  y[%d] = %.20f\n",index,y[index]);
//            }
            t += t_delta;
        }
    }
    return;
}

void GPU_RK45::run(){
    auto start = std::chrono::high_resolution_clock::now();
    //GPU memory
//    double **y_d = 0;
//    gpuErrchk(cudaMalloc ((void **)&y_d, params->number_of_ode * sizeof (double)));
//    //temp pointers
//    double **tmp_ptr = (double**)malloc (params->number_of_ode * sizeof (double));
//    for (int i = 0; i < params->number_of_ode; i++) {
//        gpuErrchk(cudaMalloc ((void **)&tmp_ptr[i], params->dimension * sizeof (double)));
//        gpuErrchk(cudaMemcpy(tmp_ptr[i], params->y[i], params->dimension * sizeof(double), cudaMemcpyHostToDevice));
//    }
//    gpuErrchk(cudaMemcpy (y_d, tmp_ptr, params->number_of_ode * sizeof (double), cudaMemcpyHostToDevice));
//    delete(params->y);
//    delete(tmp_ptr);

    //Unified memory
//    double** y_d;
//    cudaMallocManaged((void**)&y_d, gpu_threads * sizeof(double*));
//    for (int i = 0; i < gpu_threads; i++) {
//        cudaMallocManaged((void**)&y_d[i], params->dimension * sizeof(double));
////        for (int j = 0; j < params->dimension; j++) {
////            y_d[i][j] = 0.5;
////        }
//    }

    //y_test
    double *y_test_d;
    gpuErrchk(cudaMalloc ((void **)&y_test_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_test_d, params->y_test, params->dimension * sizeof (double), cudaMemcpyHostToDevice));

    GPU_Parameters* params_d;
    gpuErrchk(cudaMalloc((void **)&params_d, sizeof(GPU_Parameters)));
    gpuErrchk(cudaMemcpy(params_d, params, sizeof(GPU_Parameters), cudaMemcpyHostToDevice));

    int num_SMs;
    gpuErrchk(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
//    int numBlocks = 32*num_SMs; //multiple of 32
    int block_size = 128; //max is 1024
    int num_blocks = (params->dimension + block_size - 1) / block_size;
    printf("[GSL GPU] block_size = %d num_blocks = %d\n",block_size,num_blocks);
    dim3 dimBlock(block_size, block_size); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
    dim3 dimGrid(num_blocks, num_blocks); // 1*1 blocks in a grid

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %lld micro seconds which is %.20f seconds\n",duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();

    rk45_gpu_evolve_apply<<<num_blocks, block_size>>>(params->t0, params->t_target, 1.0, params->h, y_test_d, params_d);
    gpuErrchk(cudaDeviceSynchronize());

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for compute %d ODE with %d parameters, step %.20f in %f days on GPU: %lld micro seconds which is %.20f seconds\n",params->number_of_ode,params->dimension,params->h,params->t_target,duration.count(),(duration.count()/1e6));

    gpuErrchk(cudaMemcpy (params->y_test, y_test_d, params->dimension * sizeof (double), cudaMemcpyDeviceToHost));
    printf("Display on Host\n");
    for(int i = 0; i < params->dimension; i++){
        printf("  y[%d] = %.20f\n",i,params->y_test[i]);
    }
    return;
}