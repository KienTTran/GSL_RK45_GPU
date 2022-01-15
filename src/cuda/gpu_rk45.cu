#include "gpu_rk45.h"

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

__global__ void sumReductionDouble(double *vectIn, double *vecOut, int size)
{
    __shared__ double block[DIM];
    unsigned int globalIndex = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int i = threadIdx.x;
    if (globalIndex < size)
        block[i] = vectIn[globalIndex];
    else
        block[i] = 0;

    __syncthreads();

    for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
    {
        if (i < j)
            block[i] += block[i + j];

        __syncthreads();
    }
    if (i == 0)
        vecOut[blockIdx.x] = block[0];
}

__device__ const float d_float_min = -3.402e+38;
__global__ void maxReduceDouble(volatile double* d_data, int n)
{
    // compute max over all threads, store max in d_data[0]
    int ti = threadIdx.x;

    __shared__ volatile double max_value;

    if (ti == 0) max_value = d_float_min;

    for (int bi = 0; bi < n; bi += 32)
    {
        int i = bi + ti;
        if (i >= n) break;

        double v = d_data[i];
        __syncthreads();

        while (max_value < v)
        {
            max_value = v;
        }

        __syncthreads();
    }

    if (ti == 0) d_data[0] = max_value;
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

//    if(index == 0 || index == params_d->dimension - 1){
//        printf("    [adjust h] index = %d begin\n", index);
//        printf("      y[%d] = %.10f\n", index, y[index]);
//        printf("      y_err[%d] = %.10f\n", index, y_err[index]);
//        printf("      dydt_out[%d] = %.10f\n", index, dydt_out[index]);
//    }

    r_max[index] = 2.2250738585072014e-308;
    D0[index] = eps_rel * (a_y * fabs(y[index]) + a_dydt * fabs((h_old) * dydt_out[index])) + eps_abs;
    r[index]  = fabs(y_err[index]) / fabs(D0[index]);
//    if(index == 0 || index == params_d->dimension - 1){
//        printf("      index = %d D0[%d] = %.10f\n", index, index, D0[index]);
//        printf("      index = %d r[%d] = %.10f\n", index, index, D0[index]);
//    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    for (int i = 0; i < DIM; i++)
    {
//        const double D0 = eps_rel * (a_y * fabs(y[i]) + a_dydt * fabs((h_old) * dydt_out[i])) + eps_abs;
//        const double r  = fabs(y_err[i]) / fabs(D0);
//        printf("      index = %d i = %d compare r[%d] = %.10f r_max = %.10f\n",index,i,i,r[i],r_max[index]);
        r_max[index] = max(r[i], r_max[index]);
    }
    block.sync();

//    printf("      r_max = %.10f\n",r_max[index]);

    if (r_max[index] > 1.1) {
        /* decrease step, no more than factor of 5, but a fraction S more
           than scaling suggests (for better accuracy) */
        double r = S / pow(r_max[index], 1.0 / ord);

        if (r < 0.2)
            r = 0.2;
        h = r * (h_old);

//        printf("      index = %d decrease by %.10f, h_old is %.10f new h is %.10f\n",index, r, h_old, h);
        adjustment_out = -1;
    } else if (r_max[index] < 0.5) {
        /* increase step, no more than factor of 5 */
        double r = S / pow(r_max[index], 1.0 / (ord + 1.0));

        if (r > 5.0)
            r = 5.0;

        if (r < 1.0)  /* don't allow any decrease caused by S<1 */
            r = 1.0;

        h = r * (h_old);

//        printf("      index = %d increase by %.10f, h_old is %.10f new h is %.10f\n",index, r, h_old, h);
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
                         double y[], double y_tmp[], double y_err[], double dydt_out[],
                         double k1[], double k2[], double k3[], double k4[], double k5[], double k6[],
                         double* sum_foi, double* foi_on_susc_single_virus,
                         double* inflow_from_recovereds, double* foi_on_susc_all_viruses,
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

//    if(index == 0 || index == params_d->dimension - 1){
//        printf("    [step apply] index = %d start\n", index);
//        printf("      t = %.10f h = %.10f\n", t, h);
//    }

    y_tmp[index] = 0.0;
    y_err[index] = 0.0;
    dydt_out[index] = 0.0;
    k1[index] = 0.0;
    k2[index] = 0.0;
    k3[index] = 0.0;
    k4[index] = 0.0;
    k5[index] = 0.0;
    k6[index] = 0.0;

//    if(index == 0 || index == params_d->dimension - 1){
//        printf("      y[%d] = %.10f\n", index, y[index]);
//        printf("      y_tmp[%d] = %.10f\n", index, y_tmp[index]);
//        printf("      y_err[%d] = %.10f\n", index, y_err[index]);
//        printf("      dydt_out[%d] = %.10f\n", index, dydt_out[index]);
//    }

    /* k1 */
    gpu_func_test(t,y,k1,
                  sum_foi,foi_on_susc_single_virus,
                  inflow_from_recovereds, foi_on_susc_all_viruses,
                  index,params_d);
//    printf("      k1[%d] = %.10f\n",index,k1[index]);
    y_tmp[index] = y[index] +  ah[0] * h * k1[index];
    /* k2 */
    gpu_func_test(t + ah[0] * h, y_tmp, k2,
                  sum_foi,foi_on_susc_single_virus,
                  inflow_from_recovereds, foi_on_susc_all_viruses,
                  index,params_d);
//    printf("      k2[%d] = %.10f\n",index,k2[index]);
    y_tmp[index] = y[index] + h * (b3[0] * k1[index] + b3[1] * k2[index]);
    /* k3 */
    gpu_func_test(t + ah[1] * h, y_tmp, k3,
                  sum_foi,foi_on_susc_single_virus,
                  inflow_from_recovereds, foi_on_susc_all_viruses,
                  index,params_d);
//    printf("      k3[%d] = %.10f\n",index,k3[index]);
    y_tmp[index] = y[index] + h * (b4[0] * k1[index] + b4[1] * k2[index] + b4[2] * k3[index]);
    /* k4 */
    gpu_func_test(t + ah[2] * h, y_tmp, k4,
                  sum_foi,foi_on_susc_single_virus,
                  inflow_from_recovereds, foi_on_susc_all_viruses,
                  index,params_d);
//    printf("      k4[%d] = %.10f\n",index,k4[index]);
    y_tmp[index] = y[index] + h * (b5[0] * k1[index] + b5[1] * k2[index] + b5[2] * k3[index] + b5[3] * k4[index]);
    /* k5 */
    gpu_func_test(t + ah[3] * h, y_tmp, k5,
                  sum_foi,foi_on_susc_single_virus,
                  inflow_from_recovereds, foi_on_susc_all_viruses,
                  index,params_d);
//    printf("      k5[%d] = %.10f\n",index,k5[index]);
    y_tmp[index] = y[index] + h * (b6[0] * k1[index] + b6[1] * k2[index] + b6[2] * k3[index] + b6[3] * k4[index] + b6[4] * k5[index]);
    /* k6 */
    gpu_func_test(t + ah[4] * h, y_tmp, k6,
                  sum_foi,foi_on_susc_single_virus,
                  inflow_from_recovereds, foi_on_susc_all_viruses,
                  index,params_d);
    /* final sum */
//    printf("      k6[%d] = %.10f\n",index,k6[index]);
    const double d_i = c1 * k1[index] + c3 * k3[index] + c4 * k4[index] + c5 * k5[index] + c6 * k6[index];
    y[index] += h * d_i;
    /* Derivatives at output */
    gpu_func_test(t + h, y, dydt_out,
                  sum_foi,foi_on_susc_single_virus,
                  inflow_from_recovereds, foi_on_susc_all_viruses,
                  index,params_d);
    /* difference between 4th and 5th order */
    y_err[index] = h * (ec[1] * k1[index] + ec[3] * k3[index] + ec[4] * k4[index] + ec[5] * k5[index] + ec[6] * k6[index]);
    //debug printout
//    if(index == 0 || index == params_d->dimension - 1){
//        printf("      y[%d] = %.10f\n",index,y[index]);
//        printf("      y_err[%d] = %.10f\n",index,y_err[index]);
//        printf("      dydt_out[%d] = %.10f\n",index,dydt_out[index]);
//        printf("    [step apply] index = %d end\n",index);
//    }
    return;
}

__global__
void rk45_gpu_evolve_apply(double* t, double* t_target, double t_delta, double* h, double* y
                           , GPU_Parameters* params_d
                           ){

    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ double r_max[DIM];
    __shared__ double D0[DIM];
    __shared__ double r[DIM];
    __shared__ double y_0[DIM];
    __shared__ double y_tmp[DIM];
    __shared__ double y_err[DIM];
    __shared__ double dydt_out[DIM];
    __shared__ double k1[DIM];
    __shared__ double k2[DIM];
    __shared__ double k3[DIM];
    __shared__ double k4[DIM];
    __shared__ double k5[DIM];
    __shared__ double k6[DIM];

    __shared__ double device_t[DIM];
    __shared__ double device_t1[DIM];
    __shared__ double device_h[DIM];
    __shared__ double device_h_0[DIM];
    __shared__ int device_adjustment_out[DIM];
    __shared__ int device_final_step[DIM];

    __device__ __shared__ static double sum_foi[DIM];
    __device__ __shared__ static double foi_on_susc_single_virus[DIM];
    __device__ __shared__ static double inflow_from_recovereds[DIM];
    __device__ __shared__ static double foi_on_susc_all_viruses[DIM];

    for(int index = index_gpu; index < DIM; index += stride){

//        printf("Index = %d\n",index);
//        gpu_func_test(t[index],y,dydt_out,index,params_d);

//        printf("before y[%d] = %.10f f[%d] = %.10f\n",index,y[index],index,dydt_out[index]);
//        dydt_out[index] = -(g / l) * sin(y[index]) + (A * cos(wd * t[index]) - b * y[index]) / (m * l * l);
//        printf("after y[%d] = %.10f f[%d] = %.10f\n",index,y[index],index,dydt_out[index]);

        while(t[index] < t_target[index])
        {
            device_adjustment_out[index] = 999;
            device_final_step[index] = 0;
            device_t1[index] = t_target[index];
            device_t[index] = t[index];
            device_h[index] = h[index];

//            if(index == 0 || index == params_d->dimension - 1){
//                printf("[evolve apply t[%d] = %.10f < t_target[%d] = %.10f index = %d start\n",index,t[index],index,t_target[index],index);
//                printf("    t = %.10f t_target = %.10f  h = %.10f\n", t[index], t_target[index], h[index]);
//            }

            while(device_t[index] < device_t1[index])
            {
                const double device_t_0 = device_t[index];
                device_h_0[index] = device_h[index];
                double device_dt = device_t1[index] - device_t_0;

//                if(index == 0 || index == params_d->dimension - 1){
//                    printf("  [evolve apply device_t[%d] = %.10f < device_t1[%d] = %.10f] index = %d start\n",index,device_t[index],index,device_t1[index],index);
//                    printf("    t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f dt = %.10f\n", device_t[index], device_t_0, device_h[index], device_h_0[index], device_dt);
//                }

                y_0[index] = y[index];

                device_final_step[index] = 0;

                while(true)
                {
                    if ((device_dt >= 0.0 && device_h_0[index] > device_dt) || (device_dt < 0.0 && device_h_0[index] < device_dt)) {
                        device_h_0[index] = device_dt;
                        device_final_step[index] = 1;
                    } else {
                        device_final_step[index] = 0;
                    }

                    rk45_gpu_step_apply(device_t_0, device_h_0[index],
                                        y,y_tmp,y_err, dydt_out,
                                        k1, k2, k3, k4, k5, k6,
                                        sum_foi,foi_on_susc_single_virus,
                                        inflow_from_recovereds, foi_on_susc_all_viruses,
                                        index, params_d);

                    if (device_final_step[index]) {
                        device_t[index] = device_t1[index];
                    } else {
                        device_t[index] = device_t_0 + device_h_0[index];
                    }

                    double h_old = device_h_0[index];

//                    if(index == 0 || index == params_d->dimension - 1){
//                        printf("    before adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",
//                               device_t[index], device_t_0, device_h[index], device_h_0[index], h_old);
//                    }

                    rk45_gpu_adjust_h(y, y_err, dydt_out,
                                      device_h[index], device_h_0[index], device_adjustment_out[index], device_final_step[index],
                                      r, D0, r_max,
                                      index, params_d);

                    //Extra step to get data from h
                    device_h_0[index] = device_h[index];

//                    if(index == 0 || index == params_d->dimension - 1){
//                        printf("    after adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",
//                               device_t[index], device_t_0, device_h[index], device_h_0[index], h_old);
//                    }

                    if (device_adjustment_out[index] == -1)
                    {
                        double t_curr = (device_t[index]);
                        double t_next = (device_t[index]) + device_h_0[index];

                        if (fabs(device_h_0[index]) < fabs(h_old) && t_next != t_curr) {
                            /* Step was decreased. Undo step, and try again with new h0. */
//                            if(index == 0 || index == params_d->dimension - 1){
//                                printf("  [evolve apply] index = %d step decreased, y = y0\n", index);
//                            }
                            y[index] = y_0[index];
                        } else {
//                            if(index == 0 || index == params_d->dimension - 1){
//                                printf("  [evolve apply] index = %d step decreased h_0 = h_old\n", index);
//                            }
                            device_h_0[index] = h_old; /* keep current step size */
                            break;
                        }
                    }
                    else{
//                        if(index == 0 || index == params_d->dimension - 1){
//                            printf("  [evolve apply] index = %d step increased or no change\n", index);
//                        }
                        break;
                    }
                }
                device_h[index] = device_h_0[index];  /* suggest step size for next time-step */
                t[index] = device_t[index];
                h[index] = device_h[index];
//                if(index == 0 || index == params_d->dimension - 1){
//                    printf("    index = %d t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f\n", index, device_t, device_t_0, device_h, device_h_0);
//                    printf("    index = %d y[%d] = %.10f\n", index, index, y[index]);
//                    if (device_final_step) {
//                        printf("    [output] index = %d t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f\n", index, device_t, device_t_0, device_h, device_h_0);
//                        printf("    [output] index = %d y[%d] = %.10f\n", index, index, y[index]);
//                        printf("\n");
//                    }
//                    printf("  [evolve apply] index = %d end\n", index);
//                }
            }
//            if(index == 0 || index == params_d->dimension - 1)
//            {
//                printf("after evolve t = %.10f h = %.10f\n",t,h);
//                printf("  y[%d] = %.10f\n",index,y[index]);
//            }
            t[index] += t_delta;
        }
//        if(index == 0 || index == params_d->dimension - 1){
//            printf("[evolve apply t[%d] = %.10f < t_target[%d] = %.10f] index = %d finished\n",index,t[index],index,t_target[index],index);
//        }
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

    //y_test
    double *y_test_d;
    gpuErrchk(cudaMalloc ((void **)&y_test_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_test_d, params->y_test, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //t
    double *t_d;
    gpuErrchk(cudaMalloc ((void **)&t_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (t_d, params->t0, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //t_target
    double *t_target_d;
    gpuErrchk(cudaMalloc ((void **)&t_target_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (t_target_d, params->t_target, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //h
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
    printf("[GSL GPU] SMs = %d block_size = %d num_blocks = %d\n",num_SMs,block_size,num_blocks);
    dim3 dimBlock(block_size, block_size); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
    dim3 dimGrid(num_blocks, num_blocks); // 1*1 blocks in a grid

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %lld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
    rk45_gpu_evolve_apply<<<num_blocks, block_size>>>(t_d, t_target_d, 1.0, h_d, y_test_d
                                                      ,params_d
                                                      );
    gpuErrchk(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for compute %d ODE with %d parameters, step %.10f in %f days on GPU: %lld micro seconds which is %.10f seconds\n",params->number_of_ode,params->dimension,params->h_initial,params->t_target_initial,duration.count(),(duration.count()/1e6));

    gpuErrchk(cudaMemcpy (params->y_test, y_test_d, params->dimension * sizeof (double), cudaMemcpyDeviceToHost));
    printf("Display on Host\n");
    for(int i = 0; i < params->dimension; i++){
        printf("  y[%d] = %.10f\n",i,params->y_test[i]);
    }
    return;
}