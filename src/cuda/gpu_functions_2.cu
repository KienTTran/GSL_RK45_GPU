#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

//global parameters in device, will be instanced for each thread.
const int dim = 2;
__device__ double device_t;
__device__ double device_t1;
__device__ double device_h;
__device__ double device_h_0;
__device__ int device_adjustment_out;
__device__ int device_final_step;
__device__ double device_k1[dim];
__device__ double device_k2[dim];
__device__ double device_k3[dim];
__device__ double device_k4[dim];
__device__ double device_k5[dim];
__device__ double device_k6[dim];
__device__ double device_y[dim];
__device__ double device_y_0[dim];
__device__ double device_y_tmp[dim];
__device__ double device_y_err[dim];
__device__ double device_dydt_out[dim];

__device__
void function_2(double t, const double y[], double dydt[], int index, const int dim){
    const double m = 5.2;		// Mass of pendulum
    const double g = -9.81;		// g
    const double l = 2;		// Length of pendulum
    const double A = 0.5;		// Amplitude of driving force
    const double wd = 1;		// Angular frequency of driving force
    const double b = 0.5;		// Damping coefficient

    dydt[0] = y[1];
    dydt[1] = -(g / l) * sin(y[0]) + (A * cos(wd * t) - b * y[1]) / (m * l * l);
    return;
}

__device__
void rk45_gsl_gpu_adjust_h_2(double y[], int index, int dim){
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
    static double h_old;
    if(device_final_step){
        h_old = device_h_0;
    }
    else{
        h_old = device_h;
    }

//    printf("    [adjust h] index = %d begin\n",index);
//    for (int i = 0; i < dim; i ++)
//    {
//        printf("      y[%d] = %.10f\n",i,y[i]);
//    }
//    for (int i = 0; i < dim; i ++)
//    {
//        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
//    }
//    for (int i = 0; i < dim; i ++)
//    {
//        printf("      device_dydt_out[%d] = %.10f\n",i,device_dydt_out[i]);
//    }

    double r_max = 2.2250738585072014e-308;
    for (int i = 0; i < dim; i ++)
    {
        const double D0 = eps_rel * (a_y * fabs(y[i]) + a_dydt * fabs((h_old) * device_dydt_out[i])) + eps_abs;
        const double r  = fabs(device_y_err[i]) / fabs(D0);
//        printf("      compare r = %.10f r_max = %.10f\n",r,r_max);
        r_max = max(r, r_max);
    }

//    printf("      r_max = %.10f\n",r_max);

    if (r_max > 1.1) {
        /* decrease step, no more than factor of 5, but a fraction S more
           than scaling suggests (for better accuracy) */
        double r = S / pow(r_max, 1.0 / ord);

        if (r < 0.2)
            r = 0.2;
        device_h = r * (h_old);

//        printf("      index = %d decrease by %.10f, h_old is %.10f new h is %.10f\n",index, r, h_old, device_h);
        device_adjustment_out = -1;
    } else if (r_max < 0.5) {
        /* increase step, no more than factor of 5 */
        double r = S / pow(r_max, 1.0 / (ord + 1.0));

        if (r > 5.0)
            r = 5.0;

        if (r < 1.0)  /* don't allow any decrease caused by S<1 */
            r = 1.0;

        device_h = r * (h_old);

//        printf("      index = %d increase by %.10f, h_old is %.10f new h is %.10f\n",index, r, h_old, device_h);
        device_adjustment_out = 1;
    } else {
        /* no change */
//        printf("      index = %d no change\n",index);
        device_adjustment_out = 0;
    }
//    printf("    [adjust h] index = %d end\n",index);
    return;
}

__device__
void rk45_gsl_gpu_step_apply_2(double t, double h, double y[], int index, int dim)
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
//    printf("      t = %.10f h = %.10f\n",t,device_h);
//    for (int i = 0; i < dim; i ++)
//    {
//        printf("      y[%d] = %.10f\n",i,y[i]);
//        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
//        printf("      device_dydt_out[%d] = %.10f\n",i,device_dydt_out[i]);
//    }

    for(int i = 0; i < dim; i++){
        device_y_tmp[i] = 0.0;
        device_y_err[i] = 0.0;
        device_dydt_out[i] = 0.0;
        device_k1[i] = 0.0;
        device_k2[i] = 0.0;
        device_k3[i] = 0.0;
        device_k4[i] = 0.0;
        device_k5[i] = 0.0;
        device_k6[i] = 0.0;
    }

    /* device_k1 */
    function_2(t,y,device_k1,index,dim);
//    cudaDeviceSynchronize();
    for (int i = 0; i < dim; i ++)
    {
//        printf("      device_k1[%d] = %.10f\n",i,device_k1[i]);
        device_y_tmp[i] = y[i] +  ah[0] * h * device_k1[i];
    }
    /* device_k2 */
    function_2(t + ah[0] * h, device_y_tmp,device_k2,index,dim);
//    cudaDeviceSynchronize();
    for (int i = 0; i < dim; i ++)
    {
//        printf("      device_k2[%d] = %.10f\n",i,device_k2[i]);
        device_y_tmp[i] = y[i] + h * (b3[0] * device_k1[i] + b3[1] * device_k2[i]);
    }
    /* device_k3 */
    function_2(t + ah[1] * h, device_y_tmp,device_k3,index,dim);
//    cudaDeviceSynchronize();
    for (int i = 0; i < dim; i ++)
    {
//        printf("      device_k3[%d] = %.10f\n",i,device_k3[i]);
        device_y_tmp[i] = y[i] + h * (b4[0] * device_k1[i] + b4[1] * device_k2[i] + b4[2] * device_k3[i]);
    }
    /* device_k4 */
    function_2(t + ah[2] * h, device_y_tmp,device_k4,index,dim);
//    cudaDeviceSynchronize();
    for (int i = 0; i < dim; i ++)
    {
//        printf("      device_k4[%d] = %.10f\n",i,device_k4[i]);
        device_y_tmp[i] = y[i] + h * (b5[0] * device_k1[i] + b5[1] * device_k2[i] + b5[2] * device_k3[i] + b5[3] * device_k4[i]);
    }
    /* device_k5 */
    function_2(t + ah[3] * h, device_y_tmp,device_k5,index,dim);
//    cudaDeviceSynchronize();
    for (int i = 0; i < dim; i ++)
    {
//        printf("      device_k5[%d] = %.10f\n",i,device_k5[i]);
        device_y_tmp[i] = y[i] + h * (b6[0] * device_k1[i] + b6[1] * device_k2[i] + b6[2] * device_k3[i] + b6[3] * device_k4[i] + b6[4] * device_k5[i]);
    }
    /* device_k6 */
    function_2(t + ah[4] * h, device_y_tmp,device_k6,index,dim);
//    cudaDeviceSynchronize();
    /* final sum */
    for (int i = 0; i < dim; i ++)
    {
//        printf("      device_k6[%d] = %.10f\n",i,device_k6[i]);
        const double d_i = c1 * device_k1[i] + c3 * device_k3[i] + c4 * device_k4[i] + c5 * device_k5[i] + c6 * device_k6[i];
        y[i] += h * d_i;
    }
    /* Derivatives at output */
    function_2(t + h, y, device_dydt_out,index,dim);
//    cudaDeviceSynchronize();
    /* difference between 4th and 5th order */
    for (int i = 0; i < dim; i ++)
    {
        device_y_err[i] = h * (ec[1] * device_k1[i] + ec[3] * device_k3[i] + ec[4] * device_k4[i] + ec[5] * device_k5[i] + ec[6] * device_k6[i]);
    }
//    for (int i = 0; i < dim; i++) {
//        printf("      index = %d y[%d] = %.10f\n",index,i,y[i]);
//    }
//    for (int i = 0; i < dim; i++) {
//        printf("      index = %d device_y_err[%d] = %.10f\n",index,i,device_y_err[i]);
//    }
//    for (int i = 0; i < dim; i++) {
//        printf("      index = %d device_dydt_out[%d] = %.10f\n",index,i,device_dydt_out[i]);
//    }
//    printf("    [step apply] index = %d end\n",index);
    return;
}

__global__
void rk45_gsl_gpu_evolve_apply_2(double t1, double t, double h, double *y[], int thread_number,const int dim){
    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int index = index_gpu; index < thread_number; index += stride){
        device_t1 = t1;
        device_t = t;
        device_h = h;

        while(device_t < device_t1){
            const double t_0 = device_t;
            device_h_0 = device_h;
            double dt = device_t1 - t_0;

//            printf("\n  [evolve apply] index = %d start\n",index);

            for (int i = 0; i < dim; i ++){
                device_y[i] = y[index][i];
                device_y_0[i] = device_y[i];
            }

            device_final_step = 0;

            while(true){
                if ((dt >= 0.0 && device_h_0 > dt) || (dt < 0.0 && device_h_0 < dt)) {
                    device_h_0 = dt;
                    device_final_step = 1;
                } else {
                    device_final_step = 0;
                }

                rk45_gsl_gpu_step_apply_2(t_0,device_h_0,device_y,index,dim);
//                cudaDeviceSynchronize();

                if (device_final_step) {
                    device_t = device_t1;
                } else {
                    device_t = t_0 + device_h_0;
                }

                double h_old = device_h_0;

//                printf("    before adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",device_t,t_0,device_h,device_h_0,h_old);

                rk45_gsl_gpu_adjust_h_2(device_y, index, dim);
//                cudaDeviceSynchronize();

                //Extra step to get data from *h
                device_h_0 = device_h;

//                printf("    after adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",device_t,t_0,device_h,device_h_0,h_old);

                if (device_adjustment_out == -1)
                {
                    double t_curr = (device_t);
                    double t_next = (device_t) + device_h_0;

                    if (fabs(device_h_0) < fabs(h_old) && t_next != t_curr) {
                        /* Step was decreased. Undo step, and try again with new h0. */
//                        printf("  [evolve apply] index = %d step decreased, y = y0\n",index);
                        for (int i = 0; i < dim; i++) {
                            device_y[i] = device_y_0[i];
                        }
                    } else {
//                        printf("  [evolve apply] index = %d step decreased h_0 = h_old\n",index);
                        device_h_0 = h_old; /* keep current step size */
                        break;
                    }
                }
                else{
//                    printf("  [evolve apply] index = %d step increased or no change\n",index);
                    break;
                }
            }
            device_h = device_h_0;  /* suggest step size for next time-step */
//            printf("    index = %d t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f\n",index,device_t,t_0,device_h,device_h_0);
//            for (int i = 0; i < dim; i++){
//                printf("    index = %d y[%d][%d] = %.10f\n",index,index,i,device_y[i]);
//            }
//            printf("  [evolve apply] index = %d end\n",index);
            for (int i = 0; i < dim; i ++){
                y[index][i] = device_y[i];
            }
//            cudaDeviceSynchronize();
        }
    }
    return;
}


bool rk45_gsl_gpu_simulate_2(){
    const int gpu_thread = 1000000;
    const int gpu_block = 1;
    double t1 = 2.0;
    double t = 0.0;
    double h = 0.2;

    //Default parameters for RK45 in GSL
    //End default parameters for RK45

    double **y = new double*[gpu_thread]();
    for (int i = 0; i < gpu_thread; i++)
    {
        y[i] = new double[dim];
        for(int j = 0; j < dim; j++){
            y[i][j] = 0.5;
        }
    }

    double **y_d = 0;

    //temp pointers
    double **tmp_ptr = (double**)malloc (gpu_thread * sizeof (double));
    for (int i = 0; i < gpu_thread; i++) {
        cudaMalloc ((void **)&tmp_ptr[i], dim * sizeof (double));
        cudaMemcpy(tmp_ptr[i], y[i], dim * sizeof(double), cudaMemcpyHostToDevice);
    }
    //y
    cudaMalloc ((void **)&y_d, gpu_thread * sizeof (double));
    cudaMemcpy (y_d, tmp_ptr, gpu_thread * sizeof (double), cudaMemcpyHostToDevice);
    for (int i = 0; i < gpu_thread; i++) {
        cudaMemcpy (tmp_ptr[i], y[i], dim * sizeof (double), cudaMemcpyHostToDevice);
    }

    printf("[main] start\n");
    auto start_gpu = std::chrono::high_resolution_clock::now();
//    cudaProfilerStart();
    rk45_gsl_gpu_evolve_apply_2<<<gpu_thread,1>>>(t1, t, h,y_d,gpu_thread,dim);


//    cudaProfilerStop();
    cudaDeviceSynchronize();
    auto stop_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_gpu - start_gpu);
    printf("gpu time: %lld micro seconds which is %.10f seconds\n",duration_gpu.count(),(duration_gpu.count()/1e6));

//    double** host_y_output = (double**)malloc (gpu_thread * sizeof (double));
//    for (int i = 0; i < gpu_thread; i++) {
//        host_y_output[i] = (double *)malloc (dim * sizeof (double));
//    }
//    cudaMemcpy (tmp_ptr, y_d, gpu_thread * sizeof (double), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < gpu_thread; i++) {
//        cudaMemcpy (host_y_output[i], tmp_ptr[i], dim * sizeof (double), cudaMemcpyDeviceToHost);
//    }
//    for(int thread = 0; thread < gpu_thread; thread++){
//        for(int index = 0; index < dim; index++){
//            printf("thread %d y[%d][%d] = %.10f\n",thread,thread,index,host_y_output[thread][index]);
//        }
//    }
    // Free memory
    cudaFree(y);
    return true;
}