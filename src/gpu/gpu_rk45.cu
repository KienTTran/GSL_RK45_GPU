#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <random>
#include "gpu_rk45.h"

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
    return result;
}

GPU_RK45::GPU_RK45(){
    params = new GPU_Parameters();
}

GPU_RK45::~GPU_RK45(){
    params = nullptr;
}

void GPU_RK45::set_parameters(GPU_Parameters* params_) {
    params = &(*params_);
}

__device__
double seasonal_transmission_factor(GPU_Parameters* gpu_params, double t)
{
    /*


        We're gonna make this thing go for 40 years. 30 years of burn in and 10 years of real modeling.
        We're creating a "10-year model cycle" and need the code below to find a time point's "place" in the "cycle"
        modulus (denoted with % in C++) only works with integers, so need the acrobatics below

     */

    // This is some code that's needed to create the 10-year "cycles" in transmission.

    if(gpu_params->phis_d_length == 0){
        return 1.0;
    }

    int x = (int)t; // This is now to turn a double into an integer
    double remainder = t - (double)x;
    int xx = x % 3650; // int xx = x % NUMDAYSOUTPUT;
    double yy = (double)xx + remainder;
    // put yy into the sine function, let it return the beta value
    t = yy;
    double sine_function_value = 0.0;

    for(int i=0; i < gpu_params->phis_d_length; i++)
    {
        if( fabs( t - gpu_params->phis_d[i] ) < (gpu_params->v_d_i_epidur_d2))
        {
            // sine_function_value = sin( 2.0 * 3.141592653589793238 * (phis[i]-t+91.25) / 365.0);
            sine_function_value = sin( gpu_params->pi_x2 * (gpu_params->phis_d[i] - t + (gpu_params->v_d_i_epidur_d2)) / (gpu_params->v_d_i_epidur_x2));
//            printf("      in loop %1.3f %d  %1.3f %1.3f\n", t, i, gpu_params->phis_d[i], sine_function_value);
        }
    }
//    printf("    %f sine_function_value %1.3f\n",t,sine_function_value);
//    printf("    %f return %1.3f\n",t,1.0 + v[i_amp] * sine_function_value);
    return 1.0 + gpu_params->v_d_i_amp * sine_function_value;
}

__device__
double pop_sum( double yy[] )
{
    double sum=0.0;
    for(int i=0; i<DIM; i++) sum += yy[i];

    for(int i=STARTJ; i<STARTJ+NUMLOC*NUMSEROTYPES; i++) sum -= yy[i];
    return sum;
}

__device__
void rk45_gpu_adjust_h(double y[], double y_err[], double dydt_out[],
                             double &h, double h_0, int &adjustment_out, int final_step,
                             const int index){
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
//    for (int i = 0; i < DIM; i ++)
//    {
//        printf("      y[%d] = %.10f\n",i,y[i]);
//    }
//    for (int i = 0; i < DIM; i ++)
//    {
//        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
//    }
//    for (int i = 0; i < DIM; i ++)
//    {
//        printf("      dydt_out[%d] = %.10f\n",i,dydt_out[i]);
//    }

    double r_max = 2.2250738585072014e-308;
    for (int i = 0; i < DIM; i ++)
    {
        const double D0 = eps_rel * (a_y * fabs(y[i]) + a_dydt * fabs((h_old) * dydt_out[i])) + eps_abs;
        const double r  = fabs(y_err[i]) / fabs(D0);
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
        h = r * (h_old);

//        printf("      index = %d decrease by %.10f, h_old is %.10f new h is %.10f\n",index, r, h_old, h);
        adjustment_out = -1;
    } else if (r_max < 0.5) {
        /* increase step, no more than factor of 5 */
        double r = S / pow(r_max, 1.0 / (ord + 1.0));

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
void rk45_gpu_step_apply(double t, double h, double y[], double y_err[], double dydt_out[],
                         const int index, GPU_Parameters* params)
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
//    printf("      t = %.10f h = %.10f\n",t,h);

//    double* y_tmp = (double*)malloc(dim);
//    double* k1 = (double*)malloc(dim);
//    double* k2 = (double*)malloc(dim);
//    double* k3 = (double*)malloc(dim);
//    double* k4 = (double*)malloc(dim);
//    double* k5 = (double*)malloc(dim);
//    double* k6 = (double*)malloc(dim);
    double y_tmp[DIM];
    double k1[DIM];
    double k2[DIM];
    double k3[DIM];
    double k4[DIM];
    double k5[DIM];
    double k6[DIM];

    for(int i = 0; i < DIM; i++){
        y_tmp[i] = 0.0;
        y_err[i] = 0.0;
        dydt_out[i] = 0.0;
        k1[i] = 0.0;
        k2[i] = 0.0;
        k3[i] = 0.0;
        k4[i] = 0.0;
        k5[i] = 0.0;
        k6[i] = 0.0;
    }

//    for (int i = 0; i < DIM; i ++)
//    {
//        printf("      y[%d] = %.10f\n",i,y[i]);
//        printf("      y_tmp[%d] = %.10f\n",i,y_tmp[i]);
//        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
//        printf("      dydt_out[%d] = %.10f\n",i,dydt_out[i]);
//    }

    /* k1 */
    gpu_func_test(t,y,k1, index, params);
//    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i ++)
    {
//        printf("      k1[%d] = %.10f\n",i,k1[i]);
        y_tmp[i] = y[i] +  ah[0] * h * k1[i];
    }
    /* k2 */
    gpu_func_test(t + ah[0] * h, y_tmp,k2, index, params);
//    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i ++)
    {
//        printf("      k2[%d] = %.10f\n",i,k2[i]);
        y_tmp[i] = y[i] + h * (b3[0] * k1[i] + b3[1] * k2[i]);
    }
    /* k3 */
    gpu_func_test(t + ah[1] * h, y_tmp,k3, index, params);
//    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i ++)
    {
//        printf("      k3[%d] = %.10f\n",i,k3[i]);
        y_tmp[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i]);
    }
    /* k4 */
    gpu_func_test(t + ah[2] * h, y_tmp,k4, index, params);
//    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i ++)
    {
//        printf("      k4[%d] = %.10f\n",i,k4[i]);
        y_tmp[i] = y[i] + h * (b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] + b5[3] * k4[i]);
    }
    /* k5 */
    gpu_func_test(t + ah[3] * h, y_tmp,k5, index, params);
//    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i ++)
    {
//        printf("      k5[%d] = %.10f\n",i,k5[i]);
        y_tmp[i] = y[i] + h * (b6[0] * k1[i] + b6[1] * k2[i] + b6[2] * k3[i] + b6[3] * k4[i] + b6[4] * k5[i]);
    }
    /* k6 */
    gpu_func_test(t + ah[4] * h, y_tmp,k6, index, params);
//    cudaDeviceSynchronize();
    /* final sum */
    for (int i = 0; i < DIM; i ++)
    {
//        printf("      k6[%d] = %.10f\n",i,k6[i]);
        const double d_i = c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i];
        y[i] += h * d_i;
    }
    /* Derivatives at output */
    gpu_func_test(t + h, y, dydt_out, index, params);
//    cudaDeviceSynchronize();
    /* difference between 4th and 5th order */
    for (int i = 0; i < DIM; i ++)
    {
        y_err[i] = h * (ec[1] * k1[i] + ec[3] * k3[i] + ec[4] * k4[i] + ec[5] * k5[i] + ec[6] * k6[i]);
    }
    //debug printout
//    for (int i = 0; i < DIM; i++) {
//        printf("      index = %d y[%d] = %.10f\n",index,i,y[i]);
//    }
//    for (int i = 0; i < DIM; i++) {
//        printf("      index = %d y_err[%d] = %.10f\n",index,i,y_err[i]);
//    }
//    for (int i = 0; i < DIM; i++) {
//        printf("      index = %d dydt_out[%d] = %.10f\n",index,i,dydt_out[i]);
//    }
//    printf("    [step apply] index = %d end\n",index);
    return;
}

__device__
void rk45_gpu_evolve_apply(double t, double t_target, double t_delta, double h, double* y[], double* y_output[], int index, GPU_Parameters* params){
    double device_y[DIM];
    double device_y_0[DIM];
    double device_y_err[DIM];
    double device_dydt_out[DIM];
    while(t < t_target)
    {
      double device_t;
      double device_t1;
      double device_h;
      double device_h_0;
      double device_dt;
      int device_adjustment_out = 999;
      device_t = t;
      device_t1 = device_t + t_delta;
      device_h = h;

      int day = t;
//      printf("day %d\t", day);
//      for (int i = 0; i < params->dimension; i ++) {
//        printf("y[%d][%d] = %.1f\t", index, i, y[index][i]);
//        if(i == (params->dimension - 1)){
//          printf("\n");
//        }
//      }
      for (int i = 0; i < params->display_dimension; i ++) {
        const int y_output_index = day * params->display_dimension + i;
        if(y_output_index % params->display_dimension == 0){
          //First column
          y_output[index][y_output_index] = day*1.0;
//          printf("First day = %d index = %d i = %d y_output_index = %d y_output[%d][%d] = %.5f\n",
//                 day, index, i, y_output_index, index, y_output_index, y_output[index][y_output_index]);
        }
        else if(y_output_index % params->display_dimension == 1){
          //Second column
          y_output[index][y_output_index] = seasonal_transmission_factor(params,t);
//          printf("Second day = %d index = %d i = %d y_output_index = %d y_output[%d][%d] = %.5f\n",
//                 day, index, i, y_output_index, index, y_output_index, y_output[index][y_output_index]);
        }
        else if(y_output_index % params->display_dimension == 2){
          //Third column
          y_output[index][y_output_index] = pop_sum(y[index]);
//          printf("Third day = %d index = %d i = %d y_output_index = %d y_output[%d][%d] = %.5f\n",
//                 day, index, i, y_output_index, index, y_output_index, y_output[index][y_output_index]);
        }
        else if(day > 0 && (y_output_index % (params->display_dimension)) - (params->display_dimension - 1) == -2){
          //INC1 - last column -2
          printf("y_output_index = %d - inc1\n",y_output_index);
          y_output[index][y_output_index] = y_output[index][y_output_index - 4];
        }
        else if(day > 0 && (y_output_index % (params->display_dimension)) - (params->display_dimension - 1) == -1){
          //INC2 - last column -1
          printf("y_output_index = %d - inc2\n",y_output_index);
          y_output[index][y_output_index] = y_output[index][y_output_index - 4];
        }
        else if(day > 0 && (y_output_index % (params->display_dimension)) - (params->display_dimension - 1) == 0){
          //INC3 - last column
          printf("y_output_index = %d - inc3\n",y_output_index);
          y_output[index][y_output_index] = y_output[index][y_output_index - 4];
        }
        else{
          //Forth column onward
          const int y_index = (y_output_index - 3) % params->display_dimension;
          y_output[index][y_output_index] = y[index][y_index];
//          printf("day = %d index = %d i = %d y_output_index = %d y[%d][%d] = y[%d][%d] = %.5f\n",
//                 day, index, i, y_output_index, index, y_output_index, index, y_index, y[index][y_index]);
        }
      }

      while(device_t < device_t1)
      {
        int device_final_step = 0;
        const double device_t_0 = device_t;
        device_h_0 = device_h;
        device_dt = device_t1 - device_t_0;
        //                if(index == 0){
        //                    printf("\n  [evolve apply] index = %d start\n",index);
        //                    printf("    t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f dt = %.10f\n",device_t,device_t_0,device_h,device_h_0,device_dt);
        //                }

        for (int i = 0; i < params->dimension; i ++){
          device_y[i] = y[index][i];
          device_y_0[i] = device_y[i];
        }

        device_final_step = 0;

        while(true){
          if ((device_dt >= 0.0 && device_h_0 > device_dt) || (device_dt < 0.0 && device_h_0 < device_dt)) {
            device_h_0 = device_dt;
            device_final_step = 1;
          } else {
            device_final_step = 0;
          }

          rk45_gpu_step_apply(device_t_0,device_h_0,device_y,device_y_err,device_dydt_out,
                              index, params);

          if (device_final_step) {
            device_t = device_t1;
          } else {
            device_t = device_t_0 + device_h_0;
          }

          double h_old = device_h_0;

          //                    printf("    before adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",device_t,device_t_0,device_h,device_h_0,h_old);

          rk45_gpu_adjust_h(device_y, device_y_err, device_dydt_out,
                            device_h, device_h_0, device_adjustment_out, device_final_step,index);

          //Extra step to get data from h
          device_h_0 = device_h;

          //                    printf("    after adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",device_t,device_t_0,device_h,device_h_0,h_old);

          if (device_adjustment_out == -1)
          {
            double t_curr = (device_t);
            double t_next = (device_t) + device_h_0;

            if (fabs(device_h_0) < fabs(h_old) && t_next != t_curr) {
              /* Step was decreased. Undo step, and try again with new h0. */
              //                            printf("  [evolve apply] index = %d step decreased, y = y0\n",index);
              for (int i = 0; i < DIM; i++) {
                device_y[i] = device_y_0[i];
              }
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
        h = device_h;
        for (int i = 0; i < DIM; i++){
          y[index][i] = device_y[i];
        }
        //                printf("    index = %d t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f\n",index,device_t,device_t_0,device_h,device_h_0);
        //                for (int i = 0; i < DIM; i++){
        //                    printf("    index = %d y[%d][%d] = %.10f\n",index,index,i,device_y[i]);
        //                }
        //                printf("  [evolve apply] index = %d end\n",index);
        //                if(device_final_step){
        //                    printf("[output] index = %d t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f\n",index,device_t,device_t_0,device_h,device_h_0);
        //                    for (int i = 0; i < DIM; i++){
        //                        printf("[output] index = %d y[%d] = %.10f\n",index,i,device_y[i]);
        //                    }
        //                }
        //                device_t = device_t_0 + device_h_0;
      }
      //            if(index == 0) {
      //                printf("[evolve apply] Index = %d t = %f h = %f end one day\n", index, t, h);
      //            }
      t += t_delta;
    }
    //        if(index == 0){
    //            for (int i = 0; i < DIM; i++){
    //                printf("[output] index = %d y[%d] = %1.5f\n",index,i,device_y[i]);
    //            }
    //        }
}

__device__
void solve_ode(double* y_d[], double* y_output_d[], int index, GPU_Parameters* params){
    rk45_gpu_evolve_apply(params->t0, params->t_target, params->step, params->h, y_d, y_output_d, index, params);
    return;
}

__device__
void mcmc(double* y_output_d[], int index, GPU_Parameters* params){
    for(int i = 0; i < NUMDAYSOUTPUT * params->display_dimension; i++){
      printf("%.1f\t", y_output_d[index][i]);
      if(i > 0 && (i + 1) % params->display_dimension == 0){
        printf("\n");
      }
    }
    return;
}

__global__
void solve_ode_mcmc(double* y_d[], double* y_output_d[], GPU_Parameters* params){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;


    for(int index = index_gpu; index < NUMODE; index += stride)
    {
      for(int iter = 0; iter < 1; iter++){
        solve_ode(y_d, y_output_d, index, params);


//        mcmc(y_output_d, index, params);
      }
    }
    return;
}

void GPU_RK45::run(){
    int num_SMs;
    checkCuda(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
    //    int numBlocks = 32*num_SMs; //multiple of 32
    params->block_size = 256; //max is 1024
    params->num_blocks = (NUMODE + params->block_size - 1) / params->block_size;
    printf("[GSL GPU] block_size = %d num_blocks = %d\n",params->block_size,params->num_blocks);

    auto start = std::chrono::high_resolution_clock::now();
    double **y_d = 0;
    //temp pointers
    double **tmp_ptr = (double**)malloc (NUMODE * sizeof (double));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMalloc ((void **)&tmp_ptr[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y[i], params->dimension * sizeof(double), cudaMemcpyHostToDevice));
    }
    //y_d
    checkCuda(cudaMalloc ((void **)&y_d, NUMODE * sizeof (double)));
    checkCuda(cudaMemcpy (y_d, tmp_ptr, NUMODE * sizeof (double), cudaMemcpyHostToDevice));

    double **y_output_d = 0;
    //temp pointers
    for (int i = 0; i < NUMODE; i++) {
      checkCuda(cudaMalloc ((void **)&tmp_ptr[i], NUMDAYSOUTPUT * params->display_dimension * sizeof (double)));
      checkCuda(cudaMemcpy(tmp_ptr[i], params->y_output[i], NUMDAYSOUTPUT * params->display_dimension * sizeof(double), cudaMemcpyHostToDevice));
    }
    //y_output_d
    checkCuda(cudaMalloc ((void **)&y_output_d, NUMODE * sizeof (double)));
    checkCuda(cudaMemcpy (y_output_d, tmp_ptr, NUMODE * sizeof (double), cudaMemcpyHostToDevice));

    //params_d
    GPU_Parameters* params_d;
    checkCuda(cudaMalloc((void **) &params_d, sizeof(GPU_Parameters)));
    checkCuda(cudaMemcpy(params_d, params, sizeof(GPU_Parameters), cudaMemcpyHostToDevice));

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %ld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
//    cudaProfilerStart();
    solve_ode_mcmc<<<params->num_blocks, params->block_size>>>(y_d, y_output_d, params_d);

//    cudaProfilerStop();
    checkCuda(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for calculating %d ODE with %d parameters on GPU: %ld micro seconds which is %.10f seconds\n",NUMODE,DIM,duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
    tmp_ptr = (double**)malloc (NUMODE * sizeof (double));
    double** y_output_h = (double**)malloc (NUMODE * sizeof (double));
    for (int i = 0; i < NUMODE; i++) {
      y_output_h[i] = (double *)malloc (NUMDAYSOUTPUT * params->display_dimension * sizeof (double));
    }
    checkCuda(cudaMemcpy (tmp_ptr, y_output_d, NUMODE * sizeof (double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMemcpy (y_output_h[i], tmp_ptr[i], NUMDAYSOUTPUT * params->display_dimension * sizeof (double), cudaMemcpyDeviceToHost));
    }
    checkCuda(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for data transfer GPU to CPU: %ld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, NUMODE); // define the range

    for(int i = 0; i < params->display_number; i++){
        int random_index = 0;
        if(NUMODE == 1){
            random_index = 0;
        }
        else{
            random_index = distr(gen);
        }
        printf("Display y_output_h[%d]\n",random_index);
        for(int index = 0; index < NUMDAYSOUTPUT * params->display_dimension; index++){
          printf("%.1f\t", y_output_h[random_index][index]);
          if(index > 0 && (index + 1) % params->display_dimension == 0){
            printf("\n");
          }
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for display random results on CPU: %ld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));
    printf("\n");
    // Free memory
    checkCuda(cudaFree(y_d));
    return;
}