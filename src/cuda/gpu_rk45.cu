#include "gpu_rk45.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
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

    for(int i=0; i<gpu_params->phis_d_length; i++)
    {
        if( std::fabs( t - gpu_params->phis_d[i] ) < (gpu_params->v_d[gpu_params->i_epidur] / 2))
        {
            // sine_function_value = sin( 2.0 * 3.141592653589793238 * (phis[i]-t+91.25) / 365.0 );
            sine_function_value = std::sin( 2.0 * 3.141592653589793238 * (gpu_params->phis_d[i] - t +(gpu_params->v_d[gpu_params->i_epidur] / 2)) / (gpu_params->v_d[gpu_params->i_epidur] * 2));
//            printf("      in loop %1.3f %d  %1.3f %1.3f\n", t, i, gpu_params->phis_d[i], sine_function_value );
        }
    }
//    printf("    %f sine_function_value %1.3f\n",t,sine_function_value);
//    printf("    %f return %1.3f\n",t,1.0 + v[i_amp] * sine_function_value);
    return 1.0 + gpu_params->v_d[gpu_params->i_amp] * sine_function_value;
}


void GPU_RK45::run(){

    auto start_all = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();

    //t_d
    double* t_d;
    gpuErrchk(cudaMalloc ((void **)&t_d, sizeof (double)));
    gpuErrchk(cudaMemcpy (t_d, &params->t0, sizeof (double), cudaMemcpyHostToDevice));
    //t_0_d
    double* t_0_d;
    gpuErrchk(cudaMalloc ((void **)&t_0_d, sizeof (double)));
    gpuErrchk(cudaMemcpy (t_0_d, &params->t0, sizeof (double), cudaMemcpyHostToDevice));
    //t_0_d
    double t0_tmp = 0.0;
    double* t_0_tmp_d;
    gpuErrchk(cudaMalloc ((void **)&t_0_tmp_d, sizeof (double)));
    gpuErrchk(cudaMemcpy (t_0_tmp_d, &t0_tmp, sizeof (double), cudaMemcpyHostToDevice));
    //h_d
    double* h_d;
    gpuErrchk(cudaMalloc ((void **)&h_d, sizeof (double)));
    gpuErrchk(cudaMemcpy (h_d, &params->h, sizeof (double), cudaMemcpyHostToDevice));
    //h_0_d
    double* h_0_d;
    gpuErrchk(cudaMalloc ((void **)&h_0_d, sizeof (double)));
    gpuErrchk(cudaMemcpy (h_0_d, &params->h, sizeof (double), cudaMemcpyHostToDevice));
    //adjustment_out_d
    int adjustment_out = 999;
    int* adjustment_out_d;
    gpuErrchk(cudaMalloc ((void **)&adjustment_out_d, sizeof (double)));
    gpuErrchk(cudaMemcpy (adjustment_out_d, &(adjustment_out), sizeof (double), cudaMemcpyHostToDevice));

    //y
    double *y_d;
    gpuErrchk(cudaMalloc ((void **)&y_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //y_output
    double *y_output_d;
    gpuErrchk(cudaMalloc ((void **)&y_output_d, static_cast<int>(params->t_target) * params->display_dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_output_d, params->y_output, static_cast<int>(params->t_target) * params->display_dimension * sizeof (double), cudaMemcpyHostToDevice));
    //y_0
    double *y_0_d;
    gpuErrchk(cudaMalloc ((void **)&y_0_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_0_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //y_tmp
    double *y_tmp_d;
    gpuErrchk(cudaMalloc ((void **)&y_tmp_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_tmp_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //y_err
    double *y_err_d;
    gpuErrchk(cudaMalloc ((void **)&y_err_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_err_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //dydt_in_d
    double *dydt_in_d;
    gpuErrchk(cudaMalloc ((void **)&dydt_in_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (dydt_in_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //dydt_out_d
    double *dydt_out_d;
    gpuErrchk(cudaMalloc ((void **)&dydt_out_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (dydt_out_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //k1_d
    double *k1_d;
    gpuErrchk(cudaMalloc ((void **)&k1_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (k1_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //k2_d
    double *k2_d;
    gpuErrchk(cudaMalloc ((void **)&k2_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (k2_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //k3_d
    double *k3_d;
    gpuErrchk(cudaMalloc ((void **)&k3_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (k3_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //k4_d
    double *k4_d;
    gpuErrchk(cudaMalloc ((void **)&k4_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (k4_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //k5_d
    double *k5_d;
    gpuErrchk(cudaMalloc ((void **)&k5_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (k5_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //k6_d
    double *k6_d;
    gpuErrchk(cudaMalloc ((void **)&k6_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (k6_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //r_d
    double *r_d;
    gpuErrchk(cudaMalloc ((void **)&r_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (r_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    //r_max_d
    double r_max = 0.0;
    double *r_max_d;
    gpuErrchk(cudaMalloc ((void **)&r_max_d, params->dimension * sizeof (double)));
    gpuErrchk(cudaMemcpy (r_max_d, params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));

    //params_d
    GPU_Parameters* params_d;
    gpuErrchk(cudaMalloc((void **)&params_d, sizeof(GPU_Parameters)));
    gpuErrchk(cudaMemcpy(params_d, params, sizeof(GPU_Parameters), cudaMemcpyHostToDevice));

    int num_SMs;
    gpuErrchk(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
//    int numBlocks = 32*num_SMs; //multiple of 32
    int block_size = 256; //max is 1024
    int num_blocks = (params->dimension + block_size - 1) / block_size;
//    printf("[GSL GPU] SMs = %d block_size = %d num_blocks = %d\n",num_SMs,block_size,num_blocks);
    dim3 dimBlock(block_size, block_size); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
    dim3 dimGrid(num_blocks, num_blocks); // 1*1 blocks in a grid
    params->block_size = block_size;
    params->num_blocks = num_blocks;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %ld micro seconds which is %f seconds\n",duration.count(),(duration.count()/1e6));

//    cudaFuncSetCacheConfig(rk45_gpu_evolve_apply, cudaFuncCachePreferL1);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024000*100);

    start = std::chrono::high_resolution_clock::now();

    while(params->t0 < params->t_target)
    {
        printf("[while(params->t0 < params->t_target)] t = %f h = %f start one day\n", params->t0, params->h);
        double t;
        double t1;
        t = params->t0;
        t1 = t + 1.0;
        printf("  Will run from %f to %f, step %f\n", t, t1, params->h);
        while(t < t1)
        {
            //ODE here
            const double t0 = t;
            double h0 = params->h;
            int final_step = 0;
            double dt = t1 - t;
            gpuErrchk(cudaMemcpy(t_0_d, &t0, sizeof(double), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(y_0_d, y_d, params->dimension * sizeof(double), cudaMemcpyDeviceToDevice));

//            printf("  [while(t < t1)] t = %f t1 = %f t0 = %f h = %f h0 = %f start one time step\n", t, t1, t0, params->h, h0);
//            printf("    t = %f t0 = %f  h = %f h0 = %f dt = %f\n",t,t0,params->h,h0,dt);

            gpu_func_test<<<1,params->dimension>>>(t_0_d,y_d,dydt_in_d,params_d);
            gpuErrchk(cudaDeviceSynchronize());

            while(true)
            {
//                printf("    t = %f t0 = %f  h = %f h0 = %f dt = %f start step apply & adjust\n",t,t0,params->h,h0,dt);

                if ((dt >= 0.0 && h0 > dt) || (dt < 0.0 && h0 < dt))
                {
                    h0 = dt;
                    final_step = 1;
                }
                else
                {
                    final_step = 0;
                }
                gpuErrchk(cudaMemcpy(h_0_d, &h0, sizeof(double), cudaMemcpyHostToDevice));

                //step apply
                /* k1 */
                if (dydt_in_d != NULL)
                {
                    gpuErrchk(cudaMemcpy(k1_d, dydt_in_d, params->dimension * sizeof(double), cudaMemcpyDeviceToDevice));
//                    printf("    k1_d = dydt_in\n");
                }
                else {
                    gpu_func_test<<<1,params->dimension>>>(t_0_d,y_d,k1_d,params_d);
                    gpuErrchk(cudaDeviceSynchronize());
                }
//                printf("    k1 done\n");
                /* k2 */
                calculate_y<<<1,params->dimension>>>(y_d, y_tmp_d, nullptr, h_0_d, 1,
                                                                             k1_d,nullptr,nullptr,
                                                                                nullptr,nullptr,nullptr,
                                                                                params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    calculate k2\n");
                t0_tmp = t0 + ah[0] * h0;
                gpuErrchk(cudaMemcpy (t_0_tmp_d, &t0_tmp, sizeof (double), cudaMemcpyHostToDevice));
                gpu_func_test<<<1,params->dimension>>>(t_0_tmp_d,y_tmp_d,k2_d,params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    k2 done\n");
                /* k3 */
                calculate_y<<<1,params->dimension>>>(y_d, y_tmp_d, nullptr, h_0_d, 2,
                                                                             k1_d, k2_d,nullptr,
                                                                            nullptr,nullptr,nullptr,
                                                                            params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    calculate k3\n");
                t0_tmp = t0 + ah[1] * h0;
                gpuErrchk(cudaMemcpy (t_0_tmp_d, &t0_tmp, sizeof (double), cudaMemcpyHostToDevice));
                gpu_func_test<<<1,params->dimension>>>(t_0_tmp_d,y_tmp_d,k3_d,params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    k3 done\n");
                /* k4 */
                calculate_y<<<1,params->dimension>>>(y_d, y_tmp_d, nullptr, h_0_d, 3,
                                                     k1_d, k2_d, k3_d,
                                                    nullptr,nullptr,nullptr,
                                                    params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    calculate k4\n");
                t0_tmp = t0 + ah[2] * h0;
                gpuErrchk(cudaMemcpy (t_0_tmp_d, &t0_tmp, sizeof (double), cudaMemcpyHostToDevice));
                gpu_func_test<<<1,params->dimension>>>(t_0_tmp_d,y_tmp_d,k4_d,params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    k4 done\n");
                /* k5 */
                calculate_y<<<1,params->dimension>>>(y_d, y_tmp_d, nullptr, h_0_d, 4,
                                                                            k1_d, k2_d, k3_d,
                                                                            k4_d, nullptr, nullptr,
                                                                            params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    calculate k5\n");
                t0_tmp = t0 + ah[3] * h0;
                gpuErrchk(cudaMemcpy (t_0_tmp_d, &t0_tmp, sizeof (double), cudaMemcpyHostToDevice));
                gpu_func_test<<<1,params->dimension>>>(t_0_tmp_d,y_tmp_d,k5_d,params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    k5 done\n");
                /* k6 */
                calculate_y<<<1,params->dimension>>>(y_d, y_tmp_d, nullptr, h_0_d, 5,
                                                                            k1_d, k2_d, k3_d,
                                                                            k4_d, k5_d, nullptr,
                                                                            params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    calculate k6\n");
                t0_tmp = t0 + ah[4] * h0;
                gpuErrchk(cudaMemcpy (t_0_tmp_d, &t0_tmp, sizeof (double), cudaMemcpyHostToDevice));
                gpu_func_test<<<1,params->dimension>>>(t_0_tmp_d,y_tmp_d,k6_d,params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    k6 done\n");
                /* y */
//                printf("    calculate y\n");
                calculate_y<<<1,params->dimension>>>(y_d, y_tmp_d, nullptr, h_0_d, 6,
                                                                            k1_d, k2_d, k3_d,
                                                                            k4_d, k5_d, k6_d,
                                                                            params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    y done, calculate dydt_out\n");
                /* dydt_out */
//                printf("    calculate dydt_out\n");
                t0_tmp = t0 + h0;
                gpuErrchk(cudaMemcpy (t_0_tmp_d, &t0_tmp, sizeof (double), cudaMemcpyHostToDevice));
                gpu_func_test<<<1,params->dimension>>>(t_0_tmp_d,y_d,dydt_out_d,params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    dydt_out done\n");
                /* y_err */
//                printf("    calculate y_err\n");
                calculate_y<<<1,params->dimension>>>(y_d, y_tmp_d, y_err_d, h_0_d, 7,
                                                                            k1_d, k2_d, k3_d,
                                                                            k4_d, k5_d, k6_d,
                                                                            params_d);
                gpuErrchk(cudaDeviceSynchronize());
//                printf("    y_err done\n");

                if (final_step)
                {
                    t = t1;
                }
                else
                {
                    t = t0 + h0;
                }

                double h_old = h0;

//                printf("    calculate r\n");
                calculate_r<<<1,params->dimension>>>(y_d, y_err_d, dydt_out_d, h_0_d, h_d, final_step, r_d, params_d);
                gpuErrchk(cudaDeviceSynchronize());
                //find r_max
//                printf("    find r_max\n");
                reduce_max<<<1,params->dimension>>>(r_d,r_max_d,params->dimension);
                gpuErrchk(cudaMemcpy(&r_max, &(r_max_d[0]), sizeof(double), cudaMemcpyDeviceToHost));
                //adjust h
//                printf("    adjust h\n");
                adjust_h(r_max, h0, &params->h, final_step, &adjustment_out);
                h0 = params->h;
//                printf("    t = %f t0 = %f  h = %f h0 = %f dt = %f end step apply & adjust\n",t,t0,params->h,h0,dt);

                if (adjustment_out == -1)
                {
                    double t_curr = (t);
                    double t_next = (t) + h0;

                    if (fabs(h0) < fabs(h_old) && t_next != t_curr) {
                        /* Step was decreased. Undo step, and try again with new h0. */
//                        printf("    step decreased, y = y0\n");
                        gpuErrchk(cudaMemcpy(y_d, y_0_d, params->dimension * sizeof(double), cudaMemcpyDeviceToDevice));
                    } else {
//                        printf("    step increased, h0 = h_old\n");
                        h0 = h_old; /* keep current step size */
                        break;
                    }
                }
                else{
//                    printf("    step increased or no change\n");
                    break;
                }
            }
            gpuErrchk(cudaMemcpy(y_output_d, y_d, params->dimension * sizeof(double), cudaMemcpyDeviceToDevice));
            printf("  [while(t < t1)] t = %f t1 = %f t0 = %f h = %f h0 = %f end one time step\n", t, t1, t0, params->h, h0);
        }
        printf("[while(params->t0 < params->t_target)] t = %f h = %f end one day\n", t, params->h);
        params->t0 += 1.0;
    }

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for compute %d ODE with %d parameters, step %f in %f days on GPU: %ld micro seconds which is %f seconds\n",params->number_of_ode,params->dimension,params->h,params->t_target,duration.count(),(duration.count()/1e6));
    start = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy (params->y_output, y_output_d, static_cast<int>(params->t_target) * params->display_dimension * sizeof (double), cudaMemcpyDeviceToHost));

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for copy data from GPU on CPU: %ld micro seconds which is %f seconds\n",duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
    printf("Display on Host\n");
    for(int i = 0; i < static_cast<int>(params->t_target) * params->display_dimension; i++){
        printf("%1.1f\t",params->y_output[i]);
        //reverse position from 1D array
        if(i > 0 && (i + 1) % params->display_dimension == 0){
            printf("\n");
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for display: %ld micro seconds which is %f seconds\n",duration.count(),(duration.count()/1e6));

    auto stop_all = std::chrono::high_resolution_clock::now();
    auto duration_all = std::chrono::duration_cast<std::chrono::microseconds>(stop_all - start_all);
    printf("[GSL GPU] Time for all: %ld micro seconds which is %f seconds\n",duration_all.count(),(duration_all.count()/1e6));

    return;
}
