#include "gpu_ode_mcmc.h"

inline
cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
      assert(result == cudaSuccess);
    }
#endif
    return result;
}

GPUFlu::GPUFlu() {
    params = new GPUParameters();
}

GPUFlu::~GPUFlu() {
    params = nullptr;
}

void GPUFlu::set_parameters(GPUParameters *params_) {
    params = &(*params_);
}

double GPUFlu::rand_uniform(double range_from, double range_to) {
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_real_distribution<double>    distr(range_from, range_to);
    return distr(generator);
}

void GPUFlu::run() {
    int num_SMs;
    checkCuda(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
    //    int numBlocks = 32*num_SMs; //multiple of 32
    params->block_size = 256; //max is 1024
    params->num_blocks = (NUMODE*DATADIM_ROWS + params->block_size - 1) / params->block_size;
    printf("[GSL GPU] block_size = %d num_blocks = %d\n", params->block_size, params->num_blocks);

    auto start = std::chrono::high_resolution_clock::now();

    //y_ode_input_d
    double **y_ode_input_d = 0;
    //temp pointers
    double **tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], params->ode_dimension * sizeof(double)));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_input[i], params->ode_dimension * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_ode_input_d, NUMODE * sizeof(double)));
    checkCuda(cudaMemcpy(y_ode_input_d, tmp_ptr, NUMODE * sizeof(double), cudaMemcpyHostToDevice));

    //y_ode_output_d
    double **y_ode_output_d = 0;
    //y_ode_output_d
    //temp pointers
    tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], NUMDAYSOUTPUT * params->display_dimension * sizeof(double)));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_output[i],
                             NUMDAYSOUTPUT * params->display_dimension * sizeof(double), cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_ode_output_d, NUMODE * sizeof(double)));
    checkCuda(cudaMemcpy(y_ode_output_d, tmp_ptr, NUMODE * sizeof(double), cudaMemcpyHostToDevice));

    //y_data_input_d
    double **y_data_input_d = 0;
    //temp pointers
    tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], params->data_dimension * sizeof(double)));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_data_input[i], params->data_dimension * sizeof(double),cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_data_input_d, NUMODE * sizeof(double)));
    checkCuda(cudaMemcpy(y_data_input_d, tmp_ptr, NUMODE * sizeof(double), cudaMemcpyHostToDevice));

    //y_ode_agg_d
    double **y_ode_agg_d = 0;
    //temp pointers
    tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], NUMDAYSOUTPUT * params->agg_dimension * sizeof(double)));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_agg[i], NUMDAYSOUTPUT * params->agg_dimension * sizeof(double),cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_ode_agg_d, NUMODE * sizeof(double)));
    checkCuda(cudaMemcpy(y_ode_agg_d, tmp_ptr, NUMODE * sizeof(double), cudaMemcpyHostToDevice));

    //y_mcmc_dnorm_h1b_h3_d
    double **y_mcmc_dnorm_h1b_h3_d = 0;
    //temp pointers
    tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], params->data_dimension * sizeof(double)));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_agg[i], params->data_dimension * sizeof(double),cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_h1b_h3_d, NUMODE * sizeof(double)));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_h1b_h3_d, tmp_ptr, NUMODE * sizeof(double), cudaMemcpyHostToDevice));

    //y_mcmc_dnorm_h1_d
    double **y_mcmc_dnorm_h1_d = 0;
    //temp pointers
    tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], params->data_params.rows * sizeof(double)));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_agg[i], params->data_params.rows * sizeof(double),cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_h1_d, NUMODE * sizeof(double)));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_h1_d, tmp_ptr, NUMODE * sizeof(double), cudaMemcpyHostToDevice));

    //y_mcmc_dnorm_b_d
    double **y_mcmc_dnorm_b_d = 0;
    //temp pointers
    tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], params->data_params.rows * sizeof(double)));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_agg[i], params->data_params.rows * sizeof(double),cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_b_d, NUMODE * sizeof(double)));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_b_d, tmp_ptr, NUMODE * sizeof(double), cudaMemcpyHostToDevice));

    //y_mcmc_dnorm_h3_d
    double **y_mcmc_dnorm_h3_d = 0;
    //temp pointers
    tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], params->data_params.rows * sizeof(double)));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_agg[i], params->data_params.rows * sizeof(double),cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_h3_d, NUMODE * sizeof(double)));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_h3_d, tmp_ptr, NUMODE * sizeof(double), cudaMemcpyHostToDevice));

    double y_mcmc_dnorm_1d_h[DATADIM_ROWS];
    for (int i = 0; i < DATADIM_ROWS; i++) {
        y_mcmc_dnorm_1d_h[i] = i;
    }
    //y_mcmc_dnorm_h1_1d_d
    double *y_mcmc_dnorm_h1_1d_d = 0;
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_h1_1d_d, params->data_params.rows * sizeof(double)));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_h1_d, y_mcmc_dnorm_1d_h, params->data_params.rows * sizeof(double), cudaMemcpyHostToDevice));
    //y_mcmc_dnorm_b_1d_d
    double *y_mcmc_dnorm_b_1d_d = 0;
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_b_1d_d, params->data_params.rows * sizeof(double)));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_b_1d_d, y_mcmc_dnorm_1d_h, params->data_params.rows * sizeof(double), cudaMemcpyHostToDevice));
    //y_mcmc_dnorm_h3_1d_d
    double *y_mcmc_dnorm_h3_1d_d = 0;
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_h3_1d_d, params->data_params.rows * sizeof(double)));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_h3_1d_d, y_mcmc_dnorm_1d_h, params->data_params.rows * sizeof(double), cudaMemcpyHostToDevice));
    //data_test_sum_d
    double* data_test_sum_d = 0;
    checkCuda(cudaMalloc((void **) &data_test_sum_d, params->data_params.rows * sizeof(double)));
    checkCuda(cudaMemcpy(data_test_sum_d, y_mcmc_dnorm_1d_h, params->data_params.rows * sizeof(double), cudaMemcpyHostToDevice));
    //dnorm_sum_d
    double* dnorm_sum_d = 0;
    checkCuda(cudaMalloc((void **) &dnorm_sum_d, params->data_params.rows * sizeof(double)));
    checkCuda(cudaMemcpy(dnorm_sum_d, y_mcmc_dnorm_1d_h, params->data_params.rows * sizeof(double), cudaMemcpyHostToDevice));
    //dnorm_sum_h
    double dnorm_sum_h[DATADIM_ROWS];
    for(int i = 0; i < params->data_params.rows; i++){
        dnorm_sum_h[i] = 0.0;
    }

    //params_d
    GPUParameters *params_d;
    checkCuda(cudaMalloc((void **) &params_d, sizeof(GPUParameters)));
    checkCuda(cudaMemcpy(params_d, params, sizeof(GPUParameters), cudaMemcpyHostToDevice));

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %ld micro seconds which is %.10f seconds\n", duration.count(),
           (duration.count() / 1e6));

    start = std::chrono::high_resolution_clock::now();
    //    cudaProfilerStart();
    double r_denom = 0.0;
    double r_num = 0.0;
    GPUParameters* params_temp = params;
    for (int iter = 0; iter < params->mcmc_loop; iter++) {
        params->block_size = 256; //max is 1024
        params->num_blocks = (NUMODE + params->block_size - 1) / params->block_size;
        solve_ode<<<params->num_blocks, params->block_size>>>(
                y_ode_input_d, y_ode_output_d, y_ode_agg_d, params_d);
        checkCuda(cudaDeviceSynchronize());
        params->block_size = 512; //max is 1024
        params->num_blocks = (NUMODE*DATADIM_ROWS + params->block_size - 1) / params->block_size;
        mcmc_dnorm<<<params->num_blocks, params->block_size>>>(y_data_input_d, y_ode_agg_d, y_mcmc_dnorm_h1b_h3_d, params_d);
        checkCuda(cudaDeviceSynchronize());
        mcmc_get_dnorm_outputs_1d<<<params->num_blocks, params->block_size>>>(y_mcmc_dnorm_h1b_h3_d, y_mcmc_dnorm_h1_1d_d, y_mcmc_dnorm_b_1d_d, y_mcmc_dnorm_h3_1d_d, params_d);
        checkCuda(cudaDeviceSynchronize());
        reduce_sum<<<params->num_blocks, params->block_size>>>(data_test_sum_d, dnorm_sum_d,  params->data_params.rows);
        checkCuda(cudaMemcpy(dnorm_sum_h,dnorm_sum_d,params->data_params.rows * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());
        printf("sum reduce data_test_sum_d = %.5f\n",dnorm_sum_h[0]+dnorm_sum_h[1]);
        reduce_sum<<<params->num_blocks, params->block_size>>>(y_mcmc_dnorm_h1_1d_d, dnorm_sum_d,  params->data_params.rows);
        checkCuda(cudaMemcpy(dnorm_sum_h,dnorm_sum_d,params->data_params.rows * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());
        printf("sum reduce r_denom = %.5f\n",dnorm_sum_h[0]+dnorm_sum_h[1]);
        r_denom += dnorm_sum_h[0]+dnorm_sum_h[1];
        reduce_sum<<<params->num_blocks, params->block_size>>>(y_mcmc_dnorm_b_1d_d, dnorm_sum_d,  params->data_params.rows);
        checkCuda(cudaMemcpy(dnorm_sum_h,dnorm_sum_d,params->data_params.rows * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());
        printf("sum reduce r_denom = %.5f\n",dnorm_sum_h[0]+dnorm_sum_h[1]);
        r_denom += dnorm_sum_h[0]+dnorm_sum_h[1];
        reduce_sum<<<params->num_blocks, params->block_size>>>(y_mcmc_dnorm_h3_1d_d, dnorm_sum_d,  params->data_params.rows);
        checkCuda(cudaMemcpy(dnorm_sum_h,dnorm_sum_d,params->data_params.rows * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());
        printf("sum reduce r_denom = %.5f\n",dnorm_sum_h[0]+dnorm_sum_h[1]);
        r_denom += dnorm_sum_h[0]+dnorm_sum_h[1];

        //
        // Generate new parameters
        //

        params_temp->update();

        //
        // Copy new parameters to gpu
        //

        checkCuda(cudaMemcpy(params_d, params_temp, sizeof(GPUParameters), cudaMemcpyHostToDevice));
        checkCuda(cudaDeviceSynchronize());

        //
        // Solve ode with new parameters
        //
        params->block_size = 256; //max is 1024
        params->num_blocks = (NUMODE + params->block_size - 1) / params->block_size;
        solve_ode<<<params->num_blocks, params->block_size>>>(
                y_ode_input_d, y_ode_output_d, y_ode_agg_d, params_d);
        checkCuda(cudaDeviceSynchronize());
        params->block_size = 512; //max is 1024
        params->num_blocks = (NUMODE*DATADIM_ROWS + params->block_size - 1) / params->block_size;
        mcmc_dnorm<<<params->num_blocks, params->block_size>>>(y_data_input_d, y_ode_agg_d, y_mcmc_dnorm_h1b_h3_d, params_d);
        checkCuda(cudaDeviceSynchronize());
        mcmc_get_dnorm_outputs_1d<<<params->num_blocks, params->block_size>>>(y_mcmc_dnorm_h1b_h3_d, y_mcmc_dnorm_h1_1d_d, y_mcmc_dnorm_b_1d_d, y_mcmc_dnorm_h3_1d_d, params_d);
//        reduce_sum<<<params->num_blocks, params->block_size>>>(data_test_sum_d, dnorm_sum_d,  params->data_params.rows);
//        checkCuda(cudaMemcpy(dnorm_sum_h,dnorm_sum_d,params->data_params.rows * sizeof(double), cudaMemcpyDeviceToHost));
//        checkCuda(cudaDeviceSynchronize());
//        printf("sum reduce data_test_sum_d = %.5f\n",dnorm_sum_h[0] + dnorm_sum_h[1]);
        reduce_sum<<<params->num_blocks, params->block_size>>>(y_mcmc_dnorm_h1_1d_d, dnorm_sum_d,  params->data_params.rows);
        checkCuda(cudaMemcpy(dnorm_sum_h,dnorm_sum_d,params->data_params.rows * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());
        printf("sum reduce r_num = %.5f\n",dnorm_sum_h[0]+dnorm_sum_h[1]);
        r_num += dnorm_sum_h[0]+dnorm_sum_h[1];
        reduce_sum<<<params->num_blocks, params->block_size>>>(y_mcmc_dnorm_b_1d_d, dnorm_sum_d,  params->data_params.rows);
        checkCuda(cudaMemcpy(dnorm_sum_h,dnorm_sum_d,params->data_params.rows * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());
        printf("sum reduce r_num = %.5f\n",dnorm_sum_h[0]+dnorm_sum_h[1]);
        r_num += dnorm_sum_h[0]+dnorm_sum_h[1];
        reduce_sum<<<params->num_blocks, params->block_size>>>(y_mcmc_dnorm_h3_1d_d, dnorm_sum_d,  params->data_params.rows);
        checkCuda(cudaMemcpy(dnorm_sum_h,dnorm_sum_d,params->data_params.rows * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());
        printf("sum reduce r_num = %.5f\n",dnorm_sum_h[0]+dnorm_sum_h[1]);
        r_num += dnorm_sum_h[0]+dnorm_sum_h[1];

        double r  = r_num - r_denom;
        if(exp(r) > rand_uniform(0.0,1.0)){
            params = params_temp;
            r_denom = r_num;
            printf("iter %d accept params (r = %.5f)\n",iter,r);
        }
        else{
            printf("iter %d reject params_temp (r = %.5f)\n",iter,r);
        }
    }

    //    cudaProfilerStop();
    checkCuda(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for MCMC %d ODE with %d parameters %d times on GPU: %ld micro seconds which is %.10f seconds\n",
           NUMODE, DIM, params->mcmc_loop, duration.count(), (duration.count() / 1e6));

    start = std::chrono::high_resolution_clock::now();
    //y_ode_output_h
    tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    double **y_ode_output_h = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        y_ode_output_h[i] = (double *) malloc(NUMDAYSOUTPUT * params->display_dimension * sizeof(double));
    }
    checkCuda(cudaMemcpy(tmp_ptr, y_ode_output_d, NUMODE * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMemcpy(y_ode_output_h[i], tmp_ptr[i], NUMDAYSOUTPUT * params->display_dimension * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }
    //y_output_mcmc_h
    tmp_ptr = (double **) malloc(NUMODE * sizeof(double));
    double **y_output_agg_h = (double **) malloc(NUMODE * sizeof(double));
    for (int i = 0; i < NUMODE; i++) {
        y_output_agg_h[i] = (double *) malloc(NUMDAYSOUTPUT * params->agg_dimension * sizeof(double));
    }
    checkCuda(cudaMemcpy(tmp_ptr, y_ode_agg_d, NUMODE * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUMODE; i++) {
        checkCuda(cudaMemcpy(y_output_agg_h[i], tmp_ptr[i], NUMDAYSOUTPUT * params->agg_dimension * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }
    checkCuda(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for data transfer GPU to CPU: %ld micro seconds which is %.10f seconds\n", duration.count(),
           (duration.count() / 1e6));

    start = std::chrono::high_resolution_clock::now();
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, NUMODE); // define the range

    for (int i = 0; i < params->display_number; i++) {
        int random_index = 0;
        if (NUMODE == 1) {
            random_index = 0;
        } else {
            random_index = distr(gen);
        }
//        printf("Display y_ode_output_h[%d]\n",random_index);
//        for(int index = 0; index < NUMDAYSOUTPUT * params->display_dimension; index++){
//          printf("%.5f\t", y_ode_output_h[random_index][index]);
//          if(index > 0 && (index + 1) % params->display_dimension == 0){
//            printf("\n");
//          }
//        }
        printf("Display y_output_agg_h[%d]\n", random_index);
        for (int index = 0; index < NUMDAYSOUTPUT * params->agg_dimension; index++) {
            const int line_index = (index / params->agg_dimension) % NUMDAYSOUTPUT;
            if(line_index < 10){
                printf("%d %.5f\t",line_index, y_output_agg_h[random_index][index]);
                if (index > 0 && (index + 1) % params->agg_dimension == 0) {
                    printf("\n");
                }
            }
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for display random results on CPU: %ld micro seconds which is %.10f seconds\n",
           duration.count(), (duration.count() / 1e6));
    printf("\n");
    // Free memory
    checkCuda(cudaFree(y_ode_input_d));
    checkCuda(cudaFree(y_ode_output_d));
    checkCuda(cudaFree(y_ode_agg_d));
    checkCuda(cudaFree(y_data_input_d));
    checkCuda(cudaFree(y_mcmc_dnorm_h1_1d_d));
    checkCuda(cudaFree(y_mcmc_dnorm_b_1d_d));
    checkCuda(cudaFree(y_mcmc_dnorm_h3_1d_d));
    checkCuda(cudaFree(y_mcmc_dnorm_h1_d));
    checkCuda(cudaFree(y_mcmc_dnorm_b_d));
    checkCuda(cudaFree(y_mcmc_dnorm_h3_d));
    checkCuda(cudaFree(y_mcmc_dnorm_h1b_h3_d));
    checkCuda(cudaFree(dnorm_sum_d));
    checkCuda(cudaFree(data_test_sum_d));
    checkCuda(cudaFree(params_d));
    delete params;
//    delete params_temp;
    delete y_ode_output_h;
    delete y_output_agg_h;
    return;
}