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
    auto start = std::chrono::high_resolution_clock::now();
    size_t ode_size = params->ode_number* sizeof(double);
    //stf_h
    double stf_h[params->ode_output_day];
    for(int i = 0; i < params->ode_output_day; i++){
        stf_h[i] = 0.0;
    }
    //stf_d;
    double *stf_d = 0;
    size_t stf_d_size = params->ode_output_day*sizeof(double);
    checkCuda(cudaMalloc((void **) &stf_d, stf_d_size));
    checkCuda(cudaMemcpy(stf_d, stf_h, stf_d_size,cudaMemcpyHostToDevice));

    //y_ode_input_d
    double **y_ode_input_d = 0;
    size_t y_ode_input_d_size = params->ode_dimension * sizeof(double);
    //temp pointers
    double **tmp_ptr = (double **) malloc(ode_size);
    for (int i = 0; i < params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_ode_input_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_input[i], y_ode_input_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_ode_input_d, ode_size));
    checkCuda(cudaMemcpy(y_ode_input_d, tmp_ptr, ode_size, cudaMemcpyHostToDevice));

    //y_ode_output_d
    double **y_ode_output_d = 0;
    size_t y_ode_output_d_size = params->ode_output_day * params->display_dimension * sizeof(double);
    //y_ode_output_d
    //temp pointers
    tmp_ptr = (double **) malloc(ode_size);
    for (int i = 0; i < params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_ode_output_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_output[i],y_ode_output_d_size, cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_ode_output_d, ode_size));
    checkCuda(cudaMemcpy(y_ode_output_d, tmp_ptr, ode_size, cudaMemcpyHostToDevice));

    //y_data_input_d
    double **y_data_input_d = 0;
    size_t y_data_input_d_size = params->data_dimension * sizeof(double);
    //temp pointers
    tmp_ptr = (double **) malloc(ode_size);
    for (int i = 0; i < params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_data_input_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_data_input[i], y_data_input_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_data_input_d, ode_size));
    checkCuda(cudaMemcpy(y_data_input_d, tmp_ptr, ode_size, cudaMemcpyHostToDevice));

    //y_ode_agg_d
    double **y_ode_agg_d = 0;
    size_t y_ode_agg_d_size = params->ode_output_day * params->agg_dimension * sizeof(double);
    //temp pointers
    tmp_ptr = (double **) malloc(ode_size);
    for (int i = 0; i < params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_ode_agg_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_agg[i], y_ode_agg_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_ode_agg_d, ode_size));
    checkCuda(cudaMemcpy(y_ode_agg_d, tmp_ptr, ode_size, cudaMemcpyHostToDevice));

    //y_mcmc_dnorm_h - single 1d array for NUMODE dnorm values (NUMODE*data_dimension.rows)
    double y_mcmc_dnorm_h[params->ode_number*params->data_params.rows];
    const int y_mcmc_dnorm_h_size = params->ode_number*params->data_params.rows;
    for (int i = 0; i < y_mcmc_dnorm_h_size; i++) {
        y_mcmc_dnorm_h[i] = 0.0;
    }

    //y_mcmc_dnorm_d - single 1d array for NUMODE dnorm values (NUMODE*data_dimension.rows)
    double *y_mcmc_dnorm_d = 0;
    size_t y_mcmc_dnorm_d_size = y_mcmc_dnorm_h_size * sizeof(double);
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_d, y_mcmc_dnorm_d_size));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_d, y_mcmc_dnorm_h, y_mcmc_dnorm_d_size,cudaMemcpyHostToDevice));

    //y_mcmc_dnorm_i_d - single 1d array
    double *y_mcmc_dnorm_i_d = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_SMs = prop.multiProcessorCount;
    // pad array with zeros to allow sum algorithm to work
    int batch_size = num_SMs * 1024;
    int padding = (batch_size - (params->data_params.rows % batch_size)) % batch_size;
    size_t y_mcmc_dnorm_i_d_padding_size = (params->data_params.rows + padding) * sizeof(double);
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_i_d, y_mcmc_dnorm_i_d_padding_size));
    checkCuda(cudaMemset(y_mcmc_dnorm_i_d, 0, y_mcmc_dnorm_i_d_padding_size));
    printf("num_sm = %d batch = %d 1 ode size = %d padding = %d total size = %d\n",num_SMs, batch_size,params->data_params.rows,padding,params->data_params.rows + padding);

//    //y_test_sum_h
    int test_padding = ceil(params->data_params.rows/1024.0)*1024 - params->data_params.rows;
    double y_test_1_ode_h[params->data_params.rows];
    int y_test_1_ode_h_size = params->data_params.rows;
    for (int i = 0; i < y_test_1_ode_h_size; i++) {
        y_test_1_ode_h[i] = 1;
    }
    double y_test_1_ode_padding_h[params->data_params.rows + test_padding];
    int y_test_1_ode_padding_h_size = params->data_params.rows + test_padding;
    memcpy(y_test_1_ode_padding_h, y_test_1_ode_h, y_test_1_ode_h_size * sizeof(double));
    memset(y_test_1_ode_padding_h + y_test_1_ode_h_size,0,test_padding* sizeof(double));
//    for(int i = 0; i < y_test_1_ode_padding_h_size; i++){
//        printf("y_test_1_ode_padding_h_size[%d] = %.1f\n",i,y_test_1_ode_padding_h[i]);
//    }

    double y_test_n_ode_padding_h[params->ode_number*(params->data_params.rows + test_padding)];
    int y_test_n_ode_padding_h_size = params->ode_number*(params->data_params.rows + test_padding);
    for(int ode_index = 0; ode_index < params->ode_number; ode_index++){
        memcpy(y_test_n_ode_padding_h + ode_index*y_test_1_ode_padding_h_size, y_test_1_ode_padding_h, y_test_1_ode_padding_h_size * sizeof(double));
    }
//    for(int i = 0; i < y_test_n_ode_padding_h_size; i++){
//        if(y_test_n_ode_padding_h[i] != 0.0){
//            printf("y_test_n_ode_padding_h[%d] = %.1f\n",i,y_test_n_ode_padding_h[i]);
//        }
//    }

    double* y_test_n_ode_padding_d;
    size_t y_test_n_ode_padding_d_size = y_test_n_ode_padding_h_size * sizeof(double);
    checkCuda(cudaMalloc((void **) &y_test_n_ode_padding_d, y_test_n_ode_padding_d_size));
    checkCuda(cudaMemcpy(y_test_n_ode_padding_d, y_test_n_ode_padding_h, y_test_n_ode_padding_d_size,cudaMemcpyHostToDevice));

    double r_denom_h[params->ode_number];
    memset(r_denom_h,0,params->ode_number*sizeof(double));
    double r_num_h[params->ode_number];
    memset(r_num_h,0,params->ode_number*sizeof(double));

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
    GPUParameters* old_params = params;

    params->block_size = GPU_ODE_THREADS; //max is 1024
    params->num_blocks = (params->ode_output_day + params->block_size - 1) / params->block_size;

    int num_block = ceil(prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock / 1024);
    printf("max threads = %d block = %d\n",prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock, num_block);
    reduce_sum_n<<<num_block, 1024>>>(y_test_n_ode_padding_d,y_test_n_ode_padding_d,params->ode_number,y_test_n_ode_padding_h_size);
    checkCuda(cudaMemcpy(y_test_n_ode_padding_h,y_test_n_ode_padding_d,y_test_n_ode_padding_d_size,cudaMemcpyDeviceToHost));
    checkCuda(cudaDeviceSynchronize());
    for(int i = 0; i < y_test_n_ode_padding_h_size; i++){
        if(y_test_n_ode_padding_h[i] > 1.0){
            printf("y_test_n_ode_padding_h[%d] = %.1f\n",i,y_test_n_ode_padding_h[i]);
        }
    }


//    reduce_sum<<<num_SMs, 1024>>>(y_test_sum_d,y_test_sum_d,params->data_params.rows);


//    //Pre-calculate stf
//    calculate_stf<<<params->num_blocks, params->block_size>>>(stf_d,params_d);
//    for (int iter = 0; iter < params->mcmc_loop; iter++) {
//        solve_ode<<<params->num_blocks, params->block_size>>>(
//                y_ode_input_d, y_ode_output_d, y_ode_agg_d, stf_d, params_d);
//        mcmc_dnorm<<<params->num_blocks, params->block_size>>>(y_data_input_d, y_ode_agg_d, y_mcmc_dnorm_d, params_d);
//        for(int ode_index = 0; ode_index < params->ode_number; ode_index++){
//            checkCuda(cudaMemcpy(y_mcmc_dnorm_i_d,y_mcmc_dnorm_d + params->data_params.rows * ode_index,params->data_params.rows*sizeof(double),cudaMemcpyDeviceToDevice));
//            reduce_sum<<<num_SMs,1024>>>(y_mcmc_dnorm_i_d, y_mcmc_dnorm_i_d, params->data_params.rows);
//            checkCuda(cudaMemcpy(&r_denom_h[ode_index],y_mcmc_dnorm_i_d,sizeof(double),cudaMemcpyDeviceToHost));
////            printf("ODE %d r_denom_h = %.5f\n",ode_index,r_denom_h[ode_index]);
//        }
//
//        //
//        // Generate new parameters
//        //
//
//        old_params = params;
//        params->update();
//
//        //
//        // Copy new parameters to gpu
//        //
//
//        checkCuda(cudaMemcpy(params_d, params, sizeof(GPUParameters), cudaMemcpyHostToDevice));
//        calculate_stf<<<params->num_blocks, params->block_size>>>(stf_d,params_d);
//
//        //y_ode_input_d
//        //temp pointers
//        tmp_ptr = (double **) malloc(ode_size);
//        for (int i = 0; i < params->ode_number; i++) {
//            checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_ode_input_d_size));
//            checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_input[i], y_ode_input_d_size,cudaMemcpyHostToDevice));
//        }
//        checkCuda(cudaMalloc((void **) &y_ode_input_d, ode_size));
//        checkCuda(cudaMemcpy(y_ode_input_d, tmp_ptr, ode_size, cudaMemcpyHostToDevice));//y_ode_output_d
//
//        //y_ode_agg_d
//        //temp pointers
//        tmp_ptr = (double **) malloc(ode_size);
//        for (int i = 0; i < params->ode_number; i++) {
//            checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_ode_agg_d_size));
//            checkCuda(cudaMemcpy(tmp_ptr[i], params->y_ode_agg[i], y_ode_agg_d_size,cudaMemcpyHostToDevice));
//        }
//        checkCuda(cudaMalloc((void **) &y_ode_agg_d, ode_size));
//        checkCuda(cudaMemcpy(y_ode_agg_d, tmp_ptr, ode_size, cudaMemcpyHostToDevice));
//
//        //y_mcmc_dnorm_d - single 1d array for NUMODE dnorm values (NUMODE*data_dimension.rows)
//        checkCuda(cudaMemcpy(y_mcmc_dnorm_d, y_mcmc_dnorm_h, y_mcmc_dnorm_d_size,cudaMemcpyHostToDevice));
//
//        //
//        // Solve ode with new parameters
//        //
//
//        solve_ode<<<params->num_blocks, params->block_size>>>(
//                y_ode_input_d, y_ode_output_d, y_ode_agg_d, stf_d, params_d);
//        mcmc_dnorm<<<params->num_blocks, params->block_size>>>(y_data_input_d, y_ode_agg_d, y_mcmc_dnorm_d, params_d);
//        for(int ode_index = 0; ode_index < params->ode_number; ode_index++){
//            checkCuda(cudaMemcpy(y_mcmc_dnorm_i_d,y_mcmc_dnorm_d + params->data_params.rows * ode_index,params->data_params.rows*sizeof(double),cudaMemcpyDeviceToDevice));
//            reduce_sum<<<num_SMs,1024>>>(y_mcmc_dnorm_i_d, y_mcmc_dnorm_i_d, params->data_params.rows);
//            checkCuda(cudaMemcpy(&r_num_h[ode_index],y_mcmc_dnorm_i_d,sizeof(double),cudaMemcpyDeviceToHost));
////            printf("ODE %d r_num_h = %.5f\n",ode_index,r_denom_h[ode_index]);
//        }
//
//        for(int ode_index = 0; ode_index < params->ode_number; ode_index++) {
//            double r = r_num_h[ode_index] - r_denom_h[ode_index];
//            if (exp(r) > rand_uniform(0.0, 1.0)) {
//                params = old_params;
//                printf("iter %d ODE %d accept params (r = %.5f)\n", iter, ode_index, r);
//            } else {
//                printf("iter %d ODE %d reject params_temp (r = %.5f)\n", iter, ode_index, r);
//            }
//        }
//        checkCuda(cudaDeviceSynchronize());
//        printf("\n");
//    }

    //    cudaProfilerStop();
    checkCuda(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for MCMC %d ODE with %d parameters %d times on GPU: %ld micro seconds which is %.10f seconds\n",
           NUMODE, DIM, params->mcmc_loop, duration.count(), (duration.count() / 1e6));

    start = std::chrono::high_resolution_clock::now();
    //y_ode_output_h
    tmp_ptr = (double **) malloc(ode_size);
    double **y_ode_output_h = (double **) malloc(ode_size);
    for (int i = 0; i < params->ode_number; i++) {
        y_ode_output_h[i] = (double *) malloc(params->ode_output_day * params->display_dimension * sizeof(double));
    }
    checkCuda(cudaMemcpy(tmp_ptr, y_ode_output_d, ode_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < params->ode_number; i++) {
        checkCuda(cudaMemcpy(y_ode_output_h[i], tmp_ptr[i], params->ode_output_day * params->display_dimension * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }
    //y_output_agg_h
    tmp_ptr = (double **) malloc(ode_size);
    double **y_output_agg_h = (double **) malloc(ode_size);
    for (int i = 0; i < params->ode_number; i++) {
        y_output_agg_h[i] = (double *) malloc(params->ode_output_day * params->agg_dimension * sizeof(double));
    }
    checkCuda(cudaMemcpy(tmp_ptr, y_ode_agg_d, ode_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < params->ode_number; i++) {
        checkCuda(cudaMemcpy(y_output_agg_h[i], tmp_ptr[i], params->ode_output_day * params->agg_dimension * sizeof(double),
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
        printf("Display y_ode_output_h[%d]\n",random_index);
        for(int index = 0; index < params->ode_output_day * params->display_dimension; index++){
            const int line_index = (index / params->display_dimension) % NUMDAYSOUTPUT;
            if(line_index < 10)
            {
                printf("%.5f\t", y_ode_output_h[random_index][index]);
                if (index > 0 && (index + 1) % params->display_dimension == 0) {
                    printf("\n");
                }
            }
        }
        printf("Display y_output_agg_h[%d]\n", random_index);
        for (int index = 0; index < params->ode_output_day * params->agg_dimension; index++) {
            const int line_index = (index / params->agg_dimension) % NUMDAYSOUTPUT;
            if(line_index < 10)
            {
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
    checkCuda(cudaFree(y_mcmc_dnorm_d));
    checkCuda(cudaFree(y_mcmc_dnorm_i_d));
    checkCuda(cudaFree(params_d));
    delete params;
    delete y_ode_output_h;
    delete y_output_agg_h;
    delete tmp_ptr;
    return;
}