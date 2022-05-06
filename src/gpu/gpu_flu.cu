#include "gpu_flu.cuh"

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
    gpu_params = new GPUParameters();
    flu_params = new FluParameters*[NUMODE]();
}

GPUFlu::~GPUFlu() {
    gpu_params = nullptr;
    for(int i = 0; i < NUMODE; i++){
        flu_params[i] = nullptr;
    }
}

void GPUFlu::set_gpu_parameters(GPUParameters *gpu_params_) {
    gpu_params = &(*gpu_params_);
}

void GPUFlu::set_flu_parameters(FluParameters *flu_params_[]) {
    for(int i = 0; i < NUMODE; i++){
        flu_params[i] = &(*flu_params_[i]);
    }
}

void GPUFlu::init() {
}

void GPUFlu::run() {
    auto start = std::chrono::high_resolution_clock::now();
    size_t ode_double_size = gpu_params->ode_number* sizeof(double);
    /* stf_h - stf on host */
    double** stf_h = new double*[gpu_params->ode_number]();
    for(int i = 0; i < gpu_params->ode_number; i++){
        stf_h[i] = new double[gpu_params->ode_output_day];
        for(int j = 0; j < gpu_params->ode_output_day; j++) {
            stf_h[i][j] = 7.0;
        }
    }
    /* stf_d - stf on device */
    double **stf_d = 0;
    size_t stf_d_size = gpu_params->ode_output_day * sizeof(double);
    //temp pointers
    double **tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], stf_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], stf_h[i], stf_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &stf_d, ode_double_size));
    checkCuda(cudaMemcpy(stf_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* y_ode_input_d - device */
    double **y_ode_input_d = 0;
    size_t y_ode_input_d_size = gpu_params->ode_dimension * sizeof(double);
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_ode_input_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->y_ode_input[i], y_ode_input_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_ode_input_d, ode_double_size));
    checkCuda(cudaMemcpy(y_ode_input_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* y_ode_output_d - device */
    double **y_ode_output_d = 0;
    size_t y_ode_output_d_size = gpu_params->ode_output_day * gpu_params->display_dimension * sizeof(double);
    //y_ode_output_d
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_ode_output_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->y_ode_output[i],y_ode_output_d_size, cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_ode_output_d, ode_double_size));
    checkCuda(cudaMemcpy(y_ode_output_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* y_data_input_d - device */
    double **y_data_input_d = 0;
    size_t y_data_input_d_size = gpu_params->data_dimension * sizeof(double);
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_data_input_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->y_data_input[i], y_data_input_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_data_input_d, ode_double_size));
    checkCuda(cudaMemcpy(y_data_input_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* y_agg_input_d - device */
    double **y_agg_input_d = 0;
    size_t y_agg_d_size = gpu_params->ode_output_day * gpu_params->agg_dimension * sizeof(double);
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_agg_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->y_agg[i], y_agg_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_agg_input_d, ode_double_size));
    checkCuda(cudaMemcpy(y_agg_input_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* y_agg_output_d - device */
    double **y_agg_output_d = 0;
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_agg_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->y_agg[i], y_agg_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_agg_output_d, ode_double_size));
    checkCuda(cudaMemcpy(y_agg_output_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* dnorm 1 ode with padding - on host */
    int mcmc_dnorm_1_ode_padding_size = ceil(gpu_params->data_params.rows/1024.0)*1024 - gpu_params->data_params.rows;
    double y_mcmc_dnorm_1_ode_h[gpu_params->data_params.rows];
    int y_mcmc_dnorm_1_ode_h_size = gpu_params->data_params.rows;
    for (int i = 0; i < y_mcmc_dnorm_1_ode_h_size; i++) {
        y_mcmc_dnorm_1_ode_h[i] = 0.0;
    }
    double y_mcmc_dnorm_1_ode_padding_h[gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size];
    int y_mcmc_dnorm_1_ode_padding_h_size = gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size;
    memcpy(y_mcmc_dnorm_1_ode_padding_h, y_mcmc_dnorm_1_ode_h, y_mcmc_dnorm_1_ode_h_size * sizeof(double));
    memset(y_mcmc_dnorm_1_ode_padding_h + y_mcmc_dnorm_1_ode_h_size,0,mcmc_dnorm_1_ode_padding_size* sizeof(double));

    /* dnorm N ode with padding - on host */
    double y_mcmc_dnorm_n_ode_padding_h[gpu_params->ode_number*(gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size)];
    int y_mcmc_dnorm_n_ode_padding_h_size = gpu_params->ode_number*(gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size);
    for(int ode_index = 0; ode_index < gpu_params->ode_number; ode_index++){
        memcpy(y_mcmc_dnorm_n_ode_padding_h + ode_index*y_mcmc_dnorm_1_ode_padding_h_size, y_mcmc_dnorm_1_ode_padding_h, y_mcmc_dnorm_1_ode_padding_h_size * sizeof(double));
    }

    /* dnorm N ode with padding - on device */
    double* y_mcmc_dnorm_n_ode_padding_d;
    double* y_mcmc_dnorm_n_ode_padding_zero_d;
    size_t y_mcmc_dnorm_n_ode_padding_d_size = y_mcmc_dnorm_n_ode_padding_h_size * sizeof(double);
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_d_size));
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_n_ode_padding_zero_d, y_mcmc_dnorm_n_ode_padding_d_size));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_h, y_mcmc_dnorm_n_ode_padding_d_size,cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_n_ode_padding_zero_d, y_mcmc_dnorm_n_ode_padding_h, y_mcmc_dnorm_n_ode_padding_d_size,cudaMemcpyHostToDevice));

    /* gpu_params_d - on device */
    GPUParameters *gpu_params_d;
    checkCuda(cudaMalloc((void **) &gpu_params_d, sizeof(GPUParameters)));
    checkCuda(cudaMemcpy(gpu_params_d, gpu_params, sizeof(GPUParameters), cudaMemcpyHostToDevice));

    /* flu_params_current_d - on device */
    size_t ode_flu_params_size = gpu_params->ode_number * sizeof(FluParameters);
    size_t flu_params_d_size = sizeof(FluParameters);
    FluParameters **flu_params_current_d;
    FluParameters **flu_tmp_ptr = new FluParameters*[gpu_params->ode_number]();
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &flu_tmp_ptr[i], flu_params_d_size));
        checkCuda(cudaMemcpy(flu_tmp_ptr[i], flu_params[i], flu_params_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &flu_params_current_d, ode_flu_params_size));
    checkCuda(cudaMemcpy(flu_params_current_d, flu_tmp_ptr, ode_flu_params_size, cudaMemcpyHostToDevice));

    /* flu_params_new_d - on device */
    FluParameters **flu_params_new_d;
    flu_tmp_ptr = new FluParameters*[gpu_params->ode_number]();
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &flu_tmp_ptr[i], flu_params_d_size));
        checkCuda(cudaMemcpy(flu_tmp_ptr[i], flu_params[i], flu_params_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &flu_params_new_d, ode_flu_params_size));
    checkCuda(cudaMemcpy(flu_params_new_d, flu_tmp_ptr, ode_flu_params_size, cudaMemcpyHostToDevice));

    /* curand_state_d - on device */
    curandState *curand_state_d;
    checkCuda(cudaMalloc((void **)&curand_state_d, gpu_params->ode_number * sizeof(curandState)));

    /* r_denom/r_num - on host */
    double r_h[gpu_params->ode_number];
    memset(r_h,0,gpu_params->ode_number*sizeof(double));

    /* r_denom - on device */
    size_t r_d_size = gpu_params->ode_number * sizeof(double);
    double* r_denom_d = 0;
    checkCuda(cudaMalloc((void **) &r_denom_d, r_d_size));
    checkCuda(cudaMemcpy(r_denom_d, r_h, r_d_size,cudaMemcpyHostToDevice));

    /* r_num - on device */
    double* r_num_d = 0;
    checkCuda(cudaMalloc((void **) &r_num_d, r_d_size));
    checkCuda(cudaMemcpy(r_num_d, r_h, r_d_size,cudaMemcpyHostToDevice));

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %ld micro seconds which is %.10f seconds\n", duration.count(),
           (duration.count() / 1e6));

    start = std::chrono::high_resolution_clock::now();
    //cudaProfilerStart();

    gpu_params->block_size = GPU_ODE_THREADS; //max is 1024
    gpu_params->num_blocks = (gpu_params->ode_output_day + gpu_params->block_size - 1) / gpu_params->block_size;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_block = ceil(prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock / 1024);//
    printf("max threads = %d block = %d\n",prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock, num_block);

    /* Setup prng states */
    mcmc_setup_states_for_random<<<gpu_params->num_blocks, gpu_params->block_size>>>(curand_state_d);
    for (int iter = 0; iter < gpu_params->mcmc_loop; iter++) {
        if(iter == 0){
            /* Calculate stf */
            calculate_stf<<<gpu_params->num_blocks, gpu_params->block_size>>>(stf_d, gpu_params_d, flu_params_current_d);
            /* Calculate ODE */
            solve_ode<<<gpu_params->num_blocks, gpu_params->block_size>>>(y_ode_input_d, y_ode_output_d, y_agg_input_d, y_agg_output_d, stf_d, gpu_params_d, flu_params_current_d);
            /* Calculate dnorm */
            mcmc_dnorm_padding<<<gpu_params->num_blocks, gpu_params->block_size>>>(y_data_input_d, y_agg_output_d, y_mcmc_dnorm_n_ode_padding_d, mcmc_dnorm_1_ode_padding_size, gpu_params_d);
            /* Calculate sum dnorm */
            reduce_sum_padding<<<num_block, 1024>>>(y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_d, gpu_params_d, y_mcmc_dnorm_n_ode_padding_h_size);
            /* Calculate R_denom */
            mcmc_compute_r<<<num_block, 1024>>>(y_mcmc_dnorm_n_ode_padding_d, r_denom_d, y_mcmc_dnorm_n_ode_padding_d_size, gpu_params_d);
        }
        //
        // Generate new parameters
        //

        /* Set new flu parameters from current flu parameters */
        checkCuda(cudaMemcpy(flu_params_new_d, flu_params_current_d, ode_flu_params_size, cudaMemcpyDeviceToDevice));
        /* Reset dnorm vector on device */
        checkCuda(cudaMemcpy(y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_zero_d, y_mcmc_dnorm_n_ode_padding_d_size, cudaMemcpyDeviceToDevice));
        /* Update new flu parameters */
        mcmc_update_parameters<<<gpu_params->num_blocks, gpu_params->block_size>>>(gpu_params_d, flu_params_new_d, curand_state_d, 1234);

        //
        // Solve ode with new parameters
        //

        /* Calculate stf */
        calculate_stf<<<gpu_params->num_blocks, gpu_params->block_size>>>(stf_d, gpu_params_d, flu_params_new_d);
        /* Calculate ODE */
        solve_ode<<<gpu_params->num_blocks, gpu_params->block_size>>>(y_ode_input_d, y_ode_output_d, y_agg_input_d, y_agg_output_d, stf_d, gpu_params_d, flu_params_new_d);
        /* Calculate dnorm */
        mcmc_dnorm_padding<<<gpu_params->num_blocks, gpu_params->block_size>>>(y_data_input_d, y_agg_output_d, y_mcmc_dnorm_n_ode_padding_d, mcmc_dnorm_1_ode_padding_size, gpu_params_d);
        /* Calculate sum dnorm */
        reduce_sum_padding<<<num_block, 1024>>>(y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_d, gpu_params_d, y_mcmc_dnorm_n_ode_padding_h_size);
        /* Calculate R_num */
        mcmc_compute_r<<<num_block, 1024>>>(y_mcmc_dnorm_n_ode_padding_d, r_num_d, y_mcmc_dnorm_n_ode_padding_d_size, gpu_params_d);

        /* Accept or reject new parameters */
        mcmc_check_acceptance<<<gpu_params->num_blocks, gpu_params->block_size>>>(r_denom_d, r_num_d, gpu_params_d, flu_params_current_d, flu_params_new_d, curand_state_d, 1234);

        printf("==== iter %d done ====\n",iter);
        checkCuda(cudaDeviceSynchronize());
    }

    //    cudaProfilerStop();
    checkCuda(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for MCMC %d ODE with %d parameters %d times on GPU: %ld micro seconds which is %.10f seconds\n",
           NUMODE, DIM, gpu_params->mcmc_loop, duration.count(), (duration.count() / 1e6));

    start = std::chrono::high_resolution_clock::now();
    //y_ode_output_h
    tmp_ptr = (double **) malloc(ode_double_size);
    double **y_ode_output_h = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        y_ode_output_h[i] = (double *) malloc(gpu_params->ode_output_day * gpu_params->display_dimension * sizeof(double));
    }
    checkCuda(cudaMemcpy(tmp_ptr, y_ode_output_d, ode_double_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMemcpy(y_ode_output_h[i], tmp_ptr[i], gpu_params->ode_output_day * gpu_params->display_dimension * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }
    //y_output_agg_h
    tmp_ptr = (double **) malloc(ode_double_size);
    double **y_output_agg_h = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        y_output_agg_h[i] = (double *) malloc(gpu_params->ode_output_day * gpu_params->agg_dimension * sizeof(double));
    }
    checkCuda(cudaMemcpy(tmp_ptr, y_agg_output_d, ode_double_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMemcpy(y_output_agg_h[i], tmp_ptr[i], gpu_params->ode_output_day * gpu_params->agg_dimension * sizeof(double),
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

    for (int i = 0; i < gpu_params->display_number; i++) {
        int random_index = 0;
        if (NUMODE == 1) {
            random_index = 0;
        } else {
            random_index = distr(gen);
        }
//        printf("Display y_ode_output_h[%d]\n",random_index);
//        for(int index = 0; index < gpu_params->ode_output_day * gpu_params->display_dimension; index++){
//            const int line_index = (index / gpu_params->display_dimension) % NUMDAYSOUTPUT;
//            if(line_index < 10)
//            {
//                printf("%.5f\t", y_ode_output_h[random_index][index]);
//                if (index > 0 && (index + 1) % gpu_params->display_dimension == 0) {
//                    printf("\n");
//                }
//            }
//        }
        printf("Display y_output_agg_h[%d]\n", random_index);
        for (int index = 0; index < gpu_params->ode_output_day * gpu_params->agg_dimension; index++) {
            const int line_index = (index / gpu_params->agg_dimension);
            if(line_index < 10)
            {
                printf("%d %.5f\t",line_index, y_output_agg_h[random_index][index]);
                if (index > 0 && (index + 1) % gpu_params->agg_dimension == 0) {
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
    checkCuda(cudaFree(y_agg_input_d));
    checkCuda(cudaFree(y_agg_output_d));
    checkCuda(cudaFree(y_data_input_d));
    checkCuda(cudaFree(y_mcmc_dnorm_n_ode_padding_d));
    checkCuda(cudaFree(gpu_params_d));
    checkCuda(cudaFree(flu_params_current_d));
    checkCuda(cudaFree(flu_params_new_d));
    delete y_ode_output_h;
    delete y_output_agg_h;
    delete [] stf_h;
    delete [] tmp_ptr;
    delete [] flu_tmp_ptr;
    return;
}