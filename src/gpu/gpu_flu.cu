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
}

GPUFlu::~GPUFlu() {
}

void GPUFlu::set_gpu_parameters(GPUParameters *gpu_params_) {
    gpu_params = gpu_params_;
}

void GPUFlu::init(){
    checkCuda(cudaEventCreate(&start_event));
    checkCuda(cudaEventCreate(&stop_event));
    checkCuda(cudaEventCreate(&start_event_all));
    checkCuda(cudaEventCreate(&stop_event_all));
    checkCuda(cudaEventCreate(&start_one_ode_event));
    checkCuda(cudaEventCreate(&stop_one_ode_event));
    checkCuda(cudaEventCreate(&start_one_stf_event));
    checkCuda(cudaEventCreate(&stop_one_stf_event));
    checkCuda(cudaEventCreate(&start_one_mcmc_event));
    checkCuda(cudaEventCreate(&stop_one_mcmc_event));
    checkCuda(cudaEventCreate(&start_one_update_event));
    checkCuda(cudaEventCreate(&stop_one_update_event));
    checkCuda(cudaEventCreate(&start_one_iter_event));
    checkCuda(cudaEventCreate(&stop_one_iter_event));

    checkCuda(cudaEventRecord(start_event_all,0));
    checkCuda(cudaEventRecord(start_event,0));

    flu_params = new FluParameters();
    flu_params->init();
    gpu_params->init(flu_params);

    ode_double_size = gpu_params->ode_number* sizeof(double);

    /* stf_d - stf on device */
    stf_d_size = gpu_params->ode_output_day * sizeof(double);
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], stf_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->stf[i], stf_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &stf_d, ode_double_size));
    checkCuda(cudaMemcpy(stf_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* y_ode_input_d - device */
    y_ode_input_d_size = gpu_params->ode_dimension * sizeof(double);
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_ode_input_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->y_ode_input[i], y_ode_input_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_ode_input_d, ode_double_size));
    checkCuda(cudaMemcpy(y_ode_input_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* y_ode_output_d - device */
    y_ode_output_d_size = gpu_params->ode_output_day * gpu_params->display_dimension * sizeof(double);
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
    y_data_input_d_size = gpu_params->data_dimension * sizeof(double);
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_data_input_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->y_data_input[i], y_data_input_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_data_input_d, ode_double_size));
    checkCuda(cudaMemcpy(y_data_input_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* y_agg_input_d - device */
    y_agg_d_size = gpu_params->ode_output_day * gpu_params->agg_dimension * sizeof(double);
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_agg_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->y_agg[i], y_agg_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_agg_input_d, ode_double_size));
    checkCuda(cudaMemcpy(y_agg_input_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* y_agg_output_d - device */
    //temp pointers
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &tmp_ptr[i], y_agg_d_size));
        checkCuda(cudaMemcpy(tmp_ptr[i], gpu_params->y_agg[i], y_agg_d_size,cudaMemcpyHostToDevice));
    }
    checkCuda(cudaMalloc((void **) &y_agg_output_d, ode_double_size));
    checkCuda(cudaMemcpy(y_agg_output_d, tmp_ptr, ode_double_size, cudaMemcpyHostToDevice));

    /* dnorm 1 ode with padding - on host */
    mcmc_dnorm_1_ode_padding_size = ceil(gpu_params->data_params.rows/(GPU_REDUCE_THREADS*1.0))*GPU_REDUCE_THREADS - gpu_params->data_params.rows;
    y_mcmc_dnorm_1_ode_h = (double*)malloc(gpu_params->data_params.rows*sizeof(double));
    y_mcmc_dnorm_1_ode_h_size = gpu_params->data_params.rows;
    for (int i = 0; i < y_mcmc_dnorm_1_ode_h_size; i++) {
        y_mcmc_dnorm_1_ode_h[i] = 0.0;
    }
    y_mcmc_dnorm_1_ode_padding_h = (double*)malloc((gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size)*sizeof(double));
    y_mcmc_dnorm_1_ode_padding_h_size = gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size;
    memcpy(y_mcmc_dnorm_1_ode_padding_h, y_mcmc_dnorm_1_ode_h, y_mcmc_dnorm_1_ode_h_size * sizeof(double));
    memset(y_mcmc_dnorm_1_ode_padding_h + y_mcmc_dnorm_1_ode_h_size,0,mcmc_dnorm_1_ode_padding_size* sizeof(double));

    /* dnorm N ode with padding - on host */
    y_mcmc_dnorm_n_ode_padding_h = (double*)malloc(gpu_params->ode_number*(gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size)*sizeof(double));
    y_mcmc_dnorm_n_ode_padding_h_size = gpu_params->ode_number*(gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size);
    for(int ode_index = 0; ode_index < gpu_params->ode_number; ode_index++){
        memcpy(y_mcmc_dnorm_n_ode_padding_h + ode_index*y_mcmc_dnorm_1_ode_padding_h_size, y_mcmc_dnorm_1_ode_padding_h, y_mcmc_dnorm_1_ode_padding_h_size * sizeof(double));
    }

    /* dnorm N ode with padding - on device */
    y_mcmc_dnorm_n_ode_padding_d_size = y_mcmc_dnorm_n_ode_padding_h_size * sizeof(double);
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_d_size));
    checkCuda(cudaMalloc((void **) &y_mcmc_dnorm_n_ode_padding_zero_d, y_mcmc_dnorm_n_ode_padding_d_size));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_h, y_mcmc_dnorm_n_ode_padding_d_size,cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(y_mcmc_dnorm_n_ode_padding_zero_d, y_mcmc_dnorm_n_ode_padding_h, y_mcmc_dnorm_n_ode_padding_d_size,cudaMemcpyHostToDevice));

    /* gpu_params_d - on device */
    checkCuda(cudaMalloc((void **) &gpu_params_d, sizeof(GPUParameters)));
    checkCuda(cudaMemcpy(gpu_params_d, gpu_params, sizeof(GPUParameters), cudaMemcpyHostToDevice));

    /* flu_params_current_d - on device */
    checkCuda(cudaMalloc((void **) &flu_params_current_d, sizeof(FluParameters)));
    checkCuda(cudaMemcpy(flu_params_current_d, flu_params, sizeof(FluParameters), cudaMemcpyHostToDevice));

    /* flu_params_new_d - on device */
    checkCuda(cudaMalloc((void **) &flu_params_new_d, sizeof(FluParameters)));
    checkCuda(cudaMemcpy(flu_params_new_d, flu_params, sizeof(FluParameters), cudaMemcpyHostToDevice));

    /* r_denom/r_num - on host */
    r_h = (double*)malloc(ode_double_size);
    memset(r_h,0,ode_double_size);

    /* r_denom_d - on device */
    checkCuda(cudaMalloc((void **) &r_denom_d, ode_double_size));
    checkCuda(cudaMemcpy(r_denom_d, r_h, ode_double_size,cudaMemcpyHostToDevice));

    /* r_num_d - on device */
    checkCuda(cudaMalloc((void **) &r_num_d, ode_double_size));
    checkCuda(cudaMemcpy(r_num_d, r_h, ode_double_size,cudaMemcpyHostToDevice));

    /* norm_h - on host */
    norm_size = gpu_params->ode_number * SAMPLE_LENGTH;
    norm_h = (double*)malloc(norm_size * sizeof(double));
    for (int i = 0; i < norm_size; i++) {
        norm_h[i] = 0.0;
    }

    /* norm_d - on device */
    checkCuda(cudaMalloc((void **) &norm_d, norm_size * sizeof(double)));
    checkCuda(cudaMemcpy(norm_d, norm_h, norm_size * sizeof(double),cudaMemcpyHostToDevice));

    /* norm_and_sd_d - on device */
    checkCuda(cudaMalloc((void **) &norm_sd_d, norm_size * sizeof(double)));
    checkCuda(cudaMemcpy(norm_d, norm_h, norm_size * sizeof(double),cudaMemcpyHostToDevice));

    /* curand_state_d - on device */
    checkCuda(cudaMalloc((void **)&curand_state_d, norm_size * sizeof(curandState)));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    /* Blocks to process other things, must be equal number of ODE */
    gpu_params->block_size = GPU_ODE_THREADS; //max is 1024
    gpu_params->num_blocks = (gpu_params->ode_number + gpu_params->block_size - 1) / gpu_params->block_size;
    /* Blocks to process reduction sum with padding, must be divided by 1024 */
    gpu_reduce_num_block = ceil(prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock / GPU_REDUCE_THREADS);
    printf("GPU reduce threads = %d block = %d\n",prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock, gpu_reduce_num_block);

    checkCuda(cudaEventRecord(stop_event, 0));
    checkCuda(cudaEventSynchronize(stop_event));
    checkCuda(cudaEventElapsedTime(&transfer_h2d_ms, start_event, stop_event));
}

void GPUFlu::run() {
    checkCuda(cudaEventRecord(start_event,0));
    //cudaProfilerStart();

    /* Setup prng states */
    mcmc_setup_states_for_random<<<gpu_params->num_blocks, gpu_params->block_size>>>(curand_state_d, norm_size);
    for (int iter = 0; iter < gpu_params->mcmc_loop; iter++) {
        checkCuda(cudaEventRecord(start_one_iter_event,0));
        if(iter == 0){
            /* Calculate stf */
            calculate_stf<<<gpu_params->num_blocks, gpu_params->block_size>>>(stf_d, gpu_params_d, flu_params_current_d);
            /* Calculate ODE */
            solve_ode_n<<<gpu_params->num_blocks, gpu_params->block_size>>>(y_ode_input_d, y_ode_output_d, y_agg_input_d, y_agg_output_d, stf_d, gpu_params_d, flu_params_current_d);
            /* Calculate dnorm */
            mcmc_dnorm_padding<<<gpu_params->num_blocks, gpu_params->block_size>>>(y_data_input_d, y_agg_output_d, y_mcmc_dnorm_n_ode_padding_d, mcmc_dnorm_1_ode_padding_size, gpu_params_d);
            /* Calculate sum dnorm */
            reduce_sum_padding<<<gpu_reduce_num_block, GPU_REDUCE_THREADS>>>(y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_d, gpu_params_d, y_mcmc_dnorm_n_ode_padding_h_size);
            /* Calculate R_denom */
            mcmc_compute_r<<<gpu_reduce_num_block, GPU_REDUCE_THREADS>>>(y_mcmc_dnorm_n_ode_padding_d, r_denom_d, gpu_params_d);
        }

        //
        // Generate new parameters
        //

        checkCuda(cudaEventRecord(start_one_update_event,0));
        /* Reset dnorm vector on device */
        checkCuda(cudaMemcpy(y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_zero_d, y_mcmc_dnorm_n_ode_padding_d_size, cudaMemcpyDeviceToDevice));
        /* Update new flu parameters */
        mcmc_generate_norm<<<gpu_reduce_num_block, GPU_REDUCE_THREADS>>>(norm_d, norm_size, curand_state_d);
        mcmc_update_parameters<<<gpu_params->num_blocks, gpu_params->block_size>>>(gpu_params_d, flu_params_current_d, flu_params_new_d, curand_state_d);
        checkCuda(cudaEventRecord(stop_one_update_event,0));
        checkCuda(cudaEventSynchronize(stop_one_update_event));
        checkCuda(cudaEventElapsedTime(&one_update_ms, start_one_update_event, stop_one_update_event));

        //
        // Solve ode with new parameters
        //

        /* Calculate stf */
        checkCuda(cudaEventRecord(start_one_stf_event,0));
        calculate_stf<<<gpu_params->num_blocks, gpu_params->block_size>>>(stf_d, gpu_params_d, flu_params_new_d);
        checkCuda(cudaEventRecord(stop_one_stf_event,0));
        checkCuda(cudaEventSynchronize(stop_one_stf_event));
        checkCuda(cudaEventElapsedTime(&one_stf_ms, start_one_stf_event, stop_one_stf_event));
        /* Calculate ODE */
        checkCuda(cudaEventRecord(start_one_ode_event,0));
        solve_ode_n<<<gpu_params->num_blocks, gpu_params->block_size>>>(y_ode_input_d, y_ode_output_d, y_agg_input_d, y_agg_output_d, stf_d, gpu_params_d, flu_params_new_d);
        checkCuda(cudaEventRecord(stop_one_ode_event,0));
        checkCuda(cudaEventSynchronize(stop_one_ode_event));
        checkCuda(cudaEventElapsedTime(&one_ode_ms, start_one_ode_event, stop_one_ode_event));

        checkCuda(cudaEventRecord(start_one_mcmc_event,0));
        /* Calculate dnorm */
        mcmc_dnorm_padding<<<gpu_params->num_blocks, gpu_params->block_size>>>(y_data_input_d, y_agg_output_d, y_mcmc_dnorm_n_ode_padding_d, mcmc_dnorm_1_ode_padding_size, gpu_params_d);
        /* Calculate sum dnorm */
        reduce_sum_padding<<<gpu_reduce_num_block, GPU_REDUCE_THREADS>>>(y_mcmc_dnorm_n_ode_padding_d, y_mcmc_dnorm_n_ode_padding_d, gpu_params_d, y_mcmc_dnorm_n_ode_padding_h_size);
        /* Calculate R_num */
        mcmc_compute_r<<<gpu_reduce_num_block, GPU_REDUCE_THREADS>>>(y_mcmc_dnorm_n_ode_padding_d, r_num_d, gpu_params_d);
        /* Accept or reject new parameters */
        mcmc_check_acceptance<<<gpu_params->num_blocks, gpu_params->block_size>>>(r_denom_d, r_num_d, gpu_params_d, flu_params_current_d, flu_params_new_d, curand_state_d);
        checkCuda(cudaEventRecord(stop_one_mcmc_event,0));
        checkCuda(cudaEventSynchronize(stop_one_mcmc_event));
        checkCuda(cudaEventElapsedTime(&one_mcmc_ms, start_one_mcmc_event, stop_one_mcmc_event));

        checkCuda(cudaEventRecord(stop_one_iter_event, 0));
        checkCuda(cudaEventSynchronize(stop_one_iter_event));
        checkCuda(cudaEventElapsedTime(&one_iter_ms, start_one_iter_event, stop_one_iter_event));
	    checkCuda(cudaDeviceSynchronize());
        printf("==== iter %d update done in %f seconds ====\n",iter,(one_update_ms/1e3));
        printf("==== iter %d stf done in %f seconds ====\n",iter,(one_stf_ms/1e3));
        printf("==== iter %d ode done in %f seconds ====\n",iter,(one_ode_ms/1e3));
        printf("==== iter %d mcmc done in %f seconds ====\n",iter,(one_mcmc_ms/1e3));
	    printf("==== iter %d update-stf-ode-mcmc done in %f seconds ====\n\n",iter,(one_iter_ms/1e3));
    }

//    cudaProfilerStop();
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(stop_event, 0));
    checkCuda(cudaEventSynchronize(stop_event));
    checkCuda(cudaEventElapsedTime(&compute_ms, start_event, stop_event));

    checkCuda(cudaEventRecord(start_event,0));

    //y_ode_output_h
    y_ode_output_h = (double **) malloc(ode_double_size);
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        y_ode_output_h[i] = (double *) malloc(gpu_params->ode_output_day * gpu_params->display_dimension * sizeof(double));
    }
    checkCuda(cudaMemcpy(tmp_ptr, y_ode_output_d, ode_double_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMemcpy(y_ode_output_h[i], tmp_ptr[i], gpu_params->ode_output_day * gpu_params->display_dimension * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }

    //y_output_agg_h
    y_output_agg_h = (double **) malloc(ode_double_size);
    tmp_ptr = (double **) malloc(ode_double_size);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        y_output_agg_h[i] = (double *) malloc(gpu_params->ode_output_day * gpu_params->agg_dimension * sizeof(double));
    }
    checkCuda(cudaMemcpy(tmp_ptr, y_agg_output_d, ode_double_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMemcpy(y_output_agg_h[i], tmp_ptr[i], gpu_params->ode_output_day * gpu_params->agg_dimension * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(stop_event, 0));
    checkCuda(cudaEventSynchronize(stop_event));
    checkCuda(cudaEventElapsedTime(&transfer_d2h_ms, start_event, stop_event));

    checkCuda(cudaEventRecord(stop_event_all, 0));
    checkCuda(cudaEventSynchronize(stop_event_all));
    checkCuda(cudaEventElapsedTime(&all_ms, start_event_all, stop_event_all));

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, NUMODE); // define the range

//    for (int i = 0; i < gpu_params->display_number; i++) {
//        int random_index = 0;
//        if (NUMODE == 1) {
//            random_index = 0;
//        } else {
//            random_index = distr(gen);
//        }
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
//        printf("Display y_output_agg_h[%d]\n", random_index);
//        for (int index = 0; index < gpu_params->ode_output_day * gpu_params->agg_dimension; index++) {
//            const int line_index = (index / gpu_params->agg_dimension);
//            if(line_index < 10)
//            {
//                printf("%d %.5f\t",line_index, y_output_agg_h[random_index][index]);
//                if (index > 0 && (index + 1) % gpu_params->agg_dimension == 0) {
//                    printf("\n");
//                }
//            }
//        }
//    }
//    printf("\n");

    checkCuda(cudaDeviceSynchronize());

    printf("[GPU FLU] GPU Time for transfer data from CPU to GPU: %f milliseconds which is %f seconds\n",transfer_h2d_ms,(transfer_h2d_ms/1e3));
    printf("[GPU FLU] GPU Time for compute MCMC %d iteration with %d ODE(s) with %d parameters, step %f in %f days on GPU: %f milliseconds which is %f seconds\n",
           gpu_params->mcmc_loop,gpu_params->ode_number,gpu_params->ode_dimension,gpu_params->h,gpu_params->t_target,compute_ms,(compute_ms/1e3));
    printf("[GPU FLU] GPU Time for transfer data from GPU on CPU: %f milliseconds which is %f seconds\n",transfer_d2h_ms,(transfer_d2h_ms/1e3));
    printf("[GPU FLU] GPU Time for complete MCMC %d iteration with %d ODE(s) with %d parameters: %f milliseconds which is %f seconds\n",
           gpu_params->mcmc_loop,gpu_params->ode_number,gpu_params->ode_dimension,all_ms,(all_ms/1e3));

    // Free memory

    checkCuda(cudaEventDestroy(start_event));
    checkCuda(cudaEventDestroy(stop_event));
    checkCuda(cudaEventDestroy(start_event_all));
    checkCuda(cudaEventDestroy(stop_event_all));
    checkCuda(cudaEventDestroy(start_one_ode_event));
    checkCuda(cudaEventDestroy(stop_one_ode_event));
    checkCuda(cudaEventDestroy(start_one_stf_event));
    checkCuda(cudaEventDestroy(stop_one_stf_event));
    checkCuda(cudaEventDestroy(start_one_mcmc_event));
    checkCuda(cudaEventDestroy(stop_one_mcmc_event));
    checkCuda(cudaEventDestroy(start_one_update_event));
    checkCuda(cudaEventDestroy(stop_one_update_event));
    checkCuda(cudaEventDestroy(start_one_iter_event));
    checkCuda(cudaEventDestroy(stop_one_iter_event));

    checkCuda(cudaFree(y_ode_input_d));
    checkCuda(cudaFree(y_ode_output_d));
    checkCuda(cudaFree(y_agg_input_d));
    checkCuda(cudaFree(y_agg_output_d));
    checkCuda(cudaFree(y_data_input_d));
    checkCuda(cudaFree(y_mcmc_dnorm_n_ode_padding_d));
    checkCuda(cudaFree(gpu_params_d));
    checkCuda(cudaFree(flu_params_current_d));
    checkCuda(cudaFree(flu_params_new_d));
    checkCuda(cudaFree(norm_d));
    checkCuda(cudaFree(norm_sd_d));
    checkCuda(cudaFree(stf_d));
    checkCuda(cudaFree(curand_state_d));
    checkCuda(cudaFree(r_denom_d));
    checkCuda(cudaFree(r_num_d));
    gpu_params = nullptr;
    flu_params = nullptr;
    delete y_ode_output_h;
    delete y_output_agg_h;
    delete norm_h;
    delete r_h;

    delete [] tmp_ptr;
    return;
}
