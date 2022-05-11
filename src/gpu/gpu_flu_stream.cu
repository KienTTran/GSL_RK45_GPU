#include "gpu_flu_stream.cuh"

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

GPUStreamFlu::GPUStreamFlu() {
}

GPUStreamFlu::~GPUStreamFlu() {
}

void GPUStreamFlu::set_gpu_parameters(GPUParameters *gpu_params_) {
    gpu_params = gpu_params_;
}

void GPUStreamFlu::init() {
    flu_params = new FluParameters();
    flu_params->init();
    gpu_params->init(flu_params);
}

void GPUStreamFlu::run() {
    auto start = std::chrono::high_resolution_clock::now();
    size_t ode_double_size = gpu_params->ode_number* sizeof(double);

    /* stf_d - stf on device */
    double *stf_d[gpu_params->ode_number];
    size_t stf_size = gpu_params->ode_output_day * sizeof(double);
    double *stf_pinned_d[gpu_params->ode_number];
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMallocHost((void**)&stf_pinned_d[i], stf_size));
        memcpy(stf_pinned_d[i], gpu_params->stf[i], stf_size);
        checkCuda(cudaMalloc((void **) &stf_d[i], stf_size));
        checkCuda(cudaMemcpy(stf_d[i], stf_pinned_d[i], stf_size, cudaMemcpyHostToDevice));
    }


    /* y_ode_input_d - device */
    double *y_ode_input_d[gpu_params->ode_number];
    size_t y_ode_input_size = gpu_params->ode_dimension * sizeof(double);
    double *y_ode_input_pinned_d[gpu_params->ode_number];
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMallocHost((void**)&y_ode_input_pinned_d[i], y_ode_input_size));
        memcpy(y_ode_input_pinned_d[i], gpu_params->y_ode_input[i], y_ode_input_size);
        checkCuda(cudaMalloc((void **) &y_ode_input_d[i], y_ode_input_size));
        checkCuda(cudaMemcpy(y_ode_input_d[i], y_ode_input_pinned_d[i], y_ode_input_size, cudaMemcpyHostToDevice));
    }

    /* y_ode_output_d - device */
    double *y_ode_output_d[gpu_params->ode_number];
    double *y_ode_output_pinned_h[gpu_params->ode_number];
    size_t y_ode_output_size = gpu_params->ode_output_day * gpu_params->display_dimension * sizeof(double);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMallocHost((void **) &y_ode_output_pinned_h[i], y_ode_output_size));
        checkCuda(cudaMalloc((void **) &y_ode_output_d[i], y_ode_output_size));
        checkCuda(cudaMemcpy(y_ode_output_d[i], gpu_params->y_ode_output[i], y_ode_output_size, cudaMemcpyHostToDevice));
    }

    /* y_data_input_d - device */
    double *y_data_input_d[gpu_params->ode_number];
    size_t y_data_input_d_size = gpu_params->data_dimension * sizeof(double);
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMalloc((void **) &y_data_input_d[i], y_data_input_d_size));
        checkCuda(cudaMemcpy(y_data_input_d[i], gpu_params->y_data_input[i], y_data_input_d_size, cudaMemcpyHostToDevice));
    }

    /* y_agg_input_d - device */
    double *y_agg_input_d[gpu_params->ode_number];
    size_t y_agg_size = gpu_params->ode_output_day * gpu_params->agg_dimension * sizeof(double);
    double *y_agg_input_pinned_d[gpu_params->ode_number];
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMallocHost((void**)&y_agg_input_pinned_d[i], y_agg_size));
        memcpy(y_agg_input_pinned_d[i], gpu_params->y_agg[i], y_agg_size);
        checkCuda(cudaMalloc((void **) &y_agg_input_d[i], y_agg_size));
        checkCuda(cudaMemcpy(y_agg_input_d[i], y_agg_input_pinned_d[i], y_agg_size, cudaMemcpyHostToDevice));
    }

    /* y_agg_output_d - device */
    double *y_agg_output_d[gpu_params->ode_number];
    double *y_agg_output_pinned_h[gpu_params->ode_number];
    for (int i = 0; i < gpu_params->ode_number; i++) {
        checkCuda(cudaMallocHost((void **) &y_agg_output_pinned_h[i], y_agg_size));
        checkCuda(cudaMalloc((void **) &y_agg_output_d[i], y_agg_size));
        checkCuda(cudaMemcpy(y_agg_output_d[i], gpu_params->y_agg[i], y_agg_size, cudaMemcpyHostToDevice));
    }
    
    /* dnorm 1 ode with padding - on host */
    int mcmc_dnorm_1_ode_padding_size = ceil(gpu_params->data_params.rows/(GPU_REDUCE_THREADS*1.0))*GPU_REDUCE_THREADS - gpu_params->data_params.rows;
    double *y_mcmc_dnorm_1_ode_h = (double*)malloc(gpu_params->data_params.rows*sizeof(double));
    int y_mcmc_dnorm_1_ode_h_size = gpu_params->data_params.rows;
    for (int i = 0; i < y_mcmc_dnorm_1_ode_h_size; i++) {
        y_mcmc_dnorm_1_ode_h[i] = 0.0;
    }
    double *y_mcmc_dnorm_1_ode_padding_h = (double*)malloc((gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size)*sizeof(double));
    int y_mcmc_dnorm_1_ode_padding_h_size = gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size;
    memcpy(y_mcmc_dnorm_1_ode_padding_h, y_mcmc_dnorm_1_ode_h, y_mcmc_dnorm_1_ode_h_size * sizeof(double));
    memset(y_mcmc_dnorm_1_ode_padding_h + y_mcmc_dnorm_1_ode_h_size,0,mcmc_dnorm_1_ode_padding_size* sizeof(double));

    /* dnorm N ode with padding - on host */
    double *y_mcmc_dnorm_n_ode_padding_h = (double*)malloc(gpu_params->ode_number*(gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size)*sizeof(double));
    int y_mcmc_dnorm_n_ode_padding_h_size = gpu_params->ode_number*(gpu_params->data_params.rows + mcmc_dnorm_1_ode_padding_size);
    for(int ode_index = 0; ode_index < gpu_params->ode_number; ode_index++){
        memcpy(y_mcmc_dnorm_n_ode_padding_h + ode_index*y_mcmc_dnorm_1_ode_padding_h_size, y_mcmc_dnorm_1_ode_padding_h, y_mcmc_dnorm_1_ode_padding_h_size * sizeof(double));
    }

    /* dnorm N ode with padding - on device */
    double* y_mcmc_dnorm_n_ode_padding_d = 0;
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
    FluParameters *flu_params_current_d;
    checkCuda(cudaMalloc((void **) &flu_params_current_d, sizeof(FluParameters)));
    checkCuda(cudaMemcpy(flu_params_current_d, flu_params, sizeof(FluParameters), cudaMemcpyHostToDevice));

    /* flu_params_new_d - on device */
    FluParameters *flu_params_new_d;
    checkCuda(cudaMalloc((void **) &flu_params_new_d, sizeof(FluParameters)));
    checkCuda(cudaMemcpy(flu_params_new_d, flu_params, sizeof(FluParameters), cudaMemcpyHostToDevice));

    /* curand_state_d - on device */
    curandState *curand_state_d;
    checkCuda(cudaMalloc((void **)&curand_state_d, gpu_params->ode_number * sizeof(curandState)));

    /* r_denom/r_num - on host */
    double *r_h = (double*)malloc(ode_double_size);
    memset(r_h,0,ode_double_size);

    /* r_denom - on device */
    double* r_denom_d = 0;
    checkCuda(cudaMalloc((void **) &r_denom_d, ode_double_size));
    checkCuda(cudaMemcpy(r_denom_d, r_h, ode_double_size,cudaMemcpyHostToDevice));

    /* r_num - on device */
    double* r_num_d = 0;
    checkCuda(cudaMalloc((void **) &r_num_d, ode_double_size));
    checkCuda(cudaMemcpy(r_num_d, r_h, ode_double_size,cudaMemcpyHostToDevice));

    cudaStream_t streams[gpu_params->ode_number];
    for (int i = 0; i < gpu_params->ode_number; ++i) {
        checkCuda(cudaStreamCreate(&streams[i]));
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %ld micro seconds which is %.10f seconds\n", duration.count(),
           (duration.count() / 1e6));

    start = std::chrono::high_resolution_clock::now();
    //cudaProfilerStart();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    /* Blocks to process other things, must be equal number of ODE */
    gpu_params->block_size = GPU_ODE_THREADS; //max is 1024
    gpu_params->num_blocks = (gpu_params->ode_number + gpu_params->block_size - 1) / gpu_params->block_size;
    /* Blocks to process reduction sum with padding, must be divided by 1024 */
    int num_block = ceil(prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock / GPU_REDUCE_THREADS);
    printf("max threads = %d block = %d\n",prop.maxBlocksPerMultiProcessor * prop.maxThreadsPerBlock, num_block);

    for(int ode_index = 0; ode_index < gpu_params->ode_number; ode_index++){
        solve_ode_n_stream<<<gpu_params->num_blocks, gpu_params->block_size, 0, streams[ode_index]>>>(y_ode_input_d[ode_index], y_ode_output_d[ode_index],
                                                                 y_agg_input_d[ode_index], y_agg_output_d[ode_index],
                                                                 stf_d[ode_index],
                                                                 gpu_params_d, flu_params_new_d);
//        checkCuda(cudaStreamSynchronize(streams[ode_index]));
    }

    //    cudaProfilerStop();
    checkCuda(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for MCMC %d ODE with %d parameters %d times on GPU: %ld micro seconds which is %.10f seconds\n",
           NUMODE, DIM, gpu_params->mcmc_loop, duration.count(), (duration.count() / 1e6));

    start = std::chrono::high_resolution_clock::now();

    //y_ode_output_pinned_h
    for (int i = 0; i < gpu_params->ode_number; i++) {
//        checkCuda(cudaMemcpy(y_ode_output_pinned_h[i], y_ode_output_d[i], y_ode_output_size, cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(y_agg_output_pinned_h[i], y_agg_output_d[i], y_agg_size, cudaMemcpyDeviceToHost));
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
//                printf("%.5f\t", y_ode_output_pinned_h[random_index][index]);
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
                printf("%d %.5f\t",line_index, y_agg_output_pinned_h[random_index][index]);
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
    checkCuda(cudaDeviceSynchronize());
    // Free memory
    checkCuda(cudaFree(y_ode_input_d));
    checkCuda(cudaFree(y_ode_output_d));
    checkCuda(cudaFree(y_agg_input_d));
    checkCuda(cudaFree(y_agg_output_d));
    checkCuda(cudaFree(y_data_input_d));
    checkCuda(cudaFree(stf_d));
    checkCuda(cudaFree(y_mcmc_dnorm_n_ode_padding_d));
    checkCuda(cudaFree(gpu_params_d));
    checkCuda(cudaFree(flu_params_current_d));
    checkCuda(cudaFree(flu_params_new_d));
    gpu_params = nullptr;
    flu_params = nullptr;
    return;
}