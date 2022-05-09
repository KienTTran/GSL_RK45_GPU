//
// Created by kient on 5/2/2022.
//
#include "gpu_flu.cuh"
#include "gpu_mcmc.cuh"

__device__
double dnorm(const double x, double mean, double sd, bool use_log) {
    if (x < 0 || mean < 0) {
        return 0;
    }
    if (use_log) {
        return log((1 / (sqrtf(2 * M_PI) * sd)) * exp(-((pow((x - mean), 2)) / (2 * pow(sd, 2)))));
    } else {
        return (1 / (sqrtf(2 * M_PI) * sd)) * exp(-((pow((x - mean), 2)) / (2 * pow(sd, 2))));
    }
}

__global__ /* Each thread will calculate 1 line in y_ode_agg_d */
void mcmc_dnorm(double *y_data_input_d[], double *y_ode_agg_d[], double y_mcmc_dnorm_d[], GPUParameters *params) {
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < params->ode_number * params->data_params.rows; index += stride) {
        const int ode_index = index / params->data_params.rows;
        const int line_index = index % params->data_params.rows;

//        if(line_index % 32 == 0)
//        {
//            printf("index = %d ODE %d line %d will be processed by thread %d (block %d)\n", index, ode_index, line_index, threadIdx.x, blockIdx.x);
//        }

        //Calculate agg mean
        for (int i = 0; i < params->data_params.cols; i++) {
            const int y_agg_index = line_index*params->agg_dimension + 3;//Forth column only
            const int y_data_index = line_index*params->data_params.cols + i;
            const int y_dnorm_index = ode_index*params->data_params.rows + line_index;
            y_mcmc_dnorm_d[y_dnorm_index] += dnorm(y_data_input_d[ode_index][y_data_index],
                                          y_ode_agg_d[ode_index][y_agg_index] < 0 ? -9999.0 : y_ode_agg_d[ode_index][y_agg_index]/y_ode_agg_d[ode_index][i],
                                          0.25,
                                          true);
        }

//        if(ode_index == 0 && line_index == 1){
//            printf("Line %d data\t",line_index);
//            for (int i = 0; i < params->data_params.cols; i++) {
//                const int data_index = line_index*params->data_params.cols + i;
//                printf("%.5f\t",y_data_input_d[ode_index][data_index]);
//                if(i == params->data_params.cols - 1){
//                    printf("\n");
//                }
//            }
//            __syncthreads();
//            printf("Line %d agg\t",line_index);
//            for (int i = 0; i < params->data_params.cols; i++) {
//                const int agg_index = line_index*params->agg_dimension + 3;//Forth column only
//                printf("%.5f\t",y_ode_agg_d[ode_index][agg_index]);
//                if(i == params->data_params.cols - 1){
//                    printf("\n");
//                }
//            }
//            __syncthreads();
//            printf("Line %d max\t",line_index);
//            for (int i = 0; i < params->data_params.cols; i++) {
//                printf("%.5f\t",y_ode_agg_d[ode_index][i]);
//                if(i == params->data_params.cols - 1){
//                    printf("\n");
//                }
//            }
//            __syncthreads();
//            printf("Line %d mean\t",line_index);
//            for (int i = 0; i < params->data_params.cols; i++) {
//                const int y_agg_index = line_index*params->agg_dimension + 3;//Forth column only
//                printf("%.5f\t",y_ode_agg_d[ode_index][y_agg_index] < 0 ? -9999.0 : y_ode_agg_d[ode_index][y_agg_index]/y_ode_agg_d[ode_index][i]);
//                if(i == params->data_params.cols - 1){
//                    printf("\n");
//                }
//            }
//        }
//        if(line_index < 3)
//        {
//            printf("index %d ODE %d line %d dnorm sum %.5f\n",index,ode_index,line_index,y_mcmc_dnorm_d[index]);
//        }

    }
    return;
}


__global__ /* Each thread will calculate 1 line in y_ode_agg_d */
void mcmc_dnorm_padding(double *y_data_input_d[], double *y_ode_agg_d[], double y_mcmc_dnorm_d[], int ode_padding_size, GPUParameters *gpu_params_d) {
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number * gpu_params_d->data_params.rows; index += stride) {
        const int ode_index = index / (gpu_params_d->data_params.rows);
        const int line_index = index % gpu_params_d->data_params.rows;

//        if(line_index % 32 == 0)
//        {
//            printf("index = %d ODE %d line %d will be processed by thread %d (block %d)\n", index, ode_index, line_index, threadIdx.x, blockIdx.x);
//        }


        //Reset dnorm
        for (int i = 0; i < gpu_params_d->data_params.cols; i++) {
            const int y_dnorm_index = ode_index*(gpu_params_d->data_params.rows + ode_padding_size) + line_index;
            y_mcmc_dnorm_d[y_dnorm_index] = 0.0;
        }
        //Calculate agg mean
        for (int i = 0; i < gpu_params_d->data_params.cols; i++) {
            const int y_agg_index = line_index*gpu_params_d->agg_dimension + 3;//Forth column only
            const int y_data_index = line_index*gpu_params_d->data_params.cols + i;
            const int y_dnorm_index = ode_index*(gpu_params_d->data_params.rows + ode_padding_size) + line_index;
            y_mcmc_dnorm_d[y_dnorm_index] += dnorm(y_data_input_d[ode_index][y_data_index],
                                                   y_ode_agg_d[ode_index][y_agg_index] < 0 ? -9999.0 : y_ode_agg_d[ode_index][y_agg_index]/y_ode_agg_d[ode_index][i],
                                                   0.25,
                                                   true);
        }
//        if(line_index < 3)
//        {
//            const int y_dnorm_index = ode_index*(gpu_params_d->data_params.rows + ode_padding_size) + line_index;
//            printf("index %d ODE %d line %d dnorm sum col = %.5f\n",index,ode_index,line_index,y_mcmc_dnorm_d[y_dnorm_index]);
//        }

    }
    return;
}

__global__
void mcmc_setup_states_for_random(curandState* curand_state_d)
{
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int index = index_gpu; index < NUMODE; index += stride) {
        curand_init(clock64(), index, 0, &curand_state_d[index]);
    }
}

__global__
void mcmc_update_parameters(GPUParameters* gpu_params_d, FluParameters* flu_params_current_d, FluParameters* flu_params_new_d, curandState* curand_state_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride){
        curandState local_state = curand_state_d[index];
        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
            printf("ODE %d Old phi: ",index);
            for(int i = 0; i < SAMPLE_PHI_LENGTH; i++){
                printf("%.2f\t",flu_params_current_d->phi[i]);
            }
            printf("\n");
            for(int i = 0; i < NUMSEROTYPES; i++){
                printf("ODE %d Old flu_params_current_d->beta[%d] = %.9f\n", index, index*NUMSEROTYPES + i, flu_params_current_d->G_CLO_BETA[index*NUMSEROTYPES + i]);
            }
        }

        for(int i = 0; i < NUMSEROTYPES; i++){
            flu_params_new_d->G_CLO_BETA[index*NUMSEROTYPES + i] = flu_params_current_d->G_CLO_BETA[index*NUMSEROTYPES + i] + flu_params_current_d->beta_sd[index*NUMSEROTYPES + i]*curand_normal(&local_state);
            flu_params_new_d->beta[index*NUMSEROTYPES + i] = flu_params_new_d->G_CLO_BETA[index*NUMSEROTYPES + i] / POPSIZE_MAIN;
        }    // NOTE this is in a density-dependent transmission scheme
        flu_params_new_d->phi_0 = flu_params_current_d->phi_0 + flu_params_current_d->phi_sd * curand_normal(&local_state);
        for(int i = 0; i < SAMPLE_PHI_LENGTH; i++){
            flu_params_new_d->phi[index*SAMPLE_PHI_LENGTH] = flu_params_new_d->phi_0;
            flu_params_new_d->tau[index*SAMPLE_TAU_LENGTH + (i - 1)] = flu_params_current_d->tau[index*SAMPLE_TAU_LENGTH + (i - 1)] + flu_params_current_d->tau_sd[index*SAMPLE_TAU_LENGTH + (i - 1)]*curand_normal(&local_state);
            flu_params_new_d->phi[index*SAMPLE_PHI_LENGTH + i] = flu_params_new_d->phi[index*SAMPLE_PHI_LENGTH + (i - 1)] + flu_params_new_d->tau[index*SAMPLE_TAU_LENGTH + (i - 1)];
        }
        curand_state_d[index] = local_state;

        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
            printf("\nODE %d Updated Phi: ", index);
            for (int i = 0; i < SAMPLE_PHI_LENGTH; i++) {
                printf("%.2f\t", flu_params_new_d->phi[index*SAMPLE_PHI_LENGTH + i]);
            }
            printf("\n");
            for(int i = 0; i < NUMSEROTYPES; i++){
                printf("ODE %d Updated flu_params_new_d->beta[%d] = %.9f\n", index, index*NUMSEROTYPES + i, flu_params_new_d->G_CLO_BETA[index*NUMSEROTYPES + i]);
            }
        }
    }
    __syncthreads();
}

__global__
void mcmc_compute_r(double y_mcmc_dnorm_d[], double r_d[], GPUParameters *gpu_params_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride) {
        r_d[index] = y_mcmc_dnorm_d[index * 2];
//        printf("ODE %d r_d[%d] = %.5f\n",index, index*2, y_mcmc_dnorm_d[index * 2]);
    }
}

__global__
void mcmc_check_acceptance(double r_denom_d[], double r_num_d[], GPUParameters *gpu_params_d, FluParameters* flu_params_current_d, FluParameters* flu_params_new_d,
                           curandState* curand_state_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride) {
        curandState localState = curand_state_d[index];
        if(exp(r_num_d[index] - r_denom_d[index]) > curand_uniform_double (&localState)){
            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)){
                printf("ODE %d before copy flu_params_current_d->phi[9] = %.5f flu_params_new_d->phi[9] = %.5f\n",
                       index, flu_params_current_d->phi[9], flu_params_new_d->phi[9]);
                printf("ODE %d before copy flu_params_current_d->beta[0] = %.9f flu_params_new_d->beta[0] = %.9f\n",
                       index, flu_params_current_d->G_CLO_BETA[index*NUMSEROTYPES], flu_params_new_d->G_CLO_BETA[index*NUMSEROTYPES]);
            }
            memcpy(flu_params_current_d, flu_params_new_d, sizeof(FluParameters));
            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)){
                printf("ODE %d after copy flu_params_current_d->phi[9] = %.5f\n",
                       index, flu_params_current_d->phi[9]);
                printf("ODE %d after copy flu_params_current_d->beta[0] = %.9f\n",
                       index, flu_params_current_d->G_CLO_BETA[index*NUMSEROTYPES]);
            }
            r_denom_d[index] = r_num_d[index];
            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 4) == 0))
            {
                printf("ODE %d exp(r) = %.5f > curand_uniform_double, accepted\n", index, exp(r_num_d[index] - r_denom_d[index]));
            }
        }
        else{
            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 4) == 0))
            {
                printf("ODE %d exp(r) = %.5f <= curand_uniform_double, rejected\n", index, exp(r_num_d[index] - r_denom_d[index]));
            }
        }
        curand_state_d[index] = localState;
    }
}

__global__
void mcmc_print_r(GPUParameters* gpu_params_d, double* r_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride) {
        printf("ODE %d r = %.5f\n", index, r_d[index]);
    }

}
