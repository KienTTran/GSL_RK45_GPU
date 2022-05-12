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
//            printf("Line %d agg\t",line_index);
//            for (int i = 0; i < params->data_params.cols; i++) {
//                const int agg_index = line_index*params->agg_dimension + 3;//Forth column only
//                printf("%.5f\t",y_ode_agg_d[ode_index][agg_index]);
//                if(i == params->data_params.cols - 1){
//                    printf("\n");
//                }
//            }
//            printf("Line %d max\t",line_index);
//            for (int i = 0; i < params->data_params.cols; i++) {
//                printf("%.5f\t",y_ode_agg_d[ode_index][i]);
//                if(i == params->data_params.cols - 1){
//                    printf("\n");
//                }
//            }
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
void mcmc_setup_states_for_random(curandState* curand_state_d, int size){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int index = index_gpu; index < size; index += stride) {
        curand_init(clock64(), index, 0, &curand_state_d[index]);
    }
}

__global__
void mcmc_generate_norm(double* norm_d, size_t norm_size, curandState* curand_state_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int index = index_gpu; index < norm_size; index += stride) {
        curandState local_state = curand_state_d[index];
        norm_d[index] = curand_normal(&local_state);
        curand_state_d[index] = local_state;
//        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
//            const int ode_index = index / SAMPLE_LENGTH;
//            printf("ODE %d norm_d[%d] = %.5f\n",ode_index, index, norm_d[index]);
//            printf("Index %d Norm Random = %.5f\n",index, norm_d[index]);
//        }
    }
}

__global__
void mcmc_generate_norm_2(double* norm_d[], size_t norm_size, curandState* curand_state_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int index = index_gpu; index < norm_size; index += stride) {
        const int ode_index = index / SAMPLE_LENGTH;
        const int sample_index = index % SAMPLE_LENGTH;
        curandState local_state = curand_state_d[index];
        norm_d[ode_index][sample_index] = curand_normal(&local_state);
        curand_state_d[index] = local_state;
//        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
//            const int ode_index = index / SAMPLE_LENGTH;
//            printf("ODE %d norm_d[%d] = %.5f\n",ode_index, index, norm_d[index]);
//            printf("Index %d Norm Random = %.5f\n",index, norm_d[index]);
//        }
    }
}

__global__ void mcmc_compute_norm_sd(FluParameters* flu_params_d, double* norm_d, double* norm_sd_d, size_t norm_size){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < norm_size; index += stride){
        const int ode_index = index / SAMPLE_LENGTH;
        const int sample_index = index % SAMPLE_LENGTH;
        for(int i = 0; i < NUMSEROTYPES; i++){
            const int beta_index = ode_index * NUMSEROTYPES + i;
            const int norm_beta_index = ode_index * SAMPLE_LENGTH + i;
            norm_sd_d[norm_beta_index] = norm_d[norm_beta_index] * flu_params_d->beta_sd[i];
        }
        /* Index 3 */
        const int norm_phi0_index = ode_index * SAMPLE_LENGTH + SAMPLE_PHI_0_INDEX;
        norm_sd_d[norm_phi0_index] = norm_d[norm_phi0_index] * flu_params_d->phi_sd;
        /* Index 4 - 12 */
        for(int i = 0; i < SAMPLE_TAU_LENGTH; i++){
            const int tau_index = ode_index * SAMPLE_TAU_LENGTH + i;
            const int norm_phi_index = ode_index * SAMPLE_LENGTH + ((SAMPLE_PHI_0_INDEX + 1) + i);
            norm_sd_d[norm_phi_index] = norm_d[norm_phi_index] * flu_params_d->tau_sd[i];
        }
    }
}

__global__
void mcmc_update_parameters(GPUParameters* gpu_params_d,
                            FluParameters* flu_params_current_d, FluParameters* flu_params_new_d,
                            curandState* curand_state_d){
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
            flu_params_new_d->G_CLO_BETA[index*NUMSEROTYPES + i] = flu_params_current_d->G_CLO_BETA[index*NUMSEROTYPES + i] + flu_params_current_d->beta_sd[i]*curand_normal(&local_state);
//            flu_params_new_d->beta[index*NUMSEROTYPES + i] = flu_params_new_d->G_CLO_BETA[index*NUMSEROTYPES + i] / POPSIZE_MAIN;
            flu_params_new_d->beta[index*NUMSEROTYPES + i] = flu_params_new_d->G_CLO_BETA[index*NUMSEROTYPES + i] * BETA_OVER_POP_MAIN;
        }    // NOTE this is in a density-dependent transmission scheme
        flu_params_new_d->phi_0 = flu_params_current_d->phi_0 + flu_params_current_d->phi_sd * curand_normal(&local_state);
        flu_params_new_d->phi[index*SAMPLE_PHI_LENGTH] = flu_params_new_d->phi_0;
        for(int i = 1; i < SAMPLE_PHI_LENGTH; i++){
            flu_params_new_d->tau[index*SAMPLE_TAU_LENGTH + (i - 1)] = flu_params_current_d->tau[index*SAMPLE_TAU_LENGTH + (i - 1)] + flu_params_current_d->tau_sd[(i - 1)]*curand_normal(&local_state);
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
}

__global__
void mcmc_update_parameters_with_norm_sd(GPUParameters* gpu_params_d, FluParameters* flu_params_current_d, FluParameters* flu_params_new_d, double* norm_sd_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride){
//        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
//            printf("ODE %d Old phi: ",index);
//            for(int i = 0; i < SAMPLE_PHI_LENGTH; i++){
//                printf("%.2f\t",flu_params_current_d->phi[i]);
//            }
//            printf("\n");
//            for(int i = 0; i < NUMSEROTYPES; i++){
//                printf("ODE %d Old flu_params_current_d->beta[%d] = %.9f\n", index, index*NUMSEROTYPES + i, flu_params_current_d->G_CLO_BETA[index*NUMSEROTYPES + i]);
//            }
//        }

        /* Index 0 - 2 */
        for(int i = 0; i < NUMSEROTYPES; i++){
            const int beta_index = index * NUMSEROTYPES + i;
            const int norm_beta_index = index * SAMPLE_LENGTH + i;
            flu_params_new_d->G_CLO_BETA[beta_index] = flu_params_current_d->G_CLO_BETA[beta_index] + norm_sd_d[norm_beta_index];
            flu_params_new_d->beta[beta_index] = flu_params_new_d->G_CLO_BETA[beta_index] / POPSIZE_MAIN;
//            flu_params_new_d->beta[beta_index] = flu_params_new_d->G_CLO_BETA[beta_index] * flu_params_current_d->beta_over_pop_main;
//            printf("ODE %d beta[%d] beta_index = %d norm_index = %d\n",index, i, beta_index, norm_index);
        }
        /* Index 3 */
        const int norm_phi0_index = index * SAMPLE_LENGTH + SAMPLE_PHI_0_INDEX;
        flu_params_new_d->phi_0 = flu_params_current_d->phi_0 + flu_params_current_d->phi_sd + norm_sd_d[norm_phi0_index];
//        printf("ODE %d phi_0_index = %d norm_index = %d\n",index, phi_0_index, norm_index);
        /* Index 4 - 12 */
        flu_params_new_d->phi[index*SAMPLE_PHI_LENGTH] = flu_params_new_d->phi_0;
        for(int i = 1; i < SAMPLE_PHI_LENGTH; i++){
            const int tau_index = index * SAMPLE_TAU_LENGTH + (i - 1);
            const int norm_tau_index = index * SAMPLE_LENGTH + ((SAMPLE_PHI_0_INDEX + 1) + (i - 1));
            flu_params_new_d->tau[tau_index] = flu_params_current_d->tau[tau_index] +  norm_sd_d[norm_tau_index];
            flu_params_new_d->phi[index*SAMPLE_PHI_LENGTH + i] = flu_params_new_d->phi[index*SAMPLE_PHI_LENGTH + (i - 1)] + flu_params_new_d->tau[tau_index];
//            printf("ODE %d tau[%d] tau_index = %d norm_index = %d\n",index, (i - 1), tau_index, norm_index);
        }

//        if(index == 32){
//            for(int i = 0; i < SAMPLE_LENGTH; i++){
//                const int norm_index = index * SAMPLE_LENGTH + i;
//                printf("ODE %d norm_index %d norm_d[%d] = %.5f\n",index, norm_index, norm_index, norm_d[norm_index]);
//            }
//        }

//        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
//            printf("\nODE %d Updated Phi: ", index);
//            for (int i = 0; i < SAMPLE_PHI_LENGTH; i++) {
//                printf("%.2f\t", flu_params_new_d->phi[index*SAMPLE_PHI_LENGTH + i]);
//            }
//            printf("\n");
//            for(int i = 0; i < NUMSEROTYPES; i++){
//                printf("ODE %d Updated flu_params_new_d->beta[%d] = %.9f\n", index, index*NUMSEROTYPES + i, flu_params_new_d->G_CLO_BETA[index*NUMSEROTYPES + i]);
//            }
//        }
    }
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
//            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)){
//                printf("ODE %d before copy flu_params_current_d->phi[9] = %.5f flu_params_new_d->phi[9] = %.5f\n",
//                       index, flu_params_current_d->phi[9], flu_params_new_d->phi[9]);
//                printf("ODE %d before copy flu_params_current_d->beta[0] = %.9f flu_params_new_d->beta[0] = %.9f\n",
//                       index, flu_params_current_d->G_CLO_BETA[index*NUMSEROTYPES], flu_params_new_d->G_CLO_BETA[index*NUMSEROTYPES]);
//            }
            memcpy(flu_params_current_d, flu_params_new_d, sizeof(FluParameters));
//            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)){
//                printf("ODE %d after copy flu_params_current_d->phi[9] = %.5f\n",
//                       index, flu_params_current_d->phi[9]);
//                printf("ODE %d after copy flu_params_current_d->beta[0] = %.9f\n",
//                       index, flu_params_current_d->G_CLO_BETA[index*NUMSEROTYPES]);
//            }
            r_denom_d[index] = r_num_d[index];
//            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 4) == 0))
//            {
//                printf("ODE %d exp(r) = %.5f > curand_uniform_double, accepted\n", index, exp(r_num_d[index] - r_denom_d[index]));
//            }
        }
        else{
//            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 4) == 0))
//            {
//                printf("ODE %d exp(r) = %.5f <= curand_uniform_double, rejected\n", index, exp(r_num_d[index] - r_denom_d[index]));
//            }
        }
        curand_state_d[index] = localState;
//        if(index == 0)
//        {
//            printf("\n==== one MCMC Chain Done ====\n\n");
//        }
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

__global__
void mcmc_print_iter(int iter){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < 1; index += stride) {
        printf("\n==== iter %d done =====\n\n", iter);
    }
}
