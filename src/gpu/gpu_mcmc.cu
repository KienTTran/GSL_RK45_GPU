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
void mcmc_update_samples_test(GPUParameters* gpu_params_d, double* flu_param_samples_current_d[], double* flu_param_samples_new_d[], curandState* curand_state_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride) {
        curandState local_state = curand_state_d[index];
        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
            printf("ODE %d Old Sample: ",index);
            for(int i = 0; i < SAMPLE_LENGTH; i++){
                printf("%.9f\t",flu_param_samples_current_d[index][i]);
            }
            printf("\n");
        }
        for(int sample_index = 0; sample_index < SAMPLE_LENGTH; sample_index++){
            flu_param_samples_new_d[index][sample_index] = flu_param_samples_current_d[index][sample_index] + (0.5 * curand_normal(&local_state));
        }
        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
            printf("ODE %d New Sample: ",index);
            for(int i = 0; i < SAMPLE_LENGTH; i++){
                printf("%.9f\t",flu_param_samples_new_d[index][i]);
            }
            printf("\n");
        }
        curand_state_d[index] = local_state;
    }
}

//__global__
//void mcmc_update_parameters_from_samples(GPUParameters* gpu_params_d, double* flu_param_samples_d[], FluParameters* flu_params_d[]){
//    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = blockDim.x * gridDim.x;
//
//    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride) {
//        flu_params_d[index]->init_from_sample(flu_param_samples_d[index]);
//    }
//
//}

//__global__
//void mcmc_update_parameters(GPUParameters* gpu_params_d, FluParameters* flu_params_current_d[], FluParameters* flu_params_new_d[], curandState* curand_state_d, unsigned long seed){
//    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = blockDim.x * gridDim.x;
//
//    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride){
//        curandState local_state = curand_state_d[index];
//        // if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
//			// printf("\nODE %d Old phi: ",index);
//			// for(int i = 0; i < flu_params_current_d[index]->phi_length; i++){
//			// 	printf("%.2f\t",flu_params_current_d[index]->phi[i]);
//			// }
//			// printf("\nODE %d Old flu_params_current_d[%d]->beta[0] = %.5f\n", index, index, flu_params_current_d[index]->G_CLO_BETA1);
//			// printf("ODE %d Old flu_params_current_d[%d]->beta[1] = %.5f\n", index, index, flu_params_current_d[index]->G_CLO_BETA2);
//			// printf("ODE %d Old flu_params_current_d[%d]->beta[2] = %.5f\n", index, index, flu_params_current_d[index]->G_CLO_BETA3);
//			// printf("\nODE %d Old flu_params_current_d[%d]->sigma[0] = %.5f\n", index, index, flu_params_current_d[index]->sigma[0]);
//			// printf("ODE %d Old flu_params_current_d[%d]->sigma[1] = %.5f\n", index, index, flu_params_current_d[index]->sigma[1]);
//			// printf("ODE %d Old flu_params_current_d[%d]->sigma[2] = %.5f\n", index, index, flu_params_current_d[index]->sigma[2]);
//			// printf("ODE %d Old flu_params_current_d[%d]->amp = %.5f\n", index, index, flu_params_current_d[index]->amp);
//			// printf("ODE %d Old flu_params_current_d[%d]->nu_denom = %.5f\n", index, index, flu_params_current_d[index]->nu_denom);
//			// printf("ODE %d Old flu_params_current_d[%d]->rho_denom = %.5f\n", index, index, flu_params_current_d[index]->rho_denom);
//        // }
//        flu_params_new_d[index]->G_CLO_BETA1 = flu_params_current_d[index]->G_CLO_BETA1 + flu_params_current_d[index]->beta_sd[0]*curand_normal(&local_state);
//        flu_params_new_d[index]->G_CLO_BETA2 = flu_params_current_d[index]->G_CLO_BETA2 + flu_params_current_d[index]->beta_sd[1]*curand_normal(&local_state);
//        flu_params_new_d[index]->G_CLO_BETA3 = flu_params_current_d[index]->G_CLO_BETA3 + flu_params_current_d[index]->beta_sd[2]*curand_normal(&local_state);
//        // flu_params_new_d[index]->beta[0] = flu_params_new_d[index]->G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
//        // flu_params_new_d[index]->beta[1] = flu_params_new_d[index]->G_CLO_BETA2 / POPSIZE_MAIN;
//        // flu_params_new_d[index]->beta[2] = flu_params_new_d[index]->G_CLO_BETA3 / POPSIZE_MAIN;
//        // flu_params_new_d[index]->phi_0 = flu_params_current_d[index]->phi_0 + flu_params_current_d[index]->phi_sd * curand_normal(&local_state);
//        // flu_params_new_d[index]->phi[0] = flu_params_new_d[index]->phi_0;
//        // for(int i = 1; i < SAMPLE_PHI_LENGTH; i++){
//            // flu_params_new_d[index]->tau[i-1] = flu_params_new_d[index]->tau[i-1] + flu_params_new_d[index]->tau_sd[i-1]*curand_normal(&local_state);
//            // flu_params_new_d[index]->phi[i] = flu_params_new_d[index]->phi[i-1] + flu_params_new_d[index]->tau[i-1];
//        // }
//
//        //
//        // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
//        //
//
//        curand_state_d[index] = local_state;
//
//        // if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
//			// printf("\nODE %d Updated Phi: ", index);
//			// for (int i = 0; i < flu_params_new_d[index]->phi_length; i++) {
//				// printf("%.2f\t", flu_params_new_d[index]->phi[i]);
//			// }
//			// printf("\n");
//			// printf("\nODE %d Updated flu_params_new_d[%d]->beta[0] = %.5f\n", index, index, flu_params_new_d[index]->G_CLO_BETA1);
//			// printf("ODE %d Updated flu_params_new_d[%d]->beta[1] = %.5f\n", index, index, flu_params_new_d[index]->G_CLO_BETA2);
//			// printf("ODE %d Updated flu_params_new_d[%d]->beta[2] = %.5f\n", index, index, flu_params_new_d[index]->G_CLO_BETA3);
//			// printf("ODE %d Updated flu_params_new_d[%d]->sigma[0] = %.5f\n", index, index, flu_params_new_d[index]->sigma[0]);
//			// printf("ODE %d Updated flu_params_new_d[%d]->sigma[1] = %.5f\n", index, index, flu_params_new_d[index]->sigma[1]);
//			// printf("ODE %d Updated flu_params_new_d[%d]->sigma[2] = %.5f\n", index, index, flu_params_new_d[index]->sigma[2]);
//			// printf("ODE %d Updated flu_params_new_d[%d]->amp = %.5f\n", index, index, flu_params_new_d[index]->amp);
//			// printf("ODE %d Updated flu_params_new_d[%d]->nu_denom = %.5f\n", index, index, flu_params_new_d[index]->nu_denom);
//			// printf("ODE %d Updated flu_params_new_d[%d]->rho_denom = %.5f\n", index, index, flu_params_new_d[index]->rho_denom);
//        // }
//    }
//}

__global__
void mcmc_update_parameters_test(GPUParameters* gpu_params_d, FluParameters* flu_params_new_d, curandState* curand_state_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride){
        curandState local_state = curand_state_d[index];
        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
            printf("ODE %d Old phi: ",index);
            for(int i = 0; i < flu_params_new_d->phi_length; i++){
                printf("%.2f\t",flu_params_new_d->phi[i]);
            }
            printf("\nODE %d Old flu_params_new_d[%d]->beta[0] = %.9f\n", index, index, flu_params_new_d->G_CLO_BETA1[index]);
            printf("ODE %d Old flu_params_new_d[%d]->beta[1] = %.9f\n", index, index, flu_params_new_d->G_CLO_BETA2[index]);
            printf("ODE %d Old flu_params_new_d[%d]->beta[2] = %.9f\n", index, index, flu_params_new_d->G_CLO_BETA3[index]);
//            printf("ODE %d Old flu_params_new_d[%d]->sigma[0] = %.5f\n", index, index, flu_params_new_d->sigma[0]);
//            printf("ODE %d Old flu_params_new_d[%d]->sigma[1] = %.5f\n", index, index, flu_params_new_d->sigma[1]);
//            printf("ODE %d Old flu_params_new_d[%d]->sigma[2] = %.5f\n", index, index, flu_params_new_d->sigma[2]);
//            printf("ODE %d Old flu_params_new_d[%d]->amp = %.5f\n", index, index, flu_params_new_d->amp);
//            printf("ODE %d Old flu_params_new_d[%d]->nu_denom = %.5f\n", index, index, flu_params_new_d->nu_denom);
//            printf("ODE %d Old flu_params_new_d[%d]->rho_denom = %.5f\n", index, index, flu_params_new_d->rho_denom);
//            printf("ODE %d Old flu_params_new_d[%d]->trr = %.5f\n", index, index, flu_params_new_d->trr);
        }

//        flu_params_new_d->update(local_state);

//        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
//            gpu_params_d->update_beta();
//        }

//        flu_params_new_d->G_CLO_BETA1 = flu_params_new_d->G_CLO_BETA1 + flu_params_new_d->beta_sd[0]*curand_normal(&local_state);
//        flu_params_new_d->G_CLO_BETA2 = flu_params_new_d->G_CLO_BETA2 + flu_params_new_d->beta_sd[1]*curand_normal(&local_state);
//        flu_params_new_d->G_CLO_BETA3 = flu_params_new_d->G_CLO_BETA3 + flu_params_new_d->beta_sd[2]*curand_normal(&local_state);
//        flu_params_new_d->beta[0] = flu_params_new_d->G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
//        flu_params_new_d->beta[1] = flu_params_new_d->G_CLO_BETA2 / POPSIZE_MAIN;
//        flu_params_new_d->beta[2] = flu_params_new_d->G_CLO_BETA3 / POPSIZE_MAIN;

//        flu_params_new_d->phi_0 = flu_params_new_d->phi_0 + flu_params_new_d->phi_sd * curand_normal(&local_state);
//        flu_params_new_d->phi[0] = flu_params_new_d->phi_0;
//        for(int i = 1; i < SAMPLE_PHI_LENGTH; i++){
//            flu_params_new_d->tau[i-1] = flu_params_new_d->tau[i-1] + flu_params_new_d->tau_sd[i-1]*curand_normal(&local_state);
//            flu_params_new_d->phi[i] = flu_params_new_d->phi[i-1] + flu_params_new_d->tau[i-1];
//        }

        curand_state_d[index] = local_state;

        if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) {
            printf("\nODE %d Updated Phi: ", index);
            for (int i = 0; i < flu_params_new_d->phi_length; i++) {
                printf("%.2f\t", flu_params_new_d->phi[i]);
            }
            printf("\nODE %d Updated flu_params_new_d[%d]->beta[0] = %.9f\n", index, index, flu_params_new_d->G_CLO_BETA1[index]);
            printf("ODE %d Updated flu_params_new_d[%d]->beta[1] = %.9f\n", index, index, flu_params_new_d->G_CLO_BETA2[index]);
            printf("ODE %d Updated flu_params_new_d[%d]->beta[2] = %.9f\n", index, index, flu_params_new_d->G_CLO_BETA3[index]);
//            printf("ODE %d Updated flu_params_new_d[%d]->sigma[0] = %.5f\n", index, index, flu_params_new_d->sigma[0]);
//            printf("ODE %d Updated flu_params_new_d[%d]->sigma[1] = %.5f\n", index, index, flu_params_new_d->sigma[1]);
//            printf("ODE %d Updated flu_params_new_d[%d]->sigma[2] = %.5f\n", index, index, flu_params_new_d->sigma[2]);
//            printf("ODE %d Updated flu_params_new_d[%d]->amp = %.5f\n", index, index, flu_params_new_d->amp);
//            printf("ODE %d Updated flu_params_new_d[%d]->nu_denom = %.5f\n", index, index, flu_params_new_d->nu_denom);
//            printf("ODE %d Updated flu_params_new_d[%d]->rho_denom = %.5f\n", index, index, flu_params_new_d->rho_denom);
//            printf("ODE %d Old flu_params_new_d[%d]->trr = %.5f\n", index, index, flu_params_new_d->trr);
        }
    }
}

__global__
void mcmc_compute_r(double y_mcmc_dnorm_d[], double r_d[], int ode_padding_size, GPUParameters *gpu_params_d){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride) {
        r_d[index] = y_mcmc_dnorm_d[index * 2];
//        printf("ODE %d r_d[%d] = %.5f\n",index, index*2, y_mcmc_dnorm_d[index * 2]);
    }
}

__global__
void mcmc_check_acceptance(double r_denom_d[], double r_num_d[], GPUParameters *gpu_params_d, FluParameters* flu_params_current_d[], FluParameters* flu_params_new_d[],
                           curandState* curand_state_d, unsigned long seed){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride) {
        curandState localState = curand_state_d[index];
        if(exp(r_num_d[index] - r_denom_d[index]) > curand_uniform_double (&localState)){
            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)){
                printf("ODE %d before copy flu_params_current_d[%d]->phi[9] = %.5f flu_params_new_d[%d]->phi[9] = %.5f\n",
                       index, index, flu_params_current_d[index]->phi[9], index, flu_params_new_d[index]->phi[9]);
            }
            memcpy(flu_params_current_d[index], flu_params_new_d[index], sizeof(FluParameters));
            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)){
                printf("ODE %d after copy flu_params_current_d[%d]->phi[9] = %.5f\n",
                       index, index, flu_params_current_d[index]->phi[9]);
            }
            r_denom_d[index] = r_num_d[index];
//            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0))
            {
                printf("ODE %d exp(r) = %.5f > curand_uniform_double, accepted\n", index, exp(r_num_d[index] - r_denom_d[index]));
            }
        }
        else{
//            if(NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0))
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
