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

__global__ void mcmc_setup_states_for_random(curandState* curand_state_d)
{
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(clock64(), index_gpu, 0, &curand_state_d[index_gpu]);
}

__global__
void mcmc_update_parameters(GPUParameters* gpu_params_d, FluParameters* flu_params_current_d[], curandState* curand_state_d, unsigned long seed){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < gpu_params_d->ode_number; index += stride){
        curandState local_state = curand_state_d[index];
        for(int sample_index = 0; sample_index < flu_params_current_d[index]->sample_length; sample_index++){
            /* This formula follows rnorm.c in R */
            if(flu_params_current_d[index]->sample_sd[sample_index] == 0){
                flu_params_current_d[index]->sample[sample_index] =  flu_params_current_d[index]->sample[sample_index];
            }
            else{
                flu_params_current_d[index]->sample[sample_index] =  flu_params_current_d[index]->sample[sample_index]
                                                                     + flu_params_current_d[index]->sample_sd[sample_index] * curand_normal(&local_state);
            }
//            printf("ODE %d sample_new[%d] = %.5f\n", index, sample_index, flu_params_current_d[index]->sample[sample_index]);
        }
        flu_params_current_d[index]->beta[0] = flu_params_current_d[index]->sample[0];
        flu_params_current_d[index]->beta[1] = flu_params_current_d[index]->sample[1];
        flu_params_current_d[index]->beta[2] = flu_params_current_d[index]->sample[2];
        flu_params_current_d[index]->phi[0] = flu_params_current_d[index]->sample[3];
        flu_params_current_d[index]->tau[0] = flu_params_current_d[index]->sample[4];
        flu_params_current_d[index]->tau[1] = flu_params_current_d[index]->sample[5];
        flu_params_current_d[index]->tau[2] = flu_params_current_d[index]->sample[6];
        flu_params_current_d[index]->tau[3] = flu_params_current_d[index]->sample[7];
        flu_params_current_d[index]->tau[4] = flu_params_current_d[index]->sample[8];
        flu_params_current_d[index]->tau[5] = flu_params_current_d[index]->sample[9];
        flu_params_current_d[index]->tau[6] = flu_params_current_d[index]->sample[10];
        flu_params_current_d[index]->tau[7] = flu_params_current_d[index]->sample[11];
        flu_params_current_d[index]->tau[8] = flu_params_current_d[index]->sample[12];
        flu_params_current_d[index]->G_CLO_BETA1 = flu_params_current_d[index]->beta[0];
        flu_params_current_d[index]->G_CLO_BETA2 = flu_params_current_d[index]->beta[1];
        flu_params_current_d[index]->G_CLO_BETA3 = flu_params_current_d[index]->beta[2];
        flu_params_current_d[index]->G_CLO_SIGMA12 = flu_params_current_d[index]->sigma[0];
        flu_params_current_d[index]->G_CLO_SIGMA13 = flu_params_current_d[index]->sigma[1];
        flu_params_current_d[index]->G_CLO_SIGMA23 = flu_params_current_d[index]->sigma[2];
        flu_params_current_d[index]->G_CLO_AMPL = flu_params_current_d[index]->amp;
        flu_params_current_d[index]->G_CLO_NU_DENOM = flu_params_current_d[index]->nu_denom;
        flu_params_current_d[index]->G_CLO_RHO_DENOM = flu_params_current_d[index]->rho_denom;
        flu_params_current_d[index]->epidur = flu_params_current_d[index]->G_CLO_EPIDUR;
        flu_params_current_d[index]->phi[0] = flu_params_current_d[index]->phi_0;
        for(int i = 1; i < flu_params_current_d[index]->phi_length; i++){
            flu_params_current_d[index]->phi[i] = flu_params_current_d[index]->phi[i-1] + flu_params_current_d[index]->tau[i-1];
        }
        printf("ODE %d Updated Parameters: -beta1 %.2f -beta2 %.2f -beta3 %.2f -sigma12 %.2f -sigma13 %.2f -sigma23 %.2f -amp %.2f -nu_denom %.2f -rho_denom %.2f -phi %.2f\n",index,
               flu_params_current_d[index]->G_CLO_BETA1,flu_params_current_d[index]->G_CLO_BETA2,flu_params_current_d[index]->G_CLO_BETA3,
               flu_params_current_d[index]->G_CLO_SIGMA12,flu_params_current_d[index]->G_CLO_SIGMA13,flu_params_current_d[index]->G_CLO_SIGMA13,
               flu_params_current_d[index]->G_CLO_AMPL,flu_params_current_d[index]->G_CLO_NU_DENOM,flu_params_current_d[index]->G_CLO_RHO_DENOM,
               flu_params_current_d[index]->phi[flu_params_current_d[index]->phi_length - 1]);
        flu_params_current_d[index]->v_d_i_amp  = flu_params_current_d[index]->G_CLO_AMPL;
        flu_params_current_d[index]->beta[0] = flu_params_current_d[index]->G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
        flu_params_current_d[index]->beta[1] = flu_params_current_d[index]->G_CLO_BETA2 / POPSIZE_MAIN;
        flu_params_current_d[index]->beta[2] = flu_params_current_d[index]->G_CLO_BETA3 / POPSIZE_MAIN;

        flu_params_current_d[index]->sigma2d[0][1] = flu_params_current_d[index]->G_CLO_SIGMA12; // 0.7; // the level of susceptibility to H1 if you've had B
        flu_params_current_d[index]->sigma2d[1][0] = flu_params_current_d[index]->G_CLO_SIGMA12; // 0.7; // and vice versa

        flu_params_current_d[index]->sigma2d[1][2] = flu_params_current_d[index]->G_CLO_SIGMA23; // 0.7; // the level of susceptibility to H3 if you've had B
        flu_params_current_d[index]->sigma2d[2][1] = flu_params_current_d[index]->G_CLO_SIGMA23; // 0.7; // and vice versa

        flu_params_current_d[index]->sigma2d[0][2] = flu_params_current_d[index]->G_CLO_SIGMA13; // 0.3; // the level of susceptibility to H3 if you've had H1
        flu_params_current_d[index]->sigma2d[2][0] = flu_params_current_d[index]->G_CLO_SIGMA13; // 0.3; // and vice versa

        flu_params_current_d[index]->sigma2d[0][0] = 0;
        flu_params_current_d[index]->sigma2d[1][1] = 0;
        flu_params_current_d[index]->sigma2d[2][2] = 0;

        flu_params_current_d[index]->v_d_i_nu    = 1 / flu_params_current_d[index]->G_CLO_NU_DENOM;                // recovery rate
        flu_params_current_d[index]->v_d_i_immune_duration = flu_params_current_d[index]->G_CLO_RHO_DENOM;    // 2.5 years of immunity to recent infection'

        flu_params_current_d[index]->v_d_i_epidur = flu_params_current_d[index]->G_CLO_EPIDUR;

        //
        // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
        //

        flu_params_current_d[index]->trr = ((double)NUMR) / flu_params_current_d[index]->v_d_i_immune_duration;
//        printf("ODE %d updated phi[%d] = %.5f\n",index,9,flu_params_current_d[index]->phi[9]);
        curand_state_d[index] = local_state;
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
            flu_params_current_d[index] = flu_params_new_d[index];
            r_denom_d[index] = r_num_d[index];
            printf("ODE %d exp(r) = %.5f > curand_uniform_double, accepted\n", index, exp(r_num_d[index] - r_denom_d[index]));
        }
        else{
            printf("ODE %d exp(r) = %.5f <= curand_uniform_double, rejected\n", index, exp(r_num_d[index] - r_denom_d[index]));
        }
        curand_state_d[index] = localState;
    }
}
