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
void mcmc_dnorm_padding(double *y_data_input_d[], double *y_ode_agg_d[], double y_mcmc_dnorm_d[], int ode_padding_size, GPUParameters *params) {
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < params->ode_number * params->data_params.rows; index += stride) {
        const int ode_index = index / (params->data_params.rows);
        const int line_index = index % params->data_params.rows;

//        if(line_index % 32 == 0)
//        {
//            printf("index = %d ODE %d line %d will be processed by thread %d (block %d)\n", index, ode_index, line_index, threadIdx.x, blockIdx.x);
//        }


        //Reset dnorm
        for (int i = 0; i < params->data_params.cols; i++) {
            const int y_dnorm_index = ode_index*(params->data_params.rows + ode_padding_size) + line_index;
            y_mcmc_dnorm_d[y_dnorm_index] = 0.0;
        }
        //Calculate agg mean
        for (int i = 0; i < params->data_params.cols; i++) {
            const int y_agg_index = line_index*params->agg_dimension + 3;//Forth column only
            const int y_data_index = line_index*params->data_params.cols + i;
            const int y_dnorm_index = ode_index*(params->data_params.rows + ode_padding_size) + line_index;
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
//            const int y_dnorm_index = ode_index*(params->data_params.rows + ode_padding_size) + line_index;
//            printf("index %d ODE %d line %d dnorm sum %.5f\n",index,ode_index,line_index,y_mcmc_dnorm_d[y_dnorm_index]);
//        }

    }
    return;
}

__global__ void mcmc_setup_states_for_random(curandState* curand_state_d)
{
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, index_gpu, 0, &curand_state_d[index_gpu]);
}

__global__
void mcmc_update_parameters(FluParameters* current_flu_params[], int ode_number, curandState* curand_state_d, unsigned long seed){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < ode_number; index += stride){
        curandState local_state = curand_state_d[index];
        for(int sample_index = 0; sample_index < current_flu_params[index]->sample_length; sample_index++){
            current_flu_params[index]->sample[sample_index] =   (curand_normal(&local_state) * current_flu_params[index]->sample[sample_index])
                                                                + current_flu_params[index]->sample_sd[sample_index];
//            if(sample_index == current_flu_params[index]->sample_length - 1) {
//                printf("ODE %d random = %.5f sample_old[%d] = %.5f sample_sd[%d] = %.5f sample_new[%d] = %.5f\n", index,
//                       curand_normal(&local_state),
//                       sample_index, current_flu_params[index]->sample[sample_index],
//                       sample_index, current_flu_params[index]->sample_sd[sample_index],
//                       sample_index, current_flu_params[index]->sample[sample_index]);
//            }
        }
        current_flu_params[index]->beta[0] = current_flu_params[index]->sample[0];
        current_flu_params[index]->beta[1] = current_flu_params[index]->sample[1];
        current_flu_params[index]->beta[2] = current_flu_params[index]->sample[2];
        current_flu_params[index]->phi[0] = current_flu_params[index]->sample[3];
        current_flu_params[index]->tau[0] = current_flu_params[index]->sample[4];
        current_flu_params[index]->tau[1] = current_flu_params[index]->sample[5];
        current_flu_params[index]->tau[2] = current_flu_params[index]->sample[6];
        current_flu_params[index]->tau[3] = current_flu_params[index]->sample[7];
        current_flu_params[index]->tau[4] = current_flu_params[index]->sample[8];
        current_flu_params[index]->tau[5] = current_flu_params[index]->sample[9];
        current_flu_params[index]->tau[6] = current_flu_params[index]->sample[10];
        current_flu_params[index]->tau[7] = current_flu_params[index]->sample[11];
        current_flu_params[index]->tau[8] = current_flu_params[index]->sample[12];
        current_flu_params[index]->G_CLO_BETA1 = current_flu_params[index]->beta[0];
        current_flu_params[index]->G_CLO_BETA2 = current_flu_params[index]->beta[1];
        current_flu_params[index]->G_CLO_BETA3 = current_flu_params[index]->beta[2];
        current_flu_params[index]->G_CLO_SIGMA12 = current_flu_params[index]->sigma[0];
        current_flu_params[index]->G_CLO_SIGMA13 = current_flu_params[index]->sigma[1];
        current_flu_params[index]->G_CLO_SIGMA23 = current_flu_params[index]->sigma[2];
        current_flu_params[index]->G_CLO_AMPL = current_flu_params[index]->amp;
        current_flu_params[index]->G_CLO_NU_DENOM = current_flu_params[index]->nu_denom;
        current_flu_params[index]->G_CLO_RHO_DENOM = current_flu_params[index]->rho_denom;
        current_flu_params[index]->epidur = current_flu_params[index]->G_CLO_EPIDUR;
        current_flu_params[index]->phi[0] = current_flu_params[index]->phi_0;
        for(int i = 1; i < current_flu_params[index]->phi_length; i++){
            current_flu_params[index]->phi[i] = current_flu_params[index]->phi[i-1] + current_flu_params[index]->tau[i-1];
        }
        printf("ODE %d Updated Parameters: -beta1 %.2f -beta2 %.2f -beta3 %.2f -sigma12 %.2f -sigma13 %.2f -sigma23 %.2f -amp %.2f -nu_denom %.2f -rho_denom %.2f\n-phi\t",index,
               current_flu_params[index]->G_CLO_BETA1,current_flu_params[index]->G_CLO_BETA2,current_flu_params[index]->G_CLO_BETA3,
               current_flu_params[index]->G_CLO_SIGMA12,current_flu_params[index]->G_CLO_SIGMA13,current_flu_params[index]->G_CLO_SIGMA13,
               current_flu_params[index]->G_CLO_AMPL,current_flu_params[index]->G_CLO_NU_DENOM,current_flu_params[index]->G_CLO_RHO_DENOM);
        for(int i = 0; i < current_flu_params[index]->phi_length; i++){
//            if(i == current_flu_params[index]->phi_length - 1)
            {
                printf("[%d] %.5f\t",i,current_flu_params[index]->phi[i]);
            }
        }
        printf("\n");
        current_flu_params[index]->v_d_i_amp   = current_flu_params[index]->G_CLO_AMPL;
        current_flu_params[index]->beta[0] = current_flu_params[index]->G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
        current_flu_params[index]->beta[1] = current_flu_params[index]->G_CLO_BETA2 / POPSIZE_MAIN;
        current_flu_params[index]->beta[2] = current_flu_params[index]->G_CLO_BETA3 / POPSIZE_MAIN;

        current_flu_params[index]->sigma2d[0][1] = current_flu_params[index]->G_CLO_SIGMA12; // 0.7; // the level of susceptibility to H1 if you've had B
        current_flu_params[index]->sigma2d[1][0] = current_flu_params[index]->G_CLO_SIGMA12; // 0.7; // and vice versa

        current_flu_params[index]->sigma2d[1][2] = current_flu_params[index]->G_CLO_SIGMA23; // 0.7; // the level of susceptibility to H3 if you've had B
        current_flu_params[index]->sigma2d[2][1] = current_flu_params[index]->G_CLO_SIGMA23; // 0.7; // and vice versa

        current_flu_params[index]->sigma2d[0][2] = current_flu_params[index]->G_CLO_SIGMA13; // 0.3; // the level of susceptibility to H3 if you've had H1
        current_flu_params[index]->sigma2d[2][0] = current_flu_params[index]->G_CLO_SIGMA13; // 0.3; // and vice versa

        current_flu_params[index]->sigma2d[0][0] = 0;
        current_flu_params[index]->sigma2d[1][1] = 0;
        current_flu_params[index]->sigma2d[2][2] = 0;

        current_flu_params[index]->v_d_i_nu    = 1 / current_flu_params[index]->G_CLO_NU_DENOM;                // recovery rate
        current_flu_params[index]->v_d_i_immune_duration = current_flu_params[index]->G_CLO_RHO_DENOM;    // 2.5 years of immunity to recent infection'

        current_flu_params[index]->v_d_i_epidur = current_flu_params[index]->G_CLO_EPIDUR;

        //
        // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
        //

        current_flu_params[index]->trr = ((double)NUMR) / current_flu_params[index]->v_d_i_immune_duration;
//        printf("ODE %d updated phi[%d] = %.5f\n",index,9,current_flu_params[index]->phi[9]);
        curand_state_d[index] = local_state;
    }
}
