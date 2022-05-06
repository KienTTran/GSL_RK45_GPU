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

__global__
void mcmc_update_parameters(FluParameters* current_flu_params[], FluParameters* new_flu_params[], int ode_number, curandState curand_state_d[]){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < ode_number; index += stride){
        curand_init(clock64(), index, 0, &curand_state_d[index]);
        curandState local_state = curand_state_d[index];
        for(int sample_index = 0; sample_index < new_flu_params[index]->sample_length; sample_index++){
            new_flu_params[index]->sample[sample_index] =   (curand_normal(&local_state) * current_flu_params[index]->sample[sample_index])
                                                                + current_flu_params[index]->sample_sd[sample_index];
//            printf("ODE %d random = %.5f sample_old[%d] = %.5f sample_sd[%d] = %.5f sample_new[%d] = %.5f\n",index,curand_normal(&local_state),
//                                                                                                        sample_index,current_flu_params[index]->sample[sample_index],
//                                                                                                        sample_index,current_flu_params[index]->sample_sd[sample_index],
//                                                                                                        sample_index,new_flu_params[index]->sample[sample_index]);
        }
        new_flu_params[index]->beta[0] = new_flu_params[index]->sample[0];
        new_flu_params[index]->beta[1] = new_flu_params[index]->sample[1];
        new_flu_params[index]->beta[2] = new_flu_params[index]->sample[2];
        new_flu_params[index]->phi[0] = new_flu_params[index]->sample[3];
        new_flu_params[index]->tau[0] = new_flu_params[index]->sample[4];
        new_flu_params[index]->tau[1] = new_flu_params[index]->sample[5];
        new_flu_params[index]->tau[2] = new_flu_params[index]->sample[6];
        new_flu_params[index]->tau[3] = new_flu_params[index]->sample[7];
        new_flu_params[index]->tau[4] = new_flu_params[index]->sample[8];
        new_flu_params[index]->tau[5] = new_flu_params[index]->sample[9];
        new_flu_params[index]->tau[6] = new_flu_params[index]->sample[10];
        new_flu_params[index]->tau[7] = new_flu_params[index]->sample[11];
        new_flu_params[index]->tau[8] = new_flu_params[index]->sample[12];
        new_flu_params[index]->G_CLO_BETA1 = new_flu_params[index]->beta[0];
        new_flu_params[index]->G_CLO_BETA2 = new_flu_params[index]->beta[1];
        new_flu_params[index]->G_CLO_BETA3 = new_flu_params[index]->beta[2];
        new_flu_params[index]->G_CLO_SIGMA12 = new_flu_params[index]->sigma[0];
        new_flu_params[index]->G_CLO_SIGMA13 = new_flu_params[index]->sigma[1];
        new_flu_params[index]->G_CLO_SIGMA23 = new_flu_params[index]->sigma[2];
        new_flu_params[index]->G_CLO_AMPL = new_flu_params[index]->amp;
        new_flu_params[index]->G_CLO_NU_DENOM = new_flu_params[index]->nu_denom;
        new_flu_params[index]->G_CLO_RHO_DENOM = new_flu_params[index]->rho_denom;
        new_flu_params[index]->epidur = new_flu_params[index]->G_CLO_EPIDUR;
        new_flu_params[index]->phi_length = sizeof(new_flu_params[index]->phi)/sizeof(new_flu_params[index]->phi[0]);
        new_flu_params[index]->phi[0] = new_flu_params[index]->phi_0;
        for(int i = 1; i < new_flu_params[index]->phi_length; i++){
            new_flu_params[index]->phi[i] = new_flu_params[index]->phi[i-1] + new_flu_params[index]->tau[i-1];
        }
//        for(int i = 0; i < new_flu_params[index]->phi_length; i++){
//            if(i == new_flu_params[index]->phi_length - 1){
//                printf("ODE %d updated phi[%d] = %.5f\n",index,i,new_flu_params[index]->phi[i]);
//            }
//        }
//        printf("ODE %d updated phi[%d] = %.5f\n",index,9,new_flu_params[index]->phi[9]);
        curand_state_d[index] = local_state;
    }
}
