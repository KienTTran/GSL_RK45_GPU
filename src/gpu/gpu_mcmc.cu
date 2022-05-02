//
// Created by kient on 5/2/2022.
//
#include "gpu_ode_mcmc.h"

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
void mcmc_dnorm(double *y_data_input_d[], double *y_ode_agg_d[], double* y_mcmc_dnorm_d[], GPUParameters *params) {
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < NUMODE * DATADIM_ROWS; index += stride) {
        const int ode_index = index / DATADIM_ROWS;
        //Calculate agg mean
        for (int i = 0; i < DATADIM_COLS; i++) {
            const int y_agg_index = index*params->agg_dimension + 3;//Forth column only
            const int y_data_and_dnorm_index = index*params->data_params.cols + i;
            y_mcmc_dnorm_d[ode_index][y_data_and_dnorm_index] = dnorm(y_data_input_d[ode_index][y_data_and_dnorm_index],
                                                                      y_ode_agg_d[ode_index][y_agg_index] < 0 ? -9999.0 : y_ode_agg_d[ode_index][y_agg_index]/y_ode_agg_d[ode_index][i],
                                                                      0.25,
                                                                      true);
        }
        if(index == 73){
            printf("Line %d data\t",index);
            for (int i = 0; i < DATADIM_COLS; i++) {
                const int data_index = index*params->data_params.cols + i;
                printf("%.5f\t",y_data_input_d[ode_index][data_index]);
                if(i == DATADIM_COLS - 1){
                    printf("\n");
                }
            }
            __syncthreads();
            printf("Line %d agg\t",index);
            for (int i = 0; i < DATADIM_COLS; i++) {
                const int agg_index = index*params->agg_dimension + 3;//Forth column only
                printf("%.5f\t",y_ode_agg_d[ode_index][agg_index]);
                if(i == DATADIM_COLS - 1){
                    printf("\n");
                }
            }
            __syncthreads();
            printf("Line %d max\t",index);
            for (int i = 0; i < DATADIM_COLS; i++) {
                printf("%.5f\t",y_ode_agg_d[ode_index][i]);
                if(i == DATADIM_COLS - 1){
                    printf("\n");
                }
            }
            __syncthreads();
            printf("Line %d mean\t",index);
            for (int i = 0; i < DATADIM_COLS; i++) {
                const int y_agg_index = index*params->agg_dimension + 3;//Forth column only
                printf("%.5f\t",y_ode_agg_d[ode_index][y_agg_index] < 0 ? -9999.0 : y_ode_agg_d[ode_index][y_agg_index]/y_ode_agg_d[ode_index][i]);
                if(i == DATADIM_COLS - 1){
                    printf("\n");
                }
            }
            __syncthreads();
            printf("Line %d dnorm\t",index);
            for (int i = 0; i < DATADIM_COLS; i++) {
                const int dnorm_index = index*params->data_params.cols + i;
                printf("%.5f\t",y_mcmc_dnorm_d[ode_index][dnorm_index]);
                if(i == DATADIM_COLS - 1){
                    printf("\n");
                }
            }
            __syncthreads();
        }
    }
    return;
}

__global__ /* Each thread will calculate 1 line in y_ode_agg_d */
void mcmc_get_dnorm_outputs_2d(double* d_mcmc_dnorm_d[], double* d_mcmc_dnorm_h1_d[], double* d_mcmc_dnorm_b_d[], double* d_mcmc_dnorm_h3_d[], GPUParameters* params){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < NUMODE * DATADIM_ROWS; index += stride) {
        const int ode_index = index / DATADIM_ROWS;
        const int dnorm_h1_index = index*params->data_params.cols + 0;
        const int dnorm_b_index = index*params->data_params.cols + 1;
        const int dnorm_h3_index = index*params->data_params.cols + 2;
        d_mcmc_dnorm_h1_d[ode_index][index] = d_mcmc_dnorm_d[ode_index][dnorm_h1_index];
        d_mcmc_dnorm_b_d[ode_index][index] = d_mcmc_dnorm_d[ode_index][dnorm_b_index];
        d_mcmc_dnorm_h3_d[ode_index][index] = d_mcmc_dnorm_d[ode_index][dnorm_h3_index];
    }
    return;
}

__global__ /* Each thread will calculate 1 line in y_ode_agg_d */
void mcmc_get_dnorm_outputs_1d(double* d_mcmc_dnorm_h1_b_h3_d[], double d_mcmc_dnorm_h1_1d_d[], double d_mcmc_dnorm_b_1d_d[], double d_mcmc_dnorm_h3_1d_d[], GPUParameters* params){
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = index_gpu; index < NUMODE * DATADIM_ROWS; index += stride) {
        const int ode_index = index / DATADIM_ROWS;
        const int dnorm_h1_index = index*params->data_params.cols + 0;
        const int dnorm_b_index = index*params->data_params.cols + 1;
        const int dnorm_h3_index = index*params->data_params.cols + 2;
        d_mcmc_dnorm_h1_1d_d[index] = d_mcmc_dnorm_h1_b_h3_d[ode_index][dnorm_h1_index];
        d_mcmc_dnorm_b_1d_d[index] = d_mcmc_dnorm_h1_b_h3_d[ode_index][dnorm_b_index];
        d_mcmc_dnorm_h3_1d_d[index] = d_mcmc_dnorm_h1_b_h3_d[ode_index][dnorm_h3_index];
//        printf("index = %d dnorm_h1 = %.5f\n",index,d_mcmc_dnorm_h1_1d_d[index]);
//        printf("index = %d dnorm_b = %.5f\n",index,d_mcmc_dnorm_b_1d_d[index]);
//        printf("index = %d dnorm_h3 = %.5f\n",index,d_mcmc_dnorm_h3_1d_d[index]);
    }
    return;
}