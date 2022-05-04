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
void mcmc_dnorm(double *y_data_input_d[], double *y_ode_agg_d[], double y_mcmc_dnorm_d[], GPUParameters *params) {
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    double d_norm[3];

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
            d_norm[i] = dnorm(y_data_input_d[ode_index][y_data_index],
                             y_ode_agg_d[ode_index][y_agg_index] < 0 ? -9999.0 : y_ode_agg_d[ode_index][y_agg_index]/y_ode_agg_d[ode_index][i],
                             0.25,
                             true);
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
//            printf("Line %d dnorm\t",line_index);
//            for (int i = 0; i < params->data_params.cols; i++) {
//                printf("%.5f\t",d_norm[i]);
//                if(i == params->data_params.cols - 1){
//                    printf("\n");
//                }
//            }
//            __syncthreads();
//            printf("Line %d dnorm sum %.5f\n",line_index,y_mcmc_dnorm_d[line_index]);
//            __syncthreads();
//        }
//        if(line_index < 3)
//        {
//            printf("index %d ODE %d line %d dnorm sum %.5f\n",index,ode_index,line_index,y_mcmc_dnorm_d[index]);
//        }

    }
    return;
}

