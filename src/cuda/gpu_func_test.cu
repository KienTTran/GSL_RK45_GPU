#include <cuda_runtime.h>
#include "gpu_rk45.h"

__device__
void gpu_func_test(float t, const float y[], float f[], int index, int day, GPU_Parameters* gpu_params){

//    if(index == 0){
//        printf("Here's the info on params: \n");
//        printf("beta1 = %1.9f \n", gpu_params->beta[0]);
//        printf("beta2 = %1.9f \n", gpu_params->beta[1]);
//        printf("beta3 = %1.9f \n", gpu_params->beta[2]);
//        printf("a = %1.3f \n", gpu_params->v_d[gpu_params->i_amp]);
//        printf("sigma_H1B = %1.3f \n", gpu_params->sigma[0][1]);
//        printf("sigma_BH3 = %1.3f \n", gpu_params->sigma[1][2]);
//        printf("sigma_H1H3 = %1.3f \n", gpu_params->sigma[0][2]);
//
//        printf("phis_length = %d\n",gpu_params->phis_d_length);
//        for(int i=0; i<gpu_params->phis_d_length; i++){
//            printf("phi = %5.1f \n", gpu_params->phis_d[i]);
//        }
//    }

    // the transition rate among R-classes
    float trr = fdividef(NUMR, gpu_params->v_d[gpu_params->i_immune_duration]);
//    float stf = gpu_params->phis_d_length == 0 ? 1.0 : gpu_params->stf_d[day];
//    float stf = seasonal_transmission_factor(gpu_params,day);
    float stf = seasonal_transmission_factor(gpu_params,t);
//    float stf = gpu_params->stf;
//    float stf = 1.0;

//    if(index < STARTS)
//    {
//        printf("[function] IN y[%d] = %.20f f[%d] = %.20f\n",index,y[index],index,f[index]);
//        if(index == 0){
//            printf("\n");
//        }
//    }

//    printf("y[%d] = y %f\n",index,y[index]);

    const unsigned int START_I  = int(STARTI);
    const unsigned int START_J  = int(STARTJ);
    const unsigned int START_S  = int(STARTS);
    const unsigned int NUM_LOC  = int(NUMLOC);
    const unsigned int NUM_SEROTYPES  = int(NUMSEROTYPES);
    const unsigned int NUM_R  = int(NUMR);

    f[index] = 0.0;
    if(index < START_I){
//        int zDirection = i % zLength;
//        int yDirection = (i / zLength) % yLength;
//        int xDirection = i / (yLength * zLength);
        f[index] = index;
        int loc = index / (NUM_SEROTYPES * NUM_R);
        int vir = (index / NUM_R) % NUM_SEROTYPES;
        int stg = index % NUM_R;
        f[ index ] = - trr * y[ index ];
        if(index % NUM_R == 0){
            f[ index ] += gpu_params->v_d[gpu_params->i_nu] * y[ START_I + NUM_SEROTYPES*loc + vir ];
        }
        else{
            f[ index ] += trr * y[ NUM_SEROTYPES*NUM_R*loc + NUM_R*vir + stg - 1 ];
        }
        float sum_foi = 0.0;
        const int start_index = index * NUM_LOC*NUM_SEROTYPES;
        const int end_index = start_index + (NUM_LOC*NUM_SEROTYPES);

        for(int k = start_index; k < end_index; k++){
            sum_foi +=   gpu_params->sum_foi_sbe[k]
                        * stf
                        * y[gpu_params->sum_foi_y_index[k]];
        }

        f[index] +=  -(sum_foi) * y[index];
    }
    if(index >= START_I && index < START_J){
        int loc = (index - START_I) / NUM_SEROTYPES;
        int vir = (index - START_I) % NUM_SEROTYPES;
        f[ START_I + NUM_SEROTYPES*loc + vir ] = 0.0;
        f[ START_J + NUM_SEROTYPES*loc + vir ] = 0.0;
        float foi_on_susc_single_virus = 0.0;

        for(int l = 0; l<NUM_LOC; l++){
            foi_on_susc_single_virus += gpu_params->eta[loc][l]
                                        * stf
                                        * gpu_params->beta[vir]
                                        * y[START_I + NUM_SEROTYPES * l + vir];
        }

        f[ START_I + NUM_SEROTYPES*loc + vir ] += y[ START_S + loc ] * foi_on_susc_single_virus;
        f[ START_J + NUM_SEROTYPES*loc + vir ] += y[ START_S + loc ] * foi_on_susc_single_virus;

        const int start_index = (index % (NUM_LOC*NUM_SEROTYPES*NUM_R)) * (NUM_LOC*NUM_SEROTYPES*NUM_R);
        const int end_index = start_index + (NUM_LOC*NUM_SEROTYPES*NUM_R);

        float inflow_from_recovereds = 0.0;
        for(int k = start_index; k < end_index; k++){
            inflow_from_recovereds +=   gpu_params->inflow_from_recovereds_sbe[k]
                                        * stf
                                        * y[gpu_params->inflow_from_recovereds_y1_index[k]]
                                        * y[gpu_params->inflow_from_recovereds_y2_index[k]];
        }
        f[ START_I + NUM_SEROTYPES*loc + vir ] += inflow_from_recovereds;
        f[ START_J + NUM_SEROTYPES*loc + vir ] += inflow_from_recovereds;

        // add the recovery rate - NOTE only for I-classes
        f[ START_I + NUM_SEROTYPES*loc + vir ] += - gpu_params->v_d[gpu_params->i_nu] * y[ START_I + NUM_SEROTYPES*loc + vir ];
    }
    if(index >= START_S && index < gpu_params->dimension)
    {
        unsigned int loc = index - START_S;
        float foi_on_susc_all_viruses = 0.0;

        const int start_index = loc * NUM_LOC*NUM_SEROTYPES;
        const int end_index = start_index + (NUM_LOC*NUM_SEROTYPES);

        for(int k = start_index; k < end_index; k++){
            foi_on_susc_all_viruses +=   gpu_params->foi_on_susc_all_viruses_eb[k]
                                         * stf
                                         * y[gpu_params->foi_on_susc_all_viruses_y_index[k]];
        }

        f[ index ] = ( - foi_on_susc_all_viruses ) * y[ index ];
        for(int vir = 0; vir<NUM_SEROTYPES; vir++)
        {
            // add to dS/dt the inflow of recovereds from the final R-stage
            f[ index ] += trr * y[ NUM_SEROTYPES*NUM_R*(loc) + NUM_R*vir + (NUM_R - 1) ]; // "NUM_R-1" gets you the final R-stage only
        }
    }

//    if(index < START_S)
//    {
//        printf("[function] OUT y[%d] = %.20f f[%d] = %.20f\n",index,y[index],index,f[index]);
//        if(index == 0){
//            printf("\n");
//        }
//    }

    return;
}
