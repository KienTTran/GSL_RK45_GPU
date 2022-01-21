#include <cuda_runtime.h>
#include "gpu_rk45.h"

__device__ int get_1d_index_from_5(const int loc,const  int vir,const  int stg,const  int l,const  int v){
    return loc*NUMSEROTYPES*NUMR*NUMLOC*NUMSEROTYPES + vir*NUMR*NUMLOC*NUMSEROTYPES + stg*NUMLOC*NUMSEROTYPES + l*NUMSEROTYPES + v;
}

__device__ int get_1d_index_start_from_3(const int loc,const  int vir,const  int stg){
    return loc*NUMSEROTYPES*NUMR*NUMLOC*NUMSEROTYPES + vir*NUMR*NUMLOC*NUMSEROTYPES + stg*NUMLOC*NUMSEROTYPES;
}

__device__ int get_1d_index_end_from_3(const int loc,const  int vir,const  int stg){
    return loc*NUMSEROTYPES*NUMR*NUMLOC*NUMSEROTYPES + vir*NUMR*NUMLOC*NUMSEROTYPES + stg*NUMLOC*NUMSEROTYPES + (NUMLOC*NUMSEROTYPES);
}

__device__ double get_sum_foi_sbe_from_1(const int index_1d, const int offset, GPU_Parameters* gpu_params){
//    printf("      sum_foi_sbe[%d] = %f\n",index_1d,gpu_params->sum_foi_sbe[index_1d]);
    return gpu_params->sum_foi_sbe[index_1d];
}

__device__ double get_sum_foi_sbe_from_5(const int loc,const  int vir,const  int stg,const  int l,const  int v, GPU_Parameters* gpu_params){
    return gpu_params->sum_foi_sbe[get_1d_index_from_5(loc, vir, stg, l, v)];
}

__device__ double get_pass1_y_I(const int index, const double y[]){
    return y[STARTI + index];
}

__device__ double get_sum_foi_sbe_from_3(const int loc, const int vir, const int stg,  const double y[], GPU_Parameters* gpu_params){
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    double sum_foi_sbe = 0.0;
//    printf("  loc = %d vir = %d stg = %d sum from %d to %d\n",loc,vir,stg,get_1d_index_start_from_3(loc,vir,stg),get_1d_index_end_from_3(loc,vir,stg));
    for(int i = get_1d_index_start_from_3(loc,vir,stg); i < get_1d_index_end_from_3(loc,vir,stg); i++){
        sum_foi_sbe += get_sum_foi_sbe_from_1(i,i - get_1d_index_start_from_3(loc,vir,stg),gpu_params) * get_pass1_y_I(i - get_1d_index_start_from_3(loc,vir,stg),y);
//        printf("    loc = %d vir = %d stg = %d sum_foi_sbe index = %d y I index is %d y = %f\n",loc,vir,stg,i,STARTI + (i - get_1d_index_start_from_3(loc,vir,stg)),get_pass1_y_I(i - get_1d_index_start_from_3(loc,vir,stg),y));
    }
//    printf("  loc = %d vir = %d stg = %d sum_foi = %f\n",loc,vir,stg, sum_foi);
    block.sync();
    return sum_foi_sbe;
}

__device__
void gpu_func_test(double t, const double y[], double f[], int index, void *params){

    //    printf("gpu_function start\n");
    // just to be safe, cast the void-pointer to convert it to a prms-pointer
    GPU_Parameters* gpu_params = (GPU_Parameters*) params;

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
    double trr = ((double)NUMR) / gpu_params->v_d[gpu_params->i_immune_duration];

    double stf = gpu_params->phis_d_length == 0 ? 1.0 : gpu_params->stf_d[static_cast<int>(t)];
//    double stf = seasonal_transmission_factor(gpu_params,t);

//    R -  i = 0 f[0] = 0-R1a
//    R -  i = 1 f[1] = 0-R1b
//    R -  i = 2 f[2] = 0-R1c
//    R -  i = 3 f[3] = 0-R1d
//    R -  i = 4 f[4] = 0-R2a
//    R -  i = 5 f[5] = 0-R2b
//    R -  i = 6 f[6] = 0-R2c
//    R -  i = 7 f[7] = 0-R2d
//    R -  i = 8 f[8] = 0-R3a
//    R -  i = 9 f[9] = 0-R3b
//    R -  i = 10 f[10] = 0-R3c
//    R -  i = 11 f[11] = 0-R3d
//    I -  i = 12 f[12] = 0-I1
//    I -  i = 13 f[13] = 0-I2
//    I -  i = 14 f[14] = 0-I3
//    J -  i = 15 f[15] = 0-J1
//    J -  i = 16 f[16] = 0-J2
//    J -  i = 17 f[17] = 0-J3
//    S -  i = 18 f[18] = 0-S

    __shared__ bool step_I_done[DIM];
    __shared__ bool step_IJ_done[DIM];

//    printf("y[%d] = y %f\n",index,y[index]);
    if(index < STARTI){
//        int zDirection = i % zLength;
//        int yDirection = (i / zLength) % yLength;
//        int xDirection = i / (yLength * zLength);
        int loc = index / (NUMSEROTYPES * NUMR);
        int vir = (index / NUMR) % NUMSEROTYPES;
        int stg = index % NUMR;
//        printf("index = %d index = %d Loc %d R vir %d stg %d\n",index,index,loc,vir,stg);
//        if(index == 11)
        {
            f[ index ] = - trr * y[ index ];
            if(index % NUMR == 0){
    //            printf("  Index %d stg == 0\n",index);
                f[ index ] += gpu_params->v_d[gpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];
            }
            else{
                f[ index ] += trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg - 1 ];
            }
            double sum_foi_3 = get_sum_foi_sbe_from_3(loc,vir,stg,y,gpu_params);
            f[ index ] += ( - sum_foi_3)
                          * stf
                          * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ];
    //        printf("loc = %d vir = %d stg = %d flat y[%d] = %f\n",loc,vir,stg,index,y[ index ]);
    //        printf("loc = %d vir = %d stg = %d flat sum_foi = %f\n",loc,vir,stg,sum_foi_3);
    //        printf("loc = %d vir = %d stg = %d flat f[%d] = %f\n",loc,vir,stg,index,f[ index ]);
    //        printf("\n");
    //        if(index == STARTI - 1){
    //            printf("\n");
    //        }
        }
    }
    else if(index < STARTS){
        int loc = ((index - STARTJ) / (NUMSEROTYPES)) % NUMLOC;
        int vir = (index - NUMSEROTYPES*NUMR*NUMLOC) % NUMSEROTYPES;
//        printf("index = %d Loc %d I vir %d\n",index,loc,vir);
//        if(index == 29)
        {
            if(index < STARTJ){
                f[STARTI + NUMSEROTYPES * loc + vir] = 0.0;
//            f[STARTJ + NUMSEROTYPES * loc + vir] = 0.0;
                double foi_on_susc_single_virus = 0.0;
//                cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
                for(int l = 0; l < NUMLOC; l++){
                    double foi_on_susc_single_virus_eb = gpu_params->eta[loc][l]
                                                         * stf
                                                         * gpu_params->beta[vir];
                    foi_on_susc_single_virus += foi_on_susc_single_virus_eb * y[STARTI + NUMSEROTYPES * l + vir];
//                    printf("  loc = %d vir = %d l = %d y I index is %d y = %f\n",loc,vir,l,STARTI + NUMSEROTYPES * l + vir,y[STARTI + NUMSEROTYPES * l + vir]);
//                    printf("  loc = %d vir = %d l = %d flat foi_on_susc_single_virus_eb = %f\n",loc,vir,l,foi_on_susc_single_virus_eb);
//                    printf("  loc = %d vir = %d l = %d flat foi_on_susc_single_virus = %f\n",loc,vir,l,foi_on_susc_single_virus);
                }
                f[ STARTI + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus;
//            f[ STARTJ + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus;
//                printf("loc = %d vir = %d y I index is %d y = %f\n",loc,vir,STARTS + loc,y[STARTS + loc]);
//                printf("loc = %d vir = %d flat foi_on_susc_single_virus = %f\n",loc,vir,foi_on_susc_single_virus);
//                printf("loc = %d vir = %d flat f[%d] = %f\n", loc, vir, STARTI + NUMSEROTYPES*loc + vir,f[STARTI + NUMSEROTYPES*loc + vir]);
//            printf("loc = %d vir = %d flat f[%d] = %f\n", loc, vir, STARTJ + NUMSEROTYPES*loc + vir,f[STARTJ + NUMSEROTYPES*loc + vir]);
//                printf("\n");

                double inflow_from_recovereds = 0.0;
                for(int i = 0; i < NUMLOC*NUMSEROTYPES*NUMR; i++){
                    int multiplier = index - (NUMLOC*NUMSEROTYPES*NUMR);
                    int inflow_index = (multiplier)*NUMLOC*NUMSEROTYPES*NUMR + i;
                    inflow_from_recovereds +=   gpu_params->inflow_from_recovereds_sbe[inflow_index]
                                                * stf
                                                * y[STARTI + NUMSEROTYPES * loc + vir] * y[i];
//                    printf("loc = %d vir = %d i = %d y index = %d y = %f\n",loc,vir,i,STARTI + NUMSEROTYPES * loc + vir,y[STARTI + NUMSEROTYPES * loc + vir]);
//                    printf("loc = %d vir = %d i = %d y index = %d y = %f\n",loc,vir,i,i,y[i]);
//                    printf("loc = %d vir = %d i = %d multiplier = %d inflow_from_recovereds_sbe index = %d inflow_from_recovereds_sbe = %f\n",loc,vir,i,multiplier,inflow_index,gpu_params->inflow_from_recovereds_sbe[inflow_index]);
//                    printf("loc = %d vir = %d i = %d inflow_from_recovereds = %f\n",loc,vir,i,inflow_from_recovereds);
//                    printf("\n");
                }

//                printf("loc = %d vir = %d flat f[%d] = %f\n", loc, vir, STARTI + NUMSEROTYPES * loc + vir,f[STARTI + NUMSEROTYPES * loc + vir]);
//            printf("loc = %d vir = %d flat f[%d] = %f\n", loc, vir, STARTJ + NUMSEROTYPES * loc + vir,f[STARTJ + NUMSEROTYPES * loc + vir]);

                f[ STARTI + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds;
//            f[ STARTJ + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds;
//                printf("loc = %d vir = %d flat inflow_from_recovereds = %f\n", loc, vir, inflow_from_recovereds);
//                printf("loc = %d vir = %d flat f[%d] = %f\n", loc, vir, STARTI + NUMSEROTYPES * loc + vir,f[STARTI + NUMSEROTYPES * loc + vir]);
//            printf("loc = %d vir = %d flat f[%d] = %f\n", loc, vir, STARTJ + NUMSEROTYPES * loc + vir,f[STARTJ + NUMSEROTYPES * loc + vir]);

//                step_I_done[index] = true;
//                block.sync();
            }
            else {//MOve J to here after I completed
//                cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
                //Wait for I completed
//                while(!step_I_done[index - NUMLOC*NUMSEROTYPES]){
//                    __syncthreads();
//                }
                f[index] = index;
                f[ index ] = f[ index - (NUMLOC * NUMSEROTYPES)];
//            // add the recovery rate - NOTE only for I-classes
                f[ index - (NUMLOC * NUMSEROTYPES) ] += - gpu_params->v_d[gpu_params->i_nu] * y[ index - (NUMLOC * NUMSEROTYPES) ];

//                printf("loc = %d vir = %d flat f[%d] I only = %f\n", loc, vir, index - (NUMLOC * NUMSEROTYPES),f[index - (NUMLOC * NUMSEROTYPES)]);
//                printf("f[%d] = f[%d]\n",index,index - (NUMLOC*NUMSEROTYPES));
//                step_IJ_done[index] = true;
//                block.sync();
            }
        }
    }
    else{
//        cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        //Wait for IJ completed
//        while(!step_IJ_done[index - NUMLOC*NUMSEROTYPES]){
//            __syncthreads();
//        }
//    if(index == 36)
    {
        double foi_on_susc_all_viruses = 0.0;
        for(int i = 0; i < NUMLOC*NUMLOC*NUMSEROTYPES; i++){
            if(i < NUMLOC*NUMSEROTYPES){
                foi_on_susc_all_viruses +=  gpu_params->foi_on_susc_all_viruses_eb[i]
                                            * stf
                                            * y[STARTI + (i % (NUMLOC*NUMSEROTYPES))];
//                printf("< %d i = %d foi_on_susc_all_viruses_eb[%d] = %f\n",NUMLOC*NUMSEROTYPES,i,i,gpu_params->foi_on_susc_all_viruses_eb[i]);
//                printf("< %d i = %d y[%d] = %f\n",NUMLOC*NUMSEROTYPES,i,STARTI + (i % (NUMLOC*NUMSEROTYPES)),y[STARTI + (i % (NUMLOC*NUMSEROTYPES))]);
//                printf("< %d i = %d f[%d] = %f\n",NUMLOC*NUMSEROTYPES,i,index + (i / (NUMLOC*NUMSEROTYPES)),f[index + (i / (NUMLOC*NUMSEROTYPES))]);
                f[index + (i / (NUMLOC*NUMSEROTYPES))] = (-foi_on_susc_all_viruses) * y[index + (i / (NUMLOC*NUMSEROTYPES))];
            }
            else{
                if(i / (NUMLOC*NUMSEROTYPES) == 0){
                    foi_on_susc_all_viruses = 0.0;
                }
                foi_on_susc_all_viruses +=  gpu_params->foi_on_susc_all_viruses_eb[i]
                                            * stf
                                            * y[STARTI + (i % (NUMLOC*NUMSEROTYPES))];
//                printf(">= %d i = %d foi_on_susc_all_viruses_eb[%d] = %f\n",NUMLOC*NUMSEROTYPES,i,i,gpu_params->foi_on_susc_all_viruses_eb[i]);
//                printf(">= %d i = %d y[%d] = %f\n",NUMLOC*NUMSEROTYPES,i,STARTI + (i % (NUMLOC*NUMSEROTYPES)),y[STARTI + (i % (NUMLOC*NUMSEROTYPES))]);
//                printf(">= %d i = %d f[%d] = %f\n",NUMLOC*NUMSEROTYPES,i,index + (i / (NUMLOC*NUMSEROTYPES)),f[index + (i / (NUMLOC*NUMSEROTYPES))]);
                f[index + (i / (NUMLOC*NUMSEROTYPES))] = (-foi_on_susc_all_viruses) * y[index + (i / (NUMLOC*NUMSEROTYPES))];
            }
        }
        for(int i = 0; i < NUMLOC*NUMSEROTYPES; i++){
            f[index + (i / NUMSEROTYPES)] += trr * y[ ((i + 1) * NUMR) - 1];
        }
    }
//        f[index] =  - (stf * gpu_params->beta[0] * y[12] * y[index])
//                    - (stf * gpu_params->beta[1] * y[13] * y[index])
//                    - (stf * gpu_params->beta[2] * y[14] * y[index])
//                    + (trr * y[3]) + (trr * y[7]) + (trr * y[11]);
//        block.sync();
    }
//    if(index == 0 || index == DIM - 1){
//        printf("        [function] y[%d] = %.20f f[%d] = %.20f\n",index,y[index],index,f[index]);
//    }

    return;
}
