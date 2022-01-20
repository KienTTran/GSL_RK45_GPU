#include <cuda_runtime.h>
#include "gpu_rk45.h"

//__device__
//void gpu_func_test(double t, const double y[], double f[], void *params){
//    GPU_Parameters* params_d = (GPU_Parameters*) params;
//    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//    for(int index = index_gpu; index < params_d->dimension; index += stride){
//        const double m = 5.2;		// Mass of pendulum
//        const double g = -9.81;		// g
//        const double l = 2;		// Length of pendulum
//        const double A = 0.5;		// Amplitude of driving force
//        const double wd = 1;		// Angular frequency of driving force
//        const double b = 0.5;		// Damping coefficient
//
//        f[index] = 0.0;
//        cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
//        if(index == 0){
//            f[index] = y[index + 1];
//        }
//        cta.sync();
//
//        if(index == DIM - 1){
//            f[index] = -(g / l) * sin(y[index - 1]) + (A * cos(wd * t) - b * y[index]) / (m * l * l);
//        }
//        if(index == 0 || index == params_d->dimension - 1) {
//            printf("      [function] y[%d] = %.10f f[%d] = %.10f\n", index,y[index],index,f[index]);
//        }
//    }
//    return;
//}

//__device__
//void gpu_func_test(double t, const double y[], double f[], void *params){
//    //    printf("gpu_function start\n");
//    // just to be safe, cast the void-pointer to convert it to a prms-pointer
//
//    GPU_Parameters* gpu_params = (GPU_Parameters*) params;
//
//    // everything will be indexed by location (loc), the infecting subtype/serotype (vir), and the stage of recovery (stg) in the R-classes
//    int loc, vir, stg;
//
//    // force of infection
//    // double foi = gpu_params->v_d[i_beta] * y[NUMR+2];
//
//    // the transition rate among R-classes
//    double trr = ((double)NUMR) / gpu_params->v_d[gpu_params->i_immune_duration];
//
//    double stf = 0.0;
//
//    //for(int k=0; k<DIM; k++) f[k] = 0.0;
//
//    //
//    // ###  1.  COMPUTE THE FORCES OF INFECTION (NOTE maybe this is not necessary)
//    //
//
//    // force of infection on location loc, on immune status i, by virus vir
//    /*double foi_partial[NUMLOC][NUMSEROTYPES][NUMSEROTYPES];
//    for(loc=0; loc<NUMLOC; loc++)
//    {
//        for(vir=0; vir<NUMSEROTYPES; vir++)
//        {
//            for(stg=0; stg<NUMR; stg++)
//            {
//
//            }
//        }
//    }*/
//
//
//    //
//    // ###  2.  WRITE DOWN THE DERIVATIVES FOR ALL THE RECOVERED CLASSES
//    //
//    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//    for(loc=0; loc<NUMLOC; loc++)
//    {
//        for(vir=0; vir<NUMSEROTYPES; vir++)
//        {
//            for(stg=0; stg<NUMR; stg++)
//            {
//                if(index == NUMSEROTYPES*NUMR*loc + NUMR*vir + stg){
//                    // first add the rate at which individuals are transitioning out of the R class
//                    f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = - trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ];
//
//                    // now add the rates of individuals coming in
//                    if( stg == 0 )
//                    {
//                        // if this is the first R-class, add the recovery term for individuals coming from I
//                        f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += gpu_params->v_d[gpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];
//                    }
//                    else
//                    {
//                        // if this is not the first R-class, add a simple transition from the previous R-stage
//                        f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg - 1 ];
//                    }
//
//                    // now sum over all locations and serotypes to get the force of infection that is removing
//                    // individuals from this R-class
//
//                    sum_foi[index] = 0.0;
//                    for(int l=0; l<NUMLOC; l++) {
//                        for (int v = 0; v < NUMSEROTYPES; v++) {
//                            seasonal_transmission_factor(gpu_params,t,stf);
//                            sum_foi[index] += gpu_params->sigma[vir][v]
//                                       * stf
//                                       * gpu_params->beta[v] * gpu_params->eta[loc][l] * y[STARTI + NUMSEROTYPES * l + v];
//                        }
//                    }
//                    // now add the term to dR/dt that accounts for the force of infection removing some R-individuals
//                    f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += ( -sum_foi ) * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ];
//                }
//            }
//        }
//    }
//    block.sync();
//
//    //
//    // ###  3.  WRITE DOWN THE DERIVATIVES FOR ALL THE INFECTED CLASSES and the J-CLASSES
//    //
//    cooperative_groups::thread_block block2 = cooperative_groups::this_thread_block();
//    for(loc=0; loc<NUMLOC; loc++)
//    {
//        for(vir=0; vir<NUMSEROTYPES; vir++)
//        {
//            // initialize these derivatives to zero
//            f[ STARTI + NUMSEROTYPES*loc + vir ] = 0.0;
//            f[ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;
//
////            // sum over locations to get the force of infection of virus vir on susceptibles in location loc
//            foi_on_susc_single_virus[index] = 0.0;
//            for(int l=0; l<NUMLOC; l++){
//                seasonal_transmission_factor(gpu_params,t,stf);
//                foi_on_susc_single_virus[index] +=
//                        gpu_params->eta[loc][l]
//                        * stf
//                        * gpu_params->beta[vir] * y[STARTI + NUMSEROTYPES * l + vir];
//            }
//            // add the in-flow of new infections from the susceptible class
//            f[ STARTI + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus[index];
//            f[ STARTJ + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus[index];
//
//            // sum over locations and different types of recovered individuals to get the inflow of recovered
//            // individuals that are becoming re-infected
//            inflow_from_recovereds[index] = 0.0;
//            for(int l=0; l<NUMLOC; l++) {          // sum over locations
//                for (int v = 0; v < NUMSEROTYPES; v++) {  // sum over recent immunity
//                    for (int s = 0; s < NUMR; s++) {    // sum over R stage
//                        seasonal_transmission_factor(gpu_params,t,stf);
//                        inflow_from_recovereds[index] +=
//                                gpu_params->sigma[vir][v]
//                                * stf
//                                * gpu_params->beta[vir] * gpu_params->eta[loc][l] *
//                                y[STARTI + NUMSEROTYPES * l + vir] * y[NUMSEROTYPES * NUMR * loc + NUMR * v + s];
//                    }
//                }
//            }
//            // add the in-flow of new infections from the recovered classes (all histories, all stages)
//            f[ STARTI + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds[index];
//            f[ STARTJ + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds[index];
//
//            // add the recovery rate - NOTE only for I-classes
//            f[ STARTI + NUMSEROTYPES*loc + vir ] += - gpu_params->v_d[gpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];
//
//        }
//    }
//    block2.sync();
//    //
//    // ###  4.  WRITE DOWN THE DERIVATIVES FOR ALL THE SUSCEPTIBLE CLASSES
//    //
//    cooperative_groups::thread_block block3 = cooperative_groups::this_thread_block();
//    for(loc=0; loc<NUMLOC; loc++)
//    {
//        // compute the force of infection of all viruses at all locations on the susceptibles at the location loc
//        foi_on_susc_all_viruses[index] = 0.0;
//        for(int l=0; l<NUMLOC; l++) {
//            for (int v = 0; v < NUMSEROTYPES; v++) {
//                seasonal_transmission_factor(gpu_params,t,stf);
//                foi_on_susc_all_viruses[index] +=
//                        gpu_params->eta[loc][l]
//                        * stf
//                        * gpu_params->beta[v] * y[STARTI + NUMSEROTYPES * l + v];
//            }
//        }
//        // add to ODE dS/dt equation the removal of susceptibles by all types of infection
//        f[ STARTS + loc ] = ( - foi_on_susc_all_viruses[index] ) * y[ STARTS + loc ];
//
//        // now loop through all the recovered classes in this location (different histories, final stage only)
//        for(int vir=0; vir<NUMSEROTYPES; vir++)
//        {
//            // add to dS/dt the inflow of recovereds from the final R-stage
//            f[ STARTS + loc ] += trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + (NUMR-1) ]; // "NUMR-1" gets you the final R-stage only
//        }
//    }
//    block3.sync();
//    return;
//}

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

__device__ double get_sum_foi_from_3(const int loc, const int vir, const int stg,  const double y[], GPU_Parameters* gpu_params){
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    double sum_foi = 0.0;
//    printf("  loc = %d vir = %d stg = %d sum from %d to %d\n",loc,vir,stg,get_1d_index_start_from_3(loc,vir,stg),get_1d_index_end_from_3(loc,vir,stg));
    for(int i = get_1d_index_start_from_3(loc,vir,stg); i < get_1d_index_end_from_3(loc,vir,stg); i++){
        sum_foi += get_sum_foi_sbe_from_1(i,i - get_1d_index_start_from_3(loc,vir,stg),gpu_params) * get_pass1_y_I(i - get_1d_index_start_from_3(loc,vir,stg),y);
//        printf("    loc = %d vir = %d stg = %d sum_foi_sbe index = %d y I index is %d y = %f\n",loc,vir,stg,i,STARTI + (i - get_1d_index_start_from_3(loc,vir,stg)),get_pass1_y_I(i - get_1d_index_start_from_3(loc,vir,stg),y));
    }
//    printf("  loc = %d vir = %d stg = %d sum_foi = %f\n",loc,vir,stg, sum_foi);
    block.sync();
    return sum_foi;
}

__device__
void gpu_func_test(double t, const double y[], double f[], int index, void *params){

    //    printf("gpu_function start\n");
    // just to be safe, cast the void-pointer to convert it to a prms-pointer
    GPU_Parameters* gpu_params = (GPU_Parameters*) params;

    // the transition rate among R-classes
    double trr = ((double)NUMR) / gpu_params->v_d[gpu_params->i_immune_duration];

    double stf = 1.0;
    seasonal_transmission_factor(gpu_params,t,stf);

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
    int index_1d_pass_1 = index;
//    printf("[function] y[%d] = %.10f f[%d] = %.10f\n",index_1d_pass_1,y[index_1d_pass_1],index_1d_pass_1,f[index_1d_pass_1]);
    //fflush(stdout);
//    if(index == 0){
//        printf("[function] dimension is %d\n",gpu_params->dimension);
//    }
//    printf("[function] stf[%d] is %f\n", index_1d_pass_1, stf);
//    sum_foi = sum_foi + gpu_params->beta[v] * gpu_params->eta[loc][l] * I1/2/3;
//    f[ index_1d_pass_1 ] = f[ index_1d_pass_1 ] + ( -sum_foi ) * y[ index_1d_pass_1 ];


//    printf("y[%d] = y %f\n",index_1d_pass_1,y[index_1d_pass_1]);
    if(index_1d_pass_1 < STARTI){
//        int zDirection = i % zLength;
//        int yDirection = (i / zLength) % yLength;
//        int xDirection = i / (yLength * zLength);
        int loc = index_1d_pass_1 / (NUMSEROTYPES * NUMR);
        int vir = (index_1d_pass_1 / NUMR) % NUMSEROTYPES;
        int stg = index_1d_pass_1 % NUMR;
//        printf("index = %d index_1d_pass_1 = %d Loc %d R vir %d stg %d\n",index,index_1d_pass_1,loc,vir,stg);
//        if(index_1d_pass_1 == 11)
        {
            f[ index_1d_pass_1 ] = - trr * y[ index_1d_pass_1 ];
            if(index_1d_pass_1 % NUMR == 0){
    //            printf("  Index %d stg == 0\n",index_1d_pass_1);
                f[ index_1d_pass_1 ] += gpu_params->v_d[gpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];
            }
            else{
                f[ index_1d_pass_1 ] += trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg - 1 ];
            }
            double sum_foi_3 = get_sum_foi_from_3(loc,vir,stg,y,gpu_params);
            f[ index_1d_pass_1 ] += ( - sum_foi_3)
                                    * stf
                                    * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ];
    //        printf("loc = %d vir = %d stg = %d flat y[%d] = %f\n",loc,vir,stg,index_1d_pass_1,y[ index_1d_pass_1 ]);
    //        printf("loc = %d vir = %d stg = %d flat sum_foi = %f\n",loc,vir,stg,sum_foi_3);
    //        printf("loc = %d vir = %d stg = %d flat f[%d] = %f\n",loc,vir,stg,index_1d_pass_1,f[ index_1d_pass_1 ]);
    //        printf("\n");
    //        if(index_1d_pass_1 == STARTI - 1){
    //            printf("\n");
    //        }
        }
    }
    else if(index_1d_pass_1 < STARTS){
        int loc = ((index_1d_pass_1 - STARTJ) / (NUMSEROTYPES)) % NUMLOC;
        int vir = (index_1d_pass_1 - NUMSEROTYPES*NUMR*NUMLOC) % NUMSEROTYPES;
//        printf("index_1d_pass_1 = %d Loc %d I vir %d\n",index_1d_pass_1,loc,vir);
//        if(index_1d_pass_1 == 29)
        {
            if(index_1d_pass_1 < STARTJ){
                f[STARTI + NUMSEROTYPES * loc + vir] = 0.0;
//            f[STARTJ + NUMSEROTYPES * loc + vir] = 0.0;
                double foi_on_susc_single_virus = 0.0;
                cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
                for(int l = 0; l < NUMLOC; l++){
                    double foi_on_susc_single_virus_eb = gpu_params->eta[loc][l]
                                                         * stf
                                                         * gpu_params->beta[vir];
                    foi_on_susc_single_virus += foi_on_susc_single_virus_eb * y[STARTI + NUMSEROTYPES * l + vir];
//                    printf("  loc = %d vir = %d l = %d y I index is %d y = %f\n",loc,vir,l,STARTI + NUMSEROTYPES * l + vir,y[STARTI + NUMSEROTYPES * l + vir]);
//                    printf("  loc = %d vir = %d l = %d loop foi_on_susc_single_virus_eb = %f\n",loc,vir,l,foi_on_susc_single_virus_eb);
//                    printf("  loc = %d vir = %d l = %d loop foi_on_susc_single_virus = %f\n",loc,vir,l,foi_on_susc_single_virus);
                }
                f[ STARTI + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus;
//            f[ STARTJ + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus;
//                printf("loc = %d vir = %d y I index is %d y = %f\n",loc,vir,STARTS + loc,y[STARTS + loc]);
//                printf("loc = %d vir = %d loop foi_on_susc_single_virus = %f\n",loc,vir,foi_on_susc_single_virus);
//                printf("loc = %d vir = %d loop f[%d] = %f\n", loc, vir, STARTI + NUMSEROTYPES*loc + vir,f[STARTI + NUMSEROTYPES*loc + vir]);
//            printf("loc = %d vir = %d loop f[%d] = %f\n", loc, vir, STARTJ + NUMSEROTYPES*loc + vir,f[STARTJ + NUMSEROTYPES*loc + vir]);
//                printf("\n");

                double inflow_from_recovereds = 0.0;
                for(int i = 0; i < NUMLOC*NUMSEROTYPES*NUMR; i++){
                    int multiplier = index_1d_pass_1 - (NUMLOC*NUMSEROTYPES*NUMR);
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

//                printf("loc = %d vir = %d loop f[%d] = %f\n", loc, vir, STARTI + NUMSEROTYPES * loc + vir,f[STARTI + NUMSEROTYPES * loc + vir]);
//            printf("loc = %d vir = %d loop f[%d] = %f\n", loc, vir, STARTJ + NUMSEROTYPES * loc + vir,f[STARTJ + NUMSEROTYPES * loc + vir]);

                f[ STARTI + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds;
//            f[ STARTJ + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds;
//                printf("loc = %d vir = %d loop inflow_from_recovereds = %f\n", loc, vir, inflow_from_recovereds);
//                printf("loc = %d vir = %d loop f[%d] = %f\n", loc, vir, STARTI + NUMSEROTYPES * loc + vir,f[STARTI + NUMSEROTYPES * loc + vir]);
//            printf("loc = %d vir = %d loop f[%d] = %f\n", loc, vir, STARTJ + NUMSEROTYPES * loc + vir,f[STARTJ + NUMSEROTYPES * loc + vir]);

                step_I_done[index_1d_pass_1] = true;
                block.sync();
            }
            else {
                cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
                while(!step_I_done[index - NUMLOC*NUMSEROTYPES]){
                    __syncthreads();
                }
                f[index] = index;
                f[ index ] = f[ index - (NUMLOC * NUMSEROTYPES)];
//            // add the recovery rate - NOTE only for I-classes
                f[ index - (NUMLOC * NUMSEROTYPES) ] += - gpu_params->v_d[gpu_params->i_nu] * y[ index - (NUMLOC * NUMSEROTYPES) ];

//                printf("loc = %d vir = %d loop f[%d] I only = %f\n", loc, vir, index - (NUMLOC * NUMSEROTYPES),f[index - (NUMLOC * NUMSEROTYPES)]);
//                printf("f[%d] = f[%d]\n",index,index - (NUMLOC*NUMSEROTYPES));
                block.sync();
            }
        }
    }
    else{
        f[index_1d_pass_1] =    - (stf * gpu_params->beta[0] * y[12] * y[index_1d_pass_1])
                                - (stf * gpu_params->beta[1] * y[13] * y[index_1d_pass_1])
                                - (stf * gpu_params->beta[2] * y[14] * y[index_1d_pass_1])
                                + (trr * y[3]) + (trr * y[7]) + (trr * y[11]);
    }
//    if(index == 0 || index == DIM - 1){
//        printf("        [function] y[%d] = %.20f f[%d] = %.20f\n",index,y[index],index,f[index]);
//    }

    return;
}
