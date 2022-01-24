#include <cuda_runtime.h>
#include "gpu_rk45.h"

__device__
void gpu_func_test(double t, const double y[], double f[], int index, int day, GPU_Parameters* gpu_params){

//    printf("y[%d] = y %f\n",index,y[index]);
    // the transition rate among R-classes
    double trr = ((double)NUMR) / gpu_params->v_d[gpu_params->i_immune_duration];
    double stf = gpu_params->phis_d_length == 0 ? 1.0 : gpu_params->stf_d[day];
//    double stf = seasonal_transmission_factor(gpu_params, t);

//    if(index >= STARTS)
//    {
//        printf("[function] IN y[%d] = %f\n",index,y[index]);
////        if(index == 0){
////            printf("\n");
////        }
//    }

    if(index < STARTI){
//        int zDirection = i % zLength;
//        int yDirection = (i / zLength) % yLength;
//        int xDirection = i / (yLength * zLength);
        int loc = index / (NUMSEROTYPES * NUMR);
        int vir = (index / NUMR) % NUMSEROTYPES;
        int stg = index % NUMR;
        f[ index ] = - trr * y[ index ];
        if(index % NUMR == 0){
            f[ index ] += gpu_params->v_d[gpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];
        }
        else{
            f[ index ] += trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg - 1 ];
        }
        double sum_foi = 0.0;
//        for(int l = 0; l < NUMLOC; l++){
//            sum_foi +=      gpu_params->sigma[vir][0] * gpu_params->beta[0] * stf * gpu_params->eta[loc][l] * y[STARTI + NUMSEROTYPES*l + 0] +
//                            gpu_params->sigma[vir][1] * gpu_params->beta[1] * stf * gpu_params->eta[loc][l] * y[STARTI + NUMSEROTYPES*l + 1] +
//                            gpu_params->sigma[vir][2] * gpu_params->beta[2] * stf * gpu_params->eta[loc][l] * y[STARTI + NUMSEROTYPES*l + 2];
//        }
        sum_foi +=  gpu_params->sigma[vir][0] * gpu_params->beta[0] * stf * gpu_params->eta[loc][0] * y[STARTI + 0] +
                    gpu_params->sigma[vir][1] * gpu_params->beta[1] * stf * gpu_params->eta[loc][0] * y[STARTI + 1] +
                    gpu_params->sigma[vir][2] * gpu_params->beta[2] * stf * gpu_params->eta[loc][0] * y[STARTI + 2];
        f[index] +=  -(sum_foi) * y[index];
    }
    else if(index < STARTS){
        const int vir = (index - NUMSEROTYPES*NUMR*NUMLOC) % NUMSEROTYPES;
        if(index < STARTJ){
//            cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
            const int loc = (index / NUMSEROTYPES) % NUMLOC;
            double foi_on_susc_single_virus = 0.0;
//            for(int l=0; l<NUMLOC; l++){
//                foi_on_susc_single_virus += gpu_params->eta[loc][l] * stf * gpu_params->beta[vir] * y[STARTI + NUMSEROTYPES*l + vir];
//            }
            foi_on_susc_single_virus += gpu_params->eta[loc][0] * stf * gpu_params->beta[vir] * y[STARTI + vir];

//            printf("index %d foi_on_susc_single_virus = %f\n", index,foi_on_susc_single_virus);
            f[STARTI + NUMSEROTYPES*loc + vir] += y[STARTS + loc] * foi_on_susc_single_virus;
            f[STARTJ + NUMSEROTYPES*loc + vir] += y[ STARTS + loc ] * foi_on_susc_single_virus;
//
            double inflow_from_recovereds = 0.0;
//            for(int l = 0; l < NUMLOC; l++){
//                for(int v = 0; v < NUMSEROTYPES; v++){
//                    for (int s = 0; s < NUMR; s++) {       // sum over R stage
//                        inflow_from_recovereds += gpu_params->sigma[vir][v]
//                                                  * /* gpu_params->seasonal_transmission_factor(t) */ stf
//                                                  * gpu_params->beta[vir]
//                                                  * gpu_params->eta[loc][l]
//                                                  * y[STARTI + NUMSEROTYPES * l + vir]
//                                                  * y[NUMSEROTYPES * NUMR * loc + NUMR * v + s];
//                    }
//                }
//            }
            inflow_from_recovereds +=
            gpu_params->sigma[vir][0] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[0] +
            gpu_params->sigma[vir][0] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[1] +
            gpu_params->sigma[vir][0] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[2] +
            gpu_params->sigma[vir][0] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[3] +
            gpu_params->sigma[vir][1] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[4] +
            gpu_params->sigma[vir][1] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[5] +
            gpu_params->sigma[vir][1] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[6] +
            gpu_params->sigma[vir][1] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[7] +
            gpu_params->sigma[vir][2] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[8] +
            gpu_params->sigma[vir][2] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[9] +
            gpu_params->sigma[vir][2] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[10] +
            gpu_params->sigma[vir][2] * stf * gpu_params->beta[vir] * gpu_params->eta[0][0] * y[index] * y[11];

//            printf("index %d inflow_from_recovereds = %f\n",index,inflow_from_recovereds);
            f[STARTI + NUMSEROTYPES*loc + vir] += inflow_from_recovereds;
            f[STARTJ + NUMSEROTYPES*loc + vir] += inflow_from_recovereds;

            f[STARTI + NUMSEROTYPES*loc + vir] += - gpu_params->v_d[gpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];
//            step_I_done[index - STARTI] = true;
//            block.sync();
        }
    }
    else{
//        //Wait for IJ completed
//        while(!step_IJ_done){
//            __syncthreads();
//        }
        double foi_on_susc_all_viruses = 0.0;
//        for(int l=0; l<NUMLOC; l++) {
//            for (int v = 0; v < NUMSEROTYPES; v++) {
//                foi_on_susc_all_viruses += gpu_params->eta[index - STARTS][l] * stf * gpu_params->beta[v] *
//                                           y[STARTI + NUMSEROTYPES * l + v];
//            }
//        }
        foi_on_susc_all_viruses +=  gpu_params->eta[0][0] * stf *gpu_params->beta[0]  * y[12] +
                                    gpu_params->eta[0][0] * stf *gpu_params->beta[1]  * y[13] +
                                    gpu_params->eta[0][0] * stf *gpu_params->beta[2]  * y[14];

        f[ index ] = ( - foi_on_susc_all_viruses ) * y[ index ];
//        for(int vir=0; vir<NUMSEROTYPES; vir++)
//        {
//            // add to dS/dt the inflow of recovereds from the final R-stage
//            f[index] += trr * y[ NUMSEROTYPES*NUMR*(index - STARTS) + NUMR*vir + (NUMR-1) ]; // "NUMR-1" gets you the final R-stage only
//        }
        f[index] += trr * y[9] +
                    trr * y[9] +
                    trr * y[9];
    }

//    if(index < STARTS)
//    {
//        printf("[function] OUT f[%d] = %f\n",index,f[index]);
//        if(index == 0){
//            printf("\n");
//        }
//    }

    return;
}
