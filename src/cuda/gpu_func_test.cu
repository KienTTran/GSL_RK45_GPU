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

__global__
void gpu_func_test(double* t, const double y[], double f[], GPU_Parameters* params){
    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ bool step_I_done;
    __shared__ bool step_J_done;

    for(int index = index_gpu; index < params->dimension; index += stride)
    {
//        if(index == 0|| index == params->dimension - 1){
//            printf("    [gpu_func_test] IN index = %d t = %f y[%d] = %f\n",index,(*t),index,y[index]);
//        }
        // the transition rate among R-classes
        double trr = ((double)NUMR) / params->v_d[params->i_immune_duration];
        double stf = params->phis_d_length == 0 ? 1.0 : params->stf_d[static_cast<int>(*t)];

        const unsigned int START_I  = int(STARTI);
        const unsigned int START_J  = int(STARTJ);
        const unsigned int START_S  = int(STARTS);
        const unsigned int NUM_LOC  = int(NUMLOC);
        const unsigned int NUM_SEROTYPES  = int(NUMSEROTYPES);
        const unsigned int NUM_R  = int(NUMR);

        if(index < START_I){
            int loc = index / (NUM_SEROTYPES * NUM_R);
            int vir = (index / NUM_R) % NUM_SEROTYPES;
            int stg = index % NUM_R;
            f[index] = - trr * y[ index ];
            if(index % NUM_R == 0){
                f[index] += params->v_d[params->i_nu] * y[START_I + NUM_SEROTYPES*loc + vir];
            }
            else{
                f[index] += trr * y[NUM_SEROTYPES*NUM_R*loc + NUM_R*vir + stg - 1];
            }
            double sum_foi =    params->sigma[vir][0] * params->beta[0] * stf * params->eta[0][0] * y[12] +
                                params->sigma[vir][1] * params->beta[1] * stf * params->eta[0][0] * y[13] +
                                params->sigma[vir][2] * params->beta[2] * stf * params->eta[0][0] * y[14];
            f[index] +=  -(sum_foi) * y[index];
        }
        else if(index < START_S){
            int vir = (index - NUM_SEROTYPES*NUM_R*NUM_LOC) % NUM_SEROTYPES;
            if(index < START_J) {
                int loc = ((index - START_J) / (NUM_SEROTYPES)) % NUM_LOC;
                f[index] = 0.0;
                double foi_on_susc_single_virus = 0.0;
                foi_on_susc_single_virus = params->eta[0][0] * stf * params->beta[vir] * y[index];

                f[index] += y[START_S + loc] * foi_on_susc_single_virus;

                double inflow_from_recovereds = 0.0;
                inflow_from_recovereds +=
                        params->sigma[vir][0] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[0] +
                        params->sigma[vir][0] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[1] +
                        params->sigma[vir][0] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[2] +
                        params->sigma[vir][0] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[3] +
                        params->sigma[vir][1] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[4] +
                        params->sigma[vir][1] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[5] +
                        params->sigma[vir][1] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[6] +
                        params->sigma[vir][1] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[7] +
                        params->sigma[vir][2] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[8] +
                        params->sigma[vir][2] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[9] +
                        params->sigma[vir][2] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[10] +
                        params->sigma[vir][2] * stf * params->beta[vir] * params->eta[0][0] * y[index] * y[11];
                f[index] += inflow_from_recovereds;
                step_I_done = true;
            }
            else {
                while(!step_I_done){
                    __syncthreads();
                }
                f[index] = index;
                f[index] = f[index - (NUM_LOC * NUM_SEROTYPES)];
//            // add the recovery rate - NOTE only for I-classes
                f[index - (NUM_LOC * NUM_SEROTYPES)] += - params->v_d[params->i_nu] * y[index - (NUM_LOC * NUM_SEROTYPES)];
                step_J_done = true;
            }
        }
        else{
            while(!step_J_done){
                __syncthreads();
            }
            f[index] = -(stf * params->beta[0] * y[12] * y[index]) -
                       (stf * params->beta[1] * y[13] * y[index]) -
                       (stf * params->beta[2] * y[14] * y[index]) +
                       (trr * y[3]) + (trr * y[7]) + (trr * y[11]);

        }
        //update h before end device code
//        params->h += 0.05;
//        if(index == 0|| index == params->dimension - 1)
//        {
//            printf("    [gpu_func_test] OUT index = %d t = %f f[%d] = %f\n",index,(*t),index,f[index]);
//        }
    }
}
