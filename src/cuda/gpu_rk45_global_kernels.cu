#include "gpu_rk45_global_kernels.h"

__global__
void calculate_y(double y[], double y_tmp[], double y_err[], double* h,  int step,
                      double k1[], double k2[], double k3[],
                      double k4[], double k5[], double k6[],
                      GPU_Parameters* params){

    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const double ah[] = { 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0 };
    const double b3[] = { 3.0/32.0, 9.0/32.0 };
    const double b4[] = { 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0};
    const double b5[] = { 8341.0/4104.0, -32832.0/4104.0, 29440.0/4104.0, -845.0/4104.0};
    const double b6[] = { -6080.0/20520.0, 41040.0/20520.0, -28352.0/20520.0, 9295.0/20520.0, -5643.0/20520.0};

    const double c1 = 902880.0/7618050.0;
    const double c3 = 3953664.0/7618050.0;
    const double c4 = 3855735.0/7618050.0;
    const double c5 = -1371249.0/7618050.0;
    const double c6 = 277020.0/7618050.0;

    const double ec[] = { 0.0,
                          1.0 / 360.0,
                          0.0,
                          -128.0 / 4275.0,
                          -2197.0 / 75240.0,
                          1.0 / 50.0,
                          2.0 / 55.0
    };

    for(int index = index_gpu; index < params->dimension; index += stride){
//        printf("      [calculate_y] y[%d] = %f h = %f k1[%d] = %f\n",index,y[index],(*h),index, k1[index]);
        if(step == 1){
            y_tmp[index] = y[index] +  ah[0] * (*h) * k1[index];
        }
        else if(step == 2){
            y_tmp[index] = y[index] + (*h) * (b3[0] * k1[index] + b3[1] * k2[index]);
        }
        else if(step == 3){
            y_tmp[index] = y[index] + (*h) * (b4[0] * k1[index] + b4[1] * k2[index] + b4[2] * k3[index]);
        }
        else if(step == 4){
            y_tmp[index] = y[index] + (*h) * (b5[0] * k1[index] + b5[1] * k2[index] + b5[2] * k3[index] + b5[3] * k4[index]);
        }
        else if(step == 5){
            y_tmp[index] = y[index] + (*h) * (b6[0] * k1[index] + b6[1] * k2[index] + b6[2] * k3[index] + b6[3] * k4[index] + b6[4] * k5[index]);
        }
        else if(step == 6){
            const double d_i = c1 * k1[index] + c3 * k3[index] + c4 * k4[index] + c5 * k5[index] + c6 * k6[index];
            y[index] += (*h) * d_i;
//            printf("      [calculate_y] step %d after k%d: y[%d] = %f\n",step,step,index,y_tmp[index]);
            return;
        }
        else if(step == 7){
            y_err[index] = (*h) * (ec[1] * k1[index] + ec[3] * k3[index] + ec[4] * k4[index] + ec[5] * k5[index] + ec[6] * k6[index]);
//            printf("      [calculate_y] step %d after dydt_out: y_err[%d] = %f\n",step,index,y_err[index]);
            return;
        }
//        printf("      [calculate_y] step %d after k%d: y_tmp[%d] = %f\n",step,step,index,y_tmp[index]);
    }
    return;
}

__global__
void calculate_r(double y[], double y_err[], double dydt_out[], double* h_0, double* h, int final_step, double r[], GPU_Parameters* params)
{
    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    static double eps_abs = 1e-6;
    static double eps_rel = 0.0;
    static double a_y = 1.0;
    static double a_dydt = 0.0;
    static unsigned int ord = 5;
    const double S = 0.9;
    double h_old;
    if(final_step){
        h_old = (*h_0);
    }
    else{
        h_old = *h;
    }

    for(int index = index_gpu; index < params->dimension; index += stride){
        const double D0 = eps_rel * (a_y * fabs(y[index]) + a_dydt * fabs(h_old * dydt_out[index])) + eps_abs;
        r[index]  = fabs(y_err[index]) / fabs(D0);
//        printf("      [calculate_r] IN y[%d] = %f\n",index,y[index]);
//        printf("      [calculate_r] IN y_err[%d] = %f\n",index,y_err[index]);
//        printf("      [calculate_r] IN dydt_out[%d] = %f\n",index,dydt_out[index]);
//        printf("      [calculate_r] eps_rel[%d] = %f\n",index,eps_rel);
//        printf("      [calculate_r] a_y[%d] = %f\n",index,a_y);
//        printf("      [calculate_r] h_old = %f\n",h_old);
//        printf("      [calculate_r] h = %f\n",(*h));
//        printf("        [calculate_r] fabs((h_old) * dydt_out_d[%d])) = %f\n",index,fabs((h_old) * dydt_out[index]));
//        printf("        [calculate_r] eps_abs[%d] = %f\n",index,eps_abs);
//        printf("        [calculate_r] D0[%d] = %f\n",index,D0);
//        printf("      [calculate_r] r[%d] = %f\n",index,r);
    }
    return;
}

void adjust_h(double r_max, double h_0, double* h, int final_step, int* adjustment_out){
    double h_old;
    if(final_step){
        h_old = (h_0);
    }
    else{
        h_old = *h;
    }

//    printf("r_max = %f\n",r_max);

    if (r_max > 1.1) {
        /* decrease step, no more than factor of 5, but a fraction S more
           than scaling suggests (for better accuracy) */
        double r = S / pow(r_max, 1.0 / ord);

        if (r < 0.2)
            r = 0.2;

        *h = r * (h_old);

//        printf("    Decrease by %f, h_old is %f new h is %f\n", r, h_old, *h);
        *adjustment_out = -1;
    } else if (r_max < 0.5) {
        /* increase step, no more than factor of 5 */
        double r = S / pow(r_max, 1.0 / (ord + 1.0));

        if (r > 5.0)
            r = 5.0;

        if (r < 1.0)  /* don't allow any decrease caused by S<1 */
            r = 1.0;

        *h = r * (h_old);

//        printf("    Increase by %f, h_old is %f new h is %f\n", r, h_old, *h);
        *adjustment_out = 1;
    } else {
        /* no change */
//        printf("    No change\n");
        *adjustment_out = 0;
    }
}

__global__
void gpu_func_test(double t, const double y[], double f[], int index, GPU_Parameters* gpu_params){

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
//    double stf = gpu_params->phis_d_length == 0 ? 1.0 : gpu_params->stf_d[day];
//    double stf = seasonal_transmission_factor(gpu_params,day);
    double stf = seasonal_transmission_factor(gpu_params,t);
//    double stf = gpu_params->stf;
//    double stf = 1.0;

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
        double sum_foi = 0.0;
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
        double foi_on_susc_single_virus = 0.0;

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

        double inflow_from_recovereds = 0.0;
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
        double foi_on_susc_all_viruses = 0.0;

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
