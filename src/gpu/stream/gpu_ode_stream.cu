//
// Created by kient on 5/10/2022.
//

#include "gpu_ode_stream.cuh"

__device__
double pop_sum_stream(double yy[]) {
    double sum = 0.0;
    for (int i = 0; i < DIM; i++) sum += yy[i];

    for (int i = STARTJ; i < STARTJ + NUMLOC * NUMSEROTYPES; i++) sum -= yy[i];
    return sum;
}

__device__
void rk45_gpu_adjust_h_stream(double y[], double y_err[], double dydt_out[],
                       double &h, double h_0, int &adjustment_out, int final_step,
                       const int index) {
    /* adaptive adjustment */
    /* Available control object constructors.
       *
       * The standard control object is a four parameter heuristic
       * defined as follows:
       *    D0 = eps_abs + eps_rel * (a_y |y| + a_dydt h |y'|)
       *    D1 = |yerr|
       *    q  = consistency order of method (q=4 for 4(5) embedded RK)
       *    S  = safety factor (0.9 say)
       *
       *                      /  (D0/D1)^(1/(q+1))  D0 >= D1
       *    h_NEW = S h_OLD * |
       *                      \  (D0/D1)^(1/q)      D0 < D1
       *
       * This encompasses all the standard error scaling methods.
       *
       * The y method is the standard method with a_y=1, a_dydt=0.
       * The yp method is the standard method with a_y=0, a_dydt=1.
     */
    static double eps_abs = 1e-6;
    static double eps_rel = 0.0;
    static double a_y = 1.0;
    static double a_dydt = 0.0;
    static unsigned int ord = 5;
    const double S = 0.9;
    double h_old;
    if (final_step) {
        h_old = h_0;
    } else {
        h_old = h;
    }

    //    printf("    [adjust h] index = %d begin\n",index);
    //    for (int i = 0; i < DIM; i ++)
    //    {
    //        printf("      y[%d] = %.10f\n",i,y[i]);
    //    }
    //    for (int i = 0; i < DIM; i ++)
    //    {
    //        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
    //    }
    //    for (int i = 0; i < DIM; i ++)
    //    {
    //        printf("      dydt_out[%d] = %.10f\n",i,dydt_out[i]);
    //    }

    double r_max = 2.2250738585072014e-308;
    for (int i = 0; i < DIM; i++) {
        const double D0 = eps_rel * (a_y * fabs(y[i]) + a_dydt * fabs((h_old) * dydt_out[i])) + eps_abs;
        const double r = fabs(y_err[i]) / fabs(D0);
        //        printf("      compare r = %.10f r_max = %.10f\n",r,r_max);
        r_max = max(r, r_max);
    }

    //    printf("      r_max = %.10f\n",r_max);

    if (r_max > 1.1) {
        /* decrease step, no more than factor of 5, but a fraction S more
           than scaling suggests (for better accuracy) */
        double r = S / pow(r_max, 1.0 / ord);

        if (r < 0.2)
            r = 0.2;
        h = r * (h_old);

        //        printf("      index = %d decrease by %.10f, h_old is %.10f new h is %.10f\n",index, r, h_old, h);
        adjustment_out = -1;
    } else if (r_max < 0.5) {
        /* increase step, no more than factor of 5 */
        double r = S / pow(r_max, 1.0 / (ord + 1.0));

        if (r > 5.0)
            r = 5.0;

        if (r < 1.0)  /* don't allow any decrease caused by S<1 */
            r = 1.0;

        h = r * (h_old);

        //        printf("      index = %d increase by %.10f, h_old is %.10f new h is %.10f\n",index, r, h_old, h);
        adjustment_out = 1;
    } else {
        /* no change */
        //        printf("      index = %d no change\n",index);
        adjustment_out = 0;
    }
    //    printf("    [adjust h] index = %d end\n",index);
    return;
}

__device__
void rk45_gpu_step_apply_stream(double t, double h, double y[], double y_err[], double dydt_out[], double stf,
                         const int index, FluParameters *flu_params) {
    static const double ah[] = {1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0};
    static const double b3[] = {3.0 / 32.0, 9.0 / 32.0};
    static const double b4[] = {1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0};
    static const double b5[] = {8341.0 / 4104.0, -32832.0 / 4104.0, 29440.0 / 4104.0, -845.0 / 4104.0};
    static const double b6[] = {-6080.0 / 20520.0, 41040.0 / 20520.0, -28352.0 / 20520.0, 9295.0 / 20520.0,
                                -5643.0 / 20520.0};

    static const double c1 = 902880.0 / 7618050.0;
    static const double c3 = 3953664.0 / 7618050.0;
    static const double c4 = 3855735.0 / 7618050.0;
    static const double c5 = -1371249.0 / 7618050.0;
    static const double c6 = 277020.0 / 7618050.0;

    static const double ec[] = {0.0,
                                1.0 / 360.0,
                                0.0,
                                -128.0 / 4275.0,
                                -2197.0 / 75240.0,
                                1.0 / 50.0,
                                2.0 / 55.0
    };

    //    printf("    [step apply] index = %d start\n",index);
    //    printf("      t = %.10f h = %.10f\n",t,h);

    //    double* y_tmp = (double*)malloc(dim);
    //    double* k1 = (double*)malloc(dim);
    //    double* k2 = (double*)malloc(dim);
    //    double* k3 = (double*)malloc(dim);
    //    double* k4 = (double*)malloc(dim);
    //    double* k5 = (double*)malloc(dim);
    //    double* k6 = (double*)malloc(dim);
    double y_tmp[DIM];
    double k1[DIM];
    double k2[DIM];
    double k3[DIM];
    double k4[DIM];
    double k5[DIM];
    double k6[DIM];

    for (int i = 0; i < DIM; i++) {
        y_tmp[i] = 0.0;
        y_err[i] = 0.0;
        dydt_out[i] = 0.0;
        k1[i] = 0.0;
        k2[i] = 0.0;
        k3[i] = 0.0;
        k4[i] = 0.0;
        k5[i] = 0.0;
        k6[i] = 0.0;
    }

    //    for (int i = 0; i < DIM; i ++)
    //    {
    //        printf("      y[%d] = %.10f\n",i,y[i]);
    //        printf("      y_tmp[%d] = %.10f\n",i,y_tmp[i]);
    //        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
    //        printf("      dydt_out[%d] = %.10f\n",i,dydt_out[i]);
    //    }

    /* k1 */
    gpu_func_flu(t, y, k1, stf, index, flu_params);
    //    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i++) {
        //        printf("      k1[%d] = %.10f\n",i,k1[i]);
        y_tmp[i] = y[i] + ah[0] * h * k1[i];
    }
    /* k2 */
    gpu_func_flu(t + ah[0] * h, y_tmp, k2, stf, index, flu_params);
    //    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i++) {
        //        printf("      k2[%d] = %.10f\n",i,k2[i]);
        y_tmp[i] = y[i] + h * (b3[0] * k1[i] + b3[1] * k2[i]);
    }
    /* k3 */
    gpu_func_flu(t + ah[1] * h, y_tmp, k3, stf, index, flu_params);
    //    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i++) {
        //        printf("      k3[%d] = %.10f\n",i,k3[i]);
        y_tmp[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i]);
    }
    /* k4 */
    gpu_func_flu(t + ah[2] * h, y_tmp, k4, stf, index, flu_params);
    //    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i++) {
        //        printf("      k4[%d] = %.10f\n",i,k4[i]);
        y_tmp[i] = y[i] + h * (b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] + b5[3] * k4[i]);
    }
    /* k5 */
    gpu_func_flu(t + ah[3] * h, y_tmp, k5, stf, index, flu_params);
    //    cudaDeviceSynchronize();
    for (int i = 0; i < DIM; i++) {
        //        printf("      k5[%d] = %.10f\n",i,k5[i]);
        y_tmp[i] = y[i] + h * (b6[0] * k1[i] + b6[1] * k2[i] + b6[2] * k3[i] + b6[3] * k4[i] + b6[4] * k5[i]);
    }
    /* k6 */
    gpu_func_flu(t + ah[4] * h, y_tmp, k6, stf, index, flu_params);
    //    cudaDeviceSynchronize();
    /* final sum */
    for (int i = 0; i < DIM; i++) {
        //        printf("      k6[%d] = %.10f\n",i,k6[i]);
        const double d_i = c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i];
        y[i] += h * d_i;
    }
    /* Derivatives at output */
    gpu_func_flu(t + h, y, dydt_out, stf, index, flu_params);
    //    cudaDeviceSynchronize();
    /* difference between 4th and 5th order */
    for (int i = 0; i < DIM; i++) {
        y_err[i] = h * (ec[1] * k1[i] + ec[3] * k3[i] + ec[4] * k4[i] + ec[5] * k5[i] + ec[6] * k6[i]);
    }
    //debug printout
    //    for (int i = 0; i < DIM; i++) {
    //        printf("      index = %d y[%d] = %.10f\n",index,i,y[i]);
    //    }
    //    for (int i = 0; i < DIM; i++) {
    //        printf("      index = %d y_err[%d] = %.10f\n",index,i,y_err[i]);
    //    }
    //    for (int i = 0; i < DIM; i++) {
    //        printf("      index = %d dydt_out[%d] = %.10f\n",index,i,dydt_out[i]);
    //    }
    //    printf("    [step apply] index = %d end\n",index);
    return;
}

__device__
void rk45_gpu_evolve_apply_stream(double t, double t_target, double t_delta, double h, double *y, double *y_output,
                                  double *y_agg_input,  double *y_agg_output,  double stf[], int index, GPUParameters *gpu_params, FluParameters* flu_params) {
    double device_y[DIM];
    double device_y_0[DIM];
    double device_y_err[DIM];
    double device_dydt_out[DIM];
    double device_y_yesterday[DIM];
    int week_count = 0;
    double agg_inc_sum[DATADIM_COLS];
    double agg_inc_max[DATADIM_COLS];
    for (int i = 0; i < DATADIM_COLS; i++) {
        agg_inc_sum[i] = 0.0;
        agg_inc_max[i] = 0.0;
    }
    for (int i = 0; i < gpu_params->ode_dimension; i++) {
        device_y[i] = y[i];
    }

//    if((NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) && t == 0){
//        printf("rk45_gpu_evolve_apply flu_params: \n");
//        printf("  beta1 = %1.9f \n", flu_params->beta[index*NUMSEROTYPES + 0]);
//        printf("  beta2 = %1.9f \n", flu_params->beta[index*NUMSEROTYPES + 1]);
//        printf("  beta3 = %1.9f \n", flu_params->beta[index*NUMSEROTYPES + 2]);
//        printf("  v_d_i_amp = %1.5f \n", flu_params->v_d_i_amp);
//        printf("  sigma_H1B = %1.5f \n", flu_params->sigma2d[0][1]);
//        printf("  sigma_BH3 = %1.5f \n", flu_params->sigma2d[1][2]);
//        printf("  sigma_H1H3 = %1.5f \n", flu_params->sigma2d[0][2]);
//        printf("  eta = %1.5f \n", flu_params->eta[0][0]);
//        printf("  eta = %1.5f \n", flu_params->eta[1][1]);
//        printf("  eta = %1.5f \n", flu_params->eta[2][2]);
//        printf("  trr = %1.5f \n", flu_params->trr);
//        printf("  v_d_i_nu = %1.5f \n", flu_params->v_d_i_nu);
//        printf("phis_length = %d\n",SAMPLE_PHI_LENGTH);
//        for(int i=0; i<SAMPLE_PHI_LENGTH; i++){
//            printf("  phi = %5.1f \n", flu_params->phi[index*SAMPLE_PHI_LENGTH + i]);
//        }
//    }

    while (t < t_target) {
        double device_t;
        double device_t1;
        double device_h;
        double device_h_0;
        double device_dt;
        int device_adjustment_out = 999;
        device_t = t;
        device_t1 = device_t + t_delta;
        device_h = h;

        int day = t;
        double stf_today = stf[day];

//      printf("day %d\t", day);
//      for (int i = 0; i < gpu_params->ode_dimension; i ++) {
//        printf("y[%d][%d] = %.1f\t", index, i, device_y[i]);
//        if(i == (gpu_params->ode_dimension - 1)){
//          printf("\n");
//        }
//      }
        for (int i = 0; i < gpu_params->ode_dimension; i++) {
            device_y_yesterday[i] = device_y[i];
        }
        while (device_t < device_t1) {
            int device_final_step = 0;
            const double device_t_0 = device_t;
            device_h_0 = device_h;
            device_dt = device_t1 - device_t_0;
//            if(index == 0){
//                printf("\n  [evolve apply] index = %d start\n",index);
//                printf("    t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f dt = %.10f\n",device_t,device_t_0,device_h,device_h_0,device_dt);
//            }

            for (int i = 0; i < gpu_params->ode_dimension; i++) {
                device_y_0[i] = device_y[i];
            }

            device_final_step = 0;

            while (true) {
                if ((device_dt >= 0.0 && device_h_0 > device_dt) || (device_dt < 0.0 && device_h_0 < device_dt)) {
                    device_h_0 = device_dt;
                    device_final_step = 1;
                } else {
                    device_final_step = 0;
                }

                rk45_gpu_step_apply_stream(device_t_0, device_h_0, device_y, device_y_err, device_dydt_out, stf_today, index,flu_params);

                if (device_final_step) {
                    device_t = device_t1;
                } else {
                    device_t = device_t_0 + device_h_0;
                }

                double h_old = device_h_0;

//              printf("    before adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",device_t,device_t_0,device_h,device_h_0,h_old);

                rk45_gpu_adjust_h_stream(device_y, device_y_err, device_dydt_out, device_h, device_h_0, device_adjustment_out, device_final_step, index);

                //Extra step to get data from h
                device_h_0 = device_h;

//              printf("    after adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",device_t,device_t_0,device_h,device_h_0,h_old);

                if (device_adjustment_out == -1) {
                    double t_curr = (device_t);
                    double t_next = (device_t) + device_h_0;

                    if (fabs(device_h_0) < fabs(h_old) && t_next != t_curr) {
                        /* Step was decreased. Undo step, and try again with new h0. */
//                      printf("  [evolve apply] index = %d step decreased, y = y0\n",index);
                        for (int i = 0; i < DIM; i++) {
                            device_y[i] = device_y_0[i];
                        }
                    } else {
                        //                            printf("  [evolve apply] index = %d step decreased h_0 = h_old\n",index);
                        device_h_0 = h_old; /* keep current step size */
                        break;
                    }
                } else {
                    //                        printf("  [evolve apply] index = %d step increased or no change\n",index);
                    break;
                }
            }
            device_h = device_h_0;  /* suggest step size for next time-step */
            h = device_h;
//            printf("    index = %d t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f\n",index,device_t,device_t_0,device_h,device_h_0);
//            for (int i = 0; i < DIM; i++){
//                printf("    index = %d y[%d][%d] = %.10f\n",index,index,i,device_y[i]);
//            }
//            printf("  [evolve apply] index = %d end\n",index);
//            if(device_final_step){
//                printf("[output] index = %d t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f\n",index,device_t,device_t_0,device_h,device_h_0);
//                for (int i = 0; i < DIM; i++){
//                    printf("[output] index = %d y[%d] = %.10f\n",index,i,device_y[i]);
//                }
//            }
//            device_t = device_t_0 + device_h_0;
        }
//        if(index == 0) {
//            printf("[evolve apply] Index = %d t = %f h = %f end one day\n", index, t, h);
//        }
        if(index == 0 && t == NUMDAYSOUTPUT - 1) {
            printf("[evolve apply] Index = %d t = %f h = %f end, device_y_yesterday[%d][%d] = %.5f\n", index, t, h, index,gpu_params->ode_dimension - 4, device_y_yesterday[gpu_params->ode_dimension - 4]);
        }
        t += t_delta;

        /* y_ode_output_d*/
//        for (int i = 0; i < gpu_params->display_dimension; i ++) {
//          const int y_output_index = day * gpu_params->display_dimension + i;
//          if(y_output_index % gpu_params->display_dimension == 0){
//            //First column
//            y_output[y_output_index] = day*1.0;
//            //          printf("First day = %d index = %d i = %d y_output_index = %d y_output[%d][%d] = %.5f\n",
//            //                 day, index, i, y_output_index, index, y_output_index, y_output[y_output_index]);
//          }
//          else if(y_output_index % gpu_params->display_dimension == 1){
//            //Second column
//            y_output[y_output_index] = stf_today;
//            //          printf("Second day = %d index = %d i = %d y_output_index = %d y_output[%d][%d] = %.5f\n",
//            //                 day, index, i, y_output_index, index, y_output_index, y_output[y_output_index]);
//          }
//          else if(y_output_index % gpu_params->display_dimension >= 2 && y_output_index % gpu_params->display_dimension < gpu_params->display_dimension - 1){
//            //Third column to column next to last column
//            const int y_index = (y_output_index - 2) % gpu_params->display_dimension;
//            y_output[y_output_index] = device_y_yesterday[y_index];
//            //          printf("day = %d index = %d i = %d y_output_index = %d y[%d][%d] = y[%d][%d] = %.5f\n",
//            //                 day, index, i, y_output_index, index, y_output_index, index, y_index, device_y[y_index]);
//          }
//          else{
//            //Last column
////            y_output[y_output_index] = pop_sum(device_y);
//            y_output[y_output_index] = pop_sum(device_y);
//            //          printf("Third day = %d index = %d i = %d y_output_index = %d y_output[%d][%d] = %.5f\n",
//            //                 day, index, i, y_output_index, index, y_output_index, y_output[y_output_index]);
//          }
//        }

        /* y_ode_agg_d*/
        /* AGG Output 1-6 */
        for (int i = 0; i < gpu_params->data_params.cols; i++) {
            const int y_output_agg_index = (day + 1) * gpu_params->agg_dimension + i;
            const int y_output_agg_to_sum_index = (day) * gpu_params->agg_dimension + i;
            const int y_ode_index = gpu_params->ode_dimension - 4 + i;
            y_agg_output[y_output_agg_index] = y_agg_input[y_output_agg_index];
            if(day == 0) {
                y_agg_output[y_output_agg_to_sum_index] = 0.0;
            }
            y_agg_output[y_output_agg_index] = device_y[y_ode_index] - device_y_yesterday[y_ode_index];
            agg_inc_sum[i] += y_agg_output[y_output_agg_to_sum_index];
        }

        if ((day+1) % 7 == 0 || day == gpu_params->ode_output_day - 1) {
            for(int i = 0; i < gpu_params->data_params.cols; i++){
                //Col 3 4 5
                const int y_output_agg_col = (3 + i) + week_count * gpu_params->agg_dimension;
                y_agg_output[y_output_agg_col] = agg_inc_sum[i];
                if(agg_inc_sum[i] >= agg_inc_max[i]) agg_inc_max[i] = agg_inc_sum[i];
                agg_inc_sum[i] = 0.0;
            }
            week_count++;
        }

        //Write max agg inc to first line
        if(day == gpu_params->ode_output_day - 1){
//            printf("ODE %d agg_inc_max: ", index);
            for(int i = 0; i < DATADIM_COLS; i++){
                //Col 1 2 3
                y_agg_output[i] = agg_inc_max[i];
//                printf("%.5f\t", agg_inc_max[i]);
            }
//            printf("\n");
        }
//        if(index == 0 && t == NUMDAYSOUTPUT - 1) {
//            printf("[evolve apply agg] Index = %d t = %f h = %f end, agg_inc_max[0] = %.5f\n", index, t, h, agg_inc_max[0]);
//        }
    }
//    if(index == 0){
//        for (int i = 0; i < DIM; i++){
//            printf("[output] index = %d y[%d] = %1.5f\n",index,i,device_y[i]);
//        }
//    }
}


__device__
void solve_ode_one_stream(double *y_ode_input_d, double *y_ode_output_d, double *y_agg_input_d, double *y_agg_output_d,  double stf[], int index, GPUParameters *gpu_params, FluParameters* flu_params) {
    rk45_gpu_evolve_apply_stream(gpu_params->t0, gpu_params->t_target, gpu_params->step, gpu_params->h, y_ode_input_d, y_ode_output_d, y_agg_input_d, y_agg_output_d, stf, index, gpu_params, flu_params);
    return;
}

__global__ void
solve_ode_n_stream(double *y_ode_input_d, double *y_ode_output_d,
                        double *y_agg_input_d, double *y_agg_output_d,
                        double *stf_d,
                        GPUParameters *gpu_params, FluParameters *flu_params) {
    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int index = index_gpu; index < gpu_params->ode_number; index += stride) {
//        if(index % 32 == 0)
//        {
//            printf("ODE %d will be solved by thread index = %d blockIdx.x = %d\n", index, index, blockIdx.x);
//        }
        solve_ode_one_stream(y_ode_input_d, y_ode_output_d, y_agg_input_d, y_agg_output_d, stf_d, index, gpu_params, flu_params);
    }

}