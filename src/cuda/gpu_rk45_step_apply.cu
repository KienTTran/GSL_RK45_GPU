#include "gpu_rk45.h"

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
