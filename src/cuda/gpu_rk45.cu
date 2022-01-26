#include "gpu_rk45.h"
#include "../gpu_parameters.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
    return result;
}

GPU_RK45::GPU_RK45(){
    params = new GPU_Parameters();
}

GPU_RK45::~GPU_RK45(){
    params = nullptr;
}

void GPU_RK45::setParameters(GPU_Parameters* params_) {
    params = &(*params_);
}

__device__
double seasonal_transmission_factor(GPU_Parameters* gpu_params, double t)
{
    /*


        We're gonna make this thing go for 40 years. 30 years of burn in and 10 years of real modeling.
        We're creating a "10-year model cycle" and need the code below to find a time point's "place" in the "cycle"
        modulus (denoted with % in C++) only works with integers, so need the acrobatics below

     */

    // This is some code that's needed to create the 10-year "cycles" in transmission.

    if(gpu_params->phis_d_length == 0){
        return 1.0;
    }

    int x = (int)t; // This is now to turn a double into an integer
    double remainder = t - (double)x;
    int xx = x % 3650; // int xx = x % NUMDAYSOUTPUT;
    double yy = (double)xx + remainder;
    // put yy into the sine function, let it return the beta value
    t = yy;
    double sine_function_value = 0.0;

    for(int i=0; i<gpu_params->phis_d_length; i++)
    {
        if( std::fabs( t - gpu_params->phis_d[i] ) < (gpu_params->v_d[gpu_params->i_epidur] / 2))
        {
            // sine_function_value = sin( 2.0 * 3.141592653589793238 * (phis[i]-t+91.25) / 365.0 );
            sine_function_value = std::sin( 2.0 * 3.141592653589793238 * (gpu_params->phis_d[i] - t +(gpu_params->v_d[gpu_params->i_epidur] / 2)) / (gpu_params->v_d[gpu_params->i_epidur] * 2));
//            printf("      in loop %1.3f %d  %1.3f %1.3f\n", t, i, gpu_params->phis_d[i], sine_function_value );
        }
    }
//    printf("    %f sine_function_value %1.3f\n",t,sine_function_value);
//    printf("    %f return %1.3f\n",t,1.0 + v[i_amp] * sine_function_value);
    return 1.0 + gpu_params->v_d[gpu_params->i_amp] * sine_function_value;
}

__device__
double seasonal_transmission_factor(GPU_Parameters* gpu_params, int day)
{
    /*


        We're gonna make this thing go for 40 years. 30 years of burn in and 10 years of real modeling.
        We're creating a "10-year model cycle" and need the code below to find a time point's "place" in the "cycle"
        modulus (denoted with % in C++) only works with integers, so need the acrobatics below

     */

    // This is some code that's needed to create the 10-year "cycles" in transmission.

    if(gpu_params->phis_d_length == 0){
        return 1.0;
    }

    int x = day; // This is now to turn a double into an integer
    double remainder = day - (double)x;
    int xx = x % 3650; // int xx = x % NUMDAYSOUTPUT;
    double yy = (double)xx + remainder;
    // put yy into the sine function, let it return the beta value
    day = yy;
    double sine_function_value = 0.0;

    for(int i=0; i<gpu_params->phis_d_length; i++)
    {
        if( std::fabs( day - gpu_params->phis_d[i] ) < (gpu_params->v_d[gpu_params->i_epidur] / 2))
        {
            // sine_function_value = sin( 2.0 * 3.141592653589793238 * (phis[i]-t+91.25) / 365.0 );
            sine_function_value = std::sin( 2.0 * 3.141592653589793238 * (gpu_params->phis_d[i] - day +(gpu_params->v_d[gpu_params->i_epidur] / 2)) / (gpu_params->v_d[gpu_params->i_epidur] * 2));
//            printf("      in loop %1.3f %d  %1.3f %1.3f\n", t, i, gpu_params->phis_d[i], sine_function_value );
        }
    }
//    printf("    %f sine_function_value %1.3f\n",t,sine_function_value);
//    printf("    %f return %1.3f\n",t,1.0 + v[i_amp] * sine_function_value);
    return 1.0 + gpu_params->v_d[gpu_params->i_amp] * sine_function_value;
}

__device__
double pop_sum( double yy[] )
{
    double sum=0.0;
    for(int i=0; i<DIM; i++) sum += yy[i];

    for(int i=STARTJ; i<STARTJ+NUMLOC*NUMSEROTYPES; i++) sum -= yy[i];
    return sum;
}

__device__
void rk45_gpu_adjust_h(double y[], double y_err[], double dydt_out[],
                       double* h, double h_0, int* adjustment_out, int final_step,
                       double r[], double D0[], double r_max[],
                       const int index, GPU_Parameters* params)
                       {
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
    if(final_step){
        h_old = h_0;
    }
    else{
        h_old = *h;
    }
//    if(index == 0 || index == params->dimension - 1)
//    {
//        printf("  [gpu_adjust_h] Index = %d D0 = %f r = %f\n", index, D0, r);
//        printf("    IN y[%d] = %f\n",index,y[index]);
//        printf("    IN y_err[%d] = %f\n",index,y_err[index]);
//        printf("    IN dydt_out[%d] = %f\n",index,dydt_out[index]);
//        printf("    eps_rel[%d] = %f\n",index,eps_rel);
//        printf("    a_y[%d] = %f\n",index,a_y);
//        printf("    h_old[%d] = %f\n",index,h_old);
//        printf("    fabs((h_old) * dydt_out_d[%d])) = %f\n",index,fabs((h_old) * dydt_out[index]));
//        printf("    eps_abs[%d] = %f\n",index,eps_abs);
//        printf("    D0[%d] = %f\n",index,D0);
//        printf("    r[%d] = %f\n",index,r);
//    }

    //finding r_max
    r_max[index] = 2.2250738585072014e-308;
    D0[index] = eps_rel * (a_y * fabs(y[index]) + a_dydt * fabs((h_old) * dydt_out[index])) + eps_abs;
    r[index]  = fabs(y_err[index]) / fabs(D0[index]);

//    if(index == 0 || index == params->dimension - 1) {
//        printf("      index = %d D0[%d] = %f\n",index,index,D0[index]);
//        printf("      index = %d r[%d] = %f\n",index,index,r[index]);
//    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    for (int i = 0; i < params->dimension; i++)
    {
//        if(index == 0 || index == params->dimension - 1)
//        if(r[i] != 0)
//        {
//            printf("      compare r[%d] = %f with r_max[%d] = %f\n",i,r[i],index,r_max[index]);
//        }
        r_max[index] = max(r[i], r_max[index]);
    }
    block.sync();

//    reduce_max_device(r,r_max,params->dimension);
//    r_max[index] = r_max[0];

//    if(index == 0 || index == params->dimension - 1) {
//        printf("    Index = %d r_max =  %f\n", index, r_max[index]);
//    }

    if (r_max[index] > 1.1) {
        /* decrease step, no more than factor of 5, but a fraction S more
           than scaling suggests (for better accuracy) */
        double r = S / pow(r_max[index], 1.0 / ord);

        if (r < 0.2)
            r = 0.2;

        *h = r * (h_old);

//        if(index == 0 || index == params->dimension - 1) {
//            printf("    Index = %d decrease by %f, h_old is %f new h is %f\n", index, r, h_old, *h);
//        }
        *adjustment_out = -1;
    } else if (r_max[index] < 0.5) {
        /* increase step, no more than factor of 5 */
        double r = S / pow(r_max[index], 1.0 / (ord + 1.0));

        if (r > 5.0)
            r = 5.0;

        if (r < 1.0)  /* don't allow any decrease caused by S<1 */
            r = 1.0;

        *h = r * (h_old);

//        if(index == 0 || index == params->dimension - 1) {
//            printf("    Index = %d increase by %f, h_old is %f new h is %f\n", index, r, h_old, *h);
//        }
        *adjustment_out = 1;
    } else {
        /* no change */
//        if(index == 0 || index == params->dimension - 1) {
//            printf("    Index = %d no change\n", index);
//        }
        *adjustment_out = 0;
    }
//    if(index == 0 || index == params->dimension - 1) {
//        printf("    [adjust h] index = %d end\n",index);
//    }
    return;
}

__device__
void rk45_gpu_step_apply(double t, double h,
                         double y[], double y_tmp[], double y_err[], double dydt_in[], double dydt_out[],
                         double k1[], double k2[], double k3[], double k4[], double k5[], double k6[],
                         const int index, const int day, GPU_Parameters* params)
{
    static const double ah[] = { 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0 };
    static const double b3[] = { 3.0/32.0, 9.0/32.0 };
    static const double b4[] = { 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0};
    static const double b5[] = { 8341.0/4104.0, -32832.0/4104.0, 29440.0/4104.0, -845.0/4104.0};
    static const double b6[] = { -6080.0/20520.0, 41040.0/20520.0, -28352.0/20520.0, 9295.0/20520.0, -5643.0/20520.0};

    static const double c1 = 902880.0/7618050.0;
    static const double c3 = 3953664.0/7618050.0;
    static const double c4 = 3855735.0/7618050.0;
    static const double c5 = -1371249.0/7618050.0;
    static const double c6 = 277020.0/7618050.0;

    static const double ec[] = { 0.0,
                                 1.0 / 360.0,
                                 0.0,
                                 -128.0 / 4275.0,
                                 -2197.0 / 75240.0,
                                 1.0 / 50.0,
                                 2.0 / 55.0
    };
    y_tmp[index] = 0.0;
    y_err[index] = 0.0;
    dydt_out[index] = 0.0;
    k1[index] = 0.0;
    k2[index] = 0.0;
    k3[index] = 0.0;
    k4[index] = 0.0;
    k5[index] = 0.0;
    k6[index] = 0.0;

//    if(index == 0 || index == params->dimension - 1)
//    {
//        printf("  [gpu_step_apply] Index = %d t = %f h = %f start\n",index,t,h);
//        printf("    IN y[%d] = %f\n",index,y[index]);
//        printf("    IN y_err[%d] = %f\n",index,y_err[index]);
//        printf("    IN dydt_out[%d] = %f\n",index,dydt_out[index]);
//    }

    /* k1 */
    if (dydt_in != NULL)
    {
        k1[index] = dydt_in[index];
//        if(index == 0 || index == params->dimension - 1) {
//            printf("dydt_in != NULL\n");
//        }
    }
    else {
        gpu_func_test(t, y, k1, index, day, params);
//        gpu_func_test(t, y, k1, index, day, params->dimension);
        __syncthreads();
    }
//    if(index == 0 || index == params->dimension - 1) {
//        printf("    k1[%d] = %f\n", index, k1[index]);
//    }
    y_tmp[index] = y[index] +  ah[0] * h * k1[index];
    /* k2 */
    gpu_func_test(t + ah[0] * h, y_tmp, k2, index, day, params);
//    gpu_func_test(t + ah[0] * h, y_tmp, k2, index, day, params->dimension);
    __syncthreads();
//    if(index == 0 || index == params->dimension - 1) {
//            printf("    k2[%d] = %f\n",index,k2[index]);
//    }
    y_tmp[index] = y[index] + h * (b3[0] * k1[index] + b3[1] * k2[index]);
    /* k3 */
    gpu_func_test(t + ah[1] * h, y_tmp, k3, index, day, params);
//    gpu_func_test(t + ah[1] * h, y_tmp, k3, index, day, params->dimension);
    __syncthreads();
//    if(index == 0 || index == params->dimension - 1) {
//            printf("    k3[%d] = %f\n",index,k3[index]);
//    }
    y_tmp[index] = y[index] + h * (b4[0] * k1[index] + b4[1] * k2[index] + b4[2] * k3[index]);
    /* k4 */
    gpu_func_test(t + ah[2] * h, y_tmp, k4, index, day, params);
//    gpu_func_test(t + ah[2] * h, y_tmp, k4, index, day, params->dimension);
    __syncthreads();
//    if(index == 0 || index == params->dimension - 1) {
//            printf("    k4[%d] = %f\n",index,k4[index]);
//    }
    y_tmp[index] = y[index] + h * (b5[0] * k1[index] + b5[1] * k2[index] + b5[2] * k3[index] + b5[3] * k4[index]);
    /* k5 */
    gpu_func_test(t + ah[3] * h, y_tmp, k5, index, day, params);
//    gpu_func_test(t + ah[3] * h, y_tmp, k5, index, day, params->dimension);
    __syncthreads();
//    if(index == 0 || index == params->dimension - 1) {
//            printf("    k5[%d] = %f\n",index,k5[index]);
//    }
    y_tmp[index] = y[index] + h * (b6[0] * k1[index] + b6[1] * k2[index] + b6[2] * k3[index] + b6[3] * k4[index] + b6[4] * k5[index]);
    /* k6 */
    gpu_func_test(t + ah[4] * h, y_tmp, k6, index, day, params);
//    gpu_func_test(t + ah[4] * h, y_tmp, k6, index, day, params->dimension);
    __syncthreads();
    /* final sum */
//    if(index == 0 || index == params->dimension - 1) {
//        printf("    k6[%d] = %f\n", index, k6[index]);
//    }
    const double d_i = c1 * k1[index] + c3 * k3[index] + c4 * k4[index] + c5 * k5[index] + c6 * k6[index];
    y[index] += h * d_i;
    /* Derivatives at output */
    gpu_func_test(t + h, y, dydt_out, index, day, params);
//    gpu_func_test(t + h, y, dydt_out, index, day, params->dimension);
    __syncthreads();
    /* difference between 4th and 5th order */
    y_err[index] = h * (ec[1] * k1[index] + ec[3] * k3[index] + ec[4] * k4[index] + ec[5] * k5[index] + ec[6] * k6[index]);
    //debug printout
//    if(index == 0 || index == params->dimension - 1)
//    {
//        printf("    OUT y[%d] = %f\n",index,y[index]);
//        printf("    OUT y_err[%d] = %f\n",index,y_err[index]);
//        printf("    OUT dydt_out[%d] = %f\n",index,dydt_out[index]);
//        printf("  [gpu_step_apply] Index = %d t = %f h = %f end\n",index,t,h);
//    }
    return;
}

__global__
void rk45_gpu_evolve_apply(double t, double t_target, double t_delta, double h, double y[],
                           double y_0[], double y_err[], double y_tmp[], double dydt_in[], double dydt_out[],
                           double k1[], double k2[], double k3[], double k4[], double k5[], double k6[],
                           double D0[], double r[], double r_max[],
                           double y_output[],
                           GPU_Parameters* params){

    //shared mem version, use for less streams
//    __shared__ double y_0[DIM];
//    __shared__ double y_tmp[DIM];
//    __shared__ double y_err[DIM];
//    __shared__ double dydt_in[DIM];
//    __shared__ double dydt_out[DIM];
//    __shared__ double k1[DIM];
//    __shared__ double k2[DIM];
//    __shared__ double k3[DIM];
//    __shared__ double k4[DIM];
//    __shared__ double k5[DIM];
//    __shared__ double k6[DIM];
//    __shared__ double r_max[DIM];
//    __shared__ double D0[DIM];
//    __shared__ double r[DIM];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
//    int index_gpu = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = blockDim.x * gridDim.x;

//    for(int index = index_gpu; index < params->dimension; index += stride)
    if(index < params->dimension)
    {
        y_0[index] = 0.0;
        y_tmp[index] = 0.0;
        y_err[index] = 0.0;
        dydt_in[index] = 0.0;
        dydt_out[index] = 0.0;
        k1[index] = 0.0;
        k2[index] = 0.0;
        k3[index] = 0.0;
        k4[index] = 0.0;
        k5[index] = 0.0;
        k6[index] = 0.0;
        r_max[index] = 0.0;
        D0[index] = 0.0;
        r[index] = 0.0;

//        printf("[function] IN y[%d] = %f\n",index,y[index]);
//        for(int i = 0; i<1; i++){
//            gpu_func_test(t, y, dydt_in, index, day, params);
//            __syncthreads();
//            y[index] = dydt_in[index];
//            dydt_in[index] = 0;
//        }
//        printf("[function] OUT y[%d] = %f f[%d] = %f\n",index,y[index],index,dydt_in[index]);
//        return;

        while(t < t_target)
        {
//            if(index == 0 || index == params->dimension - 1) {
//                printf("[evolve apply] Index = %d t = %f h = %f start one day\n", index, t_start, h[index]);
//            }
            double device_t;
            double device_t1;
            double device_h;
            double device_h_0;
            double device_dt;
            int device_adjustment_out = 999;
            device_t = t;
            device_t1 = device_t + 1.0;
            device_h = h;
            params->stf = seasonal_transmission_factor(params,t);
            int day = t;

//            if(index == 0){
//                printf("index %d day %f stf = %f\n",index,t,params->stf);
//            }

//            if(index == 0 || index == params->dimension - 1) {
//                printf("\n  Will run from %f to %f, step %f\n", t, device_t1, h);
//                printf("    t = %f t_1 = %f  h = %f\n", device_t, device_t1, device_h);
//            }

            const int output_index = day * params->display_dimension + index;
            if(output_index % params->display_dimension == 0){
                //First column
                y_output[output_index] = day;
            }
            if(output_index % params->display_dimension == 1){
                //Second column
                y_output[output_index] = params->stf;
            }
            if(output_index % params->display_dimension == 2){
                //Third column
                y_output[output_index] = pop_sum(y);
            }
            //Forth column onward
            y_output[output_index + 3] = y[index];
            __syncthreads();

            while(device_t < device_t1)
            {
                int device_final_step = 0;
                const double device_t_0 = device_t;
                device_h_0 = device_h;
                device_dt = device_t1 - device_t_0;
                y_0[index] = y[index];
//                if(index == 0 || index == params->dimension - 1) {
//                    printf("[evolve apply] Index = %d t = %f t_0 = %f h = %f dt = %f start one iteration\n", index, t, t_0, h,dt);
//                }
//                if(index == 0 || index == params->dimension - 1) {
//                    printf("[evolve apply] Useydt_in\n");
//                }

                gpu_func_test(device_t_0, y, dydt_in, index, day, params);
//                gpu_func_test(device_t_0, y, dydt_in, index, day, params->dimension);
                __syncthreads();
                while(true)
                {
                    if ((device_dt >= 0.0 && device_h_0 > device_dt) || (device_dt < 0.0 && device_h_0 < device_dt)) {
                        device_h_0 = device_dt;
                        device_final_step = 1;
                    } else {
                        device_final_step = 0;
                    }
                    rk45_gpu_step_apply(device_t_0, device_h_0,
                                        y, y_tmp, y_err, dydt_in, dydt_out,
                                        k1, k2, k3, k4, k5, k6,
                                        index, day, params);
                    if (device_final_step) {
                        device_t = device_t1;
                    } else {
                        device_t = device_t_0 + device_h_0;
                    }
                    double h_old = device_h_0;
                    rk45_gpu_adjust_h(y, y_err, dydt_out,
                                      &device_h, device_h_0, &device_adjustment_out, device_final_step,
                                      r, D0, r_max,
                                      index, params);
                    //Extra step to get data from h
                    device_h_0 = device_h;
                    if (device_adjustment_out == -1)
                    {
                        double t_curr = (device_t);
                        double t_next = (device_t) + device_h_0;

                        if (fabs(device_h_0) < fabs(h_old) && t_next != t_curr) {
                            /* Step was decreased. Undo step, and try again with new h0. */
//                            if(index == 0 || index == params->dimension - 1) {
//                                printf("  [evolve apply] index = %d step decreased, y = y0\n", index);
//                            }
                            y[index] = y_0[index];
                        } else {
//                            if(index == 0 || index == params->dimension - 1) {
//                                printf("  [evolve apply] index = %d step decreased h_0 = h_old\n", index);
//                            }
                            device_h_0 = h_old; /* keep current step size */
                            break;
                        }
                    }
                    else{
//                        if(index == 0 || index == params->dimension - 1) {
//                            printf("  [evolve apply] index = %d step increased or no change\n", index);
//                        }
                        break;
                    }
                }
//                if(index == 0 || index == params->dimension - 1)
//                {
//                    printf("    index = %d t = %f t_0 = %f  h = %f h_0 = %f\n", index, device_t, device_t_0, device_h, device_h_0);
//                    printf("    index = %d y[%d] = %f\n", index, index, y[index]);
//                    printf("\n");
//                    if(device_final_step)
//                    {
//                        if(index == 0 || index == params->dimension - 1) {
//                            if(index == 0)
//                            {
//                                printf("[output] index = %d t = %f t_0 = %f  h = %f h_0 = %f\n", index, device_t,
//                                   device_t_0, device_h, device_h_0);
//                                printf("[output] index = %d y[%d] = %f\n", index, index, y[index]);
//                                printf("\n");
//                            }
//                        }
//                    }
//                    printf("  [evolve apply] index = %d end\n\n",index);
//                }
                device_h = device_h_0;  /* suggest step size for next time-step */
                h = device_h;
//                /* Test */
//                t += device_h;
//                if(index == 0){
//                    printf("Time = %d index = %d 1D index = %d\n",day,index,day*DIM + index);
//                }
            }
//            if(index == 0) {
//                printf("[evolve apply] Index = %d t = %f h = %f end one day\n", index, t, h);
//            }
            t += t_delta;
            day += 1;
        }
    }
    return;
}

__global__ void kernel(float *a, int offset)
{
    int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
    float x = (float)i;
    float s = sinf(x);
    float c = cosf(x);
    a[i] = a[i] + sqrtf(s*s+c*c);
}

__global__ void kernel2(double *a, int offset)
{
    int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
    double x = (double)i;
    double s = sinf(x);
    double c = cosf(x);
    a[i] = a[i] + sqrtf(s*s+c*c);
}

void GPU_RK45::run(){

    auto start_all = std::chrono::high_resolution_clock::now();
    auto stop_all = std::chrono::high_resolution_clock::now();
    auto start_compute = std::chrono::high_resolution_clock::now();
    auto stop_compute = std::chrono::high_resolution_clock::now();
    auto start_transfer_d2h = std::chrono::high_resolution_clock::now();
    auto stop_transfer_d2h = std::chrono::high_resolution_clock::now();
    auto start_display = std::chrono::high_resolution_clock::now();
    auto stop_display = std::chrono::high_resolution_clock::now();
    auto start_transfer_h2d = std::chrono::high_resolution_clock::now();
    auto stop_transfer_h2d = std::chrono::high_resolution_clock::now();

    cudaFuncSetCacheConfig(rk45_gpu_evolve_apply, cudaFuncCachePreferShared);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024000*100);

    const int num_streams = 4;
    const int block_size = 256;
    int num_SMs;
    checkCuda(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = (params->dimension + block_size - 1) / block_size;
//    printf("[GSL GPU] SMs = %d block_size = %d num_blocks = %d\n",num_SMs,block_size,num_blocks);

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        checkCuda(cudaStreamCreate(&streams[i]));
    }

    //y
    double* y_pinned[num_streams];
    double *y_d[num_streams];
    double *y_output_d[num_streams];
    double *y_output_host_display_pinned[num_streams];//Pinned memory
    //y_0
    double *y_0_d[num_streams];
    //y_tmp
    double *y_tmp_d[num_streams];
    //y_err
    double *y_err_d[num_streams];
    //dydt_in_d
    double *dydt_in_d[num_streams];
    //dydt_out_d
    double *dydt_out_d[num_streams];
    //k1_d
    double *k1_d[num_streams];
    //k2_d
    double *k2_d[num_streams];
    //k3_d
    double *k3_d[num_streams];
    //k4_d
    double *k4_d[num_streams];
    //k5_d
    double *k5_d[num_streams];
    //k6_d
    double *k6_d[num_streams];
    //r_d
    double *r_d[num_streams];
    //D0_d
    double *D0_d[num_streams];
    //r_max_d
    double *r_max_d[num_streams];
    //params_d
    GPU_Parameters* params_d[num_streams];

    for (int i = 0; i < num_streams; i++) {
        //Allocate pinned memory for y pinned
        checkCuda(cudaMallocHost((void**)&y_pinned[i], params->dimension * sizeof(double)));
        //Copy data from y to y_pinned
        memcpy(y_pinned[i], params->y, params->dimension * sizeof(double));

        //Allocate memory for y on device y_d
        checkCuda(cudaMalloc((void **) &y_d[i], params->dimension * sizeof(double)));
        //Copy data from y host to y device (y_pinned to y_d) - pinned version
        checkCuda(cudaMemcpy(y_d[i], y_pinned[i], params->dimension * sizeof(double), cudaMemcpyHostToDevice));

        //Allocate pinned memory for display output
        checkCuda(cudaMallocHost((void**)&y_output_host_display_pinned[i], NUMDAYSOUTPUT * params->display_dimension * sizeof(double)));
        //Allocate memory for y output on device (this one is used to store display data on device)
        checkCuda(cudaMalloc((void **) &y_output_d[i], NUMDAYSOUTPUT * params->display_dimension * sizeof(double)));
        //Copy data from y output from host to device
        checkCuda(cudaMemcpy(y_output_d[i], params->y_output, NUMDAYSOUTPUT * params->display_dimension * sizeof(double),cudaMemcpyHostToDevice));

        checkCuda(cudaMalloc ((void **)&y_0_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (y_0_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&y_tmp_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (y_tmp_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&y_err_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (y_err_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&dydt_in_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (dydt_in_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&dydt_out_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (dydt_out_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&k1_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (k1_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&k2_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (k2_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&k3_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (k3_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&k4_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (k4_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&k5_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (k5_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&k6_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (k6_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&r_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (r_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&D0_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (D0_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc ((void **)&r_max_d[i], params->dimension * sizeof (double)));
        checkCuda(cudaMemcpy (r_max_d[i], params->y, params->dimension * sizeof (double), cudaMemcpyHostToDevice));

        checkCuda(cudaMalloc((void **) &params_d[i], sizeof(GPU_Parameters)));
        checkCuda(cudaMemcpy(params_d[i], params, sizeof(GPU_Parameters), cudaMemcpyHostToDevice));
    }
    checkCuda(cudaDeviceSynchronize());

    stop_transfer_h2d = std::chrono::high_resolution_clock::now();
    auto duration_transfer_h2d = std::chrono::duration_cast<std::chrono::microseconds>(stop_transfer_h2d- start_transfer_h2d);

    start_compute = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_streams; i++) {
        rk45_gpu_evolve_apply<<<num_blocks, block_size, 0, streams[i]>>>(params->t0, params->t_target, 1.0, params->h, y_d[i],
                                                                            y_0_d[i], y_tmp_d[i], y_err_d[i], dydt_in_d[i], dydt_out_d[i],
                                                                              k1_d[i], k2_d[i], k3_d[i], k4_d[i], k5_d[i], k6_d[i],
                                                                              D0_d[i], r_d[i], r_max_d[i],
                                                                              y_output_d[i],
                                                                              params_d[i]);
    }
    checkCuda(cudaDeviceSynchronize());

    stop_compute = std::chrono::high_resolution_clock::now();
    auto duration_compute = std::chrono::duration_cast<std::chrono::microseconds>(stop_compute - start_compute);

    start_transfer_d2h = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_streams; i++) {
        checkCuda(cudaMemcpy(y_output_host_display_pinned[i], y_output_d[i], NUMDAYSOUTPUT * params->display_dimension * sizeof(double), cudaMemcpyDeviceToHost));
    }
    checkCuda(cudaDeviceSynchronize());

    stop_transfer_d2h = std::chrono::high_resolution_clock::now();
    auto duration_transfer_d2h = std::chrono::duration_cast<std::chrono::microseconds>(stop_transfer_d2h - start_transfer_d2h);

    start_display = std::chrono::high_resolution_clock::now();
    for(int s = 0; s < num_streams; s++){
        printf("Display from stream %d\n",s);
        for(int i = 0; i < NUMDAYSOUTPUT * params->display_dimension; i++){
            printf("%1.5f\t",y_output_host_display_pinned[s][i]);
            //reverse position from 1D array
            if(i > 0 && (i + 1) % params->display_dimension == 0){
                printf("\n");
            }
        }
        printf("\n");
    }
    stop_display = std::chrono::high_resolution_clock::now();
    auto duration_display = std::chrono::duration_cast<std::chrono::microseconds>(stop_display - start_display);

    stop_all = std::chrono::high_resolution_clock::now();
    auto duration_all = std::chrono::duration_cast<std::chrono::microseconds>(stop_all - start_all);

    printf("[GSL GPU] Time for transfer data from CPU to GPU: %ld micro seconds which is %f seconds\n",duration_transfer_h2d.count(),(duration_transfer_h2d.count()/1e6));
    printf("[GSL GPU] Time for compute %d ODE(s) with %d parameters, step %f in %f days on GPU: %ld micro seconds which is %f seconds\n",num_streams,params->dimension,params->h,params->t_target,duration_compute.count(),(duration_compute.count()/1e6));
    printf("[GSL GPU] Time for transfer data from GPU on CPU: %ld micro seconds which is %f seconds\n",duration_transfer_d2h.count(),(duration_transfer_d2h.count()/1e6));
    printf("[GSL GPU] Time for display %d ODE(s): %ld micro seconds which is %f seconds\n",num_streams,duration_display.count(),(duration_display.count()/1e6));
    printf("[GSL GPU] Time for complete %d ODE(s) with %d parameters: %ld micro seconds which is %f seconds\n",num_streams,params->dimension,duration_all.count(),(duration_all.count()/1e6));

    return;
}
