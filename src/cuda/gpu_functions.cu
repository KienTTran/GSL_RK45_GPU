#include "gpu_functions.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__
void rk45_gpu_adjust_h(double y[], double y_err[], double dydt_out[],
                             double &h, double h_0, int &adjustment_out, int final_step, const int index, GPU_Parameters* params){
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
        h_old = h;
    }

//    printf("    [adjust h] index = %d begin\n",index);
//    for (int i = 0; i < params->dimension; i ++)
//    {
//        printf("      y[%d] = %.10f\n",i,y[i]);
//    }
//    for (int i = 0; i < params->dimension; i ++)
//    {
//        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
//    }
//    for (int i = 0; i < params->dimension; i ++)
//    {
//        printf("      dydt_out[%d] = %.10f\n",i,dydt_out[i]);
//    }

    double r_max = 2.2250738585072014e-308;
    for (int i = 0; i < params->dimension; i ++)
    {
        const double D0 = eps_rel * (a_y * fabs(y[i]) + a_dydt * fabs((h_old) * dydt_out[i])) + eps_abs;
        const double r  = fabs(y_err[i]) / fabs(D0);
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
void rk45_gpu_step_apply(double t, double h, double y[], double y_err[], double dydt_out[],const int index, GPU_Parameters* params)
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

//    printf("    [step apply] index = %d start\n",index);
//    printf("      t = %.10f h = %.10f\n",t,h);

//    double* y_tmp = (double*)std::malloc(params->dimension);
//    double* k1 = (double*)std::malloc(params->dimension);
//    double* k2 = (double*)std::malloc(params->dimension);
//    double* k3 = (double*)std::malloc(params->dimension);
//    double* k4 = (double*)std::malloc(params->dimension);
//    double* k5 = (double*)std::malloc(params->dimension);
//    double* k6 = (double*)std::malloc(params->dimension);
    double y_tmp[DIM];
    double k1[DIM];
    double k2[DIM];
    double k3[DIM];
    double k4[DIM];
    double k5[DIM];
    double k6[DIM];

    for(int i = 0; i < params->dimension; i++){
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

//    for (int i = 0; i < params->dimension; i ++)
//    {
//        printf("      y[%d] = %.10f\n",i,y[i]);
//        printf("      y_tmp[%d] = %.10f\n",i,y_tmp[i]);
//        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
//        printf("      dydt_out[%d] = %.10f\n",i,dydt_out[i]);
//    }

    /* k1 */
    gpu_func(t,y,k1,params);
    for (int i = 0; i < params->dimension; i ++)
    {
//        printf("      k1[%d] = %.10f\n",i,k1[i]);
        y_tmp[i] = y[i] +  ah[0] * h * k1[i];
    }
    /* k2 */
    gpu_func(t + ah[0] * h, y_tmp,k2,params);
    for (int i = 0; i < params->dimension; i ++)
    {
//        printf("      k2[%d] = %.10f\n",i,k2[i]);
        y_tmp[i] = y[i] + h * (b3[0] * k1[i] + b3[1] * k2[i]);
    }
    /* k3 */
    gpu_func(t + ah[1] * h, y_tmp,k3,params);
    for (int i = 0; i < params->dimension; i ++)
    {
//        printf("      k3[%d] = %.10f\n",i,k3[i]);
        y_tmp[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i]);
    }
    /* k4 */
    gpu_func(t + ah[2] * h, y_tmp,k4,params);
    for (int i = 0; i < params->dimension; i ++)
    {
//        printf("      k4[%d] = %.10f\n",i,k4[i]);
        y_tmp[i] = y[i] + h * (b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] + b5[3] * k4[i]);
    }
    /* k5 */
    gpu_func(t + ah[3] * h, y_tmp,k5,params);
    for (int i = 0; i < params->dimension; i ++)
    {
//        printf("      k5[%d] = %.10f\n",i,k5[i]);
        y_tmp[i] = y[i] + h * (b6[0] * k1[i] + b6[1] * k2[i] + b6[2] * k3[i] + b6[3] * k4[i] + b6[4] * k5[i]);
    }
    /* k6 */
    gpu_func(t + ah[4] * h, y_tmp,k6,params);
    /* final sum */
    for (int i = 0; i < params->dimension; i ++)
    {
//        printf("      k6[%d] = %.10f\n",i,k6[i]);
        const double d_i = c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i];
        y[i] += h * d_i;
    }
    /* Derivatives at output */
    gpu_func(t + h, y, dydt_out,params);
    /* difference between 4th and 5th order */
    for (int i = 0; i < params->dimension; i ++)
    {
        y_err[i] = h * (ec[1] * k1[i] + ec[3] * k3[i] + ec[4] * k4[i] + ec[5] * k5[i] + ec[6] * k6[i]);
    }
    //debug printout
//    for (int i = 0; i < params->dimension; i++) {
//        printf("      index = %d y[%d] = %.10f\n",index,i,y[i]);
//    }
//    for (int i = 0; i < params->dimension; i++) {
//        printf("      index = %d y_err[%d] = %.10f\n",index,i,y_err[i]);
//    }
//    for (int i = 0; i < params->dimension; i++) {
//        printf("      index = %d dydt_out[%d] = %.10f\n",index,i,dydt_out[i]);
//    }
//    printf("    [step apply] index = %d end\n",index);
    return;
}

__global__
 void rk45_gpu_evolve_apply(double t, double t1, double h, double** y, GPU_Parameters* params){

    int index_gpu = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double device_t;
    double device_t1;
    double device_h;
    double device_h_0;
    int device_adjustment_out = 999;
    int device_final_step = 0;

//    double* device_y = (double*)std::malloc(params->dimension);
//    double* device_y_0 = (double*)std::malloc(params->dimension);
//    double* device_y_err = (double*)std::malloc(params->dimension);
//    double* device_dydt_out = (double*)std::malloc(params->dimension);

    double device_y[DIM];
    double device_y_0[DIM];
    double device_y_err[DIM];
    double device_dydt_out[DIM];

    for(int index = index_gpu; index < params->number_of_ode; index += stride){
        device_t1 = t1;
        device_t = t;
        device_h = h;

//        printf("    Will run from %f to %f, step %.10f\n", t, t1, h);
//        printf("    t = %.10f t_1 = %.10f  h = %.10f\n",device_t,device_t1,device_h);

        while(device_t < device_t1)
        {
            const double device_t_0 = device_t;
            device_h_0 = device_h;
            double device_dt = device_t1 - device_t_0;

//            printf("\n  [evolve apply] index = %d start\n",index);

//            printf("    t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f dt = %.10f\n",device_t,device_t_0,device_h,device_h_0,device_dt);
//            for (int i = 0; i < params->dimension; i ++){
//                printf("    y[%d][%d] = %.10f\n",index,i,y[index][i]);
//            }

            for (int i = 0; i < params->dimension; i ++){
                device_y[i] = y[index][i];
                device_y_0[i] = device_y[i];
            }

            device_final_step = 0;

            while(true)
            {
                if ((device_dt >= 0.0 && device_h_0 > device_dt) || (device_dt < 0.0 && device_h_0 < device_dt)) {
                    device_h_0 = device_dt;
                    device_final_step = 1;
                } else {
                    device_final_step = 0;
                }

                rk45_gpu_step_apply(device_t_0,device_h_0,device_y,device_y_err,device_dydt_out,index,params);
//                cudaDeviceSynchronize();

                if (device_final_step) {
                    device_t = device_t1;
                } else {
                    device_t = device_t_0 + device_h_0;
                }

                double h_old = device_h_0;

//                printf("    before adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",device_t,device_t_0,device_h,device_h_0,h_old);

                rk45_gpu_adjust_h(device_y, device_y_err, device_dydt_out,
                                        device_h, device_h_0, device_adjustment_out, device_final_step,index,params);
//                cudaDeviceSynchronize();

                //Extra step to get data from h
                device_h_0 = device_h;

//                printf("    after adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",device_t,device_t_0,device_h,device_h_0,h_old);

                if (device_adjustment_out == -1)
                {
                    double t_curr = (device_t);
                    double t_next = (device_t) + device_h_0;

                    if (fabs(device_h_0) < fabs(h_old) && t_next != t_curr) {
                        /* Step was decreased. Undo step, and try again with new h0. */
//                        printf("  [evolve apply] index = %d step decreased, y = y0\n",index);
                        for (int i = 0; i < params->dimension; i++) {
                            device_y[i] = device_y_0[i];
                        }
                    } else {
//                        printf("  [evolve apply] index = %d step decreased h_0 = h_old\n",index);
                        device_h_0 = h_old; /* keep current step size */
                        break;
                    }
                }
                else{
//                    printf("  [evolve apply] index = %d step increased or no change\n",index);
                    break;
                }
            }
            device_h = device_h_0;  /* suggest step size for next time-step */
            for (int i = 0; i < params->dimension; i++){
                y[index][i] = device_y[i];
            }
            t = device_t;
            h = device_h;
//            printf("    index = %d t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f\n",index,device_t,device_t_0,device_h,device_h_0);
//            for (int i = 0; i < params->dimension; i++){
//                printf("    index = %d y[%d][%d] = %.10f\n",index,index,i,device_y[i]);
//            }
//            if(device_final_step){
//                printf("[output] index = %d t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f\n",index,device_t,device_t_0,device_h,device_h_0);
//                for (int i = 0; i < params->dimension; i++){
//                    printf("[output] index = %d y[%d] = %.10f\n",index,i,device_y[i]);
//                }
//            }
//            printf("  [evolve apply] index = %d end\n",index);
//            cudaDeviceSynchronize();
        }
    }
    return;
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

int GPU_RK45::rk45_gpu_simulate(){

    auto start = std::chrono::high_resolution_clock::now();

    double t1 = 2.0;
    double t = 0.0;
    double h = 1e-6;
    double y_0 = 0.5;

    //CPU memory
    double **y = new double*[params->number_of_ode]();
    for (int i = 0; i < params->number_of_ode; i++)
    {
        y[i] = new double[params->dimension];
        for(int j = 0; j < params->dimension; j++){
            y[i][j] = 0.5;
        }
    }
    //GPU memory
    double **y_d = 0;
    //temp pointers
    double **tmp_ptr = (double**)malloc (params->number_of_ode * sizeof (double));
    for (int i = 0; i < params->number_of_ode; i++) {
        gpuErrchk(cudaMalloc ((void **)&tmp_ptr[i], params->dimension * sizeof (double)));
        gpuErrchk(cudaMemcpy(tmp_ptr[i], y[i], params->dimension * sizeof(double), cudaMemcpyHostToDevice));
    }
    gpuErrchk(cudaMalloc ((void **)&y_d, params->number_of_ode * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_d, tmp_ptr, params->number_of_ode * sizeof (double), cudaMemcpyHostToDevice));
    for (int i = 0; i < params->number_of_ode; i++) {
        gpuErrchk(cudaMemcpy (tmp_ptr[i], y[i], params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    }
    delete(y);
    delete(tmp_ptr);

    //Unified memory
//    double** y_d;
//    cudaMallocManaged((void**)&y_d, gpu_threads * sizeof(double*));
//    for (int i = 0; i < gpu_threads; i++) {
//        cudaMallocManaged((void**)&y_d[i], params->dimension * sizeof(double));
////        for (int j = 0; j < params->dimension; j++) {
////            y_d[i][j] = 0.5;
////        }
//    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %lld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));

    int num_SMs;
    int block_size = 128; //max is 1024
    int num_blocks = (params->number_of_ode + block_size - 1) / block_size;
    printf("[GSL GPU] block_size = %d num_blocks = %d\n",block_size,num_blocks);
    start = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
//    cudaProfilerStart();
    rk45_gpu_evolve_apply<<<num_blocks, block_size>>>(t1, t, h, y_d,params);

//    cudaProfilerStop();
    gpuErrchk(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for calculating %d ODE with %d parameters on GPU: %lld micro seconds which is %.10f seconds\n",params->number_of_ode,params->dimension,duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
    tmp_ptr = (double**)malloc (params->number_of_ode * sizeof (double));
    double** host_y_output = (double**)malloc (params->number_of_ode * sizeof (double));
    for (int i = 0; i < params->number_of_ode; i++) {
        host_y_output[i] = (double *)malloc (params->dimension * sizeof (double));
    }
    gpuErrchk(cudaMemcpy (tmp_ptr, y_d, params->number_of_ode * sizeof (double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < params->number_of_ode; i++) {
        gpuErrchk(cudaMemcpy (host_y_output[i], tmp_ptr[i], params->dimension * sizeof (double), cudaMemcpyDeviceToHost));
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for data transfer GPU to CPU: %lld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, params->number_of_ode); // define the range

    for(int i = 0; i < params->display_number; i++){
        int random_index = 0;
        if(params->number_of_ode > 1){
            //random_index = 0 + (rand() % static_cast<int>(gpu_threads - 0 + 1))
            random_index = distr(gen);
        }
        else{
            random_index = 0;
        }
        for(int index = 0; index < params->dimension; index++){
            //GPU memory
            printf("[GSL GPU] Thread %d y[%d][%d] = %.10f\n",random_index,random_index,index,host_y_output[random_index][index]);
            //unified memoery output
//            printf("[main] Thread %d y[%d][%d] = %.10f\n",random_index,random_index,index,y_d[random_index][index]);
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for get %d random results on CPU: %lld micro seconds which is %.10f seconds\n",params->display_number,duration.count(),(duration.count()/1e6));
    printf("\n");
    // Free memory
    delete(tmp_ptr);
    gpuErrchk(cudaFree(y_d));
    return 0;
}

void GPU_RK45::predict(double t0, double t1, double h, double ** y0, GPU_Parameters* params_d){
//    while (t0 < t1)
//    {
//        rk45_gpu_evolve_apply<<<1, 1>>>(t0, t1, h, y0, params_d);
//        gpuErrchk(cudaDeviceSynchronize());
//    }
}

void GPU_RK45::run(){
    auto start = std::chrono::high_resolution_clock::now();
    //GPU memory
    double **y_d = 0;
    //temp pointers
    double **tmp_ptr = (double**)malloc (params->number_of_ode * sizeof (double));
    for (int i = 0; i < params->number_of_ode; i++) {
        gpuErrchk(cudaMalloc ((void **)&tmp_ptr[i], params->dimension * sizeof (double)));
        gpuErrchk(cudaMemcpy(tmp_ptr[i], params->y[i], params->dimension * sizeof(double), cudaMemcpyHostToDevice));
    }
    gpuErrchk(cudaMalloc ((void **)&y_d, params->number_of_ode * sizeof (double)));
    gpuErrchk(cudaMemcpy (y_d, tmp_ptr, params->number_of_ode * sizeof (double), cudaMemcpyHostToDevice));
    for (int i = 0; i < params->number_of_ode; i++) {
        gpuErrchk(cudaMemcpy (tmp_ptr[i], params->y[i], params->dimension * sizeof (double), cudaMemcpyHostToDevice));
    }
    delete(tmp_ptr);

    //Unified memory
//    double** y_d;
//    cudaMallocManaged((void**)&y_d, gpu_threads * sizeof(double*));
//    for (int i = 0; i < gpu_threads; i++) {
//        cudaMallocManaged((void**)&y_d[i], params->dimension * sizeof(double));
////        for (int j = 0; j < params->dimension; j++) {
////            y_d[i][j] = 0.5;
////        }
//    }

    GPU_Parameters* params_d;
    gpuErrchk(cudaMalloc(&params_d, params->number_of_ode * sizeof(GPU_Parameters)));
    gpuErrchk(cudaMemcpy(params_d, params, params->number_of_ode * sizeof(GPU_Parameters), cudaMemcpyHostToDevice));

    int num_SMs;
    gpuErrchk(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
//    int numBlocks = 32*num_SMs; //multiple of 32
    int block_size = 128; //max is 1024
    int num_blocks = (params->number_of_ode + block_size - 1) / block_size;
    printf("[GSL GPU] block_size = %d num_blocks = %d\n",block_size,num_blocks);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for allocate mem CPU to GPU: %lld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
    while( params->t0 < params->t_target )
    {
        // integrate ODEs one day forward
//        predict( params->t0, params->t0 + 1.0, params->h, y_d, params_d);
        rk45_gpu_evolve_apply<<<num_blocks, block_size>>>(params->t0, params->t0 + 1.0, params->h, y_d, params_d);
        gpuErrchk(cudaDeviceSynchronize());
        // increment time by one day
        params->t0 += 1.0;
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("[GSL GPU] Time for compute %d ODE with %d parameters, step %.10f in %f days on GPU: %lld micro seconds which is %.10f seconds\n",params->number_of_ode,params->dimension,params->h,params->t_target,duration.count(),(duration.count()/1e6));
    return;
}