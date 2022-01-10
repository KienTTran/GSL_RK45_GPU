//
// Created by kient on 1/7/2022.
//
#include "cpu_functions.h"

int function(double t, const double y[], double dydt[], const int dim){
    // 2 dim
    const double m = 5.2;		// Mass of pendulum
    const double g = -9.81;		// g
    const double l = 2;		// Length of pendulum
    const double A = 0.5;		// Amplitude of driving force
    const double wd = 1;		// Angular frequency of driving force
    const double b = 0.5;		// Damping coefficient

    dydt[0] = y[1];
    dydt[1] = -(g / l) * sin(y[0]) + (A * cos(wd * t) - b * y[1]) / (m * l * l);
    return 0;
}

int rk45_gsl_cpu_adjust_h(double y[],double y_err[], double dydt_out[], double* h, const int dim,
           double eps_abs, double eps_rel, double a_y, double a_dydt, unsigned int ord, double scale_abs[]){
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
    const double S = 0.9;
    const double h_old = *h;

    double r_max = DBL_MIN;

    printf("    [adjust h] begin\n");
    for (int i = 0; i < dim; i ++)
    {
        printf("      y[%d] = %.10f\n",i,y[i]);
    }
    for (int i = 0; i < dim; i ++)
    {
        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
    }
    for (int i = 0; i < dim; i ++)
    {
        printf("      dydt_out[%d] = %.10f\n",i,dydt_out[i]);
    }

    for(int i=0; i<dim; i++) {
        const double D0 = eps_rel * (a_y * fabs(y[i]) + a_dydt * fabs(h_old * dydt_out[i])) + eps_abs * scale_abs[i];
        const double r  = fabs(y_err[i]) / fabs(D0);
        printf("      compare r = %.10f r_max = %.10f\n",r,r_max);
        r_max = std::max(r, r_max);
    }

    printf("      r_max = %.10f\n",r_max);

    if(r_max > 1.1) {
        /* decrease step, no more than factor of 5, but a fraction S more
           than scaling suggests (for better accuracy) */
        double r =  S / pow(r_max, 1.0/ord);

        if (r < 0.2)
            r = 0.2;

        *h = r * h_old;

        printf("      decrease by %.10f, h_old is %.10f new h is %.10f\n", r, h_old, *h);
        printf("    [adjust h] end\n");
        return -1;
    }
    else if(r_max < 0.5) {
        /* increase step, no more than factor of 5 */
        double r = S / pow(r_max, 1.0/(ord+1.0));

//        printf("  r = %.10f\n",r);

        if (r > 5.0)
            r = 5.0;

        if (r < 1.0)  /* don't allow any decrease caused by S<1 */
            r = 1.0;

        *h = r * h_old;

        printf("      increase by %.10f, h_old is %.10f new h is %.10f\n", r, h_old, *h);
        printf("    [adjust h] end\n");
        return 1;
    }
    else {
        /* no change */
        printf("  no change\n");
        printf("    [adjust h] end\n");
        return 0;
    }
}

int rk45_gsl_cpu_step_apply(double t, double h, double y[], double y_err[], double dydt_out[], const int dim){
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
    double* y_tmp = new double[dim]();
    double* k1 = new double[dim]();
    double* k2 = new double[dim]();
    double* k3 = new double[dim]();
    double* k4 = new double[dim]();
    double* k5 = new double[dim]();
    double* k6 = new double[dim]();

    printf("    [step apply] start\n");
    printf("      t = %.10f h = %.10f\n",t,h);
    for (int i = 0; i < dim; i ++)
    {
        printf("      y[%d] = %.10f\n",i,y[i]);
        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
        printf("      dydt_out[%d] = %.10f\n",i,dydt_out[i]);
    }

    /* k1 */
    bool s = function(t,y,k1,dim);
    if(s != 0) return s;
    for (int i = 0; i < dim; i++){
        printf("      k1[%d] = %.10f\n",i,k1[i]);
        y_tmp[i] = y[i] +  ah[0] * h * k1[i];
    }
    /* k2 */
    s = function(t + ah[0] * h, y_tmp,k2,dim);
    if(s != 0) return s;
    for (int i = 0; i < dim; i++){
        printf("      k2[%d] = %.10f\n",i,k2[i]);
        y_tmp[i] = y[i] + h * (b3[0] * k1[i] + b3[1] * k2[i]);
    }
    /* k3 */
    s = function(t + ah[1] * h, y_tmp,k3,dim);
    if(s != 0) return s;
    for (int i = 0; i < dim; i++){
        printf("      k3[%d] = %.10f\n",i,k3[i]);
        y_tmp[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i]);
    }
    /* k4 */
    s = function(t + ah[2] * h, y_tmp,k4,dim);
    if(s != 0) return s;
    for (int i = 0; i < dim; i++){
        printf("      k4[%d] = %.10f\n",i,k4[i]);
        y_tmp[i] = y[i] + h * (b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] + b5[3] * k4[i]);
    }
    /* k5 */
    s = function(t + ah[3] * h, y_tmp,k5,dim);
    if(s != 0) return s;
    for (int i = 0; i < dim; i++){
        printf("      k5[%d] = %.10f\n",i,k5[i]);
        y_tmp[i] = y[i] + h * (b6[0] * k1[i] + b6[1] * k2[i] + b6[2] * k3[i] + b6[3] * k4[i] + b6[4] * k5[i]);
    }
    /* k6 */
    s = function(t + ah[4] * h, y_tmp,k6,dim);
    if(s != 0) return s;
    for (int i = 0; i < dim; i++){
        printf("      k6[%d] = %.10f\n",i,k6[i]);
        y_tmp[i] = y[i] + h * (b6[0] * k1[i] + b6[1] * k2[i] + b6[2] * k3[i] + b6[3] * k4[i] + b6[4] * k5[i]);
    }
    /* final sum */
    for (int i = 0; i < dim; i++){
        const double d_i = c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i];
        y[i] += h * d_i;
    }
    /* Derivatives at output */
    s = function(t + h, y, dydt_out,dim);
    if(s != 0) return s;
    /* difference between 4th and 5th order */
    for (int i = 0; i < dim; i++){
        y_err[i] = h * (ec[1] * k1[i] + ec[3] * k3[i] + ec[4] * k4[i] + ec[5] * k5[i] + ec[6] * k6[i]);
    }
    for (int i = 0; i < dim; i++) {
        printf("      y[%d] = %.10f\n",i,y[i]);
    }
    for (int i = 0; i < dim; i++) {
        printf("      y_err[%d] = %.10f\n",i,y_err[i]);
    }
    for (int i = 0; i < dim; i++) {
        printf("      dydt_out[%d] = %.10f\n",i,dydt_out[i]);
    }
    printf("    [step apply] end\n");
    return 0;
}

int rk45_gsl_cpu_evolve_apply(double* t, double t1, double *h, double y[], const int dim,
                              double eps_abs, double eps_rel, double a_y, double a_dydt, unsigned int ord, double scale_abs[]){
    const double t_0 = *t;
    double h_0 = *h;
    int step_status;
    int final_step = 0;
    double dt = t1 - t_0;  /* remaining time, possibly less than h */
    double* y0 = new double[dim]();
    double* y_err = new double[dim]();
    double* dydt_out = new double[dim]();

    printf("  [evolve apply] start\n");

    for (int i = 0; i < dim; i++) {
        y0[i] = y[i];
    }

    int h_adjust_status;
    while(true){
        if ((dt >= 0.0 && h_0 > dt) || (dt < 0.0 && h_0 < dt))
        {
            h_0 = dt;
            final_step = 1;
        }
        else
        {
            final_step = 0;
        }

        step_status = rk45_gsl_cpu_step_apply(t_0, h_0, y, y_err, dydt_out, dim);

        if (step_status != 0)
        {
            *h = h_0;  /* notify user of step-size which caused the failure */
            *t = t_0;  /* restore original t value */
            return step_status;
        }

        if (final_step)
        {
            *t = t1;
        }
        else
        {
            *t = t_0 + h_0;
        }

        double h_old = h_0;

        printf("    after adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",*t,t_0,*h,h_0,h_old);
        
        h_adjust_status = rk45_gsl_cpu_adjust_h(y, y_err, dydt_out, &h_0, dim, eps_abs, eps_rel, a_y, a_dydt, ord, scale_abs);

        printf("    after adjust t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f h_old = %.10f\n",*t,t_0,*h,h_0,h_old);

        if (h_adjust_status == -1)
        {
            double t_curr = *t;
            double t_next = (*t) + h_0;

            if (fabs(h_0) < fabs(h_old) && t_next != t_curr)
            {
                /* Step was decreased. Undo step, and try again with new h_0. */
                printf("  [evolve apply] step decreased, y = y0\n");
                for (int i = 0; i < dim; i++) {
                    y[i] = y0[i];
                }
            }
            else
            {
                printf("  [evolve apply] step decreased h_0 = h_old\n");
                h_0 = h_old; /* keep current step size */
                break;
            }
        }
        else{
            printf("  [evolve apply] step increased or no change\n");
            break;
        }
    }
    *h = h_0;  /* suggest step size for next time-step */
    printf("    t = %.10f t_0 = %.10f  h = %.10f h_0 = %.10f dt = %.10f\n",*t,t_0,*h,h_0,dt);
    printf("    ");
    for (int i = 0; i < dim; i++){
        printf("y[%d] = %.10f\t",i,y[i]);
    }
    printf("\n");
    printf("  [evolve apply] end\n");
    return step_status;
}

bool rk45_gsl_cpu_simulate(){
//    const int dim = 1; //1 dim
    const int dim = 2; //2 dim

    //Default parameters for RK45 in GSL
    double eps_abs = 1e-6;
    double eps_rel = 0.0;
    double a_y = 1.0;
    double a_dydt = 0.0;
    unsigned int ord = 5;
    double scale_abs[dim];
    std::fill_n(scale_abs, dim, 1.0);
    //End default parameters for RK45

    double* y = new double[dim]();
    y[0] = 0.5; // 1 dim
    y[1] = 0.5; // 2 dim

    double t = 0.0;
    double t1 = 2.0;
    double h = 0.2;

//    auto start_cpu = std::chrono::high_resolution_clock::now();
    int step_count = 0;
    while(t < t1){
        printf ("\n[main cpu] step %d\n", step_count);
        rk45_gsl_cpu_evolve_apply(&t, t1, &h, y, dim, eps_abs, eps_rel, a_y, a_dydt, ord, scale_abs);
        printf ("[main cpu] step %d t = %.10f \t  h = %.10f\n", step_count, t, h);
        for (int i = 0; i < dim; i++){
            printf("\t y = %.10f",y[i]);
        }
        printf("\n");
        step_count++;
    }
//    auto stop_cpu = std::chrono::high_resolution_clock::now();
//    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);
//    printf("cpu time: %d micro seconds which is %.10f seconds\n",duration_cpu.count(),(duration_cpu.count()/1e6));
    return true;
}