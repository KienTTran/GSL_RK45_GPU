#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_odeiv.h>
#include <math.h>
#include <chrono>

int odefunc (double t, const double y[], double dydt[], void *params)
{
    //1 dimension
//    dydt[0] = y[0] - std::pow(t,2) + 1;

    //2 dimentsion
    const double m = 5.2;		// Mass of pendulum
    const double g = -9.81;		// g
    const double l = 2;		// Length of pendulum
    const double A = 0.5;		// Amplitude of driving force
    const double wd = 1;		// Angular frequency of driving force
    const double b = 0.5;		// Damping coefficient
    const int dimension = 2;	// Dimension of system
    dydt[0] = y[1];
    dydt[1] = -(g / l) * sin(y[0]) + (A * cos(wd * t) - b * y[1]) / (m * l * l);
//    printf("function dydt[0] = %.10f\n",dydt[0]);
//    printf("function dydt[1] = %.10f\n",dydt[1]);
    return GSL_SUCCESS;
}


int * jac;

#define thread 200000

int rk45_gsl_simulate ()
{
    int dim = 2;

//    printf("\n=====ODEIV=====\n");
    // Define GSL odeiv parameters
    const gsl_odeiv_step_type * step_type = gsl_odeiv_step_rkf45;
    gsl_odeiv_step * step = gsl_odeiv_step_alloc (step_type, dim);
    gsl_odeiv_control * control = gsl_odeiv_control_y_new (1e-6, 0.0);
    gsl_odeiv_evolve * evolve = gsl_odeiv_evolve_alloc (dim);
    gsl_odeiv_system sys = {odefunc, NULL, dim, NULL};

    double t[thread];
    double t_target[thread];
    double h[thread];
    //1 dimension
//    double y[1] = {0.5};
    //2 dimension
    double y[thread][2];

    //Integration up to intervention
    int status(GSL_SUCCESS);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < thread; i++){
        y[i][0] = 0.5;
        y[i][1] = 0.5;
        t[i] = 0.0;
        t_target[i] = 2.0;
        h[i] = 0.2;
        while (t[i] < t_target[i]) {
            status = gsl_odeiv_evolve_apply (evolve,control,step,&sys,&t[i],t_target[i],&h[i],y[i]);
            //1 dimension
//            printf ("after evolve: %.10f %.10f\n", h, y[0]);
            //2 dimension
            if (status != GSL_SUCCESS) {
                break;
            }
        }
//        printf("%d t = %.10f h = %.10f y[0] = %.10f y[1] = %.10f\n",i,t[i], h[i], y[i][0], y[i][1]);
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);
    printf("cpu time: %lld micro seconds which is %.10f seconds\n",duration_cpu.count(),(duration_cpu.count()/1e6));

//    for (double t_target = t+t_step; t_target < tc; t_target += t_step ) {
//        while (t < t_target) {
//            status = gsl_odeiv_evolve_apply (evolve,control,step,&sys,&t,t_target,&h,y);
//            if (status != GSL_SUCCESS) {
//                break;
//            }
//        }
//        printf ("%.8f %.8f\n", x, y[0]);
//    }

//    printf("\n=====ODEIV2=====\n");

//    gsl_odeiv2_system sys2 = {odefunc, NULL, dim, NULL};
//    gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new (&sys2, gsl_odeiv2_step_rkf45, 0.2, 1e-6, 0.0);
//    int i;
//    double x0 = 0.0,  xf = 2.0; /* start and end of integration interval */
//    double x = x0;
//    double y2[1] = { 0.5  };  /* initial value */
//    for (i = 1; i <= 10; i++)
//    {
//        double xi = x0 + i * (xf-x0) / 10.0;
//        int status = gsl_odeiv2_driver_apply (d, &x, xi, y2);
//
//        if (status != GSL_SUCCESS)
//        {
//            printf ("error, return value=%d\n", status);
//            break;
//        }
//
//        printf ("%.8f %.8f\n", x, y[0]);
//    }
//    gsl_odeiv2_driver_free (d);
    return 0;
}