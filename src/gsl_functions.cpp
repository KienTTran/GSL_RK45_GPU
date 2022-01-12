#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_odeiv.h>
#include <math.h>
#include <chrono>
#include <random>

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

#define DIMS 2

bool rk45_gsl_simulate(const int cpu_threads)
{
    auto start = std::chrono::high_resolution_clock::now();
    // Define GSL odeiv parameters
    const gsl_odeiv_step_type * step_type = gsl_odeiv_step_rkf45;
    gsl_odeiv_step * step = gsl_odeiv_step_alloc (step_type, DIMS);
    gsl_odeiv_control * control = gsl_odeiv_control_y_new (1e-6, 0.0);
    gsl_odeiv_evolve * evolve = gsl_odeiv_evolve_alloc (DIMS);
    gsl_odeiv_system sys = {odefunc, NULL, DIMS, NULL};

    double t[cpu_threads];
    double t_target[cpu_threads];
    double h[cpu_threads];
    //1 dimension
//    double y[1] = {0.5};
    //2 dimension
    double y[cpu_threads][DIMS];

    //Integration up to intervention
    int status(GSL_SUCCESS);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Time for allocate mem on CPU: %lld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < cpu_threads; i++){
        for(int j = 0; j < DIMS; j++){
            y[i][j] = 0.5;
        }
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
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Time for ODE on CPU: %lld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));

    start = std::chrono::high_resolution_clock::now();
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, cpu_threads); // define the range

    for(int i = 0; i < 10; i++) {
        int random_index = distr(gen);
        for(int index = 0; index < DIMS; index++){
            printf("thread %d y[%d][%d] = %.10f\n",random_index,random_index,index,y[random_index][index]);
        }
        printf("\n");
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Time for display random results on CPU: %lld micro seconds which is %.10f seconds\n",duration.count(),(duration.count()/1e6));

    return 0;
}