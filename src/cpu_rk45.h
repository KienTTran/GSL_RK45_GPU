#ifndef CPU_FUNCTION_H
#define CPU_FUNCTION_H
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <chrono>
#include "cpu_parameters.h"
#include "flu_default_params.h"

#ifdef ON_CLUSTER
    bool rk45_gsl_simulate(const int number_of_ode,const int display_numbers);
#endif

class CPU_RK45{
public:
    explicit CPU_RK45();
    ~CPU_RK45();
    int rk45_cpu_simulate();
    int rk45_cpu_evolve_apply(double& t, double t1, double& h, double y[]);
    int rk45_cpu_step_apply(double t, double h, double y[], double y_err[], double dydt_out[]);
    int rk45_cpu_adjust_h(double y[],double y_err[], double dydt_out[], double& h, int final_step, double h_0);
    void setParameters(CPU_Parameters* params);
    void predict(double t0, double t1, double* y0);
    void run();
private:
    CPU_Parameters* params;
};

#endif
