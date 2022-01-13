//
// Created by kient on 1/12/2022.
//

#ifndef RK45_CUDA_CPU_PARAMETERS_H
#define RK45_CUDA_CPU_PARAMETERS_H
#include <cuda_runtime.h>
#include "flu_cpu_prms.h"
#include "flu_default_params.h"

class CPU_Parameters {
public:
    explicit CPU_Parameters();
    ~CPU_Parameters();
    int number_of_ode;
    int dimension;
    int display_number;
    double t_target;
    double t0;
    double h;
    double y[NUMODE][DIM];
    cpu_prms* ppc;
    int (* cpu_function) (double t, const double y[], double dydt[], void * params);
    bool isFloat( string myString);
    void ParseArgs(int argc, char **argv);
    void initPPC();
private:
};


#endif //RK45_CUDA_CPU_PARAMETERS_H
