//
// Created by kient on 1/12/2022.
//

#ifndef RK45_CUDA_FLU_DEFAULT_PARAMS_H
#define RK45_CUDA_FLU_DEFAULT_PARAMS_H

#include <string>
#include <vector>
#include <assert.h>
#include <sstream>
#include <fstream>

namespace {
    // global variables - CLO means it is a command-line option
    double G_CLO_BETA1 = 1.20;
    double G_CLO_BETA2 = 1.40;
    double G_CLO_BETA3 = 1.60;

    double G_CLO_SIGMA12 = 0.70;
    double G_CLO_SIGMA13 = 0.30;
    double G_CLO_SIGMA23 = 0.70;

    double G_CLO_AMPL = 0.10;

    double G_CLO_NU_DENOM = 5;
    double G_CLO_RHO_DENOM = 900;

    double G_CLO_EPIDUR = 365;

    bool G_CLO_CHECKPOP_MODE = false;

//double G_CLO_OUTPUT_ALL_TRAJ = false;

    double G_CLO_ADJUST_BURNIN = -100.0; // add this number of days to the t0 variable (currently set at -3000.0)
    double G_CLO_ISS = 10.0; // this is the initial step size in the iterator

    std::string G_CLO_STR_PARAMSFILE;
    int G_CLO_INT_PARAMSFILE_INDEX = 0;

    bool G_PHIS_INITIALIZED_ON_COMMAND_LINE = false;
}
// number of locations
#define NUMLOC 1

// number OF types/subtypes of influenza (this will always be three - H1, H3, and B)
// for generality (and to avoid constantly having to specify type/subtype) we call this serotypes
#define NUMSEROTYPES 3

// number of R stages in the recovery class
#define NUMR 4

// the start index of the infected classes; the I-classes start after all of the R-classes
// have been listed
#define STARTI (NUMLOC*NUMSEROTYPES*NUMR) // 9

// the start index of the cumulative infected classes, i.e. the J-classes
#define STARTJ (STARTI + (NUMLOC*NUMSEROTYPES)) // 12

// the start index of the suceptible (S) classes; there will be one of these for every location
#define STARTS (STARTJ + (NUMLOC*NUMSEROTYPES)) // 15

// this is the dimensionality of the ODE system
#define DIM (STARTS+NUMLOC) // 16

//#define DIM 2

// this is the number of days of simulation that will be sent to standard output (and used for model fitting)
#define NUMDAYSOUTPUT 3650*2// use this to define "cycle" lengths
//#define NUMDAYSOUTPUT 3650// use this to define "cycle" lengths
//#define NUMDAYSOUTPUT 520// use this to define "cycle" lengths

//#define NUMDAYSOUTPUT 2// use this to define "cycle" lengths

#define NUMODE 1// GPU Streams
#define DATADIM_ROWS 520
#define DATADIM_COLS 3


// Two population sizes: main population and all outer populations have the same size
#define POPSIZE_MAIN 1000000.00
#define POPSIZE_OUT 100000.00

//
//
// this function contains the ode system
//
int func(double t, const double y[], double f[], void *params);

// void* jac;	// do this for C-compilation
//
// for C++ compilation we are replacing the void* declaration above with
// the inline dummy declaration below
inline int jac(double a1, const double* a2, double* a3, double* a4, void* a5)
{
    return 0;
};

inline double popsum( double yy[] )
{
    double sum=0.0;
    for(int i=0; i<DIM; i++) sum += yy[i];

    for(int i=STARTJ; i<STARTJ+NUMLOC*NUMSEROTYPES; i++) sum -= yy[i];

    return sum;
}


#endif //RK45_CUDA_FLU_DEFAULT_PARAMS_H
