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

// this is the number of days of simulation that will be sent to standard output (and used for model fitting)
#define NUMDAYSOUTPUT 3650*2// use this to define "cycle" lengths
//#define NUMDAYSOUTPUT 520// use this to define "cycle" lengths

// number of locations
#define NUMLOC 3

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

// Two population sizes: main population and all outer populations have the same size
#define POPSIZE_MAIN 1000000.00 /* Remember to change BETA_OVER_POP_MAIN = 1 / POPSIZE_MAIN*/
#define POPSIZE_OUT 100000.00
/* This is to improve speed in the kernel.
 * Instead of beta = clo_beta / POPSIZE_MAIN,
 * using beta  = clo_beta * (1/POPSIZE_MAIN)
 * increased speed to double time
 * */
#define BETA_OVER_POP_MAIN 0.0000001

// GPU CONFIG
#define NUMODE 128// GPU Streams
#define DATADIM_ROWS 520
#define DATADIM_COLS 3
#define GPU_ODE_THREADS (NUMODE)
#define GPU_REDUCE_THREADS 1024
#define MCMC_ITER 10
#define SAMPLE_TAU_LENGTH 9
#define SAMPLE_PHI_0_INDEX 3
#define SAMPLE_PHI_LENGTH (SAMPLE_TAU_LENGTH + 1)
#define SAMPLE_LENGTH (NUMSEROTYPES + 1 + SAMPLE_TAU_LENGTH) /* beta + phi_0 + tau */

#endif //RK45_CUDA_FLU_DEFAULT_PARAMS_H
