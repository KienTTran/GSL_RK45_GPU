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

#ifdef GSL_ENABLE
    bool rk45_gsl_simulate(const int cpu_threads,const int display_numbers);
#endif
bool rk45_cpu_simulate(int numbers);

#endif
