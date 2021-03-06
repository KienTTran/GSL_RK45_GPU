//
// Created by kient on 1/12/2022.
//

#ifndef RK45_CUDA_CPU_PARAMETERS_H
#define RK45_CUDA_CPU_PARAMETERS_H
#include "flu_default_params.h"
#include <cstring>
#include <cmath>

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
    double **y;
    int (* cpu_function) (double t, const double y[], double dydt[], void * params);
    bool isFloat( std::string myString);
    void initFlu(int argc, char **argv);

    std::vector<double> v;           // this holds some of the parameters -- they are indexed by the enums above

    double beta[NUMSEROTYPES];                  // the transmission parameter for each serotype

    double sigma[NUMSEROTYPES][NUMSEROTYPES];   // this is the symmetric sigma-matrix which tells you the susceptibility of
    // a person recently infected with serotype i to infection with serotype j

    double eta[NUMLOC][NUMLOC];                 // this is the non-symmetric eta-matrix which tells you the fraction of people in
    // location b (second index) that mix into location a (first index)

    double N[NUMLOC];                           // population size at each location

    std::vector<double> phis;                        // this is the list of peaktimes for the transmission parameter beta

    // auxiliary functions for this class
    double seasonal_transmission_factor( double t );

    enum parameter_index {	i_phi01, i_phi02, i_phi03, i_phi04, i_phi05, i_phi06, i_phi07, i_phi08, i_phi09, i_phi10,        // the peak epidemic times
        i_phi11, i_phi12, i_phi13, i_phi14, i_phi15, i_phi16, i_phi17, i_phi18, i_phi19, i_phi20,
        i_amp, i_nu, i_immune_duration, num_params };

    typedef enum parameter_index prm_index;
private:
};


#endif //RK45_CUDA_CPU_PARAMETERS_H
