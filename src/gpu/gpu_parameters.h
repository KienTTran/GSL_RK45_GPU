//
// Created by kient on 1/12/2022.
//

#ifndef RK45_CUDA_GPU_PARAMETERS_H
#define RK45_CUDA_GPU_PARAMETERS_H
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <iostream>
#include <chrono>
#include "../flu_default_params.h"
#include "../csv/csv_data.h"

struct Parameters{
    double N = 1e6; //  Total population size
    double Na = 0.8 * 1e6; //  Subpopulation 1 (not used)
    double Nb = 0.2 * 1e6; //  Subpopulation 2 (not used)
    double beta_H1 = 0.24; //  transmission for Flu A/H1
    double beta_B = 0.22; //  transmission for Flu B
    double beta_H3 = 0.26; //  transmission for Flu A/H3
    double nu_denom = 5; //  Duration of infection in days
    double amp = 0.07; //  Quantity of increase in betas during eidemic times
    double rho_denom = 730; //  Duration of immunity after infection in days
    double sigma12 = 0.75; //  Cross protection between H1 and B
    double sigma13 = 0.50; //  Cross protection between H1 and H3
    double sigma23 = 0.75; //  Cross protection between B and H3
    double etaab = 0.01; //  Movement between subpopulations (not used)
    double etaba = 0.05; //  Movement between subpopulations (not used)
    std::vector<double> phi; //  Timing (in days) of first epidemic peak
    double phi_0 = 175; //  Timing (in days) of first epidemic peak
    std::vector<double> tau = {270, 420, 330, 280, 440, 300, 250, 330, 400};
    double epidur = 60; //  Duration of increased transmmission around each peak
};

class GPUParameters {
public:
    explicit GPUParameters();
    ~GPUParameters();
    int num_blocks;
    int block_size;
    int number_of_ode;
    int ode_dimension;
    int display_dimension;
    int agg_dimension;
    int data_dimension;
    CSVParameters data_params;
    Parameters default_prams;
    Parameters current_params;
    Parameters new_params;
    int display_number;
    double t_target;
    double t0;
    double h;
    double step;
    double** y_ode_input;
    double** y_data_input;
    double** y_ode_output;
    double** y_ode_agg;
    double seasonal_transmission_factor(double t);
    double sum_foi_sbe[NUMLOC * NUMSEROTYPES * NUMR * NUMLOC * NUMSEROTYPES];
    int sum_foi_y_index[NUMLOC * NUMSEROTYPES * NUMR * NUMLOC * NUMSEROTYPES];
    double inflow_from_recovereds_sbe[NUMLOC*NUMSEROTYPES*NUMLOC*NUMSEROTYPES*NUMR];
    int inflow_from_recovereds_y1_index[NUMLOC*NUMSEROTYPES*NUMLOC*NUMSEROTYPES*NUMR];
    int inflow_from_recovereds_y2_index[NUMLOC*NUMSEROTYPES*NUMLOC*NUMSEROTYPES*NUMR];
    double foi_on_susc_all_viruses_eb [NUMLOC*NUMLOC*NUMSEROTYPES];
    int foi_on_susc_all_viruses_y_index [NUMLOC*NUMLOC*NUMSEROTYPES];

    bool is_float( std::string myString);
    void init();
    void init_from_cmd(int argc, char **argv);

    //from Flu
    std::vector<double> v;           // this holds some of the parameters -- they are indexed by the enums above
//    thrust::host_vector<double> v;           // this holds some of the parameters -- they are indexed by the enums above
//    thrust::device_vector<double> v_temp;
//    double* v_d;

    double beta[NUMSEROTYPES];                  // the transmission parameter for each serotype

    double sigma[NUMSEROTYPES][NUMSEROTYPES];   // this is the symmetric sigma-matrix which tells you the susceptibility of
    // a person recently infected with serotype i to infection with serotype j

    double eta[NUMLOC][NUMLOC];                 // this is the non-symmetric eta-matrix which tells you the fraction of people in
    // location b (second index) that mix into location a (first index)

    double N[NUMLOC];                           // population size at each location

    thrust::host_vector<double> phis;                        // this is the list of peaktimes for the transmission parameter beta
    thrust::device_vector<double> phis_temp;
    double* phis_d;
    int phis_d_length;
    double stf = 0.0;
    double trr = 0.0;
    double v_d_i_nu = 0.0;
    double v_d_i_amp = 0.0;
    double v_d_i_epidur_x2 = 0.0;
    double v_d_i_epidur_d2 = 0.0;
    double pi_x2 = 0.0;

    enum parameter_index {	i_phi01, i_phi02, i_phi03, i_phi04, i_phi05, i_phi06, i_phi07, i_phi08, i_phi09, i_phi10,        // the peak epidemic times
        i_phi11, i_phi12, i_phi13, i_phi14, i_phi15, i_phi16, i_phi17, i_phi18, i_phi19, i_phi20,
        i_amp, i_nu, i_epidur, i_immune_duration, num_params };


private:
};


#endif //RK45_CUDA_GPU_PARAMETERS_H
