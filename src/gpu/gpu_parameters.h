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
    double N[NUMLOC]; //Total population size
    double beta[NUMSEROTYPES] = {
                                 0.24, // transmission for Flu H1
                                 0.22, // transmission for Flu B
                                 0.26
                                }; //  transmission for Flu A/H3
    double sigma[NUMSEROTYPES] = {
                                 0.75, // sigma 12 Cross protection between H1 and B
                                 0.50, // sigma 13 Cross protection between H1 and H3
                                 0.75 // sigma 23 Cross protection between B and H3
                                 }; //  transmission for Flu A/H3
    double sigma2d[NUMSEROTYPES][NUMSEROTYPES];
    double eta[NUMLOC][NUMLOC];
    double nu_denom = 5; //  Duration of infection in days
    double amp = 0.07; //  Quantity of increase in betas during eidemic times
    double rho_denom = 730; //  Duration of immunity after infection in days
    double phi[10]; //  Timing (in days) of first epidemic peak
    double phi_0 = 175; //  Timing (in days) of first epidemic peak
    double tau[9] = {270, 420, 330, 280, 440, 300, 250, 330, 400};
    double epidur = 60; //  Duration of increased transmmission around each peak
    int phi_length = 0;
    static const int sample_length = 13;
    double sample[sample_length] {
            beta[0],
            beta[1],
            beta[2],
            phi[0],
            tau[0],
            tau[1],
            tau[2],
            tau[3],
            tau[4],
            tau[5],
            tau[6],
            tau[7],
            tau[8]
    };
    double sample_sd[sample_length] = {
            0.05, // for beta_H1
            0.05, // for beta_B
            0.05, // for beta_H3
            15, // for phi.1
            25, // for tau.1
            25, // for tau.2
            25, // for tau.3
            25, // for tau.4
            25, // for tau.5
            25, // for tau.6
            25, // for tau.7
            25, // for tau.8
            25 // for tau.9
    };
    double stf = 0.0;
    double trr = 0.0;
    double v_d_i_nu = 0.0;
    double v_d_i_amp = 0.0;
    double v_d_i_epidur_x2 = 0.0;
    double v_d_i_epidur_d2 = 0.0;
    double pi_x2 = 0.0;
};

class GPUParameters {
public:
    explicit GPUParameters();
    ~GPUParameters();
    int num_blocks;
    int block_size;
    int ode_dimension;
    int display_dimension;
    int agg_dimension;
    int data_dimension;
    CSVParameters data_params;
    Parameters flu_params;
    int display_number;
    double t_target;
    double t0;
    double h;
    double step;
    double** y_ode_input;
    double** y_data_input;
    double** y_ode_output;
    double** y_ode_agg;

    bool is_float( std::string myString);
    void init_from_cmd(int argc, char **argv);
    void init();
    void update();

    //from Flu
    std::vector<double> v;
    std::vector<double> phis;
    enum parameter_index {	i_phi01, i_phi02, i_phi03, i_phi04, i_phi05, i_phi06, i_phi07, i_phi08, i_phi09, i_phi10,        // the peak epidemic times
        i_phi11, i_phi12, i_phi13, i_phi14, i_phi15, i_phi16, i_phi17, i_phi18, i_phi19, i_phi20,
        i_amp, i_nu, i_epidur, i_immune_duration, num_params };


private:
};


#endif //RK45_CUDA_GPU_PARAMETERS_H
