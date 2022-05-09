//
// Created by kient on 5/5/2022.
//

#ifndef GPU_FLU_FLU_PARAMETERS_CUH
#define GPU_FLU_FLU_PARAMETERS_CUH

#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <curand_kernel.h>
#include "../flu_default_params.h"

class Test{
public:
    double beta[3] = {0.1,0.2,0.3};
    double gamma = 0.4;
};

class FluParameters {
public:
    __device__ __host__ explicit FluParameters();
    ~FluParameters();
public:
    double N[NUMLOC]; //Total population size
    double beta[NUMODE][NUMSEROTYPES];
//    0.24, // transmission for Flu H1
//    0.22, // transmission for Flu B
//    0.26
    double sigma[NUMSEROTYPES] = {
            0.75, // sigma 12 Cross protection between H1 and B
            0.50, // sigma 13 Cross protection between H1 and H3
            0.75 // sigma 23 Cross protection between B and H3
    }; //  transmission for Flu A/H3
    const int phi_length = SAMPLE_PHI_LENGTH;
    double sigma2d[NUMSEROTYPES][NUMSEROTYPES];
    double eta[NUMLOC][NUMLOC];
    double nu_denom = 5; //  Duration of infection in days
    double amp = 0.07; //  Quantity of increase in betas during eidemic times
    double rho_denom = 730; //  Duration of immunity after infection in days
    double phi[SAMPLE_PHI_LENGTH]; //  Timing (in days) of first epidemic peak
    double phi_0 = 175; //  Timing (in days) of first epidemic peak
    double tau[SAMPLE_TAU_LENGTH] = {270, 420, 330, 280, 440, 300, 250, 330, 400};
    double trr = 0.0;
    double v_d_i_nu = 0.0;
    double v_d_i_amp = 0.0;
    double v_d_i_immune_duration = 0.0;
    double v_d_i_epidur = 0.0;
    double v_d_i_epidur_x2 = 0.0;
    double v_d_i_epidur_d2 = 0.0;
    double pi_x2 = 0.0;
    double beta_sd[NUMLOC] = {
            0.05, // for beta_H1
            0.05, // for beta_B
            0.05, // for beta_H3
    };
    double phi_sd = 15;
    double tau_sd[SAMPLE_TAU_LENGTH] = {
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

    //from Flu
    // global variables - CLO means it is a command-line option
    double G_CLO_BETA1[NUMODE];// = new double(1.20);
    double G_CLO_BETA2[NUMODE];// = new double(1.40);
    double G_CLO_BETA3[NUMODE];// = new double(1.60);
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
    std::vector<double> phis_h;
    enum parameter_index {	i_phi01, i_phi02, i_phi03, i_phi04, i_phi05, i_phi06, i_phi07, i_phi08, i_phi09, i_phi10,        // the peak epidemic times
        i_phi11, i_phi12, i_phi13, i_phi14, i_phi15, i_phi16, i_phi17, i_phi18, i_phi19, i_phi20,
        i_amp, i_nu, i_epidur, i_immune_duration, num_params };

    bool is_float( std::string myString);
//    void init_from_cmd(int argc, char **argv);
    void init();
//    __host__ __device__ void init_from_sample(double sample[]);
    __device__ void update(curandState curand_state_d);
};

#endif //GPU_FLU_FLU_PARAMETERS_CUH
