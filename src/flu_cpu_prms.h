#ifndef PRMS
#define PRMS

#include "flu_default_params.h"

#include <vector>
#include <math.h>
//#include <gsl/gsl_errno.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_odeiv.h>
//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_rng.h> // random number generators from Gnu Scientific Library
//#include <gsl/gsl_cdf.h>
//#include <gsl/gsl_randist.h>  // this includes the gsl_ran_gaussian function


using namespace std;

// the list below is all of the parameters that the model allows
//
// there are currently 20 peak times allowed, phi01 to phi20
//
// the parameterization assumes there are three types/subtypes or three serotypes
//
// the parameterization allows for four locations currently, with location 1 being the central location
// and locations 2, 3, 4 being separate rural locations connected to the central (urban) location

//extern gsl_rng *G_RNG;

class cpu_prms
{
public:    
    explicit cpu_prms();    // constructor
    ~cpu_prms();         	// destructor

    //int dim;		            // the number of parameters

    vector<double> v;           // this holds some of the parameters -- they are indexed by the enums above

    double beta[NUMSEROTYPES];                  // the transmission parameter for each serotype
    
    double sigma[NUMSEROTYPES][NUMSEROTYPES];   // this is the symmetric sigma-matrix which tells you the susceptibility of
                                                // a person recently infected with serotype i to infection with serotype j
                                                
    double eta[NUMLOC][NUMLOC];                 // this is the non-symmetric eta-matrix which tells you the fraction of people in 
                                                // location b (second index) that mix into location a (first index)
                                                
    double N[NUMLOC];                           // population size at each location
    
    vector<double> phis;                        // this is the list of peaktimes for the transmission parameter beta
    
    // these are pointers to GSL ODE structures that integrate the ODE system
//    gsl_odeiv_step* 	os;
//    gsl_odeiv_control* 	oc;
//    gsl_odeiv_evolve*	oe;
//
    
    // auxiliary functions for this class
    double seasonal_transmission_factor( double t );

    enum parameter_index {	i_phi01, i_phi02, i_phi03, i_phi04, i_phi05, i_phi06, i_phi07, i_phi08, i_phi09, i_phi10,        // the peak epidemic times
        i_phi11, i_phi12, i_phi13, i_phi14, i_phi15, i_phi16, i_phi17, i_phi18, i_phi19, i_phi20,
        i_amp, i_nu, i_immune_duration, num_params };

    typedef enum parameter_index prm_index;
};

#endif // PRMS
