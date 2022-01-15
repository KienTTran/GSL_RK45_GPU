#include <cuda_runtime.h>
#include "gpu_rk45.h"

__host__ __device__
void seasonal_transmission_factor(GPU_Parameters* gpu_params, double t, double &factor )
{
    /*


        We're gonna make this thing go for 40 years. 30 years of burn in and 10 years of real modeling.
        We're creating a "10-year model cycle" and need the code below to find a time point's "place" in the "cycle"
        modulus (denoted with % in C++) only works with integers, so need the acrobatics below

     */

    // This is some code that's needed to create the 10-year "cycles" in transmission.

    if(gpu_params->phis_d == nullptr){
        factor = 1.0;
        return;
    }

    int x = (int)t; // This is now to turn a double into an integer
    double remainder = t - (double)x;
    int xx = x % 3650; // int xx = x % NUMDAYSOUTPUT;
    double yy = (double)xx + remainder;
    // put yy into the sine function, let it return the beta value
    t = yy;
    double sine_function_value = 0.0;

    int phis_length = sizeof(gpu_params->phis_d)/ sizeof(gpu_params->phis_d[0]);

    for(int i=0; i<phis_length; i++)
    {
        if( fabs( t - gpu_params->phis_d[i] ) < (gpu_params->v_d[gpu_params->i_epidur] / 2))
        {
            // sine_function_value = sin( 2.0 * 3.141592653589793238 * (phis[i]-t+91.25) / 365.0 );
            sine_function_value = sin( 2.0 * 3.141592653589793238 * (gpu_params->phis_d[i]-t+(gpu_params->v_d[gpu_params->i_epidur] / 2)) / (gpu_params->v_d[gpu_params->i_epidur] * 2));
            // printf("\n\t\t\t %1.3f %1.3f %1.3f \n\n", t, phis[i], sine_function_value );
        }
    }
    factor = 1.0 + gpu_params->v_d[gpu_params->i_amp] * sine_function_value;
    return;
}

__device__
void gpu_func_test(double t, const double y[], double f[],
                   double* sum_foi, double* foi_on_susc_single_virus,
                   double* inflow_from_recovereds, double* foi_on_susc_all_viruses,
                   int index, void *params){
    //    printf("gpu_function start\n");
    // just to be safe, cast the void-pointer to convert it to a prms-pointer

    GPU_Parameters* gpu_params = (GPU_Parameters*) params;

    // everything will be indexed by location (loc), the infecting subtype/serotype (vir), and the stage of recovery (stg) in the R-classes
    int loc, vir, stg;

    // force of infection
    // double foi = gpu_params->v_d[i_beta] * y[NUMR+2];

    // the transition rate among R-classes
    double trr = ((double)NUMR) / gpu_params->v_d[gpu_params->i_immune_duration];

    double stf = 0.0;

    //for(int k=0; k<DIM; k++) f[k] = 0.0;

    //
    // ###  1.  COMPUTE THE FORCES OF INFECTION (NOTE maybe this is not necessary)
    //

    // force of infection on location loc, on immune status i, by virus vir
    /*double foi_partial[NUMLOC][NUMSEROTYPES][NUMSEROTYPES];
    for(loc=0; loc<NUMLOC; loc++)
    {
        for(vir=0; vir<NUMSEROTYPES; vir++)
        {
            for(stg=0; stg<NUMR; stg++)
            {

            }
        }
    }*/


    //
    // ###  2.  WRITE DOWN THE DERIVATIVES FOR ALL THE RECOVERED CLASSES
    //
//    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//    for(loc=0; loc<NUMLOC; loc++)
//    {
//        for(vir=0; vir<NUMSEROTYPES; vir++)
//        {
//            for(stg=0; stg<NUMR; stg++)
//            {
//                if(index == NUMSEROTYPES*NUMR*loc + NUMR*vir + stg){
//                    // first add the rate at which individuals are transitioning out of the R class
//                    f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = - trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ];
//
//                    // now add the rates of individuals coming in
//                    if( stg == 0 )
//                    {
//                        // if this is the first R-class, add the recovery term for individuals coming from I
//                        f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += gpu_params->v_d[gpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];
//                    }
//                    else
//                    {
//                        // if this is not the first R-class, add a simple transition from the previous R-stage
//                        f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg - 1 ];
//                    }
//
//                    // now sum over all locations and serotypes to get the force of infection that is removing
//                    // individuals from this R-class
//
//                    sum_foi[index] = 0.0;
//                    for(int l=0; l<NUMLOC; l++) {
//                        for (int v = 0; v < NUMSEROTYPES; v++) {
//                            seasonal_transmission_factor(gpu_params,t,stf);
//                            sum_foi[index] += gpu_params->sigma[vir][v]
//                                       * stf
//                                       * gpu_params->beta[v] * gpu_params->eta[loc][l] * y[STARTI + NUMSEROTYPES * l + v];
//                        }
//                    }
//                    // now add the term to dR/dt that accounts for the force of infection removing some R-individuals
//                    f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += ( -sum_foi[index] ) * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ];
//                }
//            }
//        }
//    }
//    block.sync();

    //
    // ###  3.  WRITE DOWN THE DERIVATIVES FOR ALL THE INFECTED CLASSES and the J-CLASSES
    //
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    for(loc=0; loc<NUMLOC; loc++)
    {
        for(vir=0; vir<NUMSEROTYPES; vir++)
        {
            // initialize these derivatives to zero
            f[ STARTI + NUMSEROTYPES*loc + vir ] = 0.0;
            f[ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;

////            // sum over locations to get the force of infection of virus vir on susceptibles in location loc
//            foi_on_susc_single_virus[index] = 0.0;
//            for(int l=0; l<NUMLOC; l++){
//                seasonal_transmission_factor(gpu_params,t,stf);
//                foi_on_susc_single_virus[index] +=
//                        gpu_params->eta[loc][l]
//                        * stf
//                        * gpu_params->beta[vir] * y[STARTI + NUMSEROTYPES * l + vir];
//            }
//            // add the in-flow of new infections from the susceptible class
//            f[ STARTI + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus[index];
//            f[ STARTJ + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus[index];

//            // sum over locations and different types of recovered individuals to get the inflow of recovered
//            // individuals that are becoming re-infected
//            inflow_from_recovereds[index] = 0.0;
//            for(int l=0; l<NUMLOC; l++) {          // sum over locations
//                for (int v = 0; v < NUMSEROTYPES; v++) {  // sum over recent immunity
//                    for (int s = 0; s < NUMR; s++) {    // sum over R stage
//                        seasonal_transmission_factor(gpu_params,t,stf);
//                        inflow_from_recovereds[index] +=
//                                gpu_params->sigma[vir][v]
//                                * stf
//                                * gpu_params->beta[vir] * gpu_params->eta[loc][l] *
//                                y[STARTI + NUMSEROTYPES * l + vir] * y[NUMSEROTYPES * NUMR * loc + NUMR * v + s];
//                    }
//                }
//            }
//            // add the in-flow of new infections from the recovered classes (all histories, all stages)
//            f[ STARTI + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds[index];
//            f[ STARTJ + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds[index];

            // add the recovery rate - NOTE only for I-classes
            f[ STARTI + NUMSEROTYPES*loc + vir ] += - gpu_params->v_d[gpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];

        }
    }
    block.sync();
//    //
//    // ###  4.  WRITE DOWN THE DERIVATIVES FOR ALL THE SUSCEPTIBLE CLASSES
//    //
//    for(loc=0; loc<NUMLOC; loc++)
//    {
//        // compute the force of infection of all viruses at all locations on the susceptibles at the location loc
//        foi_on_susc_all_viruses[index] = 0.0;
//        for(int l=0; l<NUMLOC; l++) {
//            for (int v = 0; v < NUMSEROTYPES; v++) {
//                seasonal_transmission_factor(gpu_params,t,stf);
//                foi_on_susc_all_viruses[index] +=
//                        gpu_params->eta[loc][l]
//                        * stf
//                        * gpu_params->beta[v] * y[STARTI + NUMSEROTYPES * l + v];
//            }
//        }
//        // add to ODE dS/dt equation the removal of susceptibles by all types of infection
//        f[ STARTS + loc ] = ( - foi_on_susc_all_viruses[index] ) * y[ STARTS + loc ];
//
//        // now loop through all the recovered classes in this location (different histories, final stage only)
//        for(int vir=0; vir<NUMSEROTYPES; vir++)
//        {
//            // add to dS/dt the inflow of recovereds from the final R-stage
//            f[ STARTS + loc ] += trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + (NUMR-1) ]; // "NUMR-1" gets you the final R-stage only
//        }
//    }
    return;
}

//__host__ __device__
//void gpu_func_test(double t, const double y[], double f[], int index, void *params){
//    const double m = 5.2;		// Mass of pendulum
//    const double g = -9.81;		// g
//    const double l = 2;		// Length of pendulum
//    const double A = 0.5;		// Amplitude of driving force
//    const double wd = 1;		// Angular frequency of driving force
//    const double b = 0.5;		// Damping coefficient
//
////    printf("before y[%d] = %.10f f[%d] = %.10f\n",index,y[index],index,f[index]);
////    f[0] = y[1];
////    f[1] = -(g / l) * sin(y[0]) + (A * cos(wd * t) - b * y[1]) / (m * l * l);
//    f[index] = -(g / l) * sin(y[index]) + (A * cos(wd * t) - b * y[index]) / (m * l * l);
////    printf("after y[%d] = %.10f f[%d] = %.10f\n",index,y[index],index,f[index]);
//    return;
//}
