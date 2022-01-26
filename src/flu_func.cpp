#include "flu_default_params.h"
#include "cpu_parameters.h"
#include <chrono>
#include <random>

#define PI 3.141592653589793238

// this returns the seasonality factor -- will be 1.0 if it is outside the high transmission season
double seasonal_transmission_factor( double t);

//
//  "func" is the system of ordinary differential equations and it must be defined
//  with these four arguments for it to work in the GSL ODE routines
//
//  t is the current time, in days
//
//  y[] is the vector that holds the current value of all the state variables (S, E, I, R, etc) at time t
//
//  f[] is the vector that is populated in the function below
//
//
int func(double t, const double y[], double f[], void *params)
{
//    printf("        function start\n");
//    for(int i = 0; i < DIM; i++){
//        printf("          f[%d] = %.10f\n",i,f[i]);
//    }

    // just to be safe, cast the void-pointer to convert it to a prms-pointer
    CPU_Parameters* cpu_params = (CPU_Parameters*) params;

    // everything will be indexed by location (loc), the infecting subtype/serotype (vir), and the stage of recovery (stg) in the R-classes
    int loc, vir, stg;

    // force of infection
    // double foi = cpu_params->v[i_beta] * y[NUMR+2];

    // the transition rate among R-classes
    double trr = ((double)NUMR) / cpu_params->v[cpu_params->i_immune_duration];

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


    for(loc=0; loc<NUMLOC; loc++)
    {
        for(vir=0; vir<NUMSEROTYPES; vir++)
        {
            for(stg=0; stg<NUMR; stg++)
            {
                // first add the rate at which individuals are transitioning out of the R class
                f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = - trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ];

                // now add the rates of individuals coming in
                if( stg==0 )
                {
                    // if this is the first R-class, add the recovery term for individuals coming from I
                    f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += cpu_params->v[cpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];
                }
                else
                {
                    // if this is not the first R-class, add a simple transition from the previous R-stage
                    f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg - 1 ];
                }

                // now sum over all locations and serotypes to get the force of infection that is removing
                // individuals from this R-class
                double sum_foi = 0.0;
                for(int l=0; l<NUMLOC; l++)
                    for(int v=0; v<NUMSEROTYPES; v++)
                        sum_foi += cpu_params->sigma[vir][v] * cpu_params->beta[v] * cpu_params->eta[loc][l] * y[ STARTI + NUMSEROTYPES*l + v ];

                // now add the term to dR/dt that accounts for the force of infection removing some R-individuals
                f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += ( -sum_foi ) * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ];

            }
        }
    }


    //
    // ###  3.  WRITE DOWN THE DERIVATIVES FOR ALL THE INFECTED CLASSES and the J-CLASSES
    //

    for(loc=0; loc<NUMLOC; loc++)
    {
        for(vir=0; vir<NUMSEROTYPES; vir++)
        {
            // initialize these derivatives to zero
            f[ STARTI + NUMSEROTYPES*loc + vir ] = 0.0;
            f[ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;

            // sum over locations to get the force of infection of virus vir on susceptibles in location loc
            double foi_on_susc_single_virus = 0.0;
            for(int l=0; l<NUMLOC; l++)
                foi_on_susc_single_virus += cpu_params->eta[loc][l] * cpu_params->beta[vir] * y[ STARTI + NUMSEROTYPES*l + vir ];

            // add the in-flow of new infections from the susceptible class
            f[ STARTI + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus;
            f[ STARTJ + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus;

            // sum over locations and different types of recovered individuals to get the inflow of recovered
            // individuals that are becoming re-infected
            double inflow_from_recovereds = 0.0;
            for(int l=0; l<NUMLOC; l++)             // sum over locations
                for(int v=0; v<NUMSEROTYPES; v++)   // sum over recent immunity
                    for(int s=0; s<NUMR; s++)       // sum over R stage
                        inflow_from_recovereds += cpu_params->sigma[vir][v] * cpu_params->beta[vir] * cpu_params->eta[loc][l] * y[ STARTI + NUMSEROTYPES*l + vir ] * y[ NUMSEROTYPES*NUMR*loc + NUMR*v + s ];

            // add the in-flow of new infections from the recovered classes (all histories, all stages)
            f[ STARTI + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds;
            f[ STARTJ + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds;

            // add the recovery rate - NOTE only for I-classes
            f[ STARTI + NUMSEROTYPES*loc + vir ] += - cpu_params->v[cpu_params->i_nu] * y[ STARTI + NUMSEROTYPES*loc + vir ];

        }
    }



    //
    // ###  4.  WRITE DOWN THE DERIVATIVES FOR ALL THE SUSCEPTIBLE CLASSES
    //

    for(loc=0; loc<NUMLOC; loc++)
    {
        // compute the force of infection of all viruses at all locations on the susceptibles at the location loc
        double foi_on_susc_all_viruses = 0.0;
        for(int l=0; l<NUMLOC; l++)
            for(int v=0; v<NUMSEROTYPES; v++)
                foi_on_susc_all_viruses += cpu_params->eta[loc][l] * cpu_params->beta[v] * y[ STARTI + NUMSEROTYPES*l + v ];

        // add to ODE dS/dt equation the removal of susceptibles by all types of infection
        f[ STARTS + loc ] = ( - foi_on_susc_all_viruses ) * y[ STARTS + loc ];

        // now loop through all the recovered classes in this location (different histories, final stage only)
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
            // add to dS/dt the inflow of recovereds from the final R-stage
            f[ STARTS + loc ] += trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + (NUMR-1) ]; // "NUMR-1" gets you the final R-stage only
        }
    }

//    printf("        function end\n");
//    for(int i = 0; i < DIM; i++){
//        printf("          f[%d] = %.10f\n",i,f[i]);
//    }
    return 0;
}




