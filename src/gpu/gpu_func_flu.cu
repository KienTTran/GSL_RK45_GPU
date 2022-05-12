#include <cuda_runtime.h>
#include "gpu_flu.cuh"
#include "gpu_ode.cuh"


__device__
void gpu_func_flu(double t, const double y[], double f[], double stf, int index, FluParameters* flu_params){
    // everything will be indexed by location (loc), the infecting subtype/serotype (vir), and the stage of recovery (stg) in the R-classes
    int loc, vir, stg;

//    if((NUMODE == 1  || (index > 0 && index % (NUMODE / 2) == 0)) && t == 0){
//        printf("gpu_func_flu flu_params: \n");
//        printf("  beta1 = %1.9f \n", flu_params->beta[index*NUMSEROTYPES + 0]);
//        printf("  beta2 = %1.9f \n", flu_params->beta[index*NUMSEROTYPES + 1]);
//        printf("  beta3 = %1.9f \n", flu_params->beta[index*NUMSEROTYPES + 2]);
//        printf("  v_d_i_amp = %1.5f \n", flu_params->v_d_i_amp);
//        printf("  sigma_H1B = %1.5f \n", flu_params->sigma2d[0][1]);
//        printf("  sigma_BH3 = %1.5f \n", flu_params->sigma2d[1][2]);
//        printf("  sigma_H1H3 = %1.5f \n", flu_params->sigma2d[0][2]);
//        printf("  eta = %1.5f \n", flu_params->eta[0][0]);
//        printf("  eta = %1.5f \n", flu_params->eta[1][1]);
//        printf("  eta = %1.5f \n", flu_params->eta[2][2]);
//        printf("  trr = %1.5f \n", flu_params->trr);
//        printf("  v_d_i_nu = %1.5f \n", flu_params->v_d_i_nu);
//        printf("phis_length = %d\n",SAMPLE_PHI_LENGTH);
//        for(int i=0; i<SAMPLE_PHI_LENGTH; i++){
//            printf("  phi = %5.1f \n", flu_params->phi[index*SAMPLE_PHI_LENGTH + i]);
//        }
//    }

//    for(int j=0;j<DIM;j++) {
////        if(j == 0 || j == DIM -1)
//        {
//            printf("[function] IN y[%d] = %f \n",j,y[j]);
//        }
//    }
//    printf("\n");

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
                f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = - flu_params->trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ];

                // now add the rates of individuals coming in
                if( stg==0 )
                {
                    // if this is the first R-class, add the recovery term for individuals coming from I
                    f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += flu_params->v_d_i_nu * y[ STARTI + NUMSEROTYPES*loc + vir ];
                }
                else
                {
                    // if this is not the first R-class, add a simple transition from the previous R-stage
                    f[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] += flu_params->trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg - 1 ];
                }

                // now sum over all locations and serotypes to get the force of infection that is removing
                // individuals from this R-class
                double sum_foi = 0.0;
                for(int l=0; l<NUMLOC; l++)
                    for(int v=0; v<NUMSEROTYPES; v++)
                        sum_foi += flu_params->sigma2d[vir][v]
                                   * flu_params->beta[index*NUMSEROTYPES + v]
                                   * stf
                                   * flu_params->eta[loc][l]
                                   * y[ STARTI + NUMSEROTYPES*l + v ];

//                printf("index %d sum_foi = %f\n", NUMSEROTYPES*NUMR*loc + NUMR*vir + stg,sum_foi);
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
            for(int l=0; l<NUMLOC; l++) {
                foi_on_susc_single_virus +=
                        flu_params->eta[loc][l]
                        * stf
                        * flu_params->beta[index*NUMSEROTYPES + vir]
                        * y[STARTI + NUMSEROTYPES * l + vir];
//                printf("index %d foi_on_susc_single_virus += flu_params->eta[%d][%d]"
//                       " * stf"
//                       " * flu_params->beta[index*NUMSEROTYPES + %d]"
//                       " * y[%d] = %f\n",
//                       STARTI + NUMSEROTYPES*loc + vir, loc,l,vir,STARTI + NUMSEROTYPES*l + vir,foi_on_susc_single_virus);
//                printf("index %d loc %d vir %d l %d Y[%d] = %f\n",STARTI + NUMSEROTYPES*loc + vir,loc,vir,l,STARTI + NUMSEROTYPES * l + vir,y[STARTI + NUMSEROTYPES * l + vir]);
            }

//            printf("index %d foi_on_susc_single_virus = %f\n", STARTI + NUMSEROTYPES*loc + vir,foi_on_susc_single_virus);
//            printf("index %d loc %d vir %d Y[%d] = %f\n",STARTI + NUMSEROTYPES*loc + vir,loc,vir,STARTS + loc,y[STARTS + loc]);
            // add the in-flow of new infections from the susceptible class
            f[ STARTI + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus;
            f[ STARTJ + NUMSEROTYPES*loc + vir ] += y[ STARTS + loc ] * foi_on_susc_single_virus;

            // sum over locations and different types of recovered individuals to get the inflow of recovered
            // individuals that are becoming re-infected
            double inflow_from_recovereds = 0.0;
            for(int l=0; l<NUMLOC; l++){          // sum over locations
                for (int v = 0; v < NUMSEROTYPES; v++) { // sum over recent immunity
                    for (int s = 0; s < NUMR; s++) {       // sum over R stage
                        inflow_from_recovereds += flu_params->sigma2d[vir][v]
                                                  * stf
                                                  * flu_params->beta[index*NUMSEROTYPES + vir]
                                                  * flu_params->eta[loc][l]
                                                  * y[STARTI + NUMSEROTYPES * l + vir]
                                                  * y[NUMSEROTYPES * NUMR * loc + NUMR * v + s];
//                        printf("index = %d inflow_from_recovereds += inflow_from_recovereds_sbe = %f * y[%d] = %f * y[%d] = %f\n",STARTI + NUMSEROTYPES*loc + vir,flu_params->sigma2d[vir][v]
//                                                                                                                                                              * stf
//                                                                                                                                                              * flu_params->beta[index*NUMSEROTYPES + vir]
//                                                                                                                                                              * flu_params->eta[loc][l],
//                               STARTI + NUMSEROTYPES * l + vir,NUMSEROTYPES * NUMR * loc + NUMR * v + s,
//                                y[STARTI + NUMSEROTYPES * l + vir],y[NUMSEROTYPES * NUMR * loc + NUMR * v + s]);
                    }
                }
            }

//            printf("index %d inflow_from_recovereds = %f\n",STARTI + NUMSEROTYPES*loc + vir,inflow_from_recovereds);
            // add the in-flow of new infections from the recovered classes (all histories, all stages)
            f[ STARTI + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds;
            f[ STARTJ + NUMSEROTYPES*loc + vir ] += inflow_from_recovereds;

            // add the recovery rate - NOTE only for I-classes
            f[ STARTI + NUMSEROTYPES*loc + vir ] += - flu_params->v_d_i_nu * y[ STARTI + NUMSEROTYPES*loc + vir ];

        }
    }



    //
    // ###  4.  WRITE DOWN THE DERIVATIVES FOR ALL THE SUSCEPTIBLE CLASSES
    //


    for(loc=0; loc<NUMLOC; loc++)
    {
        // compute the force of infection of all viruses at all locations on the susceptibles at the location loc
        double foi_on_susc_all_viruses = 0.0;
        for(int l=0; l<NUMLOC; l++) {
            for (int v = 0; v < NUMSEROTYPES; v++) {
                foi_on_susc_all_viruses += flu_params->eta[loc][l] * stf * flu_params->beta[index*NUMSEROTYPES + v] *
                                           y[STARTI + NUMSEROTYPES * l + v];
//                printf(" loop l-v index %d loc %d foi_on_susc_all_viruses = %f\n",STARTS + loc,loc,foi_on_susc_all_viruses);
            }
        }


//        printf("index %d loc %d foi_on_susc_all_viruses = %f\n",STARTS + loc,loc,foi_on_susc_all_viruses);
//        printf("index %d loc %d y[%d] = %f\n",STARTS + loc,loc,STARTS + loc,y[STARTS + loc]);
        // add to ODE dS/dt equation the removal of susceptibles by all types of infection
        f[ STARTS + loc ] = ( - foi_on_susc_all_viruses ) * y[ STARTS + loc ];

//        printf("index %d loc %d f[%d] = %f\n",STARTS + loc,loc,STARTS + loc,f[STARTS + loc]);
        // now loop through all the recovered classes in this location (different histories, final stage only)
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
            // add to dS/dt the inflow of recovereds from the final R-stage
            f[ STARTS + loc ] += flu_params->trr * y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + (NUMR-1) ]; // "NUMR-1" gets you the final R-stage only
//            printf("loop vir index %d loc %d f[%d] = %f\n",STARTS + loc,loc,STARTS + loc,f[STARTS + loc]);
        }
    }

//    for(int j=0;j<DIM;j++) {
//        if(j == 0 || j == DIM -1)
//        {
//            printf("[function] OUT f[%d] = %f\n",j,f[j]);
//        }
//    }

    return;
}
