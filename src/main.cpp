#include "cuda/gpu_rk45.h"
#include "cpu_rk45.h"
#include <iostream>
#include <thread>
#include <chrono>
#include "cpu_parameters.h"
#include "cuda/gpu_rk45.h"

int main(int argc, char* argv[])
{
    int threads = 1;
    int display = 1;
#ifdef ON_CLUSTER
    std::cout << "Running GSL on CPU" << std::endl;
    rk45_gsl_simulate(threads,display);
#endif
    int loc,vir,stg;
    std::string f[] = {"0-R1a","0-R1b","0-R1c","0-R1d",
                       "0-R2a","0-R2b","0-R2c","0-R2d",
                       "0-R3a","0-R3b","0-R3c","0-R3d",
                       "1-R1a","1-R1b","1-R1c","1-R1d",
                       "1-R2a","1-R2b","1-R2c","1-R2d",
                       "1-R3a","1-R3b","1-R3c","1-R3d",
//                       "2-R1a","2-R1b","2-R1c","2-R1d",
//                       "2-R2a","2-R2b","2-R2c","2-R2d",
//                       "2-R3a","2-R3b","2-R3c","2-R3d",
//                       "3-R1a","3-R1b","3-R1c","3-R1d",
//                       "3-R2a","3-R2b","3-R2c","3-R2d",
//                       "3-R3a","3-R3b","3-R3c","3-R3d",
//                       "4-R1a","4-R1b","4-R1c","4-R1d",
//                       "4-R2a","4-R2b","4-R2c","4-R2d",
//                       "4-R3a","4-R3b","4-R3c","4-R3d",
                       "0-I1", "0-I2", "0-I3",
                       "1-I1", "1-I2", "1-I3",
//                       "2-I1", "2-I2", "2-I3",
//                       "3-I1", "3-I2", "3-I3",
//                       "4-I1", "4-I2", "4-I3",
                       "0-J1", "0-J2", "0-J3",
                       "1-J1", "1-J2", "1-J3",
//                       "2-J1", "2-J2", "2-J3",
//                       "3-J1", "3-J2", "3-J3",
//                       "4-J1", "4-J2", "4-J3",
                       "0-S",
                       "1-S",
//                       "2-S",
//                       "3-S",
//                       "4-S"
                        };

//    for(int i = 0; i < DIM ; i++){
//        if(i < STARTI){
//            printf("R -  i = %d f[%d] = %s\n",i,i,f[i].c_str());
//        }
//        else if(i < STARTJ){
//            printf("I -  i = %d f[%d] = %s\n",i,i,f[i].c_str());
//        }
//        else if(i < STARTS){
//            printf("J -  i = %d f[%d] = %s\n",i,i,f[i].c_str());
//        }
//        else{
//            printf("S -  i = %d f[%d] = %s\n",i,i,f[i].c_str());
//        }
//    }

//    int sum_foi_count = 0;
//    for(loc=0; loc<NUMLOC; loc++) //1
//    {
//        printf("location %d\n",loc);
//        for(vir=0; vir<NUMSEROTYPES; vir++) //3
//        {
//            printf("  virus type %d\n",vir);
//            for(stg=0; stg<NUMR; stg++) //1
//            {
//                printf("    stage %d\n",stg);
//                printf("      f index = %d, calculate %s\n",NUMSEROTYPES*NUMR*loc + NUMR*vir + stg,f[NUMSEROTYPES*NUMR*loc + NUMR*vir + stg].c_str());
//                // first add the rate at which individuals are transitioning out of the R class
//                printf("      f[%d] = -trr * y[%d]\n",NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ,NUMSEROTYPES*NUMR*loc + NUMR*vir + stg);
//
//                // now add the rates of individuals coming in
//                if( stg==0 ){
//                    printf("      if( stg==0 )\n");
//                    printf("        f[%d] += ppc->v[i_nu] * y[%d]\n",NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ,STARTI + NUMSEROTYPES*loc + vir);
//                }
//                else{
//                    printf("      else\n");
//                    printf("        f[%d] += trr * y[%d]\n",NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ,NUMSEROTYPES*NUMR*loc + NUMR*vir + stg - 1);
//                }
//
//                // now sum over all locations and serotypes to get the force of infection that is removing
//                // individuals from this R-class
//                double sum_foi = 0.0;
//                for(int l=0; l<NUMLOC; l++){
//                    printf("      for location %d\n",l);
//                    for(int v=0; v<NUMSEROTYPES; v++){
//                        printf("        for virus %d\n",v);
//                        printf("          sum_foi += ppc->sigma[%d][%d] * ppc->beta[%d] * ppc->eta[%d][%d] * y[%d]\n",vir,v,v,loc,l,STARTI + NUMSEROTYPES*l + v);
//                        printf("        end virus %d\n",v);
//                        sum_foi_count++;
//                    }
//                    printf("      end location %d\n",l);
//                }
//
//                // now add the term to dR/dt that accounts for the force of infection removing some R-individuals
//                printf("      f[%d] += ( -sum_foi ) * y[%d]\n",NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ,NUMSEROTYPES*NUMR*loc + NUMR*vir + stg);
//                printf("    end stage %d\n",stg);
//            }
//            printf("  end virus type %d\n",vir);
//        }
//        printf("end location %d\n",loc);
//    }
//
//
//    printf("\nstep 2\n");
//
//    for(loc=0; loc<NUMLOC; loc++)
//    {
//        printf("location %d\n",loc);
//        for(vir=0; vir<NUMSEROTYPES; vir++)
//        {
//            printf("  virus type %d\n",vir);
//            printf("    f I index = %d, calculate %s\n",STARTI + NUMSEROTYPES*loc + vir,f[STARTI + NUMSEROTYPES*loc + vir].c_str());
//            printf("    f J index = %d, calculate %s\n",STARTJ + NUMSEROTYPES*loc + vir,f[STARTJ + NUMSEROTYPES*loc + vir].c_str());
//
//            // initialize these derivatives to zero
//            printf("    f[%d] = 0.0\n",STARTI + NUMSEROTYPES*loc + vir);
//            printf("    f[%d] = 0.0\n",STARTJ + NUMSEROTYPES*loc + vir);
//
//            // sum over locations to get the force of infection of virus vir on susceptibles in location loc
//            printf("    foi_on_susc_single_virus = 0.0\n");
//            for(int l=0; l<NUMLOC; l++){
//                printf("    for location %d\n",l);
//                printf("      foi_on_susc_single_virus += ppc->eta[%d][%d] * ppc->beta[%d] * y[%d]\n",loc,l,vir,STARTI + NUMSEROTYPES*l + vir);
//                printf("    end location %d\n",l);
//            }
//
//            // add the in-flow of new infections from the susceptible class
//            printf("    f[%d] += y[%d] * foi_on_susc_single_virus\n",STARTI + NUMSEROTYPES*loc + vir,STARTS + loc);
//            printf("    f[%d] += y[%d] * foi_on_susc_single_virus\n",STARTJ + NUMSEROTYPES*loc + vir,STARTS + loc);
//
//            // sum over locations and different types of recovered individuals to get the inflow of recovered
//            // individuals that are becoming re-infected
//            printf("    inflow_from_recovereds = 0.0\n");
//            for(int l=0; l<NUMLOC; l++){
//                printf("    for location %d\n",l);
//                for(int v=0; v<NUMSEROTYPES; v++){
//                    printf("      for virus type %d\n",v);
//                    for(int s=0; s<NUMR; s++){
//                        printf("        for recovery stage %d\n",s);
//                        printf("          inflow_from_recovereds += ppc->sigma[%d][%d] * ppc->beta[%d] * ppc->eta[%d][%d] * y[%d] * y[%d]\n",vir,v,vir,loc,l,STARTI + NUMSEROTYPES*l + vir,NUMSEROTYPES*NUMR*loc + NUMR*v + s);
//                        printf("        end recovery stage %d\n",s);
//                    }
//                    printf("      end virus type %d\n",v);
//                }
//                printf("    end location %d\n",l);
//            }
//
//
//            // add the in-flow of new infections from the recovered classes (all histories, all stages)
//            printf("    f[%d] += inflow_from_recovereds\n",STARTI + NUMSEROTYPES*loc + vir);
//            printf("    f[%d] += inflow_from_recovereds\n",STARTJ + NUMSEROTYPES*loc + vir);
//
//            // add the recovery rate - NOTE only for I-classes
//            printf("    f[%d] += - ppc->v[i_nu] * y[%d]\n",STARTI + NUMSEROTYPES*loc + vir,STARTI + NUMSEROTYPES*loc + vir);
//            printf("  end virus type %d\n",vir);
//        }
//        printf("end location %d\n",loc);
//    }
//
//    printf("\nstep 3\n");
//
//    for(loc=0; loc<NUMLOC; loc++)
//    {
//        printf("location %d\n",loc);
//        printf("  f S index = %d, calculate %s\n",STARTS + loc,f[STARTS + loc].c_str());
//
//        printf("  foi_on_susc_all_viruses = 0.0\n");
//        for(int l=0; l<NUMLOC; l++){
//            printf("    for location %d\n",l);
//            for(int v=0; v<NUMSEROTYPES; v++){
//                printf("      for virus type %d\n",v);
//                printf("        foi_on_susc_all_viruses += ppc->eta[%d][%d] * ppc->beta[%d] * y[%d]\n",loc,l,v,STARTI + NUMSEROTYPES * l + v);
//                printf("      end virus type %d\n",v);
//            }
//            printf("    end location %d\n",l);
//        }
//        // add to ODE dS/dt equation the removal of susceptibles by all types of infection
//        printf("  f[%d] = ( - foi_on_susc_all_viruses ) * y[%d]\n",STARTS + loc,STARTS + loc);
//
//        // now loop through all the recovered classes in this location (different histories, final stage only)
//        for(int vir=0; vir<NUMSEROTYPES; vir++)
//        {
//            printf("    for virus type %d\n",vir);
//            // add to dS/dt the inflow of recovereds from the final R-stage
//            printf("      f[%d] += trr * y[%d]\n",STARTS + loc,NUMSEROTYPES*NUMR*loc + NUMR*vir + (NUMR-1));
//            printf("    end virus type %d\n",vir);
//        }
//        printf("location %d\n",loc);
//    }
//    printf("sum_foi_count = %d\n",sum_foi_count);

    std::cout << "Running TEST on GPU" << std::endl;
    GPU_RK45* gpu_rk45 = new GPU_RK45();
    GPU_Parameters* gpu_params_test = new GPU_Parameters();
    gpu_params_test->number_of_ode = 1;
    gpu_params_test->dimension = DIM;
    gpu_params_test->initTest(argc,argv);
    gpu_params_test->display_number = display;
    gpu_params_test->t_target = NUMDAYSOUTPUT;
    gpu_params_test->t0 = 0.0;
    gpu_params_test->h = 1e-6;
    gpu_rk45->setParameters(gpu_params_test);
    gpu_rk45->run();

    delete gpu_rk45;
    return 0;

}
