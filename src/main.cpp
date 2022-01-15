#include "cuda/gpu_rk45.h"
#include "cpu_rk45.h"
#include <iostream>
#include <thread>
#include <chrono>
#include "cpu_parameters.h"
#include "cuda/gpu_rk45.h"

int ode_function(double t, const double y[], double dydt[], void* params){
    // 2 dim
    const double m = 5.2;		// Mass of pendulum
    const double g = -9.81;		// g
    const double l = 2;		// Length of pendulum
    const double A = 0.5;		// Amplitude of driving force
    const double wd = 1;		// Angular frequency of driving force
    const double b = 0.5;		// Damping coefficient

    dydt[0] = y[1];
    dydt[1] = -(g / l) * sin(y[0]) + (A * cos(wd * t) - b * y[1]) / (m * l * l);
    return 0;
}

int main(int argc, char* argv[])
{
    int threads = 1;
    int display = 1;
//    std::cout << "Enter number of ODE equations to run: " << std::endl;
//    std::cin >> threads;
//    std::cout << "Enter number of results to display randomly: " << std::endl;
//    std::cin >> display;
#ifdef ON_CLUSTER
    std::cout << "Running GSL on CPU" << std::endl;
    rk45_gsl_simulate(threads,display);
#endif
//    std::cout << "Running PEN on CPU" << std::endl;
//    CPU_RK45* cpu_rk45 = new CPU_RK45();
//    CPU_Parameters* cpu_params = new CPU_Parameters();
//    cpu_params->number_of_ode = 1;
//    cpu_params->dimension = 2;
//    cpu_params->initFlu(argc, argv);
//    cpu_params->display_number = display;
//    cpu_params->cpu_function = ode_function;
//    cpu_params->t_target = 2;
//    cpu_params->t0 = 0.0;
//    cpu_params->h = 0.2;
//    cpu_rk45->setParameters(cpu_params);
//    cpu_rk45->run();

//    std::cout << "Running FLU on CPU" << std::endl;
//    CPU_RK45* cpu_rk45 = new CPU_RK45();
//    CPU_Parameters* cpu_params = new CPU_Parameters();
//    cpu_params->number_of_ode = 1;
//    cpu_params->dimension = DIM;
//    cpu_params->initFlu(argc, argv);
//    cpu_params->display_number = display;
//    cpu_params->cpu_function = func;
//    cpu_params->t_target = 10;
//    cpu_params->t0 = 0.0;
//    cpu_params->h = 0.2;
//    cpu_rk45->setParameters(cpu_params);
//    cpu_rk45->run();
//
//    // 0 - R1
//    // 1 - R2
//    // 2 - R3
//    // 3 - I1
//    // 4 - I2
//    // 5 - I3
//    // 6 - J1
//    // 7 - J2
//    // 8 - J3
//    // 9 - S
//    std::string f[] = {"R1", "R2", "R3", "I1", "I2", "I3", "J1", "J2", "J3", "S"};
//    int loc,vir,stg;
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
//                    }
//                    printf("      end location %d\n",l);
//                }
//
//                // now add the term to dR/dt that accounts for the force of infection removing some R-individuals
//                printf("      f[%d] += ( -sum_foi ) * y[%d]\n",NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ,NUMSEROTYPES*NUMR*loc + NUMR*vir + stg);
//                printf("    end stage %d\n",stg);
//            }
//            printf("  end virus type %d\n",stg);
//        }
//        printf("end location %d\n",stg);
//    }
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
//                printf("    for location %d\n",loc);
//                printf("      foi_on_susc_single_virus += ppc->eta[%d][%d] * ppc->beta[%d] * y[%d]\n",loc,l,vir,STARTI + NUMSEROTYPES*l + vir);
//                printf("    end location %d\n",loc);
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
//            printf("  end virus type %d\n",stg);
//        }
//        printf("end location %d\n",stg);
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

//    std::cout << std::endl;
//    std::cout << "Performing test 1 on GPU" << std::endl;
////    test_cuda_1();
//    std::cout << std::endl;
//    std::cout << "Performing test 2 on GPU" << std::endl;
////    test_cuda_2();
//    std::cout << std::endl;

//    std::cout << "Running PEN on GPU" << std::endl;
//    GPU_RK45* gpu_rk45_pen = new GPU_RK45();
//    GPU_Parameters* gpu_params_pen = new GPU_Parameters();
//    gpu_params_pen->number_of_ode = 1;
//    gpu_params_pen->dimension = 2;
//    gpu_params_pen->initPen();
//    gpu_params_pen->display_number = display;
//    gpu_params_pen->t_target = 2.0;
//    gpu_params_pen->t0 = 0.0;
//    gpu_params_pen->h = 0.2;
//    gpu_rk45_pen->setParameters(gpu_params_pen);
//    gpu_rk45_pen->run();

//    std::cout << "Running FLU on GPU" << std::endl;
//    GPU_RK45* gpu_rk45_flu = new GPU_RK45();
//    GPU_Parameters* gpu_params_flu = new GPU_Parameters();
//    gpu_params_flu->number_of_ode = 1024;
//    gpu_params_flu->dimension = 16;
//    gpu_params_flu->initFlu(argc, argv);
//    gpu_params_flu->display_number = display;
//    gpu_params_flu->t_target = 1.0;
//    gpu_params_flu->t0 = 0.0;
//    gpu_params_flu->h = 1e-6;
//    gpu_rk45_flu->setParameters(gpu_params_flu);
//    gpu_rk45_flu->run();

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

    return 0;

}