//
// Created by kient on 1/12/2022.
//

#include <cmath>
#include "GPU_Parameters.h"
#include "flu_default_params.h"

GPU_Parameters::GPU_Parameters(){
    dimension = 0;
    number_of_ode = 0;
    t_target = 0.0;
    t0 = 0.0;
    h = 1e-6;
}

GPU_Parameters::~GPU_Parameters(){
    dimension = 0;
    number_of_ode = 0;
    t_target = 0.0;
    t0 = 0.0;
    h = 1e-6;
}

bool GPU_Parameters::isFloat( std::string myString ) {
    std::istringstream iss(myString);
    double f;
    iss >> std::noskipws >> f; // noskipws considers leading whitespace invalid
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail();
}

void GPU_Parameters::initPen(){
    y = new double[dimension]();
    for(int j = 0; j < dimension; j++){
        y[j] = 0.5;
    }
}

void GPU_Parameters::initTest(int argc, char **argv){
    y = new double[dimension]();
    for(int j = 0; j < dimension; j++){
        y[j] = 0.5;
    }

    //Todo check this before change output format, added 2 field for time and seasonal factor
    display_dimension = dimension + 2;
    y_output = new double[static_cast<int>(t_target) * display_dimension]();
    for(int j = 0; j < t_target * display_dimension; j++){
        y_output[j] = 0.0;
    }

    v.insert( v.begin(), num_params, 0.0 );
    assert( v.size()==num_params );

    //dim = num_params;

    for(int i=0; i<NUMSEROTYPES; i++)
    {
        for(int j=0; j<NUMSEROTYPES; j++)
        {
            sigma[i][j] = 0.0; // this initializes the system with full cross-immunity among serotypes
        }
    }

    for(int i=0; i<NUMLOC; i++)
    {
        for(int j=0; j<NUMLOC; j++)
        {
            if(i==j) eta[i][j] = 1.0;  // the diagonal elements have to be 1.0, as a population mixes fully with itself
            if(i!=j) eta[i][j] = 0.0;  // these are initialized to 0.0 indicating that the different sub-populations do not mix at all
        }
    }

    // set initial values for the population sizes, 1 million for first location and 100K for others
    N[0] = POPSIZE_MAIN;
    for(int i=1; i<NUMLOC; i++) N[i] = POPSIZE_OUT;

    //Parse begin
    std::string str;
    int i, start;
    i=1;    // this is the position where you start reading command line options
    // you skip "i=0" which is the text string "odesim"

    /*if( argc<start )
    {
        PrintUsageModes();
        exit(-1);
    }*/


    // read in options from left to right
    while(i<argc)
    {
        str = argv[i]; // read the ith text string into the variable "str"

        //BEGIN MAIN IF-BLOCK BELOW

        // ### 1 ### IF BLOCK FOR PHI
        if( str == "-phi" )
        {
            phis.clear();
            i++;

            //BEGIN LOOPING THROUGH THE LIST OF BETAS
            while(i<argc)
            {
                std::string s( argv[i] );    // convert argv[i] into a normal string object
                if( isFloat(s) )        // if the current string is a floating point number, write it into the phis array
                {
                    // if the command line argument is <0, just set it back to zero
                    double d = atof( argv[i] );
                    if( d < 0.0 ){
                        d = 0.0;
                        //TODO print warning here and probably should exit
                        fprintf(stderr, "\n\n \t Don't make phis less than zero! \n\n");
                    }

                    phis.push_back( d );

                    // increment and move on in this sub-loop
                    i++;
                }
                else
                {
                    // if the current string is NOT a float, set the counter back down (so the string can be handled by the next if block
                    // and break out of the loop
                    i--;
                    break;
                }

            }
            //END OF LOOPING THROUGH THE LIST OF BETAS

            // make sure at least one phi-value was read in
            if( phis.size() == 0 )
            {
                fprintf(stderr,"\n\n\tWARNING : No phi-values were read in after the command-line option \"-phi\".\n\n");
            }
        }
            // ### 2 ### BLOCKS FOR FOR THE OTHER NON-PHI COMMAND-LINE OPTIONS
        else if( str == "-checkpop" )
        {
            G_CLO_CHECKPOP_MODE = true;
        }
        else if( str == "-beta1" )    {     G_CLO_BETA1 = atof( argv[++i] );        }
        else if( str == "-beta2" )    {     G_CLO_BETA2 = atof( argv[++i] );        } //atof is character to floating point
        else if( str == "-beta3" )    {     G_CLO_BETA3 = atof( argv[++i] );        } //atoi changes it to integer
        else if( str == "-sigma12" )    {   G_CLO_SIGMA12 = atof( argv[++i] );      } //atoi changes it to integer
        else if( str == "-sigma13" )    {   G_CLO_SIGMA13 = atof( argv[++i] );      }
        else if( str == "-sigma23" )    {   G_CLO_SIGMA23 = atof( argv[++i] );      }
        else if( str == "-amp" )        {   G_CLO_AMPL = atof( argv[++i] );         }
        else if( str == "-nu_denom")     {   G_CLO_NU_DENOM = atof( argv[++i] );     }
        else if( str == "-rho_denom")    {   G_CLO_RHO_DENOM = atof( argv[++i]);      }
        else if( str == "-epidur")      {   G_CLO_EPIDUR = atof( argv[++i]);        }


        else
        {
            fprintf(stderr, "\n\tUnknown option [%s] on command line.\n\n", argv[i]);
            exit(-1);
        }
        //END OF MAIN WHILE-LOOP BLOCK; INCREMENT AND MOVE ON TO THE NEXT COMMAND-LINE ARGUMENT

        // increment i so we can look at the next command-line option
        i++;
    }
    //Parse end

    //
    // ###  3.  INITIALIZE PARAMETERS - these are the default/starting values
    //

    // if the phi-parameters are not initialized on the command line
    if( !G_PHIS_INITIALIZED_ON_COMMAND_LINE )
    {
        for(int i=0;  i<10; i++) v[i] = ((double)i)*365.0 + 240.0; // sets the peak epidemic time in late August for the first 10 years
        for(int i=10; i<20; i++) v[i] = -99.0;
    }

    v[ i_amp ]   = G_CLO_AMPL;
    beta[0] = G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
    beta[1] = G_CLO_BETA2 / POPSIZE_MAIN;
    beta[2] = G_CLO_BETA3 / POPSIZE_MAIN;

    sigma[0][1] = G_CLO_SIGMA12; // 0.7; // the level of susceptibility to H1 if you've had B
    sigma[1][0] = G_CLO_SIGMA12; // 0.7; // and vice versa

    sigma[1][2] = G_CLO_SIGMA23; // 0.7; // the level of susceptibility to H3 if you've had B
    sigma[2][1] = G_CLO_SIGMA23; // 0.7; // and vice versa

    sigma[0][2] = G_CLO_SIGMA13; // 0.3; // the level of susceptibility to H3 if you've had H1
    sigma[2][0] = G_CLO_SIGMA13; // 0.3; // and vice versa

    sigma[0][0] = 0;
    sigma[1][1] = 0;
    sigma[2][2] = 0;

    v[ i_nu ]    = 1 / G_CLO_NU_DENOM;                // recovery rate
    v[ i_immune_duration ] = G_CLO_RHO_DENOM;    // 2.5 years of immunity to recent infection'

    v[ i_epidur ] = G_CLO_EPIDUR;

//    fprintf(stderr, "Here's the info on params: \n");
//    fprintf(stderr, "beta1 = %1.9f \n", beta[0]);
//    fprintf(stderr, "beta2 = %1.9f \n", beta[1]);
//    fprintf(stderr, "beta3 = %1.9f \n", beta[2]);
//    fprintf(stderr, "a = %1.3f \n", v[i_amp]);
//    fprintf(stderr, "sigma_H1B = %1.3f \n", sigma[0][1]);
//    fprintf(stderr, "sigma_BH3 = %1.3f \n", sigma[1][2]);
//    fprintf(stderr, "sigma_H1H3 = %1.3f \n", sigma[0][2]);
//     for (int i = 0; i<phis.size(); i++) {
//        fprintf(stderr, "phi = %5.1f \n", phis[i]);
//     }


    //
    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
    //

    // declare the vector y that holds the values of all the state variables (at the current time)
    // below you are declaring a vector of size DIM

    for(int loc=0; loc<NUMLOC; loc++)
    {
        // put half of the individuals in the susceptible class
        y[ STARTS + loc ] = 0.5 * N[loc];

        // put small number (but slightly different amounts each time) of individuals into the infected classes
        // double r = rand() % 50 + 10;
        //double x = r / 1000.0; // double x = 0.010;
        double x = 0.010;
        double sumx = 0.0;
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
            double r = rand() % 50 + 1;
            x = r / 1000.0;

            if (vir == 0) { x = 35 / 1000.0; }
            if (vir == 1) { x = 25 / 1000.0; }
            if (vir == 2) { x = 21 / 1000.0; }

            // fprintf(stderr, "r = %1.4f, x = %1.6f", r, x);

            sumx += x;
            y[ STARTI + NUMSEROTYPES*loc + vir ] = x * N[loc];
            y[ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;     // initialize all of the J-variables to zero

            x += 0.001;
        }
        x=0.010; // reset x

        // distribute the remainder of individuals into the different recovered stages equally
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
            double z = (0.5 - sumx)/((double)NUMR*NUMSEROTYPES);  // this is the remaining fraction of individuals to be distributed
            for(int stg=0; stg<NUMR; stg++)
            {
                y[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = z * N[loc];
            }
        }
    }

//    printf("v size = %d\n",v.size());
//    printf("phis size = %d\n",phis.size());
    //copy cpu vector to gpu vector
    v_temp = v;
    v_d = thrust::raw_pointer_cast(v_temp.data());

    printf("phis size = %lu\n",phis.size());

    if(phis.empty()){
        phis_d = nullptr;
        phis_d_length = 0;
    }
    else{
        phis_temp = phis;
        phis_d_length = phis.size();
        phis_d = thrust::raw_pointer_cast(phis_temp.data());
        for(int t = 0; t < static_cast<int>(t_target); t++){
            stf_d[t] = seasonal_transmission_factor(t);
        }
    }

    printf("\nsum_foi_sbe:\n");
    //sum_foi
//    m0,m1,.. are dimensions
//    A(i,j,k,...) -> A0[i + j*m0 + k*m0*m1 + ...]
    int index = 0;
    for(int loc = 0; loc < NUMLOC; loc++) {
        for (int vir = 0; vir < NUMSEROTYPES; vir++) {
            for (int stg = 0; stg < NUMR; stg++) {
                for(int l = 0; l < NUMLOC; l++) {
                    for (int v = 0; v < NUMSEROTYPES; v++) {
                        index = loc*NUMSEROTYPES*NUMR*NUMLOC*NUMSEROTYPES + vir*NUMR*NUMLOC*NUMSEROTYPES + stg*NUMLOC*NUMSEROTYPES + l*NUMSEROTYPES + v;
                        sum_foi_sbe[index] = sigma[vir][v]
                                                * beta[v]
                                                * eta[loc][l];
                        printf("loc = %d vir = %d stg = %d l = %d v = %d index = %d sum_foi[%d] = %f\n",loc,vir,stg,l,v,index,index,sum_foi_sbe[index]);
                    }
                }
            }
        }
    }
    printf("\ninflow_from_recovereds_sbe:\n");
    //inflow_from_recovereds_sbe
    index = 0;
    for(int loc = 0; loc < NUMLOC; loc++) {
        for (int vir = 0; vir < NUMSEROTYPES; vir++) {
            for(int l=0; l<NUMLOC; l++) {        // sum over locations
                for (int v = 0; v < NUMSEROTYPES; v++) { // sum over recent immunity
                    for (int s = 0; s < NUMR; s++) {    // sum over R stage
                        index = loc*NUMSEROTYPES*NUMLOC*NUMSEROTYPES*NUMR + vir*NUMLOC*NUMSEROTYPES*NUMR + l*NUMSEROTYPES*NUMR + v*NUMR + s;
                        inflow_from_recovereds_sbe[index] = sigma[vir][v]
                                                            * beta[vir]
                                                            * eta[loc][l] ;
                        printf("loc = %d vir = %d l = %d v = %d s = %d index = %d inflow_from_recovereds[%d] = %f\n",loc,vir,l,v,s,index,index,inflow_from_recovereds_sbe[index]);
                    }
                }
            }

        }
    }
    printf("\nfoi_on_susc_all_viruses_eb:\n");
    index = 0;
    for(int loc = 0; loc < NUMLOC; loc++) {
        for(int l=0; l<NUMLOC; l++){
            for(int v=0; v<NUMSEROTYPES; v++){
                index = loc*NUMLOC*NUMSEROTYPES + l*NUMSEROTYPES + v;
                foi_on_susc_all_viruses_eb[index] =    eta[loc][l]
                                                    * beta[v];
                printf("loc = %d vir = %d index = %d foi_on_susc_all_viruses[%d] = %f\n",loc,v,index,index,foi_on_susc_all_viruses_eb[index]);
            }
        }
    }

    // Copy host_vector H to device_vector D
//    printf("copy cpu data to gpu\n");
}

double GPU_Parameters::seasonal_transmission_factor(double t)
{
    /*


        We're gonna make this thing go for 40 years. 30 years of burn in and 10 years of real modeling.
        We're creating a "10-year model cycle" and need the code below to find a time point's "place" in the "cycle"
        modulus (denoted with % in C++) only works with integers, so need the acrobatics below

     */

    // This is some code that's needed to create the 10-year "cycles" in transmission.

    if(phis.size() == 0){
        return 1.0;
    }

    int x = (int)t; // This is now to turn a double into an integer
    double remainder = t - (double)x;
    int xx = x % 3650; // int xx = x % NUMDAYSOUTPUT;
    double yy = (double)xx + remainder;
    // put yy into the sine function, let it return the beta value
    t = yy;
    double sine_function_value = 0.0;

    for(int i=0; i<phis.size(); i++)
    {
        if( std::fabs( t - phis[i] ) < (v[i_epidur] / 2))
        {
            // sine_function_value = sin( 2.0 * 3.141592653589793238 * (phis[i]-t+91.25) / 365.0 );
            sine_function_value = std::sin( 2.0 * 3.141592653589793238 * (phis[i] - t +(v[i_epidur] / 2)) / (v[i_epidur] * 2));
//            printf("      in loop %1.3f %d  %1.3f %1.3f\n", t, i, phis[i], sine_function_value );
        }
    }
//    printf("    %f sine_function_value %1.3f\n",t,sine_function_value);
//    printf("    %f return %1.3f\n",t,1.0 + v[i_amp] * sine_function_value);
    return 1.0 + v[i_amp] * sine_function_value;
}
