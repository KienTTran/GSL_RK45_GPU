//
// Created by kient on 1/12/2022.
//

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
    y = new double*[number_of_ode]();
    for(int i = 0; i < number_of_ode; i++){
        y[i] = new double[dimension];
        for(int j = 0; j < dimension; j++){
            y[i][j] = 0.0;
        }
    }
}

void GPU_Parameters::initTest(int argc, char **argv){
    y_test = new double[dimension]();
    for(int j = 0; j < dimension; j++){
        y_test[j] = 0.5;
    }

    std::string str;
    int i, start;
    i=1;    // this is the position where you start reading command line options
    // you skip "i=0" which is the text string "odesim"

    /*if( argc<start )
    {
        PrintUsageModes();
        exit(-1);
    }*/

    v.insert( v.begin(), num_params, 0.0 );
    assert( v.size() == num_params );
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
                    if( d < 0.0 ) d = 0.0;
                    //TODO print warning here and probably should exit

                    phis.push_back( d );

                    // increment and move on in this sub-loop
                    i++;
                }
                else
                {
                    // if the current string is NOT a double, set the counter back down (so the string can be handled by the next if block
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
        else if( str == "-sigma12" )    {     sigma[0][1] = atof( argv[++i] );        } //atoi changes it to integer
        else if( str == "-sigma13" )    {     sigma[0][2] = atof( argv[++i] );        } //atoi changes it to integer
        else if( str == "-sigma23" )    {     sigma[1][2] = atof( argv[++i] );        } //atoi changes it to integer
        else if( str == "-amp" )    {     v[i_amp] = atof( argv[++i] );        } //atoi changes it to integer
        else if( str == "-nu" )    {     v[i_nu] = atof( argv[++i] );        }
        else if( str == "-rho_denom" )    {     v[i_immune_duration] = atof( argv[++i] );        }
        else
        {
            fprintf(stderr, "\n\tUnknown option [%s] on command line.\n\n", argv[i]);
            exit(-1);
        }
        //END OF MAIN WHILE-LOOP BLOCK; INCREMENT AND MOVE ON TO THE NEXT COMMAND-LINE ARGUMENT

        // increment i so we can look at the next command-line option
        i++;
    }

    if (sigma[0][1] > 1) {
        fprintf(stderr,"\n\n\tWARNING : Sigma can't be over 1. %1.3f\n\n", sigma[0][1]); // %1.3f is a placeholder for what is being printed
        exit(-1);
    }

    //
    // ###  3.  INITIALIZE PARAMETERS - these are the default/starting values
    //

    // if the phi-parameters are not initialized on the command line
    if( !G_PHIS_INITIALIZED_ON_COMMAND_LINE )
    {
        for(int i=0;  i<10; i++) v[i] = ((double)i)*365.0 + 240.0; // sets the peak epidemic time in late August for the first 10 years
        for(int i=10; i<20; i++) v[i] = -99.0;
    }

    v[ i_amp ]   = 0.1;
    beta[0] = G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
    beta[1] = G_CLO_BETA2 / POPSIZE_MAIN;
    beta[2] = G_CLO_BETA3 / POPSIZE_MAIN;

    sigma[0][1] = 0.7; // the level of susceptibility to H1 if you've had B
    sigma[1][0] = 0.7; // and vice versa

    sigma[1][2] = 0.7; // the level of susceptibility to H3 if you've had B
    sigma[2][1] = 0.7; // and vice versa

    sigma[0][2] = 0.3; // the level of susceptibility to H3 if you've had H1
    sigma[2][0] = 0.3; // and vice versa

    sigma[0][0] = 0;
    sigma[1][1] = 0;
    sigma[2][2] = 0;

    v[ i_nu ]    = 0.2;                // recovery rate
    v[ i_immune_duration ] = 900.0;    // 2.5 years of immunity to recent infection

    //
    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
    //

    // declare the vector y that holds the values of all the state variables (at the current time)
    // below you are declaring a vector of size DIM

    //init already
//    double y[DIM];

    for(int loc=0; loc<NUMLOC; loc++)
    {
        // put half of the individuals in the susceptible class
        y_test[ STARTS + loc ] = 0.5 * N[loc];

        // put small number (but slightly different amounts each time) of individuals into the infected classes
        double x = 0.010;
        double sumx = 0.0;
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
            sumx += x;
            y_test[ STARTI + NUMSEROTYPES*loc + vir ] = x * N[loc];
            y_test[ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;     // initialize all of the J-variables to zero

            x += 0.001;
        }
        x=0.010; // reset x

        // distribute the remainder of individuals into the different recovered stages equally
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
            double z = (0.5 - sumx)/((double)NUMR*NUMSEROTYPES);  // this is the remaining fraction of individuals to be distributed
            for(int stg=0; stg<NUMR; stg++)
            {
                y_test[ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = z * N[loc];
            }
        }
    }

    //copy cpu vector to gpu vector
    v_temp = v;
    v_d = thrust::raw_pointer_cast(v_temp.data());
    phis_temp = phis;
    phis_d = thrust::raw_pointer_cast(phis_temp.data());

    // Copy host_vector H to device_vector D
    printf("copy cpu vector to gpu vector\n");
}
void GPU_Parameters::initFlu(int argc, char **argv){
    std::string str;
    int i, start;
    i=1;    // this is the position where you start reading command line options
    // you skip "i=0" which is the text string "odesim"

    /*if( argc<start )
    {
        PrintUsageModes();
        exit(-1);
    }*/

    v.insert( v.begin(), num_params, 0.0 );
    assert( v.size() == num_params );
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
            if( d < 0.0 ) d = 0.0;
            //TODO print warning here and probably should exit

            phis.push_back( d );

            // increment and move on in this sub-loop
            i++;
          }
          else
          {
            // if the current string is NOT a double, set the counter back down (so the string can be handled by the next if block
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
      else if( str == "-sigma12" )    {     sigma[0][1] = atof( argv[++i] );        } //atoi changes it to integer
      else if( str == "-sigma13" )    {     sigma[0][2] = atof( argv[++i] );        } //atoi changes it to integer
      else if( str == "-sigma23" )    {     sigma[1][2] = atof( argv[++i] );        } //atoi changes it to integer
      else if( str == "-amp" )    {     v[i_amp] = atof( argv[++i] );        } //atoi changes it to integer
      else if( str == "-nu" )    {     v[i_nu] = atof( argv[++i] );        }
      else if( str == "-rho_denom" )    {     v[i_immune_duration] = atof( argv[++i] );        }
      else
      {
        fprintf(stderr, "\n\tUnknown option [%s] on command line.\n\n", argv[i]);
        exit(-1);
      }
      //END OF MAIN WHILE-LOOP BLOCK; INCREMENT AND MOVE ON TO THE NEXT COMMAND-LINE ARGUMENT

      // increment i so we can look at the next command-line option
      i++;
    }

    if (sigma[0][1] > 1) {
        fprintf(stderr,"\n\n\tWARNING : Sigma can't be over 1. %1.3f\n\n", sigma[0][1]); // %1.3f is a placeholder for what is being printed
        exit(-1);
    }

    //
    // ###  3.  INITIALIZE PARAMETERS - these are the default/starting values
    //

    // if the phi-parameters are not initialized on the command line
    if( !G_PHIS_INITIALIZED_ON_COMMAND_LINE )
    {
        for(int i=0;  i<10; i++) v[i] = ((double)i)*365.0 + 240.0; // sets the peak epidemic time in late August for the first 10 years
        for(int i=10; i<20; i++) v[i] = -99.0;
    }

    v[ i_amp ]   = 0.1;
    beta[0] = G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
    beta[1] = G_CLO_BETA2 / POPSIZE_MAIN;
    beta[2] = G_CLO_BETA3 / POPSIZE_MAIN;

    sigma[0][1] = 0.7; // the level of susceptibility to H1 if you've had B
    sigma[1][0] = 0.7; // and vice versa

    sigma[1][2] = 0.7; // the level of susceptibility to H3 if you've had B
    sigma[2][1] = 0.7; // and vice versa

    sigma[0][2] = 0.3; // the level of susceptibility to H3 if you've had H1
    sigma[2][0] = 0.3; // and vice versa

    sigma[0][0] = 0;
    sigma[1][1] = 0;
    sigma[2][2] = 0;

    v[ i_nu ]    = 0.2;                // recovery rate
    v[ i_immune_duration ] = 900.0;    // 2.5 years of immunity to recent infection

    //
    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
    //

    // declare the vector y that holds the values of all the state variables (at the current time)
    // below you are declaring a vector of size DIM

    //init already
//    double y[DIM];

    y = new double*[number_of_ode]();
    for(int i = 0; i < number_of_ode; i++){
        y[i] = new double[dimension];
        for(int j = 0; j < dimension; j++){
            y[i][j] = 0.0;
        }
    }

    for(int i = 0; i < number_of_ode; i++){
        for(int loc=0; loc<NUMLOC; loc++)
        {
            // put half of the individuals in the susceptible class
            y[i][ STARTS + loc ] = 0.5 * N[loc];

            // put small number (but slightly different amounts each time) of individuals into the infected classes
            double x = 0.010;
            double sumx = 0.0;
            for(int vir=0; vir<NUMSEROTYPES; vir++)
            {
                sumx += x;
                y[i][ STARTI + NUMSEROTYPES*loc + vir ] = x * N[loc];
                y[i][ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;     // initialize all of the J-variables to zero

                x += 0.001;
            }
            x=0.010; // reset x

            // distribute the remainder of individuals into the different recovered stages equally
            for(int vir=0; vir<NUMSEROTYPES; vir++)
            {
                double z = (0.5 - sumx)/((double)NUMR*NUMSEROTYPES);  // this is the remaining fraction of individuals to be distributed
                for(int stg=0; stg<NUMR; stg++)
                {
                    y[i][ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = z * N[loc];
                }
            }
        }
    }

    //copy cpu vector to gpu vector
    v_temp = v;
    v_d = thrust::raw_pointer_cast(v_temp.data());
    phis_temp = phis;
    phis_d = thrust::raw_pointer_cast(phis_temp.data());

    // Copy host_vector H to device_vector D
    printf("copy cpu vector to gpu vector\n");
}

double GPU_Parameters::seasonal_transmission_factor( double t )
{
    /*
     Ok, here's what's going down:
        We're gonna make this thing go for 40 years. 30 years of burn in and 10 years of real modeling.
        We're creating a "10-year model cycle" and need the code below to find a time point's "place" in the "cycle"
        modulus (denoted with % in C++) only works with integers, so need the acrobatics below

    // This is some code that's needed to create the 10-year "cycles" in transmission.

    int x = (int)t; // This is now to turn a double into an integer
    double remainder = t - (double)x;
    int xx = x % NUMDAYSOUTPUT;
    double yy = (double)xx + remainder
    // put yy into the sine function, let it return the beta value
    */
    double sine_function_value = 0.0;

    for(int i=0; i<phis.size(); i++)
    {
        if( fabs( t - phis[i] ) < 91.25 )
        {
            sine_function_value = sin( 2.0 * 3.141592653589793238 * (phis[i]-t+91.25) / 365.0 );
            //printf("\n\t\t\t %1.3f %1.3f \n\n", t, phis[i] );
        }
    }

    return 1.0 + v[i_amp] * sine_function_value;
}
