//
// Created by kient on 1/12/2022.
//

#include <cmath>
#include "gpu_parameters.h"

GPUParameters::GPUParameters(){
    mcmc_loop = 0;
    ode_dimension = 0;
    t_target = 0.0;
    t0 = 0.0;
    h = 1e-6;
    num_blocks = 256;
    block_size = 1;
}

GPUParameters::~GPUParameters(){
    mcmc_loop = 0;
    ode_dimension = 0;
    t_target = 0.0;
    t0 = 0.0;
    h = 1e-6;
    num_blocks = 256;
    block_size = 1;
}

bool GPUParameters::is_float(std::string myString ) {
    std::istringstream iss(myString);
    double f;
    iss >> std::noskipws >> f; // noskipws considers leading whitespace invalid
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail();
}

void GPUParameters::init_from_cmd(int argc, char **argv){
  y_ode_input = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
      y_ode_input[i] = new double[ode_dimension];
    }
    for (int i = 0; i < NUMODE; i++) {
      for (int j = 0; j < ode_dimension; j++) {
        y_ode_input[i][j] = 0.5;
      }
    }
    CSV_Data* csv_data = new CSV_Data();
    csv_data->read_csv_data();
    data_dimension = csv_data->get_params().cols * csv_data->get_params().rows;
    data_params.cols = csv_data->get_params().cols;
    data_params.rows = csv_data->get_params().rows;
    y_data_input = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
      y_data_input[i] = new double[data_dimension];
    }
    csv_data->load_csv_data(NUMODE, y_data_input);

    display_dimension = ode_dimension + 3;//3 for 0,1,2 columns
    y_ode_output = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
      y_ode_output[i] = new double[NUMDAYSOUTPUT * display_dimension];
    }
    for (int i = 0; i < NUMODE; i++) {
      for (int j = 0; j < NUMDAYSOUTPUT * display_dimension; j++) {
        y_ode_output[i][j] = -(j * 1.0);
      }
    }
    y_ode_agg = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
      y_ode_agg[i] = new double[NUMDAYSOUTPUT * agg_dimension];
    }
    for (int i = 0; i < NUMODE; i++) {
      for (int j = 0; j < NUMDAYSOUTPUT * agg_dimension; j++) {
        y_ode_agg[i][j] = -9999.0;
      }
    }
    printf("ODE numbers = %d\n",NUMODE);
    printf("1 ODE parameters = %d\n", ode_dimension);
    printf("1 ODE lines = %d\n", NUMDAYSOUTPUT);
    printf("Display dimension = %d\n",display_dimension);
    printf("Total display dimension = %d x %d x %d = %d\n",NUMODE, NUMDAYSOUTPUT, display_dimension, NUMODE * NUMDAYSOUTPUT * display_dimension);

    v.clear();
    v.insert( v.begin(), num_params, 0.0 );
    assert( v.size()==num_params );

    //dim = num_params;

    for(int i=0; i<NUMSEROTYPES; i++)
    {
        for(int j=0; j<NUMSEROTYPES; j++)
        {
            flu_params.sigma2d[i][j] = 0.0; // this initializes the system with full cross-immunity among serotypes
        }
    }

    for(int i=0; i<NUMLOC; i++)
    {
        for(int j=0; j<NUMLOC; j++)
        {
            if(i==j) flu_params.eta[i][j] = 1.0;  // the diagonal elements have to be 1.0, as a population mixes fully with itself
            if(i!=j) flu_params.eta[i][j] = 0.0;  // these are initialized to 0.0 indicating that the different sub-populations do not mix at all
        }
    }

    // set initial values for the population sizes, 1 million for first location and 100K for others
    flu_params.N[0] = POPSIZE_MAIN;
    for(int i=1; i<NUMLOC; i++) flu_params.N[i] = POPSIZE_OUT;

    //Parse begin
    std::string str;
    int i, start;
    i=1;    // this is the position where you start reading command line options
    // you skip "i=0" which is the text string "odesim"

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
                if( is_float(s) )        // if the current string is a floating point number, write it into the phis array
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
    flu_params.beta[0] = G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
    flu_params.beta[1] = G_CLO_BETA2 / POPSIZE_MAIN;
    flu_params.beta[2] = G_CLO_BETA3 / POPSIZE_MAIN;

    flu_params.sigma2d[0][1] = G_CLO_SIGMA12; // 0.7; // the level of susceptibility to H1 if you've had B
    flu_params.sigma2d[1][0] = G_CLO_SIGMA12; // 0.7; // and vice versa

    flu_params.sigma2d[1][2] = G_CLO_SIGMA23; // 0.7; // the level of susceptibility to H3 if you've had B
    flu_params.sigma2d[2][1] = G_CLO_SIGMA23; // 0.7; // and vice versa

    flu_params.sigma2d[0][2] = G_CLO_SIGMA13; // 0.3; // the level of susceptibility to H3 if you've had H1
    flu_params.sigma2d[2][0] = G_CLO_SIGMA13; // 0.3; // and vice versa

    flu_params.sigma2d[0][0] = 0;
    flu_params.sigma2d[1][1] = 0;
    flu_params.sigma2d[2][2] = 0;

    v[ i_nu ]    = 1 / G_CLO_NU_DENOM;                // recovery rate
    v[ i_immune_duration ] = G_CLO_RHO_DENOM;    // 2.5 years of immunity to recent infection'
    v[ i_epidur ] = G_CLO_EPIDUR;

    //
    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
    //

    // declare the vector y that holds the values of all the state variables (at the current time)
    // below you are declaring a vector of size DIM

    for(int i = 0; i < NUMODE; i++){
      for(int loc=0; loc<NUMLOC; loc++)
      {
        // put half of the individuals in the susceptible class
        y_ode_input[i][ STARTS + loc ] = 0.5 * flu_params.N[loc];

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
          y_ode_input[i][ STARTI + NUMSEROTYPES*loc + vir ] = x * flu_params.N[loc];
          y_ode_input[i][ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;     // initialize all of the J-variables to zero

          x += 0.001;
        }
        x=0.010; // reset x

        // distribute the remainder of individuals into the different recovered stages equally
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
          double z = (0.5 - sumx)/((double)NUMR*NUMSEROTYPES);  // this is the remaining fraction of individuals to be distributed
          for(int stg=0; stg<NUMR; stg++)
          {
            y_ode_input[i][ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = z * flu_params.N[loc];
          }
        }
      }
    }

    flu_params.trr = ((double)NUMR) / v[i_immune_duration];
    flu_params.v_d_i_nu = v[i_nu];
    flu_params.v_d_i_amp = v[i_amp];
    flu_params.v_d_i_epidur_d2 = v[i_epidur] / 2.0;
    flu_params.v_d_i_epidur_x2 = v[i_epidur] * 2.0;
    flu_params.pi_x2 = 2.0 * M_PI;
}

void GPUParameters::init(){
    y_ode_input = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
      y_ode_input[i] = new double[ode_dimension];
    }
    for (int i = 0; i < NUMODE; i++) {
      for (int j = 0; j < ode_dimension; j++) {
        y_ode_input[i][j] = 0.5;
      }
    }
    CSV_Data* csv_data = new CSV_Data();
    csv_data->read_csv_data();
    data_dimension = csv_data->get_params().cols * csv_data->get_params().rows;
    data_params.cols = csv_data->get_params().cols;
    data_params.rows = csv_data->get_params().rows;
    y_data_input = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
      y_data_input[i] = new double[data_dimension];
    }
    csv_data->load_csv_data(NUMODE, y_data_input);

    display_dimension = ode_dimension + 3;//3 for 0,1,2 columns
    y_ode_output = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
      y_ode_output[i] = new double[NUMDAYSOUTPUT * display_dimension];
    }
    for (int i = 0; i < NUMODE; i++) {
      for (int j = 0; j < NUMDAYSOUTPUT * display_dimension; j++) {
        y_ode_output[i][j] = -(j * 1.0);
      }
    }
    y_ode_agg = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
      y_ode_agg[i] = new double[NUMDAYSOUTPUT * agg_dimension];
    }
    for (int i = 0; i < NUMODE; i++) {
      for (int j = 0; j < NUMDAYSOUTPUT * agg_dimension; j++) {
        y_ode_agg[i][j] = -9999.0;
      }
    }
    printf("ODE numbers = %d\n",NUMODE);
    printf("1 ODE parameters = %d\n", ode_dimension);
    printf("1 ODE lines = %d\n", NUMDAYSOUTPUT);
    printf("Display dimension = %d\n",display_dimension);
    printf("Total display dimension = %d x %d x %d = %d\n",NUMODE, NUMDAYSOUTPUT, display_dimension, NUMODE * NUMDAYSOUTPUT * display_dimension);

    v.clear();
    v.insert( v.begin(), num_params, 0.0 );
    assert( v.size()==num_params );

    //dim = num_params;

    for(int i=0; i<NUMSEROTYPES; i++)
    {
        for(int j=0; j<NUMSEROTYPES; j++)
        {
            flu_params.sigma2d[i][j] = 0.0; // this initializes the system with full cross-immunity among serotypes
        }
    }

    for(int i=0; i<NUMLOC; i++)
    {
        for(int j=0; j<NUMLOC; j++)
        {
            if(i==j) flu_params.eta[i][j] = 1.0;  // the diagonal elements have to be 1.0, as a population mixes fully with itself
            if(i!=j) flu_params.eta[i][j] = 0.0;  // these are initialized to 0.0 indicating that the different sub-populations do not mix at all
        }
    }

    // set initial values for the population sizes, 1 million for first location and 100K for others
    flu_params.N[0] = POPSIZE_MAIN;
    for(int i=1; i<NUMLOC; i++) flu_params.N[i] = POPSIZE_OUT;

    //Load default params
    G_CLO_BETA1 = flu_params.beta[0];
    G_CLO_BETA2 = flu_params.beta[1];
    G_CLO_BETA3 = flu_params.beta[2];
    G_CLO_SIGMA12 = flu_params.sigma[0];
    G_CLO_SIGMA13 = flu_params.sigma[1];
    G_CLO_SIGMA23 = flu_params.sigma[2];
    G_CLO_AMPL = flu_params.amp;
    G_CLO_NU_DENOM = flu_params.nu_denom;
    G_CLO_RHO_DENOM = flu_params.rho_denom;
    G_CLO_EPIDUR = flu_params.epidur;
    flu_params.phi_length = sizeof(flu_params.phi)/sizeof(flu_params.phi[0]);
    flu_params.phi[0] = flu_params.phi_0;
    for(int i = 1; i < flu_params.phi_length; i++){
        flu_params.phi[i] = flu_params.phi[i-1] + flu_params.tau[i-1];
    }
//    for(int i = 0; i < flu_params.phi_length; i++){
//        printf("phi[%d] = %.5f\n",i,flu_params.phi[i]);
//    }

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
    flu_params.beta[0] = G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
    flu_params.beta[1] = G_CLO_BETA2 / POPSIZE_MAIN;
    flu_params.beta[2] = G_CLO_BETA3 / POPSIZE_MAIN;

    flu_params.sigma2d[0][1] = G_CLO_SIGMA12; // 0.7; // the level of susceptibility to H1 if you've had B
    flu_params.sigma2d[1][0] = G_CLO_SIGMA12; // 0.7; // and vice versa

    flu_params.sigma2d[1][2] = G_CLO_SIGMA23; // 0.7; // the level of susceptibility to H3 if you've had B
    flu_params.sigma2d[2][1] = G_CLO_SIGMA23; // 0.7; // and vice versa

    flu_params.sigma2d[0][2] = G_CLO_SIGMA13; // 0.3; // the level of susceptibility to H3 if you've had H1
    flu_params.sigma2d[2][0] = G_CLO_SIGMA13; // 0.3; // and vice versa

    flu_params.sigma2d[0][0] = 0;
    flu_params.sigma2d[1][1] = 0;
    flu_params.sigma2d[2][2] = 0;

    v[ i_nu ]    = 1 / G_CLO_NU_DENOM;                // recovery rate
    v[ i_immune_duration ] = G_CLO_RHO_DENOM;    // 2.5 years of immunity to recent infection'

    v[ i_epidur ] = G_CLO_EPIDUR;

    //
    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
    //

    // declare the vector y that holds the values of all the state variables (at the current time)
    // below you are declaring a vector of size DIM

    for(int i = 0; i < NUMODE; i++){
      for(int loc=0; loc<NUMLOC; loc++)
      {
        // put half of the individuals in the susceptible class
        y_ode_input[i][ STARTS + loc ] = 0.5 * flu_params.N[loc];

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
          y_ode_input[i][ STARTI + NUMSEROTYPES*loc + vir ] = x * flu_params.N[loc];
          y_ode_input[i][ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;     // initialize all of the J-variables to zero

          x += 0.001;
        }
        x=0.010; // reset x

        // distribute the remainder of individuals into the different recovered stages equally
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
          double z = (0.5 - sumx)/((double)NUMR*NUMSEROTYPES);  // this is the remaining fraction of individuals to be distributed
          for(int stg=0; stg<NUMR; stg++)
          {
            y_ode_input[i][ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = z * flu_params.N[loc];
          }
        }
      }
    }

    flu_params.trr = ((double)NUMR) / v[i_immune_duration];
    flu_params.v_d_i_nu = v[i_nu];
    flu_params.v_d_i_amp = v[i_amp];
    flu_params.v_d_i_epidur_d2 = v[i_epidur] / 2.0;
    flu_params.v_d_i_epidur_x2 = v[i_epidur] * 2.0;
    flu_params.pi_x2 = 2.0 * M_PI;
}

void GPUParameters::update(){
    double new_sample[flu_params.sample_length];
    for(int i = 0; i < flu_params.sample_length; i++){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> d(flu_params.sample[i], flu_params.sample_sd[i]);
        new_sample[i] = d(gen);
    }
    flu_params.beta[0] = new_sample[0];
    flu_params.beta[1] = new_sample[1];
    flu_params.beta[2] = new_sample[2];
    flu_params.phi[0] = new_sample[3];
    flu_params.tau[0] = new_sample[4];
    flu_params.tau[1] = new_sample[5];
    flu_params.tau[2] = new_sample[6];
    flu_params.tau[3] = new_sample[7];
    flu_params.tau[4] = new_sample[8];
    flu_params.tau[5] = new_sample[9];
    flu_params.tau[6] = new_sample[10];
    flu_params.tau[7] = new_sample[11];
    flu_params.tau[8] = new_sample[12];
    
    y_ode_input = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
        y_ode_input[i] = new double[ode_dimension];
    }
    for (int i = 0; i < NUMODE; i++) {
        for (int j = 0; j < ode_dimension; j++) {
            y_ode_input[i][j] = 0.5;
        }
    }
    y_ode_agg = new double*[NUMODE]();
    for (int i = 0; i < NUMODE; i++) {
      y_ode_agg[i] = new double[NUMDAYSOUTPUT * agg_dimension];
    }
    for (int i = 0; i < NUMODE; i++) {
      for (int j = 0; j < NUMDAYSOUTPUT * agg_dimension; j++) {
        y_ode_agg[i][j] = -9999.0;
      }
    }
    v.clear();
    v.insert( v.begin(), num_params, 0.0 );
    assert( v.size()==num_params );

    for(int i=0; i<NUMSEROTYPES; i++)
    {
        for(int j=0; j<NUMSEROTYPES; j++)
        {
            flu_params.sigma2d[i][j] = 0.0; // this initializes the system with full cross-immunity among serotypes
        }
    }

    for(int i=0; i<NUMLOC; i++)
    {
        for(int j=0; j<NUMLOC; j++)
        {
            if(i==j) flu_params.eta[i][j] = 1.0;  // the diagonal elements have to be 1.0, as a population mixes fully with itself
            if(i!=j) flu_params.eta[i][j] = 0.0;  // these are initialized to 0.0 indicating that the different sub-populations do not mix at all
        }
    }

    // set initial values for the population sizes, 1 million for first location and 100K for others
    flu_params.N[0] = POPSIZE_MAIN;
    for(int i=1; i<NUMLOC; i++) flu_params.N[i] = POPSIZE_OUT;

    //Load default params
    G_CLO_BETA1 = flu_params.beta[0];
    G_CLO_BETA2 = flu_params.beta[1];
    G_CLO_BETA3 = flu_params.beta[2];
    G_CLO_SIGMA12 = flu_params.sigma[0];
    G_CLO_SIGMA13 = flu_params.sigma[1];
    G_CLO_SIGMA23 = flu_params.sigma[2];
    G_CLO_AMPL = flu_params.amp;
    G_CLO_NU_DENOM = flu_params.nu_denom;
    G_CLO_RHO_DENOM = flu_params.rho_denom;
    G_CLO_EPIDUR = flu_params.epidur;
    flu_params.phi_length = sizeof(flu_params.phi)/sizeof(flu_params.phi[0]);
    flu_params.phi[0] = flu_params.phi_0;
    for(int i = 1; i < flu_params.phi_length; i++){
        flu_params.phi[i] = flu_params.phi[i-1] + flu_params.tau[i-1];
    }
//    for(int i = 0; i < flu_params.phi_length; i++){
//        printf("phi[%d] = %.5f\n",i,flu_params.phi[i]);
//    }


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
    flu_params.beta[0] = G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
    flu_params.beta[1] = G_CLO_BETA2 / POPSIZE_MAIN;
    flu_params.beta[2] = G_CLO_BETA3 / POPSIZE_MAIN;

    flu_params.sigma2d[0][1] = G_CLO_SIGMA12; // 0.7; // the level of susceptibility to H1 if you've had B
    flu_params.sigma2d[1][0] = G_CLO_SIGMA12; // 0.7; // and vice versa

    flu_params.sigma2d[1][2] = G_CLO_SIGMA23; // 0.7; // the level of susceptibility to H3 if you've had B
    flu_params.sigma2d[2][1] = G_CLO_SIGMA23; // 0.7; // and vice versa

    flu_params.sigma2d[0][2] = G_CLO_SIGMA13; // 0.3; // the level of susceptibility to H3 if you've had H1
    flu_params.sigma2d[2][0] = G_CLO_SIGMA13; // 0.3; // and vice versa

    flu_params.sigma2d[0][0] = 0;
    flu_params.sigma2d[1][1] = 0;
    flu_params.sigma2d[2][2] = 0;

    v[ i_nu ]    = 1 / G_CLO_NU_DENOM;                // recovery rate
    v[ i_immune_duration ] = G_CLO_RHO_DENOM;    // 2.5 years of immunity to recent infection'

    v[ i_epidur ] = G_CLO_EPIDUR;

    //
    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
    //

    // declare the vector y that holds the values of all the state variables (at the current time)
    // below you are declaring a vector of size DIM

    for(int i = 0; i < NUMODE; i++){
      for(int loc=0; loc<NUMLOC; loc++)
      {
        // put half of the individuals in the susceptible class
        y_ode_input[i][ STARTS + loc ] = 0.5 * flu_params.N[loc];

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
          y_ode_input[i][ STARTI + NUMSEROTYPES*loc + vir ] = x * flu_params.N[loc];
          y_ode_input[i][ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;     // initialize all of the J-variables to zero

          x += 0.001;
        }
        x=0.010; // reset x

        // distribute the remainder of individuals into the different recovered stages equally
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
          double z = (0.5 - sumx)/((double)NUMR*NUMSEROTYPES);  // this is the remaining fraction of individuals to be distributed
          for(int stg=0; stg<NUMR; stg++)
          {
            y_ode_input[i][ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = z * flu_params.N[loc];
          }
        }
      }
    }

    flu_params.trr = ((double)NUMR) / v[i_immune_duration];
    flu_params.v_d_i_nu = v[i_nu];
    flu_params.v_d_i_amp = v[i_amp];
    flu_params.v_d_i_epidur_d2 = v[i_epidur] / 2.0;
    flu_params.v_d_i_epidur_x2 = v[i_epidur] * 2.0;
    flu_params.pi_x2 = 2.0 * M_PI;
}
