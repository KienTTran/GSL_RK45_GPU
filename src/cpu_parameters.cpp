//
// Created by kient on 1/12/2022.
//

#include "cpu_parameters.h"
#include "flu_default_params.h"

CPU_Parameters::CPU_Parameters(){
    dimension = 0;
    number_of_ode = 0;
    cpu_function = 0;
    t_target = 1.0;
    t0 = 0.0;
    h = 1e-6;
    ppc = new cpu_prms();
}

CPU_Parameters::~CPU_Parameters(){
    dimension = 0;
    number_of_ode = 0;
    cpu_function = 0;
    t_target = 1.0;
    t0 = 0.0;
    h = 1e-6;
    ppc = nullptr;
}

bool CPU_Parameters::isFloat( string myString ) {
    std::istringstream iss(myString);
    float f;
    iss >> noskipws >> f; // noskipws considers leading whitespace invalid
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail();
}

void CPU_Parameters::ParseArgs(int argc, char **argv)
{
    string str;
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
            ppc->phis.clear();
            i++;

            //BEGIN LOOPING THROUGH THE LIST OF BETAS
            while(i<argc)
            {
                string s( argv[i] );    // convert argv[i] into a normal string object
                if( isFloat(s) )        // if the current string is a floating point number, write it into the phis array
                {
                    // if the command line argument is <0, just set it back to zero
                    double d = atof( argv[i] );
                    if( d < 0.0 ) d = 0.0;
                    //TODO print warning here and probably should exit

                    ppc->phis.push_back( d );

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
            if( ppc->phis.size() == 0 )
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
        else if( str == "-sigma12" )    {     ppc->sigma[0][1] = atof( argv[++i] );        } //atoi changes it to integer
        else if( str == "-sigma13" )    {     ppc->sigma[0][2] = atof( argv[++i] );        } //atoi changes it to integer
        else if( str == "-sigma23" )    {     ppc->sigma[1][2] = atof( argv[++i] );        } //atoi changes it to integer
        else if( str == "-amp" )    {     ppc->v[ppc->i_amp] = atof( argv[++i] );        } //atoi changes it to integer
        else if( str == "-nu" )    {     ppc->v[ppc->i_nu] = atof( argv[++i] );        }
        else if( str == "-rho_denom" )    {     ppc->v[ppc->i_immune_duration] = atof( argv[++i] );        }
        else
        {
            fprintf(stderr, "\n\tUnknown option [%s] on command line.\n\n", argv[i]);
            exit(-1);
        }
        //END OF MAIN WHILE-LOOP BLOCK; INCREMENT AND MOVE ON TO THE NEXT COMMAND-LINE ARGUMENT

        // increment i so we can look at the next command-line option
        i++;
    }
    return;
}


void CPU_Parameters::initPPC(){
    if (ppc->sigma[0][1] > 1) {
        fprintf(stderr,"\n\n\tWARNING : Sigma can't be over 1. %1.3f\n\n", ppc->sigma[0][1]); // %1.3f is a placeholder for what is being printed
        exit(-1);
    }

    //
    // ###  3.  INITIALIZE PARAMETERS - these are the default/starting values
    //

    // if the phi-parameters are not initialized on the command line
    if( !G_PHIS_INITIALIZED_ON_COMMAND_LINE )
    {
        for(int i=0;  i<10; i++) ppc->v[i] = ((double)i)*365.0 + 240.0; // sets the peak epidemic time in late August for the first 10 years
        for(int i=10; i<20; i++) ppc->v[i] = -99.0;
    }

    ppc->v[ ppc->i_amp ]   = 0.1;
    ppc->beta[0] = G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
    ppc->beta[1] = G_CLO_BETA2 / POPSIZE_MAIN;
    ppc->beta[2] = G_CLO_BETA3 / POPSIZE_MAIN;

    ppc->sigma[0][1] = 0.7; // the level of susceptibility to H1 if you've had B
    ppc->sigma[1][0] = 0.7; // and vice versa

    ppc->sigma[1][2] = 0.7; // the level of susceptibility to H3 if you've had B
    ppc->sigma[2][1] = 0.7; // and vice versa

    ppc->sigma[0][2] = 0.3; // the level of susceptibility to H3 if you've had H1
    ppc->sigma[2][0] = 0.3; // and vice versa

    ppc->sigma[0][0] = 0;
    ppc->sigma[1][1] = 0;
    ppc->sigma[2][2] = 0;

    ppc->v[ ppc->i_nu ]    = 0.2;                // recovery rate
    ppc->v[ ppc->i_immune_duration ] = 900.0;    // 2.5 years of immunity to recent infection

    //
    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
    //

    // declare the vector y that holds the values of all the state variables (at the current time)
    // below you are declaring a vector of size DIM

    //init already
//    double y[DIM];

    for(int i = 0; i < NUMODE; i++){
        for(int j = 0; j < DIM; j++){
            y[i][j] = 0.0;
        }
    }


    for(int i = 0; i < NUMODE; i++) {
        for (int loc = 0; loc < NUMLOC; loc++) {
            // put half of the individuals in the susceptible class
            y[i][STARTS + loc] = 0.5 * ppc->N[loc];

            // put small number (but slightly different amounts each time) of individuals into the infected classes
            double x = 0.010;
            double sumx = 0.0;
            for (int vir = 0; vir < NUMSEROTYPES; vir++) {
                sumx += x;
                y[i][STARTI + NUMSEROTYPES * loc + vir] = x * ppc->N[loc];
                y[i][STARTJ + NUMSEROTYPES * loc + vir] = 0.0;     // initialize all of the J-variables to zero

                x += 0.001;
            }
            x = 0.010; // reset x

            // distribute the remainder of individuals into the different recovered stages equally
            for (int vir = 0; vir < NUMSEROTYPES; vir++) {
                double z = (0.5 - sumx) / ((double) NUMR *
                                           NUMSEROTYPES);  // this is the remaining fraction of individuals to be distributed
                for (int stg = 0; stg < NUMR; stg++) {
                    y[i][NUMSEROTYPES * NUMR * loc + NUMR * vir + stg] = z * ppc->N[loc];
                }
            }
        }
    }
}
