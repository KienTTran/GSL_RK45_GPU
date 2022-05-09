//
// Created by kient on 5/5/2022.
//

#include "flu_parameters.cuh"


FluParameters::FluParameters() {
}

FluParameters::~FluParameters() {

}

bool FluParameters::is_float(std::string myString) {
    return false;
}

//void FluParameters::init_from_cmd(int argc, char **argv) {
//    //dim = num_params;
//
//    for(int i=0; i<NUMSEROTYPES; i++)
//    {
//        for(int j=0; j<NUMSEROTYPES; j++)
//        {
//            sigma2d[i][j] = 0.0; // this initializes the system with full cross-immunity among serotypes
//        }
//    }
//
//    for(int i=0; i<NUMLOC; i++)
//    {
//        for(int j=0; j<NUMLOC; j++)
//        {
//            if(i==j) eta[i][j] = 1.0;  // the diagonal elements have to be 1.0, as a population mixes fully with itself
//            if(i!=j) eta[i][j] = 0.0;  // these are initialized to 0.0 indicating that the different sub-populations do not mix at all
//        }
//    }
//
//    // set initial values for the population sizes, 1 million for first location and 100K for others
//    N[0] = POPSIZE_MAIN;
//    for(int i=1; i<NUMLOC; i++) N[i] = POPSIZE_OUT;
//
//    //Parse begin
//    std::string str;
//    int i;
//    i=1;    // this is the position where you start reading command line options
//    // you skip "i=0" which is the text string "odesim"
//
//    // read in options from left to right
//    while(i<argc)
//    {
//        str = argv[i]; // read the ith text string into the variable "str"
//
//        //BEGIN MAIN IF-BLOCK BELOW
//
//        // ### 1 ### IF BLOCK FOR PHI
//        if( str == "-phi" )
//        {
////            phis_h.clear();
//            phi[i] = 0;
//            i++;
//
//            //BEGIN LOOPING THROUGH THE LIST OF BETAS
//            while(i<argc)
//            {
//                std::string s( argv[i] );    // convert argv[i] into a normal string object
//                if( is_float(s) )        // if the current string is a floating point number, write it into the phis array
//                {
//                    // if the command line argument is <0, just set it back to zero
//                    double d = atof( argv[i] );
//                    if( d < 0.0 ){
//                        d = 0.0;
//                        //TODO print warning here and probably should exit
//                        fprintf(stderr, "\n\n \t Don't make phis less than zero! \n\n");
//                    }
//
//                    phis_h.push_back( d );
//
//                    // increment and move on in this sub-loop
//                    i++;
//                }
//                else
//                {
//                    // if the current string is NOT a float, set the counter back down (so the string can be handled by the next if block
//                    // and break out of the loop
//                    i--;
//                    break;
//                }
//
//            }
//            //END OF LOOPING THROUGH THE LIST OF BETAS
//
//            // make sure at least one phi-value was read in
//            if( phi_length == 0 )
//            {
//                fprintf(stderr,"\n\n\tWARNING : No phi-values were read in after the command-line option \"-phi\".\n\n");
//            }
//        }
//            // ### 2 ### BLOCKS FOR FOR THE OTHER NON-PHI COMMAND-LINE OPTIONS
//        else if( str == "-checkpop" )
//        {
//            G_CLO_CHECKPOP_MODE = true;
//        }
//        else if( str == "-beta1" )    {     G_CLO_BETA1 = atof( argv[++i] );        }
//        else if( str == "-beta2" )    {     G_CLO_BETA2 = atof( argv[++i] );        } //atof is character to floating point
//        else if( str == "-beta3" )    {     G_CLO_BETA3 = atof( argv[++i] );        } //atoi changes it to integer
//        else if( str == "-sigma12" )    {   G_CLO_SIGMA12 = atof( argv[++i] );      } //atoi changes it to integer
//        else if( str == "-sigma13" )    {   G_CLO_SIGMA13 = atof( argv[++i] );      }
//        else if( str == "-sigma23" )    {   G_CLO_SIGMA23 = atof( argv[++i] );      }
//        else if( str == "-amp" )        {   G_CLO_AMPL = atof( argv[++i] );         }
//        else if( str == "-nu_denom")     {   G_CLO_NU_DENOM = atof( argv[++i] );     }
//        else if( str == "-rho_denom")    {   G_CLO_RHO_DENOM = atof( argv[++i]);      }
//        else if( str == "-epidur")      {   G_CLO_EPIDUR = atof( argv[++i]);        }
//
//
//        else
//        {
//            fprintf(stderr, "\n\tUnknown option [%s] on command line.\n\n", argv[i]);
//            exit(-1);
//        }
//        //END OF MAIN WHILE-LOOP BLOCK; INCREMENT AND MOVE ON TO THE NEXT COMMAND-LINE ARGUMENT
//
//        // increment i so we can look at the next command-line option
//        i++;
//    }
//
//    //Parse end
//    printf("Parameters from CMD: -beta1 %.2f -beta2 %.2f -beta3 %.2f -sigma12 %.2f -sigma13 %.2f -sigma23 %.2f -amp %.2f -nu_denom %.2f -rho_denom %.2f\n -phi ",
//           G_CLO_BETA1,G_CLO_BETA2,G_CLO_BETA3,G_CLO_SIGMA12,G_CLO_SIGMA13,G_CLO_SIGMA13,
//           G_CLO_AMPL,G_CLO_NU_DENOM,G_CLO_RHO_DENOM);
//    for(int i = 0; i < phi_length; i++){
//        phi[i] = phis_h.at(i);
//        printf("%.2f\t",phi[i]);
//    }
//    printf("\n");
//
//    //
//    // ###  3.  INITIALIZE PARAMETERS - these are the default/starting values
//    //
//
////    // if the phi-parameters are not initialized on the command line
////    if( !G_PHIS_INITIALIZED_ON_COMMAND_LINE )
////    {
////        for(int i=0;  i<10; i++) v[i] = ((double)i)*365.0 + 240.0; // sets the peak epidemic time in late August for the first 10 years
////        for(int i=10; i<20; i++) v[i] = -99.0;
////    }
//
//    v_d_i_amp   = G_CLO_AMPL;
//    beta[0] = G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
//    beta[1] = G_CLO_BETA2 / POPSIZE_MAIN;
//    beta[2] = G_CLO_BETA3 / POPSIZE_MAIN;
//
//    sigma2d[0][1] = G_CLO_SIGMA12; // 0.7; // the level of susceptibility to H1 if you've had B
//    sigma2d[1][0] = G_CLO_SIGMA12; // 0.7; // and vice versa
//
//    sigma2d[1][2] = G_CLO_SIGMA23; // 0.7; // the level of susceptibility to H3 if you've had B
//    sigma2d[2][1] = G_CLO_SIGMA23; // 0.7; // and vice versa
//
//    sigma2d[0][2] = G_CLO_SIGMA13; // 0.3; // the level of susceptibility to H3 if you've had H1
//    sigma2d[2][0] = G_CLO_SIGMA13; // 0.3; // and vice versa
//
//    sigma2d[0][0] = 0;
//    sigma2d[1][1] = 0;
//    sigma2d[2][2] = 0;
//
//    v_d_i_nu    = 1 / G_CLO_NU_DENOM;                // recovery rate
//    v_d_i_immune_duration = G_CLO_RHO_DENOM;    // 2.5 years of immunity to recent infection'
//    v_d_i_epidur = G_CLO_EPIDUR;
//
//    //
//    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
//    //
//
//    // declare the vector y that holds the values of all the state variables (at the current time)
//    // below you are declaring a vector of size DIM
//
//    trr = ((double)NUMR) / v_d_i_immune_duration;
////    v_d_i_nu = v_d_i_nu;
////    v_d_i_amp = v_d_i_amp;
//    v_d_i_epidur_d2 = v_d_i_epidur / 2.0;
//    v_d_i_epidur_x2 = v_d_i_epidur * 2.0;
//    pi_x2 = 2.0 * M_PI;
//}

void FluParameters::init() {
    //dim = num_params;

    for(int i=0; i<NUMSEROTYPES; i++)
    {
        for(int j=0; j<NUMSEROTYPES; j++)
        {
            sigma2d[i][j] = 0.0; // this initializes the system with full cross-immunity among serotypes
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


    for(int i = 0; i < NUMODE; i++){
        G_CLO_BETA1[i] = 1.2;
        G_CLO_BETA2[i] = 1.4;
        G_CLO_BETA3[i] = 1.6;
        beta[i][0] = 0.24;
        beta[i][1] = 0.22;
        beta[i][2] = 0.26;
        G_CLO_BETA1[i] = beta[i][0];
        G_CLO_BETA2[i] = beta[i][1];
        G_CLO_BETA3[i] = beta[i][2];
        beta[i][0] = G_CLO_BETA1[i] / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
        beta[i][1] = G_CLO_BETA2[i] / POPSIZE_MAIN;
        beta[i][2] = G_CLO_BETA3[i] / POPSIZE_MAIN;

    }
    //Load default params
    G_CLO_SIGMA12 = sigma[0];
    G_CLO_SIGMA13 = sigma[1];
    G_CLO_SIGMA23 = sigma[2];
    G_CLO_AMPL = amp;
    G_CLO_NU_DENOM = nu_denom;
    G_CLO_RHO_DENOM = rho_denom;
    phi[0] = phi_0;
    for(int i = 1; i < phi_length; i++){
        phi[i] = phi[i-1] + tau[i-1];
    }
//    printf("Parameters: -beta1 %.2f -beta2 %.2f -beta3 %.2f -sigma12 %.2f -sigma13 %.2f -sigma23 %.2f -amp %.2f -nu_denom %.2f -rho_denom %.2f\n",
//           G_CLO_BETA1,G_CLO_BETA2,G_CLO_BETA3,G_CLO_SIGMA12,G_CLO_SIGMA13,G_CLO_SIGMA13,
//           G_CLO_AMPL,G_CLO_NU_DENOM,G_CLO_RHO_DENOM);
//    printf("Sample: ");
//    for(int i = 0; i < sample_length; i++){
//        printf("%.5f\t",sample[i]);
//    }
//    printf("\nPhi: ");
//    for(int i = 0; i < phi_length; i++){
//        printf("%.2f\t",phi[i]);
//    }
//    printf("\n");

    //
    // ###  3.  INITIALIZE PARAMETERS - these are the default/starting values
    //

    // if the phi-parameters are not initialized on the command line
//    if( !G_PHIS_INITIALIZED_ON_COMMAND_LINE )
//    {
//        for(int i=0;  i<10; i++) v[i] = ((double)i)*365.0 + 240.0; // sets the peak epidemic time in late August for the first 10 years
//        for(int i=10; i<20; i++) v[i] = -99.0;
//    }

    v_d_i_amp   = G_CLO_AMPL;

    sigma2d[0][1] = G_CLO_SIGMA12; // 0.7; // the level of susceptibility to H1 if you've had B
    sigma2d[1][0] = G_CLO_SIGMA12; // 0.7; // and vice versa

    sigma2d[1][2] = G_CLO_SIGMA23; // 0.7; // the level of susceptibility to H3 if you've had B
    sigma2d[2][1] = G_CLO_SIGMA23; // 0.7; // and vice versa

    sigma2d[0][2] = G_CLO_SIGMA13; // 0.3; // the level of susceptibility to H3 if you've had H1
    sigma2d[2][0] = G_CLO_SIGMA13; // 0.3; // and vice versa

    sigma2d[0][0] = 0;
    sigma2d[1][1] = 0;
    sigma2d[2][2] = 0;

    v_d_i_nu    = 1 / G_CLO_NU_DENOM;                // recovery rate
//    v_d_i_immune_duration = G_CLO_RHO_DENOM;    // 2.5 years of immunity to recent infection'
    v_d_i_epidur = G_CLO_EPIDUR;

    //
    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
    //

    trr = ((double)NUMR) / G_CLO_RHO_DENOM;
//    v_d_i_nu = v_d_i_nu;
//    v_d_i_amp = v_d_i_amp;
    v_d_i_epidur_d2 = v_d_i_epidur / 2.0;
    v_d_i_epidur_x2 = v_d_i_epidur * 2.0;
    pi_x2 = 2.0 * M_PI;
}

//__host__ __device__
//void FluParameters::init_from_sample(double sample[]) {
//    //dim = num_params;
//
//    for(int i=0; i<NUMSEROTYPES; i++)
//    {
//        for(int j=0; j<NUMSEROTYPES; j++)
//        {
//            sigma2d[i][j] = 0.0; // this initializes the system with full cross-immunity among serotypes
//        }
//    }
//
//    for(int i=0; i<NUMLOC; i++)
//    {
//        for(int j=0; j<NUMLOC; j++)
//        {
//            if(i==j) eta[i][j] = 1.0;  // the diagonal elements have to be 1.0, as a population mixes fully with itself
//            if(i!=j) eta[i][j] = 0.0;  // these are initialized to 0.0 indicating that the different sub-populations do not mix at all
//        }
//    }
//
//    // set initial values for the population sizes, 1 million for first location and 100K for others
//    N[0] = POPSIZE_MAIN;
//    for(int i=1; i<NUMLOC; i++) N[i] = POPSIZE_OUT;
//
//    //Load default params
//    G_CLO_BETA1 = sample[0];
//    G_CLO_BETA2 = sample[1];
//    G_CLO_BETA3 = sample[2];
//    G_CLO_SIGMA12 = sigma[0];
//    G_CLO_SIGMA13 = sigma[1];
//    G_CLO_SIGMA23 = sigma[2];
//    G_CLO_AMPL = amp;
//    G_CLO_NU_DENOM = nu_denom;
//    G_CLO_RHO_DENOM = rho_denom;
//    phi[0] = sample[3];
//    for(int i = 1; i < phi_length; i++){
//        tau[i-1] = sample[4 + (i-1)];
//        phi[i] = phi[i-1] + tau[i-1];
//    }
////    printf("Parameters: -beta1 %.2f -beta2 %.2f -beta3 %.2f -sigma12 %.2f -sigma13 %.2f -sigma23 %.2f -amp %.2f -nu_denom %.2f -rho_denom %.2f\n",
////           G_CLO_BETA1,G_CLO_BETA2,G_CLO_BETA3,G_CLO_SIGMA12,G_CLO_SIGMA13,G_CLO_SIGMA13,
////           G_CLO_AMPL,G_CLO_NU_DENOM,G_CLO_RHO_DENOM);
////    printf("Sample: ");
////    for(int i = 0; i < sample_length; i++){
////        printf("%.5f\t",sample[i]);
////    }
////    printf("\nPhi: ");
////    for(int i = 0; i < phi_length; i++){
////        printf("%.2f\t",phi[i]);
////    }
////    printf("\n");
//
//    //
//    // ###  3.  INITIALIZE PARAMETERS - these are the default/starting values
//    //
//
//    // if the phi-parameters are not initialized on the command line
////    if( !G_PHIS_INITIALIZED_ON_COMMAND_LINE )
////    {
////        for(int i=0;  i<10; i++) v[i] = ((double)i)*365.0 + 240.0; // sets the peak epidemic time in late August for the first 10 years
////        for(int i=10; i<20; i++) v[i] = -99.0;
////    }
//
//    v_d_i_amp   = G_CLO_AMPL;
//    beta[0] = G_CLO_BETA1 / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
//    beta[1] = G_CLO_BETA2 / POPSIZE_MAIN;
//    beta[2] = G_CLO_BETA3 / POPSIZE_MAIN;
//
//    sigma2d[0][1] = G_CLO_SIGMA12; // 0.7; // the level of susceptibility to H1 if you've had B
//    sigma2d[1][0] = G_CLO_SIGMA12; // 0.7; // and vice versa
//
//    sigma2d[1][2] = G_CLO_SIGMA23; // 0.7; // the level of susceptibility to H3 if you've had B
//    sigma2d[2][1] = G_CLO_SIGMA23; // 0.7; // and vice versa
//
//    sigma2d[0][2] = G_CLO_SIGMA13; // 0.3; // the level of susceptibility to H3 if you've had H1
//    sigma2d[2][0] = G_CLO_SIGMA13; // 0.3; // and vice versa
//
//    sigma2d[0][0] = 0;
//    sigma2d[1][1] = 0;
//    sigma2d[2][2] = 0;
//
//    v_d_i_nu    = 1 / G_CLO_NU_DENOM;                // recovery rate
////    v_d_i_immune_duration = G_CLO_RHO_DENOM;    // 2.5 years of immunity to recent infection'
//    v_d_i_epidur = G_CLO_EPIDUR;
//
//    //
//    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
//    //
//
//    trr = ((double)NUMR) / G_CLO_RHO_DENOM;
////    v_d_i_nu = v_d_i_nu;
////    v_d_i_amp = v_d_i_amp;
//    v_d_i_epidur_d2 = v_d_i_epidur / 2.0;
//    v_d_i_epidur_x2 = v_d_i_epidur * 2.0;
//    pi_x2 = 2.0 * M_PI;
//}

__device__
void FluParameters::update(curandState curand_state) {
    //Load default params
    for(int i = 0; i < NUMODE; i++){
        G_CLO_BETA1[i] = G_CLO_BETA1[i] + curand_normal(&curand_state) * beta_sd[0];
        G_CLO_BETA2[i] = G_CLO_BETA2[i] + curand_normal(&curand_state) * beta_sd[1];
        G_CLO_BETA3[i] = G_CLO_BETA3[i] + curand_normal(&curand_state) * beta_sd[2];
        beta[i][0] = G_CLO_BETA1[i] / POPSIZE_MAIN;    // NOTE this is in a density-dependent transmission scheme
        beta[i][1] = G_CLO_BETA2[i] / POPSIZE_MAIN;
        beta[i][2] = G_CLO_BETA3[i] / POPSIZE_MAIN;
    }
}
