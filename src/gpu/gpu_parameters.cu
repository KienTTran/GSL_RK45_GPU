//
// Created by kient on 1/12/2022.
//

#include <cmath>
#include "gpu_parameters.cuh"

GPUParameters::GPUParameters(){
    mcmc_loop = 0;
    ode_output_day = 0;
    ode_number = 0;
    ode_dimension = 0;
    t_target = 0.0;
    t0 = 0.0;
    h = 1e-6;
    num_blocks = 256;
    block_size = 1;
}

GPUParameters::~GPUParameters(){
    mcmc_loop = 0;
    ode_output_day = 0;
    ode_number = 0;
    ode_dimension = 0;
    t_target = 0.0;
    t0 = 0.0;
    h = 1e-6;
    num_blocks = 256;
    block_size = 1;
}

void GPUParameters::init(FluParameters *flu_params){
    y_ode_input = new double*[ode_number]();
    for (int i = 0; i < ode_number; i++) {
      y_ode_input[i] = new double[ode_dimension];
        for (int j = 0; j < ode_dimension; j++) {
            y_ode_input[i][j] = 0.5;
        }
    }
    CSV_Data* csv_data = new CSV_Data();
    csv_data->read_csv_data();
    data_dimension = csv_data->get_params().cols * csv_data->get_params().rows;
    data_params.cols = csv_data->get_params().cols;
    data_params.rows = csv_data->get_params().rows;
    y_data_input = new double*[ode_number]();
    for (int i = 0; i < ode_number; i++) {
      y_data_input[i] = new double[data_dimension];
    }
    csv_data->load_csv_data(ode_number, y_data_input);

    display_dimension = ode_dimension + 3;//3 for 0,1,2 columns
    y_ode_output = new double*[ode_number]();
    for (int i = 0; i < ode_number; i++) {
      y_ode_output[i] = new double[ode_output_day * display_dimension];
        for (int j = 0; j < ode_output_day * display_dimension; j++) {
            y_ode_output[i][j] = -(j * 1.0);
        }
    }
    for (int i = 0; i < ode_number; i++) {
    }
    y_agg = new double*[ode_number]();
    for (int i = 0; i < ode_number; i++) {
      y_agg[i] = new double[ode_output_day * agg_dimension];
        for (int j = 0; j < ode_output_day * agg_dimension; j++) {
            y_agg[i][j] = -9999.0;
        }
    }
    printf("ODE numbers = %d\n",ode_number);
    printf("1 ODE parameters = %d\n", ode_dimension);
    printf("1 ODE lines = %d\n", ode_output_day);
    printf("Display dimension = %d\n",display_dimension);
    printf("Total display dimension = %d x %d x %d = %d\n",ode_number, ode_output_day, display_dimension, ode_number * ode_output_day * display_dimension);

    //
    // ###  4.  SET INITIAL CONDITIONS FOR ODE SYSTEM
    //

    // declare the vector y that holds the values of all the state variables (at the current time)
    // below you are declaring a vector of size DIM

    for(int ode_index = 0; ode_index < ode_number; ode_index++){
      for(int loc=0; loc<NUMLOC; loc++)
      {
        // put half of the individuals in the susceptible class
        y_ode_input[ode_index][ STARTS + loc ] = 0.5 * flu_params->N[loc];

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
          y_ode_input[ode_index][ STARTI + NUMSEROTYPES*loc + vir ] = x * flu_params->N[loc];
          y_ode_input[ode_index][ STARTJ + NUMSEROTYPES*loc + vir ] = 0.0;     // initialize all of the J-variables to zero

          x += 0.001;
        }
        x=0.010; // reset x

        // distribute the remainder of individuals into the different recovered stages equally
        for(int vir=0; vir<NUMSEROTYPES; vir++)
        {
          double z = (0.5 - sumx)/((double)NUMR*NUMSEROTYPES);  // this is the remaining fraction of individuals to be distributed
          for(int stg=0; stg<NUMR; stg++)
          {
            y_ode_input[ode_index][ NUMSEROTYPES*NUMR*loc + NUMR*vir + stg ] = z * flu_params->N[loc];
          }
        }
      }
    }
}

