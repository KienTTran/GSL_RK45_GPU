#include "gpu/gpu_rk45.h"
#include "cpu/cpu_functions.h"
#include <iostream>
#include <thread>
#include <chrono>

using MyDataFrame = hmdf::StdDataFrame<unsigned long>;

int main(int argc, char* argv[])
{

    GPU_RK45* gpu_rk45 = new GPU_RK45();
    GPU_Parameters* gpu_params_test = new GPU_Parameters();
    gpu_params_test->number_of_ode = 1;
    gpu_params_test->dimension = DIM;
    gpu_params_test->display_number = 1;
    gpu_params_test->t_target = NUMDAYSOUTPUT;
    gpu_params_test->t0 = 0.0;
    gpu_params_test->step = 1.0;
    gpu_params_test->h = 1e-6;
    gpu_params_test->initTest(argc,argv);
    MyDataFrame df;
    try  {
      gpu_params_test->csv_dataframe.load_data(MyDataFrame::gen_sequence_index(0,520,1));
      gpu_params_test->csv_dataframe.read("vndat.csv", hmdf::io_format::csv2,true);
    }
    catch (const hmdf::DataFrameError &ex)  {
      std::cout << ex.what() << std::endl;
    }
    gpu_rk45->set_parameters(gpu_params_test);
    gpu_rk45->run();

    delete gpu_rk45;
    return 0;

}