//
// Created by kient on 4/29/2022.
//

#include "mcmc.h"

MCMC::MCMC(){

}

MCMC::~MCMC() {

}

void MCMC::read_csv_data(){
  try  {
    csv_dataframe.load_data(DataFrame::gen_sequence_index(0,520,1));
    csv_dataframe.read("vndat.csv", hmdf::io_format::csv2,true);
  }
  catch (const hmdf::DataFrameError &ex)  {
    std::cout << ex.what() << std::endl;
  }
}

void MCMC::process_csv_data(){
  csv_data.ILI_p_H1 = csv_dataframe.get_column<double>("ILI_p_H1");
  csv_data.ILI_p_B = csv_dataframe.get_column<double>("ILI_p_B");
  csv_data.ILI_p_H3 = csv_dataframe.get_column<double>("ILI_p_H3");

}

void MCMC::test()  {

  std::cout << "\nTesting Rotating Up/Down ..." << std::endl;

  std::vector<unsigned long>  idx =
      { 123450, 123451, 123452, 123453, 123454, 123455, 123456, 123457, 123458, 123459, 123460, 123461, 123462, 123466 };
  std::vector<double>         d1 = { 15, 16, 15, 18, 19, 16, 21, 0.34, 1.56, 0.34, 2.3, 0.34, 19.0 };
  std::vector<int>            i1 = { 22, 23, 24, 25, 99 };
  std::vector<std::string>    s1 =
      { "qqqq", "wwww", "eeee", "rrrr", "tttt", "yyyy", "uuuu", "iiii", "oooo", "pppp", "2222", "aaaa", "dddd", "ffff" };
  DataFrame                 df;

  df.load_data(std::move(idx),
               std::make_pair("dbl_col", d1),
               std::make_pair("int_col", i1),
               std::make_pair("str_col", s1));

  std::cout << "Original DF:" << std::endl;
  df.write<std::ostream, double, int, std::string>(std::cout);

  auto    rudf = df.rotate<double, int, std::string>(3, hmdf::shift_policy::up);

  std::cout << "Rotated Up DF:" << std::endl;
  rudf.write<std::ostream, double, int, std::string>(std::cout);

  auto    rddf = df.rotate<double, int, std::string>(1, hmdf::shift_policy::right);

  std::cout << "Rotated Right DF:" << std::endl;
  rddf.write<std::ostream, double, int, std::string>(std::cout);
}
