//
// Created by kient on 4/29/2022.
//

#ifndef RK45_CUDA_CSV_Data_H
#define RK45_CUDA_CSV_Data_H
#include <DataFrame/DataFrame.h>  // Main DataFrame header
#include <DataFrame/DataFrameStatsVisitors.h>
#define STATS_GO_INLINE
#include "../lib/stats/include/stats.hpp"
#include "../flu_default_params.h"

using DataFrame = hmdf::StdDataFrame<unsigned long>;

struct Parameters{
  int cols;
  int rows;
};

class CSV_Data {
public:
  explicit CSV_Data();
  ~CSV_Data();
  void read_csv_data();
  void process_csv_data();
  void load_csv_data(int numode, double* y[]);
  Parameters get_params();
  std::vector<std::string> split_string(const std::string &s, char delim);
  void test();
public:
  DataFrame csv_dataframe;
  Parameters csv_params;
  double csv_data[DATADIM_ROWS][DATADIM_COLS];
};

#endif // RK45_CUDA_CSV_Data_H
