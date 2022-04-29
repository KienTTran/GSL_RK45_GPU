//
// Created by kient on 4/29/2022.
//

#ifndef RK45_CUDA_MCMC_H
#define RK45_CUDA_MCMC_H
#include <DataFrame/DataFrame.h>  // Main DataFrame header
#include <DataFrame/DataFrameStatsVisitors.h>
#define STATS_GO_INLINE
#include "../lib/stats/include/stats.hpp"

using DataFrame = hmdf::StdDataFrame<unsigned long>;

struct Data{
  std::vector<double> ILI_p_H1;
  std::vector<double> ILI_p_B;
  std::vector<double> ILI_p_H3;
};

struct RData{
  double num;
  double denom;
};

class MCMC {
public:
  explicit MCMC();
  ~MCMC();
  void read_csv_data();
  void process_csv_data();
  void test();
public:
  DataFrame csv_dataframe;
  Data csv_data;
};

#endif // RK45_CUDA_MCMC_H
