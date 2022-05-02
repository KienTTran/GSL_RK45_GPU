//
// Created by kient on 4/29/2022.
//

#ifndef RK45_CUDA_CSV_Data_H
#define RK45_CUDA_CSV_Data_H
#include "../flu_default_params.h"

struct CSVParameters{
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
    CSVParameters get_params();
  std::vector<std::string> split_string(const std::string &s, char delim);
public:
    CSVParameters csv_params;
  double csv_data[DATADIM_ROWS][DATADIM_COLS];
};

#endif // RK45_CUDA_CSV_Data_H
