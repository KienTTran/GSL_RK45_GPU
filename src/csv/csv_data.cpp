//
// Created by kient on 4/29/2022.
//

#include "csv_data.h"

CSV_Data::CSV_Data(){
}

CSV_Data::~CSV_Data() {

}

std::vector<std::string> CSV_Data::split_string(const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss (s);
  std::string item;
  while (getline (ss, item, delim)) {
    result.push_back (item);
  }
  return result;
}

void CSV_Data::read_csv_data(){
    csv_params.cols = DATADIM_COLS;
    csv_params.rows = DATADIM_ROWS;
  // Create an input filestream
  std::ifstream csv_file("vndat.csv");
  // Make sure the file is open
  if(!csv_file.is_open()) throw std::runtime_error("Could not open file");

  std::string row;
  for(int i = 0; i < csv_params.rows; i++){
    std::getline(csv_file, row);
    if (csv_file.bad() || csv_file.fail()) {
      break;
    }
//    printf("row = %s\n",row.c_str());
    std::vector<std::string> words = split_string(row,',');

    for(int j = 0; j < csv_params.cols; j++){
//      printf("words[%d] = %s\n",j,words[j].c_str());
      if (words[j].find("NA") != std::string::npos){
        csv_data[i][j] = -9999;
      }
      else{
        try{
          csv_data[i][j] = std::stod(words[j].c_str());
        }
        catch(const std::exception& e){
          printf("Error reading %s\n",words[j].c_str());
        }
      }
    }
  }
  // Close file
  csv_file.close();
}


void CSV_Data::load_csv_data(int ode_number, double* y[]){
  printf("Load data from CSV\n");
  for (int i = 0; i < ode_number; i++) {
    for (int j = 0; j < csv_params.rows; j++) {
      for (int k = 0; k < csv_params.cols; k++) {
        const int index = j*csv_params.cols + k;
          y[i][index] = csv_data[j][k];
      }
//      printf("day = %d H1 = %.5f B = %.5f H3 = %.5f\n",j,csv_data[j][0], csv_data[j][1],csv_data[j][2]);
    }
  }
}

CSVParameters CSV_Data::get_params(){
  return csv_params;
}