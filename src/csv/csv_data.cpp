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
  for(int i = 0; i < DATADIM_ROWS; i++){
    std::getline(csv_file, row);
    if (csv_file.bad() || csv_file.fail()) {
      break;
    }
//    printf("row = %s\n",row.c_str());
    std::vector<std::string> words = split_string(row,',');

    for(int j = 0; j < DATADIM_COLS; j++){
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


void CSV_Data::load_csv_data(int numode, double* y[]){
  printf("Load data from CSV\n");
  for (int i = 0; i < numode; i++) {
    for (int j = 0; j < csv_params.rows; j++) {
      for (int k = 0; k < csv_params.cols; k++) {
        const int index = j*csv_params.cols + k;
          y[i][index] = csv_data[j][k];
      }
//      printf("day = %d H1 = %.5f B = %.5f H3 = %.5f\n",j,csv_data[j][0], csv_data[j][1],csv_data[j][2]);
    }
  }
}

Parameters CSV_Data::get_params(){
  return csv_params;
}
void CSV_Data::test()  {

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
