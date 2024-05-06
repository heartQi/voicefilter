#include "stdio.h"
#include "voice_filter.h"
#include <fstream>

void read_dump(const char* filename, float* data, int len) {
  std::ifstream file(filename);
  int idx = 0;
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      auto startPos = line.find('(');
      auto endPos = line.rfind(')');
      data[idx++] = std::stof(line.substr(startPos + 1, endPos - startPos - 1));
    }
    file.close();
  }
}

bool check_output(const char* filename, float* data, int len) {
  for (int i = 0; i < len; i++) {
    printf("%.4f\n", data[i]);
  }
}

int main(int argc, char** argv) {
  VoiceFilterNcnn vf;
  vf.init();

  float input[601] = {0};
  float dvec_test[256] = {0};
//  read_dump(ROOT_DIR"/../input.data", input, 601);
  read_dump(ROOT_DIR"/../dvec.data", dvec_test, 256);

  float output[601] = {0};
  vf.inference(input, 601, output, dvec_test);
  check_output(ROOT_DIR"/../model/output.data", output, 601);
  vf.uninit();
  
  return 0;
}
