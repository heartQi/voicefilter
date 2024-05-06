#include "stdio.h"
#include "voice_filter_i.h"
#include <fstream>
#include <assert.h>
#include <string>
#include "wavreader.h"
#include <iostream>

#include <chrono>
#include <numeric>
#include <algorithm>
#include <vector>

void saveToPCM(const std::vector<short>& data, const std::string& filename) {
    std::ofstream pcmFile(filename, std::ios::out | std::ios::binary);
    if (!pcmFile) {
        std::cerr << "Failed to open PCM file for writing." << std::endl;
        return;
    }

    pcmFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(short));
    pcmFile.close();
}

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
  float ref[601] = {0};
  read_dump(filename, ref, len);
  
  float sum = 0;
  for (int i = 0; i < len; i++) {
    sum += fabs(data[i] - ref[i]);
  }
  printf("%.4f\n", sum);
  assert(sum < 0.05);
  return true;
}
using namespace std;

std::vector<float> getDataBatch(const std::vector<float>& data, size_t batch_size, size_t start_index) {
    std::vector<float> batch_data;
    size_t end_index = start_index + batch_size;
    if (end_index > data.size()) {
        end_index = data.size();
    }

    for (size_t i = start_index; i < end_index; ++i) {
        batch_data.push_back(data[i]);
    }

    return batch_data;
}


int main(int argc, char** argv) {
    void* h_x = wav_read_open(ROOT_DIR"/samples/mia-mervin-mixed.wav");

    int format, channels, sr, bits_per_sample;
    unsigned int data_length;
    int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
    if (!res)
    {
        cerr << "get ref header error: " << res << endl;
        return -1;
    }

    int samples = data_length * 8 / bits_per_sample;
    std::vector<int16_t> tmp(samples);
    res = wav_read_data(h_x, reinterpret_cast<unsigned char*>(tmp.data()), data_length);
    if (res < 0)
    {
        cerr << "read wav file error: " << res << endl;
        return -1;
    }
    std::vector<float> x(samples);
    std::transform(tmp.begin(), tmp.end(), x.begin(),
        [](int16_t a) {
            return static_cast<float>(a) / 32768;//32767.f
        });

    std::cout << "Sample rate: " << sr << "Hz" << std::endl;
    
    
  VoiceFilter* vf = VoiceFilter::create();
  float output[601] = { 0 };

  float dvec_test[256] = { 0 };
  read_dump(ROOT_DIR"/../dvec.data", dvec_test, 256);
  

  size_t batch_size = 160;
  size_t start_index = 0;
  std::vector<short> shorts;
  while (start_index < x.size() - 3200) {
      std::vector<float> batch = getDataBatch(x, batch_size, start_index);

      std::vector<float> ifft_out = vf->inference(batch, dvec_test);
      //std::vector<short> shorts;
      for (float value : ifft_out) {
          short short_value = static_cast<short>(round(value)); // 舍入操作
          shorts.push_back(short_value);
      }
      start_index += batch_size;
  }



  FILE* vad_fp_input_rnn;
  vad_fp_input_rnn = fopen("voice_filter_output.pcm", "wb");
  fwrite(shorts.data(), sizeof(short), shorts.size(), vad_fp_input_rnn);
  std::cout << "Data saved to PCM file!" << std::endl;
  /*
  std::vector<float> ifft_out = vf->inference(x, dvec_test);
  std::vector<short> shorts;
  for (float value : ifft_out) {
      short short_value = static_cast<short>(round(value)); // 舍入操作
      shorts.push_back(short_value);
  }

  saveToPCM(shorts, "out_put.pcm");
  */
/*
  void* h_x = wav_read_open("samples/mia-mervin-mixed.wav");

  float input[601] = {0};
  float dvec_test[256] = {0};
  read_dump(ROOT_DIR"/../input.data", input, 601);
  read_dump(ROOT_DIR"/../dvec.data", dvec_test, 256);

  float output[601] = {0};
  vf->inference(input, 601, output, dvec_test);
  check_output(ROOT_DIR"/../output.data", output, 601);
  */
  return 0;
}
