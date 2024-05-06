#pragma once

#include <string>
#include <iostream>
#include <chrono>

namespace cmp {
std::string getCurrFilePath();

class ChronoProfiler {
public:
  ChronoProfiler(const std::string& title) : title_(title) {
    start_ = std::chrono::system_clock::now();
  }

  virtual ~ChronoProfiler() {
    std::cout << "[" << title_ << "]: " << duration() << " ms\n";
  }
  
  void watch(const std::string& tag) {
    std::cout << "[" << title_ << "," << tag << "]: " << duration() << " ms\n";
  }
  
  long duration() {
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
  }

private:
  std::string title_;
  std::chrono::system_clock::time_point start_;
};

}
