#ifndef VOICE_FILTER_ONNX_H
#define VOICE_FILTER_ONNX_H

#pragma once

#include <string>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <vector>
#include <mutex>
#include "voice_filter_i.h"

class VoiceFilterONNX : public VoiceFilter
{
public:
  VoiceFilterONNX();
  virtual ~VoiceFilterONNX();
  
  virtual std::vector<float> inference(std::vector<float> input, float dvec[256]);
  
protected:
  std::vector<std::vector<float>> elementWiseProduct(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B);
  
  Ort::Env env;
  std::unique_ptr<Ort::Session> session_;
  bool is_first_;
};

#endif // !defined(VOICE_FILTER_ONNX_H)
