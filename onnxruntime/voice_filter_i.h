#pragma once
#include <vector>
#ifdef VOICE_FILTER_EXPORT
#define VOICE_FILTER_MODEL_EXPORT __declspec(dllexport)
#else
#define VOICE_FILTER_MODEL_EXPORT __declspec(dllimport)
#endif

class VOICE_FILTER_MODEL_EXPORT VoiceFilter
{
public:
  virtual ~VoiceFilter() {};
  
  virtual std::vector<float> inference(std::vector<float> input, float dvec[256]) = 0;

  static VoiceFilter* create();
};
