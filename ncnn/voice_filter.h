#ifndef VOICE_FILTER_H
#define VOICE_FILTER_H

#pragma once

#include <string>
#include "net.h"
#include "gpu.h"
#include "layer.h"

#include <algorithm>
#include <vector>
#include <mutex>

class VoiceFilterNcnn
{
public:
  VoiceFilterNcnn() = default;
  virtual ~VoiceFilterNcnn() = default;
  
  virtual void inference(void *input, int len, void *output, float dvec[256]);
  
  virtual int init();
  
  virtual void uninit();

protected:
  std::mutex mutex_;
  bool inited_ = false;
  bool useVulkan_ = false;

  ncnn::UnlockedPoolAllocator blob_pool_allocator_;
  ncnn::PoolAllocator workspace_pool_allocator_;

  std::unique_ptr<ncnn::Net> net;
};

#endif // !defined(VOICE_FILTER_H)
