#include "voice_filter.h"
#include "sys_utils.h"

int VoiceFilterNcnn::init()
{
  if (inited_) {
    return 1;
  }
  std::lock_guard<std::mutex> theGuard(mutex_);

  net.reset(new ncnn::Net());

  useVulkan_ = false;
  
  net->opt.use_vulkan_compute = useVulkan_;
  if (useVulkan_) {
    net->opt.use_fp16_packed = false;
    net->opt.use_fp16_storage = true;
    net->opt.use_fp16_arithmetic = true;
    net->opt.use_int8_storage = false;
    net->opt.use_int8_arithmetic = false;
  }
  else {
    //net->opt.num_threads = 8;
    net->opt.blob_allocator = &blob_pool_allocator_;
    net->opt.workspace_allocator = &workspace_pool_allocator_;
  }

  std::string resFolder = cmp::getCurrFilePath() + "/Resources/models/";
  std::cout << "load models from resource folder:" << resFolder << std::endl;

  int ret0 = net->load_param((resFolder + "voicefilter.param").c_str());
  int ret1 = net->load_model((resFolder + "voicefilter.bin").c_str());

  std::cout << "load voicefilter, load param=" << ret0 << ", load bin=" << ret1 << std::endl;
  if (ret0 != 0 || ret1 != 0)
      return -1;
  
  inited_ = true;
  return 0;
}

void VoiceFilterNcnn::uninit()
{
  std::lock_guard<std::mutex> theGuard(mutex_);
  if (!inited_)
      return;

  net = nullptr;
  inited_ = false;
}

void VoiceFilterNcnn::inference(void *input, int len, void *output, float dvec[256])
{
  ncnn::Mat magMat(len, input);
  magMat.reshape(1, 1, len);
  ncnn::Mat dvecMat(256, dvec);

  ncnn::Extractor ex = net->create_extractor();
  int hr = ex.input("onnx::Unsqueeze_0", magMat);
  std::cout << "Set mel hr=" << hr << std::endl;
  hr = ex.input("onnx::Unsqueeze_1", dvecMat);
  std::cout << "Set dvec hr=" << hr << std::endl;
  
  ncnn::Mat out;
  hr = ex.extract("/lstm/Transpose_2_output_0", out);
  std::cout << "extract output hr=" << hr << std::endl;

  memcpy(output, out.data, len * sizeof(float));
}
