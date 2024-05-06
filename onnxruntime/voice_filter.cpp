#include "voice_filter.h"
#include <iostream>
#include <chrono>
#include <librosa/librosa.h>
#include <assert.h>

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

VoiceFilterONNX::VoiceFilterONNX():is_first_(true)
{
  session_.reset(new Ort::Session(env, L"D:/hackthon/voicefilter/model/voice_filter_257_sim.onnx", Ort::SessionOptions()));
  librosa::Feature::init();
}

VoiceFilterONNX::~VoiceFilterONNX()
{
}

std::vector<float> VoiceFilterONNX::inference(std::vector<float> input, float dvec[256])
{
    int n_fft = 512;
    int n_hop = 160;
    int n_mel = 40;
    int fmin = 80;
    int fmax = 7600;
    int win_length = 256;
    std::vector<std::vector<float>> transposeAValues_out;
    std::vector<std::vector<float>> fft_out = librosa::Feature::stft(input, transposeAValues_out, n_fft, n_hop, win_length, "hann", true, "constant");
    static std::vector<std::vector<float>> transposeAValues_out_model;
    static std::vector<std::vector<float>> fft_out_model;
    static std::vector<std::vector<float>> model_out_vec2d(20, std::vector<float>(257, 0.0f));
    static std::vector<std::vector<float>> ifft_in;// (20, std::vector<float>(257, 0.0f));
    if (is_first_) {
        is_first_ = false;
    }
    else {
        fft_out_model.insert(fft_out_model.end(), fft_out.begin(), fft_out.end());
        transposeAValues_out_model.insert(transposeAValues_out_model.end(), transposeAValues_out.begin(), transposeAValues_out.end());
    }


    if (fft_out_model.size() == 20) {
        assert(ifft_in.empty());
        std::vector<float> model_in;
        for (const auto& row : fft_out_model) {
            model_in.insert(model_in.end(), row.begin(), row.end());
        }

        ChronoProfiler profile("VoiceFilterONNX");
        Ort::AllocatorWithDefaultOptions allocator;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::AllocatedStringPtr> input_names_ptr;
        std::vector<const char*> input_node_names;
        std::vector<Ort::Value> input_datas;

        for (int i = 0; i < 2; i++) {
            Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
            //printf("input %d name=%s\n", i, input_name.get());
            input_names_ptr.push_back(std::move(input_name));

            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            //std::cout << "Input " << i << " : type = " << type << std::endl;
            auto input_node_dims = tensor_info.GetShape();
            if (i == 0) {
                auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)model_in.data(), model_in.size(), input_node_dims.data(), input_node_dims.size());
                input_datas.push_back(std::move(input_tensor));
            }
            else {
                auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, dvec, 256, input_node_dims.data(), input_node_dims.size());
                input_datas.push_back(std::move(input_tensor));
            }
        }
        Ort::AllocatedStringPtr outputNameMask = session_->GetOutputNameAllocated(0, allocator);
        std::vector<const char*> output_node_names = { outputNameMask.get() };

        auto output_tensors = session_->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_datas.data(), 2, output_node_names.data(), 1);
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        std::vector<float> vec(model_in.size());

        //memcpy(vec.data(), model_in.data(), model_in.size() * sizeof(float));
        memcpy(vec.data(), floatarr, model_in.size() * sizeof(float));
        
        size_t index = 0;
        for (size_t i = 0; i < 20; ++i) {
            for (size_t j = 0; j < 257; ++j) {
                model_out_vec2d[i][j] = vec[index++];
            }
        }

        ifft_in = elementWiseProduct(fft_out_model, model_out_vec2d);

        fft_out_model.clear();
    }
   
    std::vector<float> ifft_out;
    if (ifft_in.size() > 0) {
        std::vector<float> ifft_first_vec = ifft_in.front();
        std::vector<std::vector<float>> new_first_ifft_vec2d;
        new_first_ifft_vec2d.push_back(std::move(ifft_first_vec));

        std::vector<float> transposeAValues_out_first_vec = transposeAValues_out_model.front();
        std::vector<std::vector<float>> new_transposeAValues_out_vec2d;
        new_transposeAValues_out_vec2d.push_back(std::move(transposeAValues_out_first_vec));

        ifft_out = librosa::Feature::istft(new_first_ifft_vec2d, new_transposeAValues_out_vec2d, n_fft, n_hop, win_length, "hann", true, "constant");
        
        ifft_in.erase(ifft_in.begin());
        transposeAValues_out_model.erase(transposeAValues_out_model.begin());
    }

    return ifft_out;
}

VoiceFilter* VoiceFilter::create() {
  return new VoiceFilterONNX();
}


std::vector<std::vector<float>> VoiceFilterONNX::elementWiseProduct(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();

    if (B.size() != rows || B[0].size() != cols) {
        std::cerr << "Error: The dimensions of the matrices are not compatible for element-wise product." << std::endl;
        return {};
    }
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] * B[i][j];
        }
    }

    return result;
}
