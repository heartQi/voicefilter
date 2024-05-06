/* ------------------------------------------------------------------
* Copyright (C) 2020 ewan xu<ewan_xu@outlook.com>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
* express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
* -------------------------------------------------------------------
*/

#ifndef LIBROSA_H_
#define LIBROSA_H_

#include "eigen3/Eigen/Core"
#include "eigen3/unsupported/Eigen/FFT"

#include <vector>
#include <complex>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

//#include <torch/torch.h>

///
/// \brief c++ implemention of librosa
///
namespace librosa{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // !M_PI

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixcf;

#define FFT_LENGTH 512
#define FRAME_LENGTH 160
int frames = 0;
Vectorf fft_buffer(FFT_LENGTH / 2);
Vectorf fft_buffer_old(FRAME_LENGTH * 2 - FFT_LENGTH / 2);
Vectorf fft_buffer_pad(FFT_LENGTH);
int frames_ifft = 0;
Eigen::FFT<float> fft;

Vectorf paded_win(FFT_LENGTH);
Vectorf paded_win_ifft(FFT_LENGTH);


namespace internal{

static Vectorf pad(Vectorf &x, int left, int right, const std::string &mode, float value){
  Vectorf x_paded = Vectorf::Constant(left+x.size()+right, value);
  x_paded.segment(left, x.size()) = x;

  if (mode.compare("reflect") == 0){
    for (int i = 0; i < left; ++i){
      x_paded[i] = x[left-i];
    }
    for (int i = left; i < left+right; ++i){
      x_paded[i+x.size()] = x[x.size()-2-i+left];
    }
  }

  if (mode.compare("symmetric") == 0){
    for (int i = 0; i < left; ++i){
      x_paded[i] = x[left-i-1];
    }
    for (int i = left; i < left+right; ++i){
      x_paded[i+x.size()] = x[x.size()-1-i+left];
    }
  }

  if (mode.compare("edge") == 0){
    for (int i = 0; i < left; ++i){
      x_paded[i] = x[0];
    }
    for (int i = left; i < left+right; ++i){
      x_paded[i+x.size()] = x[x.size()-1];
    }
  }
  if (mode.compare("constant") == 0) {
	  for (int i = 0; i < left; ++i) {
		  x_paded[i] = 0.0;
	  }
  }
  return x_paded;
}

static Vectorf win_hann(int n_fft, int n_hop, int win_length, const std::string& win, bool center, const std::string& mode) {
	 // hanning
  Vectorf window = 0.5*(1.f-(Vectorf::LinSpaced(win_length, 0.f, static_cast<float>(win_length -1))*2.f*M_PI/ win_length).array().cos());

  int pad_length = (n_fft - win_length) / 2;
  Vectorf padded_window(n_fft);

  for (int i = 0; i < pad_length; ++i) {
      padded_window[i] = 0.0f;
  }

  for (int i = 0; i < win_length; ++i) {
      padded_window[pad_length + i] = window[i];
  }

  for (int i = 0; i < pad_length; ++i) {
      padded_window[pad_length + win_length + i] = 0.0f;
  }
  return padded_window;
}

static Vectorf win_hann_iff(int n_fft, int n_hop, int win_length, const std::string& win, bool center, const std::string& mode) {
	// hanning
	Vectorf window = 0.5 * (1.f - (Vectorf::LinSpaced(win_length, 0.f, static_cast<float>(win_length - 1)) * 2.f * M_PI / win_length).array().cos());

	int pad_length = (n_fft - win_length) / 2;
	Vectorf padded_window(n_fft);

	for (int i = 0; i < pad_length; ++i) {
		padded_window[i] = 0.0f;
	}

	for (int i = 0; i < win_length; ++i) {
		padded_window[pad_length + i] = window[i];
	}

	for (int i = 0; i < pad_length; ++i) {
		padded_window[pad_length + win_length + i] = 0.0f;
	}
	Vectorf padded_window_iff(n_fft);
	for (int i = 0; i < n_fft; ++i) {
		padded_window_iff[i] = padded_window[i] * padded_window[i];
	}
	return padded_window_iff;
}

static Matrixcf stft(Vectorf &x, int n_fft, int n_hop,int win_length, const std::string &win, bool center, const std::string &mode, Vectorf& padded_window){
	int pad_len_left = center ? n_fft / 2 : 0;//600

	int n_f = n_fft / 2 + 1;
	Matrixcf X(1, n_fft);
	static Eigen::FFT<float> fft;
	std::vector<float> test(FRAME_LENGTH);
	if (frames == 0)
	{
		for (int j = 0; j < FRAME_LENGTH; j++) {
			fft_buffer[j] = x[j];
		}
	}
	if (frames == 1)
	{
		for (int j = 0; j < (FFT_LENGTH / 2 - FRAME_LENGTH); j++) {//96
			fft_buffer[frames * FRAME_LENGTH + j] = x[j];
		}
		for (int j = 0; j < (FRAME_LENGTH * 2 - FFT_LENGTH / 2); j++) {//64
			fft_buffer_old[j] = x[FFT_LENGTH / 2 - FRAME_LENGTH + j];
		}
		fft_buffer_pad = pad(fft_buffer, pad_len_left, 0, mode, 0.f);
		Vectorf x_frame = padded_window.array() * fft_buffer_pad.array();
		X.row(0) = fft.fwd(x_frame);
	}
	if (frames > 1) {
		for (int j = 0; j < (FFT_LENGTH - FRAME_LENGTH); j++) {//160
			fft_buffer_pad[j] = fft_buffer_pad[FRAME_LENGTH + j];
		}
		for (int j = 0; j < (FRAME_LENGTH * 2 - FFT_LENGTH / 2); j++) {//64
			fft_buffer_pad[FFT_LENGTH - FRAME_LENGTH + j] = fft_buffer_old[j];
		}
		for (int j = 0; j < (FFT_LENGTH / 2 - FRAME_LENGTH); j++) {
			fft_buffer_pad[FRAME_LENGTH + FFT_LENGTH / 2 + j] = x[j];
		}
		for (int j = 0; j < (FRAME_LENGTH * 2 - FFT_LENGTH / 2); j++) {
			fft_buffer_old[j] = x[FFT_LENGTH / 2 - FRAME_LENGTH + j];
		}


		Vectorf x_frame = padded_window.array() * fft_buffer_pad.array();
		X.row(0) = fft.fwd(x_frame);
	}
	frames++;
	return X.leftCols(n_f);
}

static void window_ss_fill(std::vector<float>& x, const std::vector<float>& win_sq, int n_frames, int hop_length) {
	int n = x.size();
	int n_fft = win_sq.size();
	for (int i = 0; i < n_frames; ++i) {
		int sample = i * hop_length;
		for (int j = 0; j < n_fft && sample + j < n; ++j) {
			x[sample + j] += (sample + j < n) ? win_sq[j] : 0;
		}
	}
}
static void overlapAdd(std::vector<float>& y, const std::vector<std::vector<float>>& ytmp, int hop_length) {
	int n_fft = ytmp[0].size();  // Assuming ytmp is a 2D vector where each sub-vector is a frame.
	int N = n_fft;
	int num_frames = ytmp.size();

	for (int frame = 0; frame < num_frames; ++frame) {
		int sample = frame * hop_length;
		if (N > y.size() - sample) {
			N = y.size() - sample;
		}

		for (int i = 0; i < N; ++i) {
			y[sample + i] += ytmp[frame][i];
		}
	}
}

static void isft_init()
{
	fft.SetFlag(fft.HalfSpectrum);
}

//static Matrixcf istft(Matrixcf& input, int n_fft, int n_hop, int win_length, const std::string& win, bool center, const std::string& mode) {
 static std::vector<float>  istft(Matrixcf& input, int n_fft, int n_hop, int win_length, const std::string& win, bool center, const std::string& mode, Vectorf& padded_window, Vectorf& padded_window_iff) {
	 int rows = 2861;

	 //////////
	 static int start_frames1 = int(std::ceil((float)n_fft / 2.0f / n_hop));
	 static std::vector<std::vector<float>> output_start1(start_frames1, std::vector<float>(n_fft));

	 static int start_out_length1 = n_fft + n_hop * (start_frames1 - 1);

	 static int offset1 = start_frames1 * n_hop - n_fft / 2;//64
	 static std::vector<float> output_buffer(n_hop);
	 static std::vector<float> output_buffer_overlap(n_fft);
	 static std::vector<float> output_buffer_old(n_fft);

	 //for win_iff
	 static std::vector<float> output_buffer_win_long(n_fft * 2);//0
	 static std::vector<float> output_buffer_win(n_hop * 3 - n_fft / 2);//0
	 static std::vector<float> output_buffer_overlap_win(n_fft);//0
	 static std::vector<float> output_buffer_ifft_win(n_hop);
	 static int iff_count = 0;
	 for (int i = 0; i < n_hop; i++) {
		 output_buffer_ifft_win[i] = 1.0f;
	 }
	 //for test
	 iff_count++;
	 Matrixcf output(1, n_fft);
	 Vectorf out_inv = fft.inv(input.row(0));
	 output.row(0) = out_inv;
	 std::vector<float> output_ifft_frames(n_fft);
	 //for win_ifft overlap//////////
	 if (iff_count == 1)
	 {
		 for (int j = 0; j < n_fft; j++)//overlap
		 {
			 output_buffer_overlap_win[j] = padded_window_iff[j];//current overlap
		 }

	 }
	 else {

		 for (int j = 0; j < n_fft; j++)//overlap
		 {
			 output_buffer_overlap_win[j] += padded_window_iff[j];//current overlap
		 }
	 }


	 if (iff_count <= 3)
	 {
		 for (int j = 0; j < n_hop; j++)//out to win long
		 {
			 output_buffer_win_long[(iff_count - 1) * n_hop + j] = output_buffer_overlap_win[j];//current overlap
		 }

		 //for next overlap
		 for (int j = 0; j < n_fft - n_hop; j++)//for next overlap
		 {
			 output_buffer_overlap_win[j] = padded_window_iff[n_hop + j];//current overlap
		 }
		 for (int j = 0; j < n_hop; j++)
		 {
			 output_buffer_overlap_win[n_fft - n_hop + j] = 0.0f;//current overlap
		 }

		 if (iff_count == 3) {
			 for (int j = 0; j < (n_hop * 3 - n_fft / 2); j++)
			 {
				 output_buffer_win[j] = output_buffer_win_long[n_fft / 2 + j];
			 }
			 for (int j = 0; j < n_hop; j++) {
				 output_buffer_ifft_win[j] = output_buffer_win[j];
			 }
		 }

	 }
	 else {
		 for (int j = 0; j < n_hop * 2 - n_fft / 2; j++)//overlap 
		 {
			 output_buffer_win[j] = output_buffer_win[n_hop + j];
		 }
		 for (int j = 0; j < n_hop; j++)//out to win long
		 {
			 output_buffer_win[n_hop * 2 - n_fft / 2 + j] = output_buffer_overlap_win[j];
		 }

		 //for next overlap
		 for (int j = 0; j < n_fft - n_hop; j++)//for next overlap
		 {
			 output_buffer_overlap_win[j] = padded_window_iff[n_hop + j];//current overlap
		 }
		 for (int j = 0; j < n_hop; j++)
		 {
			 output_buffer_overlap_win[n_fft - n_hop + j] = 0.0f;//current overlap
		 }

		 for (int j = 0; j < n_hop; j++) {
			 output_buffer_ifft_win[j] = output_buffer_win[j];
		 }
	 }

	 /////////////////////////////



	 if (iff_count <= start_frames1)
	 {
		 for (int j = 0; j < n_fft; ++j) {
			 output_start1[iff_count - 1][j] = output(0, j).real() * padded_window[j];  
		 }
		 for (int j = 0; j < n_hop; j++)
		 {
			 output_buffer[j] = 0.0f;//start 64 sapmples remain
		 }
	 }
	 else {
		 for (int j = 0; j < n_fft; ++j) {
			 output_ifft_frames[j] = output(0, j).real() * padded_window[j];  
		 }
	 }

	 if (iff_count < start_frames1) {
		 //for first frame output zero
		 for (int j = 0; j < n_hop; j++) {
			 output_buffer[j] = 0.0f;
		 }

	 }
	 else if (iff_count == start_frames1)//the firt two frames no output
	 {
		 std::vector<float> overlap_out_start1(start_out_length1);
		 overlapAdd(overlap_out_start1, output_start1, n_hop);//160+ 512
		 std::vector<float> output_ifftf_2(start_out_length1);
		 for (int j = 0; j < (start_out_length1 - n_fft / 2); j++) {
			 output_ifftf_2[j] = overlap_out_start1[n_fft / 2 + j];//160+512 - 256 
		 }
		 for (int j = 0; j < offset1; j++)
		 {
			 output_buffer_old[j] = output_ifftf_2[j];//start 64 sapmples remain
		 }
		 for (int j = 0; j < start_out_length1 - n_fft / 2 - offset1; j++)
		 {
			 output_buffer_overlap[j] = output_ifftf_2[j + offset1];//for next ovelap 352
		 }
		 for (int j = 0; j < n_hop; j++)
		 {
			 output_buffer_overlap[start_out_length1 - n_fft / 2 - offset1 + j] = 0.0f;//for next ovelap 352
		 }

		 ///for second frames output zero
		 for (int j = 0; j < n_hop; j++) {
			 output_buffer[j] = 0.0f;
		 }
	 }
	 else if (iff_count > start_frames1) {//from third frame output

		 for (int j = 0; j < n_fft; j++)//overlap
		 {
			 output_buffer_overlap[j] += output_ifft_frames[j];//current overlap
		 }
		 for (int j = 0; j < offset1; j++)//output
		 {
			 output_buffer[j] = output_buffer_old[j];
		 }
		 for (int j = 0; j < n_hop - offset1; j++)//output
		 {
			 output_buffer[offset1 + j] = output_buffer_overlap[j];
		 }
		 for (int j = 0; j < offset1; j++)
		 {
			 output_buffer_old[j] = output_buffer_overlap[n_hop - offset1 + j];//start 64 sapmples remain for  next output
		 }

		 // for next overlap
		 for (int j = 0; j < n_fft - n_hop; j++)
		 {
			 output_buffer_overlap[j] = output_buffer_overlap[n_hop + j];
		 }
		 for (int j = 0; j < n_hop; j++)
		 {
			 output_buffer_overlap[n_fft - n_hop + j] = 0.0f;
		 }

		 for (int j = 0; j < n_hop; j++) {
			 //output_buffer[j] *= 32768;
			 //if (output_buffer_ifft_win[j] > 1.754944e-38)
				// output_buffer[j] = output_buffer[j] / output_buffer_ifft_win[j];
		 }

	 }
	 return output_buffer;
}

static Matrixf spectrogram(Matrixcf &X, float power = 1.f){
  return X.cwiseAbs().array().pow(power);
}

static Matrixf melfilter(int sr, int n_fft, int n_mels, int fmin, int fmax){
  int n_f = n_fft/2+1;
  Vectorf fft_freqs = (Vectorf::LinSpaced(n_f, 0.f, static_cast<float>(n_f-1))*sr)/n_fft;

  float f_min = 0.f;
  float f_sp = 200.f/3.f;
  float min_log_hz = 1000.f;
  float min_log_mel = (min_log_hz-f_min)/f_sp;
  float logstep = logf(6.4f)/27.f;

  auto hz_to_mel = [=](int hz, bool htk = false) -> float {
    if (htk){
      return 2595.0f*log10f(1.0f+hz/700.0f);
    }
    float mel = (hz-f_min)/f_sp;
    if (hz >= min_log_hz){
      mel = min_log_mel+logf(hz/min_log_hz)/logstep;
    }
    return mel;
  };
  auto mel_to_hz = [=](Vectorf &mels, bool htk = false) -> Vectorf {
    if (htk){
      return 700.0f*(Vectorf::Constant(n_mels+2, 10.f).array().pow(mels.array()/2595.0f)-1.0f);
    }
    return (mels.array()>min_log_mel).select(((mels.array()-min_log_mel)*logstep).exp()*min_log_hz, (mels*f_sp).array()+f_min);
  };

  float min_mel = hz_to_mel(fmin);
  float max_mel = hz_to_mel(fmax);
  Vectorf mels = Vectorf::LinSpaced(n_mels+2, min_mel, max_mel);
  Vectorf mel_f = mel_to_hz(mels);
  Vectorf fdiff = mel_f.segment(1, mel_f.size() - 1) - mel_f.segment(0, mel_f.size() - 1);
  Matrixf ramps = mel_f.replicate(n_f, 1).transpose().array() - fft_freqs.replicate(n_mels + 2, 1).array();

  Matrixf lower = -ramps.topRows(n_mels).array()/fdiff.segment(0, n_mels).transpose().replicate(1, n_f).array();
  Matrixf upper = ramps.bottomRows(n_mels).array()/fdiff.segment(1, n_mels).transpose().replicate(1, n_f).array();
  Matrixf weights = (lower.array()<upper.array()).select(lower, upper).cwiseMax(0);

  auto enorm = (2.0/(mel_f.segment(2, n_mels)-mel_f.segment(0, n_mels)).array()).transpose().replicate(1, n_f);
  weights = weights.array()*enorm;

  return weights;
}
/*
static Matrixf melspectrogram(Vectorf &x, int sr, int n_fft, int n_hop, int win_length,
                        const std::string &win, bool center,
                        const std::string &mode, float power,
                        int n_mels, int fmin, int fmax){
  Matrixcf X = stft(x, n_fft, n_hop, win_length,win, center, mode);
  Matrixf mel_basis = melfilter(sr, n_fft, n_mels, fmin, fmax);
  Matrixf sp = spectrogram(X, power);
  Matrixf mel = mel_basis*sp.transpose();
  return mel;
}*/

static Matrixf power2db(Matrixf& x) {
  auto log_sp = 10.0f*x.array().max(1e-10).log10();
  return log_sp.cwiseMax(log_sp.maxCoeff() - 80.0f);
}

static Matrixf dct(Matrixf& x, bool norm, int type) {
  int N = x.cols();
  Matrixf xi = Matrixf::Zero(N, N);
  xi.rowwise() += Vectorf::LinSpaced(N, 0.f, static_cast<float>(N-1));
  // type 2
  Matrixf coeff = 2*(M_PI*xi.transpose().array()/N*(xi.array()+0.5)).cos();
  Matrixf dct = x*coeff.transpose();
  // ortho
  if (norm) {
    Vectorf ortho = Vectorf::Constant(N, std::sqrtf(0.5f/N));
    ortho[0] = std::sqrtf(0.25f/N);
    dct = dct*ortho.asDiagonal();
  }
  return dct;
}

static Matrixcf convertToMatrix(std::vector<std::vector<std::complex<float>>>& vec) {
	int rows = vec.size();
	int cols = rows ? vec[0].size() : 0;
	Matrixcf mat(rows, cols);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			mat(i, j) = vec[i][j];
		}
	}
	return mat;
}

// Function to convert an Eigen::MatrixXcf back to a std::vector<std::vector<std::complex<float>>>
static void matrixToVector(Matrixcf& mat, std::vector<std::vector<std::complex<float>>>& vec) {
	int rows = mat.rows();
	int cols = mat.cols();
	vec.resize(rows);
	for (int i = 0; i < rows; ++i) {
		vec[i].resize(cols);
		for (int j = 0; j < cols; ++j) {
			vec[i][j] = mat(i, j);
		}
	}
}


} // namespace internal

class Feature
{
public:
static float clip(float value, float min, float max) {
		return std::min(std::max(value, min), max);
	}
static float magnitudeToDecibel(const std::complex<float>& z) {
	float magnitude = std::abs(z);
	float maxMagnitude = std::max(1e-5f, magnitude);
    float decibel = 20.0 * std::log10(maxMagnitude) - 20;
    float decibel_nomarlize = clip(decibel / 100, -1.0, 0.0) + 1.0;
	return decibel_nomarlize;
}
static double angle(const std::complex<double>& z, bool deg = false) {
	double radians = std::atan2(z.imag(), z.real());
	if (deg) {
		return radians * (180.0 / M_PI);
	}
	return radians;
}

static std::vector<std::vector<float>> transposeVector(std::vector<std::vector<float>>& original) {
if (original.empty()) return original;  

size_t rows = original.size();
size_t cols = original[0].size(); 

std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows, 0.0f));


for (size_t i = 0; i < rows; ++i) {
for (size_t j = 0; j < cols; ++j) {
	transposed[j][i] = original[i][j];
}
}

  return transposed;
}

static void init() {
	internal::isft_init();
  }

  /// \brief      short-time fourier transform similar with librosa.feature.stft
  /// \param      x             input audio signal
  /// \param      n_fft         length of the FFT size
  /// \param      n_hop         number of samples between successive frames
  /// \param      win           window function. currently only supports 'hann'
  /// \param      center        same as librosa
  /// \param      mode          pad mode. support "reflect","symmetric","edge"
  /// \return     complex-valued matrix of short-time fourier transform coefficients.
  static std::vector<std::vector<float>> stft(std::vector<float> &x, std::vector<std::vector<float>> &t_A_Values,
                                                            int n_fft, int n_hop,
                                                            int win_length,
                                                            const std::string &win, bool center,
                                                            const std::string &mode){
	  
	  paded_win = internal::win_hann(n_fft, n_hop, win_length, win, center, mode);
	  paded_win_ifft = internal::win_hann_iff(n_fft, n_hop, win_length, win, center, mode);
	  int count_ifft = 0;
	  int count_fft = 0;
	  
	
	  
	  
	Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixcf X = internal::stft(map_x, n_fft, n_hop, win_length, win, center, mode, paded_win);
    std::vector<std::vector<std::complex<float>>> X_vector(X.rows(), std::vector<std::complex<float>>(X.cols(), 0));
    for (int i = 0; i < X.rows(); ++i){
      auto &row = X_vector[i];
      Eigen::Map<Vectorcf>(row.data(), row.size()) = X.row(i);
    }

	std::vector<std::vector<float>> decibels(X.rows(), std::vector<float>(X.cols(), 0.0f));
    std::vector<std::vector<float>> angle_out(X.rows(), std::vector<float>(X.cols(), 0.0f));
	for (int i = 0; i < X_vector.size(); ++i) {
		for (int j = 0; j < X_vector[i].size(); ++j) {
			decibels[i][j] = magnitudeToDecibel(X_vector[i][j]);
            angle_out[i][j] = angle(X_vector[i][j]);
		//	std::cout << "Decibel value at (" << i << ", " << j << "): " << decibels[i][j] << std::endl;
		}
	}

    //std::vector<std::vector<float>> transposedValues = transposeVector(decibels);
    //std::vector<std::vector<float>> transposeAValues = transposeVector(angle_out);
	t_A_Values = angle_out;
	return decibels;


  }
  
  static std::vector<float> istft(std::vector<std::vector<float>>& transposedValues_in, std::vector<std::vector<float>> t_A_Values,
										  int n_fft, int n_hop,
										  int win_length,
										  const std::string& win, bool center,
										  const std::string& mode) {
	  
	  //std::vector<std::vector<float>> transposedValues_T = transposeVector(transposedValues_in);
	  //std::vector<std::vector<float>> transposeAValues_T = transposeVector(t_A_Values);

	  std::vector<std::vector<float>> mag(transposedValues_in.size(), std::vector<float>(transposedValues_in[0].size(), 0.0f));
	  for (int i = 0; i < transposedValues_in.size(); ++i) {
		  for (int j = 0; j < transposedValues_in[i].size(); ++j) {
			  float decibel_nomarlize = (clip(transposedValues_in[i][j], 0.0, 1.0) - 1.0) * 100 + 20.0f;
			  mag[i][j] = std::pow(10.0, decibel_nomarlize * 0.05);
		  }
	  }

	  std::vector<std::vector<std::complex<float>>> stft_matrix(transposedValues_in.size(), std::vector<std::complex<float>>(transposedValues_in[0].size()));

	  for (size_t i = 0; i < mag.size(); i++) {
		  for (size_t j = 0; j < mag[i].size(); j++) {
			  std::complex<float> exp_part = std::exp(std::complex<float>(0.0f, t_A_Values[i][j]));
			  stft_matrix[i][j] = mag[i][j] * exp_part;
		  }
	  }

	  Matrixcf ifft_input = internal::convertToMatrix(stft_matrix);
	  std::vector<float>  ifft_output = internal::istft(ifft_input, n_fft, n_hop, win_length, win, center, mode, paded_win, paded_win_ifft);
	  return ifft_output;
  }
  
  /// \brief      compute mel spectrogram similar with librosa.feature.melspectrogram
  /// \param      x             input audio signal
  /// \param      sr            sample rate of 'x'
  /// \param      n_fft         length of the FFT size
  /// \param      n_hop         number of samples between successive frames
  /// \param      win           window function. currently only supports 'hann'
  /// \param      center        same as librosa
  /// \param      mode          pad mode. support "reflect","symmetric","edge"
  /// \param      power         exponent for the magnitude melspectrogram
  /// \param      n_mels        number of mel bands
  /// \param      f_min         lowest frequency (in Hz)
  /// \param      f_max         highest frequency (in Hz)
  /// \return     mel spectrogram matrix

/*  static std::vector<std::vector<float>> melspectrogram(std::vector<float>& x, int sr,
                                                        int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode,
                                                        float power, int n_mels, int fmin, int fmax){
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixf mel = internal::melspectrogram(map_x, sr, n_fft, n_hop, n_fft, win, center, mode, power, n_mels, fmin, fmax).transpose();
    std::vector<std::vector<float>> mel_vector(mel.rows(), std::vector<float>(mel.cols(), 0.f));
    for (int i = 0; i < mel.rows(); ++i){
      auto &row = mel_vector[i];
      Eigen::Map<Vectorf>(row.data(), row.size()) = mel.row(i);
    }
    return mel_vector;
  }
 */

  /// \brief      compute mfcc similar with librosa.feature.mfcc
  /// \param      x             input audio signal
  /// \param      sr            sample rate of 'x'
  /// \param      n_fft         length of the FFT size
  /// \param      n_hop         number of samples between successive frames
  /// \param      win           window function. currently only supports 'hann'
  /// \param      center        same as librosa
  /// \param      mode          pad mode. support "reflect","symmetric","edge"
  /// \param      power         exponent for the magnitude melspectrogram
  /// \param      n_mels        number of mel bands
  /// \param      f_min         lowest frequency (in Hz)
  /// \param      f_max         highest frequency (in Hz)
  /// \param      n_mfcc        number of mfccs
  /// \param      norm          ortho-normal dct basis
  /// \param      type          dct type. currently only supports 'type-II'
  /// \return     mfcc matrix
 /* static std::vector<std::vector<float>> mfcc(std::vector<float>& x, int sr,
                                              int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode,
                                              float power, int n_mels, int fmin, int fmax,
                                              int n_mfcc, bool norm, int type) {
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixf mel = internal::melspectrogram(map_x, sr, n_fft, n_hop, n_fft, win, center, mode, power, n_mels, fmin, fmax).transpose();
    Matrixf mel_db = internal::power2db(mel);
    Matrixf dct = internal::dct(mel_db, norm, type).leftCols(n_mfcc);
    std::vector<std::vector<float>> mfcc_vector(dct.rows(), std::vector<float>(dct.cols(), 0.f));
    for (int i = 0; i < dct.rows(); ++i) {
      auto &row = mfcc_vector[i];
      Eigen::Map<Vectorf>(row.data(), row.size()) = dct.row(i);
    }
    return mfcc_vector;
  }
*/
};

} // namespace librosa

#endif
