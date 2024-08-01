import numpy as np
import librosa
import soundfile as sf

# 从 WAV 文件中读取数据
y, sr = librosa.load('result.wav', sr=16000)

# 设置帧长度和帧移
frame_length = 1024
hop_length = 512

# 对音频进行分帧和重叠处理
frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

# 逐帧进行短时傅里叶变换和逆变换
reconstructed_signal = []
for frame in frames.T:  # 按行迭代，即逐帧处理
    # 进行短时傅里叶变换
    stft_matrix = librosa.stft(frame, n_fft=frame_length, hop_length=hop_length)

    # 进行逆短时傅里叶变换
    reconstructed_frame = librosa.istft(stft_matrix, hop_length=hop_length)

    # 将重构的帧添加到重构信号中
    reconstructed_signal.extend(reconstructed_frame)

# 将重构的信号转换为 numpy 数组
reconstructed_signal = np.array(reconstructed_signal)

# 将重构的信号写入 WAV 文件
sf.write('reconstructed_audio.wav', reconstructed_signal, sr)
