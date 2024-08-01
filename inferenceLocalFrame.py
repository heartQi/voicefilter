import os
import glob
import torch
import librosa
import argparse
import soundfile as sf

from utils.audio import Audio
from utils.hparams import HParam
from model.model import VoiceFilter
#from model.modelmosformer import VoiceFilter
#from model.modelmish import VoiceFilter
#from model.modeleis import VoiceFilter
#from model.modelsnake import VoiceFilter

from model.embedder import SpeechEmbedder

def dump_mel(dvec, path):
   with open(path, "w") as file:
       # 将数据按行写入文件
       for value in dvec:
           # 将每个值转换为字符串，并写入文件，格式为 "tensor(0.0240)"
           file.write("tensor({:.18f})\n".format(value.item()))
def main(args, hp):
    with torch.no_grad():
        device = torch.device('cpu')
        model = VoiceFilter(hp).to(device)
        chkpt_model = torch.load(args.checkpoint_path, map_location=device)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        embedder = SpeechEmbedder(hp).to(device)
        chkpt_embed = torch.load(args.embedder_path, map_location=device)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()

        audio = Audio(hp)

        if 1:
            dvec_wav, _ = librosa.load(args.reference_file, sr=16000)

        else:
            import scipy.io.wavfile as wavfile
            import numpy as np
            fs_wavfile, dvec_wav = wavfile.read(args.reference_file)
            dvec_wav = dvec_wav.astype(np.float32)

        dvec_mel = audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().to(device)
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)
        dump_mel(dvec[0, :], "./dvec_rc_win_admin.data")


        if 1:
            mixed_wav, _ = librosa.load(args.mixed_file, sr=16000)
            mag, phase = audio.wav2spec(mixed_wav)
        else:
            n_fft = 512
            hop_length = 160
            win_length = 256
            fs_wavfile, mixed_wav = wavfile.read(args.mixed_file)
            mixed_wav = mixed_wav.astype(np.float32)
            #fft_result = np.fft.fft(mixed_wav)
            fft_result = np.array(
                [np.fft.fft(mixed_wav[i:i + win_length], n_fft) for i in range(0, len(mixed_wav) - win_length, hop_length)])

            # 获取振幅谱
            mag = np.abs(fft_result)[:, :n_fft // 2 + 1]

            # 获取相位谱
            phase = np.angle(fft_result)[:, :n_fft // 2 + 1]

        mag = torch.from_numpy(mag).float().to(device)

        if 0:
            # 对于每个时间帧，分别计算遮罩并应用到幅度谱上
            for i in range(mag.shape[0]):
                # 获取当前时间帧的幅度谱
                mag_frame = mag[i].unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
                # 使用模型计算遮罩
                mask = model(mag_frame, dvec)
                # 将遮罩应用到当前时间帧的幅度谱上
                est_mag_frame = mag_frame * mask
                # 将处理后的幅度谱放回原始 mag 中对应的位置
                mag[i] = est_mag_frame.squeeze(0).squeeze(0)
        else:
            n = 20
            # 对于每5个时间帧，一起计算遮罩并应用到幅度谱上
            for i in range(0, mag.shape[0], n):
                # 获取当前5个时间帧的幅度谱
                mag_frames = mag[i:i + n].unsqueeze(0)  # 添加 channel 维度
                # 使用模型计算遮罩
                mask = model(mag_frames, dvec)
                # 将遮罩应用到当前5个时间帧的幅度谱上
                est_mag_frames = mag_frames * mask
                # 将处理后的幅度谱放回原始 mag 中对应的位置
                mag[i:i + n] = est_mag_frames.squeeze(1)

        # 将处理后的 mag 转换为 NumPy 数组
        mag = mag.cpu().numpy()
        # 将处理后的 mag 和相位信息转换回音频信号
        est_wav = audio.spec2wav(mag, phase)

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, 'result.wav')
        sf.write(out_path, est_wav, samplerate=16000)
        # librosa.output.write_wav(out_path, est_wav, sr=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--mixed_file', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-r', '--reference_file', type=str, required=True,
                        help='path of reference wav file')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='directory of output')

    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)
