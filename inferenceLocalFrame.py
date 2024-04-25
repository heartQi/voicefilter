import os
import glob
import torch
import librosa
import argparse
import soundfile as sf

from utils.audio import Audio
from utils.hparams import HParam
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder


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
        dvec_wav, _ = librosa.load(args.reference_file, sr=16000)
        dvec_mel = audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().to(device)
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)

        mixed_wav, _ = librosa.load(args.mixed_file, sr=16000)
        mag, phase = audio.wav2spec(mixed_wav)
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
