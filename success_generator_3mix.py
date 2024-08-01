import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
import soundfile as sf
from multiprocessing import Pool, cpu_count

from utils.audio import Audio
from utils.hparams import HParam


def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def mix(hp, args, audio, num, s1_dvec, s1_target, s2, s3, train):
    srate = hp.audio.sample_rate
    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')

    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    w3, _ = librosa.load(s3, sr=srate)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == len(w3.shape) == 1, \
        'wav files must be mono, not stereo'

    d, _ = librosa.effects.trim(d, top_db=20)
    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)
    w3, _ = librosa.effects.trim(w3, top_db=20)

    # if reference for d-vector is too short, discard it
    if d.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
        return

    # LibriSpeech dataset have many silent interval, so let's vad-merge them
    # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
    if args.vad == 1:
        w1, w2, w3 = vad_merge(w1), vad_merge(w2), vad_merge(w3)

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L or w3.shape[0] < L:
        return
    w1, w2, w3 = w1[:L], w2[:L], w3[:L]

    mixed = w1 + w2 + w3

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, w3, mixed = w1/norm, w2/norm, w3/norm, mixed/norm

    # save vad & normalized wav files
    target_wav_path = formatter(dir_, hp.form.target.wav, num)
    mixed_wav_path = formatter(dir_, hp.form.mixed.wav, num)
    sf.write(target_wav_path, w1, srate)
    sf.write(mixed_wav_path, mixed, srate)

    # save magnitude spectrograms
    target_mag, _ = audio.wav2spec(w1)
    mixed_mag, _ = audio.wav2spec(mixed)
    target_mag_path = formatter(dir_, hp.form.target.mag, num)
    mixed_mag_path = formatter(dir_, hp.form.mixed.mag, num)
    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

    # save selected sample as text file. d-vec will be calculated soon
    dvec_text_path = formatter(dir_, hp.form.dvec, num)
    with open(dvec_text_path, 'w') as f:
        f.write(s1_dvec)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-d', '--libri_dir', type=str, default=None,
                        help="Directory of LibriSpeech dataset, containing folders of train-clean-100, train-clean-360, dev-clean.")
    parser.add_argument('-v', '--voxceleb_dir', type=str, default=None,
                        help="Directory of VoxCeleb2 dataset, ends with 'aac'")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-p', '--process_num', type=int, default=None,
                        help='number of processes to run. default: cpu_count')
    parser.add_argument('--vad', type=int, default=0,
                        help='apply vad to wav file. yes(1) or no(0, default)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)

    # 从配置文件中加载超参数
    hp = HParam(args.config)

    # 根据超参数创建音频处理对象
    audio = Audio(hp)

    # 获取CPU核心数量
    cpu_num = cpu_count() if args.process_num is None else args.process_num

    # 以下为原代码中的数据加载和处理部分

    # 加载数据集
    if args.libri_dir is None and args.voxceleb_dir is None:
        raise Exception("Please provide directory of data")

    if args.libri_dir is not None:
        train_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-100', '*'))
                            if os.path.isdir(x)] + \
                        [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-360', '*'))
                            if os.path.isdir(x)]
                        # we recommned to exclude train-other-500
                        # See https://github.com/mindslab-ai/voicefilter/issues/5#issuecomment-497746793
                        # + \
                        #[x for x in glob.glob(os.path.join(args.libri_dir, 'train-other-500', '*'))
                        #    if os.path.isdir(x)]
        test_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'dev-clean', '*'))]

    elif args.voxceleb_dir is not None:
        all_folders = [x for x in glob.glob(os.path.join(args.voxceleb_dir, '*'))
                            if os.path.isdir(x)]
        train_folders = all_folders[:-20]
        test_folders = all_folders[-20:]

    train_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in train_folders]
    train_spk = [x for x in train_spk if len(x) >= 2]

    test_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in test_folders]
    test_spk = [x for x in test_spk if len(x) >= 2]

    # 定义数据处理函数
    def train_wrapper(num, hp, args, audio, train_spk):
        spk1, spk2, spk3 = random.sample(train_spk, 3)
        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)
        s3 = random.choice(spk3)
        mix(hp, args, audio, num, s1_dvec, s1_target, s2, s3, train=True)

    def test_wrapper(num, hp, args, audio, test_spk):
        spk1, spk2, spk3 = random.sample(test_spk, 3)
        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)
        s3 = random.choice(spk3)
        mix(hp, args, audio, num, s1_dvec, s1_target, s2, s3, train=False)

    # 遍历数据集并生成训练和测试样本
    for i in tqdm.tqdm(range(10**2), desc='Generating train samples'):
        train_wrapper(i, hp, args, audio, train_spk)

    for i in tqdm.tqdm(range(10**1), desc='Generating test samples'):
        test_wrapper(i, hp, args, audio, test_spk)




