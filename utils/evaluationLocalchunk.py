import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources


def validate(audio, model, embedder, testloader, writer, step, n):
    device = torch.device('cpu')
    model.eval()
    
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in testloader:
            dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase = batch[0]

            dvec_mel = dvec_mel.to(device)
            target_mag = target_mag.unsqueeze(0).to(device)
            mixed_mag = mixed_mag.unsqueeze(0).to(device)

            dvec = embedder(dvec_mel)
            dvec = dvec.unsqueeze(0)

            est_list = []
            target_list = []
            mask_list = []
            for i in range(0, mixed_mag.shape[1], n):
                # 检查剩余帧数是否小于n
                if mixed_mag.shape[1] - i < n:
                    break
                mixed_tmp = mixed_mag[:, i:i + n, :]
                est_mask = model(mixed_tmp, dvec)
                est_mag = mixed_tmp * est_mask
                est_list.append(est_mag)
                mask_list.append(est_mask)
                target_list.append(target_mag[:, i:i + n, :])

            # 将列表中的所有 output_tmp 和 target_mag 合并为一个大的张量
            est_tensor = torch.cat(est_list, dim=1)
            target_tensor = torch.cat(target_list, dim=1)
            mask_tensor = torch.cat(mask_list, dim=1)

            test_loss = criterion(target_tensor, est_tensor).item()

            mixed_mag = mixed_mag[0].cpu().detach().numpy()
            target_mag = target_tensor[0].cpu().detach().numpy()
            est_mag = est_tensor[0].cpu().detach().numpy()
            mixed_phase = mixed_phase[0:est_mag.shape[0],:]
            est_wav = audio.spec2wav(est_mag, mixed_phase)
            est_mask = mask_tensor[0].cpu().detach().numpy()

            mixed_wav = mixed_wav[0:est_wav.shape[0]]
            target_wav = target_wav[0:est_wav.shape[0]]
            sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
            writer.log_evaluation(test_loss, sdr,
                                  mixed_wav, target_wav, est_wav,
                                  mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
                                  step)
            break

    model.train()
