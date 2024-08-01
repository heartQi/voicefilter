import os
import math
import torch
import torch.nn as nn
import traceback
import numpy as np

from .adabound import AdaBound
from .audio import Audio
from .evaluationLocalchunk import validate
from model.model import VoiceFilter
#from model.modelmosformer import VoiceFilter
from model.embedder import SpeechEmbedder


def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    device = torch.device('cpu')
    # load embedder
    embedder_pt = torch.load(args.embedder_path, map_location=torch.device('cpu'))
    embedder = SpeechEmbedder(hp).to(device)
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)
    model = VoiceFilter(hp).to(device)
    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    try:
        criterion = nn.MSELoss()
        while True:
            model.train()
            for dvec_mels, target_mag, mixed_mag in trainloader:
                target_mag = target_mag.to(device)
                mixed_mag = mixed_mag.to(device)

                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.to(device)
                    dvec = embedder(mel)
                    dvec_list.append(dvec)
                dvec = torch.stack(dvec_list, dim=0)
                dvec = dvec.detach()

                n = 20
                output_list = []
                target_list = []
                for i in range(0, mixed_mag.shape[1], n):
                    # 检查剩余帧数是否小于n
                    if mixed_mag.shape[1] - i < n:
                        break
                    mixed_tmp = mixed_mag[:, i:i + n, :]
                    mask = model(mixed_tmp, dvec)
                    output_tmp = mixed_tmp * mask
                    output_list.append(output_tmp)
                    target_list.append(target_mag[:, i:i + n, :])

                # 将列表中的所有 output_tmp 和 target_mag 合并为一个大的张量
                output_tensor = torch.cat(output_list, dim=1)
                target_tensor = torch.cat(target_list, dim=1)

                # 计算损失
                loss = criterion(output_tensor, target_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                print(loss)
                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                # write loss to tensorboard
                if step % hp.train.summary_interval == 0:
                    writer.log_training(loss, step)
                    logger.info("Wrote summary at step %d" % step)

                # 1. save checkpoint file to resume training
                # 2. evaluate and save sample to tensorboard
                if step % hp.train.checkpoint_interval == 0:
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'hp_str': hp_str,
                    }, save_path)
                    logger.info("Saved checkpoint to: %s" % save_path)
                    validate(audio, model, embedder, testloader, writer, step, n)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
