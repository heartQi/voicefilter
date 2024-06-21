import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceFilter(nn.Module):
    def __init__(self, hp):
        super(VoiceFilter, self).__init__()
        self.hp = hp
        assert hp.audio.n_fft // 2 + 1 == hp.audio.num_freq == hp.model.fc2_dim, \
            "stft-related dimension mismatch"

        self.lstm = nn.LSTM(
            hp.audio.num_freq + hp.embedder.emb_dim,
            hp.model.lstm_dim,
            num_layers=3,  # 3层LSTM
            batch_first=True,
            bidirectional=False)  # 单向的

        self.fc = nn.Linear(hp.model.lstm_dim, hp.model.fc2_dim)

    def forward(self, x, dvec):
        # x: [B, T, num_freq]

        # dvec: [B, emb_dim]
        dvec = dvec.unsqueeze(1)
        dvec = dvec.repeat(1, x.size(1), 1)
        # dvec: [B, T, emb_dim]

        x = torch.cat((x, dvec), dim=2)  # [B, T, num_freq + emb_dim]

        x, _ = self.lstm(x)  # [B, T, lstm_dim]
        x = F.relu(x)

        x = self.fc(x)  # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        return x
