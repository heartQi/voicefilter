import torch
import torch.nn as nn
import torch.nn.functional as F

class VoiceFilter(nn.Module):
    def __init__(self, hp):
        super(VoiceFilter, self).__init__()
        self.hp = hp
        assert hp.audio.n_fft // 2 + 1 == hp.audio.num_freq == hp.model.fc2_dim, \
            "stft-related dimension mismatch"

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hp.audio.num_freq + hp.embedder.emb_dim,  # 输入维度
            nhead=1,  # 头的数量
            dim_feedforward=2048,  # 前馈维度
            activation='gelu',  # 激活函数
            dropout=0.1,  # 丢弃率
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.fc1 = nn.Linear(hp.audio.num_freq + hp.embedder.emb_dim, hp.model.fc1_dim)
        self.fc2 = nn.Linear(hp.model.fc1_dim, hp.model.fc2_dim)

    def forward(self, x, dvec):
        # x: [B, T, num_freq]
        # dvec: [B, emb_dim]

        # 将 dvec 维度从 [B, emb_dim] 转换为 [B, T, emb_dim]
        dvec = dvec.unsqueeze(1).expand(-1, x.size(1), -1)

        # 将 x 和 dvec 连接在最后一个维度上，得到输入张量
        x = torch.cat((x, dvec), dim=-1)  # [B, T, num_freq + emb_dim]

        # TransformerEncoder 需要的输入形状是 [序列长度, 批量大小, 嵌入维度]
        # 在这里我们将输入张量的维度转换为 [T, B, num_freq + emb_dim]
        x = x.permute(1, 0, 2)

        # 将输入张量传递给 TransformerEncoder
        x = self.transformer_encoder(x)

        # 将输出张量的维度转换回 [B, T, num_freq]
        x = x.permute(1, 0, 2)

        # 使用全连接层进行最终的输出
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

