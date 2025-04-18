import torch
import torch.nn as nn
from architectures.generic_spatiotemporal_decoder import GenericCNNEncoder, TransformerDecoder

class SpatioTemporalRegressor(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embedding_dim), nn.ReLU()
        )

        self.temporal_fc = nn.Sequential(
            nn.Linear(1, embedding_dim), nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, 13 * 13)
        )

    def forward(self, x0, t):
        # x0: (B, C, 13, 13)
        # t:  (B, 1)
        x_feat = self.encoder(x0)       # (B, D)
        t_feat = self.temporal_fc(t)    # (B, D)
        combined = torch.cat([x_feat, t_feat], dim=1)  # (B, 2D)
        out = self.mlp(combined)        # (B, 169)
        return out.view(-1, 1, 13, 13)  # (B, 1, 13, 13)


class GenericSpatioTemporalRegressor(nn.Module):
    def __init__(self, embedding_dim=512, num_layers=6, nhead=8, in_channels=3):
        super().__init__()
        self.encoder = GenericCNNEncoder(in_channels=in_channels, embedding_dim=embedding_dim)

        self.temporal_fc = nn.Sequential(
            nn.Linear(1, embedding_dim), nn.ReLU()
        )

        self.temporal_decoder = TransformerDecoder(
            embedding_dim=embedding_dim * 2,
            num_layers=num_layers,
            nhead=nhead
        )

    def forward(self, x0, t):  # x0: (B, C, 13, 13), t: (B, 1)
        B = x0.size(0)
        x_feat = self.encoder(x0)         # (B, D)
        t_feat = self.temporal_fc(t)      # (B, D)
        combined = torch.cat([x_feat, t_feat], dim=-1)  # (B, 2D)
        combined = combined.unsqueeze(1)  # (B, 1, 2D)

        y_seq = self.temporal_decoder(combined)  # (B, 1, 1, 13, 13)
        return y_seq.squeeze(1)  # (B, 1, 13, 13)
