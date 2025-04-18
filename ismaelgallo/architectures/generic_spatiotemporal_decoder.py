import torch
import torch.nn as nn


class GenericCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, embedding_dim)
        )

    def forward(self, x):  # x: (B*T, C, 13, 13)
        return self.encoder(x)  # (B*T, D)


def add_positional_encoding(x):
    B, T, D = x.shape
    position = torch.arange(T, dtype=torch.float, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2, device=x.device) * -(torch.log(torch.tensor(10000.0)) / D))
    pe = torch.zeros(T, D, device=x.device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).expand(B, -1, -1)  # (B, T, D)
    return x + pe


def add_temporal_channel(x):
    B, T, C, H, W = x.shape
    t = torch.linspace(0, 1, steps=T, device=x.device).view(1, T, 1, 1, 1)
    t_channel = t.expand(B, T, 1, H, W)
    return torch.cat([x, t_channel], dim=2)


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim=512, num_layers=6, nhead=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_map = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 13 * 13)
        )

    def forward(self, x):  # x: (B, T, D)
        x = self.transformer(x)         # (B, T, D)
        x = self.to_map(x)              # (B, T, 169)
        x = x.view(x.size(0), x.size(1), 1, 13, 13)  # (B, T, 1, 13, 13)
        return x


class GenericSpatioTemporalDecoder(nn.Module):
    def __init__(self, embedding_dim=512, num_layers=6, nhead=8, in_channels=3, use_temporal_channel=False):
        super().__init__()
        self.use_temporal_channel = use_temporal_channel
        final_in_channels = in_channels + 1 if use_temporal_channel else in_channels
        self.encoder = GenericCNNEncoder(in_channels=final_in_channels, embedding_dim=embedding_dim)
        self.temporal_decoder = TransformerDecoder(embedding_dim=embedding_dim, num_layers=num_layers, nhead=nhead)

    def forward(self, x):  # x: (B, T, C, 13, 13)
        if self.use_temporal_channel:
            x = add_temporal_channel(x)  # (B, T, C+1, 13, 13)

        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)            # (B*T, C, 13, 13)
        encoded = self.encoder(x)             # (B*T, D)
        encoded = encoded.view(B, T, -1)      # (B, T, D)
        encoded = add_positional_encoding(encoded)  # (B, T, D)
        out = self.temporal_decoder(encoded)  # (B, T, 1, 13, 13)
        return out
