import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class GenericCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embedding_dim)
        )

    def forward(self, x):  # x: (B*T, 3, 13, 13)
        return self.encoder(x)  # (B*T, D)


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim=128, num_layers=2, nhead=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_map = nn.Linear(embedding_dim, 13 * 13)

    def forward(self, x):  # x: (B, T, D)
        x = self.transformer(x)         # (B, T, D)
        x = self.to_map(x)              # (B, T, 169)
        x = x.view(x.size(0), x.size(1), 1, 13, 13)  # (B, T, 1, 13, 13)
        return x


class GenericSpatioTemporalDecoder(nn.Module):
    def __init__(self, embedding_dim=128, num_layers=2, nhead=4, in_channels=3):
        super().__init__()
        self.encoder = GenericCNNEncoder(in_channels=in_channels, embedding_dim=embedding_dim)
        self.temporal_decoder = TransformerDecoder(embedding_dim=embedding_dim, num_layers=num_layers, nhead=nhead)

    def forward(self, x):  # x: (B, T, 3, 13, 13)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)            # (B*T, 3, 13, 13)
        encoded = self.encoder(x)             # (B*T, D)
        encoded = encoded.view(B, T, -1)      # (B, T, D)
        out = self.temporal_decoder(encoded)  # (B, T, 1, 13, 13)
        return out


# # === ENTRENAMIENTO DE EJEMPLO ===
# class DummyThermalDataset(Dataset):
#     def __init__(self, num_samples=1000, num_steps=100):
#         self.x = torch.randn(num_samples, num_steps, 3, 13, 13)
#         self.y = torch.randn(num_samples, num_steps, 1, 13, 13)

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]


# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     for x, y in dataloader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         y_pred = model(x)
#         loss = criterion(y_pred, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * x.size(0)
#     return total_loss / len(dataloader.dataset)


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = GenericSpatioTemporalDecoder().to(device)
#     dataset = DummyThermalDataset(num_samples=500, num_steps=100)
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)

#     for epoch in range(5):
#         loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
#         print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

#     # Simulación de inferencia
#     x0 = torch.randn(1, 100, 3, 13, 13).to(device)  # (1, T, 3, 13, 13)
#     y_pred = model(x0).squeeze(0)  # (T, 1, 13, 13)
#     print("Predicted sequence shape:", y_pred.shape)
