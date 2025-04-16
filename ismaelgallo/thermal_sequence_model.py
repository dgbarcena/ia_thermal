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

    def forward(self, x):  # x: (B, 3, 13, 13)
        return self.encoder(x)  # (B, D)


class GenericTemporalEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_steps, embedding_dim)

    def forward(self, T):
        time_ids = torch.arange(T, device=self.embedding.weight.device)  # (T,)
        return self.embedding(time_ids)  # (T, D_time)


class GenericSequenceDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_size=13 * 13):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):  # x: (B, T, D+D_time)
        B, T, D = x.shape
        x = x.view(B * T, D)
        out = self.decoder(x)  # (B*T, 13*13)
        return out.view(B, T, 13, 13)


class GenericSpatioTemporalDecoder(nn.Module):
    def __init__(self, num_steps=100, encoder_dim=128, time_emb_dim=32):
        super().__init__()
        self.encoder = GenericCNNEncoder(in_channels=3, embedding_dim=encoder_dim)
        self.time_embedding = GenericTemporalEmbedding(num_steps, time_emb_dim)
        self.decoder = GenericSequenceDecoder(input_dim=encoder_dim + time_emb_dim)
        self.num_steps = num_steps

    def forward(self, x):  # x: (B, 3, 13, 13)
        B = x.shape[0]
        z = self.encoder(x)  # (B, encoder_dim)
        t_embed = self.time_embedding(self.num_steps)  # (T, time_emb_dim)
        t_embed = t_embed.unsqueeze(0).expand(B, -1, -1)  # (B, T, D_time)
        z = z.unsqueeze(1).expand(-1, self.num_steps, -1)  # (B, T, D)
        z_cat = torch.cat([z, t_embed], dim=-1)  # (B, T, D+D_time)
        out = self.decoder(z_cat)  # (B, T, 13, 13)
        return out


# === ENTRENAMIENTO DE EJEMPLO ===
class DummyThermalDataset(Dataset):
    def __init__(self, num_samples=1000, num_steps=100):
        self.x = torch.randn(num_samples, 3, 13, 13)
        self.y = torch.randn(num_samples, num_steps, 13, 13)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenericSpatioTemporalDecoder(num_steps=100).to(device)
    dataset = DummyThermalDataset(num_samples=500, num_steps=100)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    # Simulación de inferencia
    x0 = torch.randn(3, 13, 13).unsqueeze(0).to(device)  # (1, 3, 13, 13)
    y_pred = model(x0).squeeze(0)  # (100, 13, 13)
    print("Predicted sequence shape:", y_pred.shape)