import os
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# Step 1: Load data
# -----------------------------
train_data = np.load("data/processed/train.npy")  # (N, T, C)

# Convert to tensor: (batch, channels, time)
train_data = torch.tensor(train_data, dtype=torch.float32).permute(0, 2, 1)

print("Train tensor shape:", train_data.shape)  # (N, C, T)

# -----------------------------
# Step 2: TCN Block
# -----------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        # Remove extra padding to keep length same (causal effect)
        out = out[:, :, :x.size(2)]

        res = x if self.downsample is None else self.downsample(x)
        return out + res

# -----------------------------
# Step 3: Autoencoder
# -----------------------------
class TCNAutoencoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.encoder = nn.Sequential(
            TCNBlock(in_ch, 32, 3, 1),
            TCNBlock(32, 64, 3, 2)
        )

        self.decoder = nn.Sequential(
            TCNBlock(64, 32, 3, 1),
            nn.Conv1d(32, in_ch, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# -----------------------------
# Step 4: Training setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TCNAutoencoder(in_ch=train_data.shape[1]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# -----------------------------
# Step 5: Training loop
# -----------------------------
epochs = 10
batch_size = 32

for epoch in range(epochs):
    perm = torch.randperm(train_data.size(0))

    total_loss = 0

    for i in range(0, train_data.size(0), batch_size):
        batch_idx = perm[i:i+batch_size]
        batch = train_data[batch_idx].to(device)

        output = model(batch)
        loss = loss_fn(output, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# -----------------------------
# Step 6: Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/tcn_autoencoder.pth")

print("✅ Model training complete & saved!")