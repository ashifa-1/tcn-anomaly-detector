import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# -----------------------------
# Step 1: Load test data
# -----------------------------
test_data = np.load("data/processed/test.npy")  # (N, T, C)
test_tensor = torch.tensor(test_data, dtype=torch.float32).permute(0, 2, 1)

print("Test tensor shape:", test_tensor.shape)

# -----------------------------
# Step 2: Load model
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
        out = out[:, :, :x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        return out + res


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
        return self.decoder(self.encoder(x))


device = "cuda" if torch.cuda.is_available() else "cpu"

model = TCNAutoencoder(in_ch=test_tensor.shape[1]).to(device)
model.load_state_dict(torch.load("models/tcn_autoencoder.pth"))
model.eval()

# -----------------------------
# Step 3: Reconstruction + error
# -----------------------------
errors = []

with torch.no_grad():
    for i in range(test_tensor.size(0)):
        x = test_tensor[i:i+1].to(device)
        recon = model(x)

        error = torch.mean((x - recon) ** 2).item()
        errors.append(error)

errors = np.array(errors)

# -----------------------------
# Step 4: Smoothing (EMA)
# -----------------------------
def ema(data, alpha=0.1):
    smoothed = []
    prev = data[0]
    for val in data:
        prev = alpha * val + (1 - alpha) * prev
        smoothed.append(prev)
    return np.array(smoothed)

smoothed_errors = ema(errors)

# -----------------------------
# Step 5: Threshold
# -----------------------------
threshold = np.percentile(smoothed_errors, 95)

anomalies = np.where(smoothed_errors > threshold)[0]

print("Threshold:", threshold)
print("Number of anomalies:", len(anomalies))

# -----------------------------
# Step 6: Save results
# -----------------------------
os.makedirs("results", exist_ok=True)

df = pd.DataFrame({
    "index": range(len(errors)),
    "raw_error": errors,
    "smoothed_error": smoothed_errors
})

df.to_csv("results/anomaly_scores.csv", index=False)

pd.DataFrame({
    "index": anomalies,
    "score": smoothed_errors[anomalies]
}).to_csv("results/anomalies_percentile.csv", index=False)

print("✅ Evaluation complete!")