import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Step 1: Paths
# -----------------------------
DATA_DIR = "data/raw/nasa"

train_path = os.path.join(DATA_DIR, "train")
test_path = os.path.join(DATA_DIR, "test")

print("Train path exists:", os.path.exists(train_path))
print("Test path exists:", os.path.exists(test_path))


# -----------------------------
# Step 2: Load MULTIPLE files
# -----------------------------
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)

print(f"Total train files: {len(train_files)}")

# Load multiple files (start with 5 to avoid memory issues)
selected_files = train_files[:5]

train_list = []
test_list = []

for file in selected_files:
    train_data = np.load(os.path.join(train_path, file))
    test_data = np.load(os.path.join(test_path, file))

    train_list.append(train_data)
    test_list.append(test_data)

# Combine all
train_data = np.concatenate(train_list, axis=0)
test_data = np.concatenate(test_list, axis=0)

print("Combined Train shape:", train_data.shape)
print("Combined Test shape:", test_data.shape)

# -----------------------------
# Step 3: Normalize
# -----------------------------
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# -----------------------------
# Step 4: Windowing
# -----------------------------
def create_windows(data, window_size=50):
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i+window_size])
    return np.array(windows)

window_size = 50

train_windows = create_windows(train_scaled, window_size)
test_windows = create_windows(test_scaled, window_size)

print("Windowed Train shape:", train_windows.shape)
print("Windowed Test shape:", test_windows.shape)

# -----------------------------
# Step 5: Save
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/train.npy", train_windows)
np.save("data/processed/test.npy", test_windows)

print("✅ Preprocessing complete!")