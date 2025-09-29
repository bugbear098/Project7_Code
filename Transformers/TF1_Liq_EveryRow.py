import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Parameters ===
DATA_DIR = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Processed_Transformer"
CONFIDENCE_THRESHOLD = 0.6
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Data ===
def load_csvs(data_dir, split):
    X = pd.read_csv(os.path.join(data_dir, f"X_{split}.csv"))
    y = pd.read_csv(os.path.join(data_dir, f"y_{split}.csv"))
    return X.values.astype(np.float32), y.values.astype(np.int64).flatten()

X_train, y_train = load_csvs(DATA_DIR, "train")
X_val, y_val = load_csvs(DATA_DIR, "val")

print("Before cleaning:")
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("NaNs in X_train:", np.isnan(X_train).sum(), " NaNs in y_train:", np.isnan(y_train).sum())

# === Clean data ===
# Remove rows with NaN in features or labels
mask = (~np.isnan(X_train).any(axis=1)) & (~np.isnan(y_train))
X_train, y_train = X_train[mask], y_train[mask]

mask = (~np.isnan(X_val).any(axis=1)) & (~np.isnan(y_val))
X_val, y_val = X_val[mask], y_val[mask]

print("After cleaning:")
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("NaNs in X_train:", np.isnan(X_train).sum(), " NaNs in y_train:", np.isnan(y_train).sum())

# === Map labels: {-1, +1} → {0, 1} ===
y_train = np.where(y_train == -1, 0, 1)
y_val = np.where(y_val == -1, 0, 1)

assert set(np.unique(y_train)).issubset({0,1}), f"Unexpected labels in y_train: {np.unique(y_train)}"
assert set(np.unique(y_val)).issubset({0,1}), f"Unexpected labels in y_val: {np.unique(y_val)}"

# === Scale Features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

n_features = X_train.shape[1]

# === Dataset ===
class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = PriceDataset(X_train, y_train)
val_ds = PriceDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === Transformer Model ===
class TransformerClassifier(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.input_fc = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_fc(x).unsqueeze(1)  # (batch, seq=1, d_model)
        x = self.transformer(x)            # (batch, seq=1, d_model)
        x = x.mean(dim=1)                  # (batch, d_model)
        return self.cls_head(x)

model = TransformerClassifier(n_features).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Training Loop ===
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        if torch.isnan(loss):
            print("NaN detected in train loss — skipping batch")
            continue
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / max(1, len(train_loader))

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for Xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            preds = model(Xb)
            loss = criterion(preds, yb)
            if torch.isnan(loss):
                print("NaN detected in val loss — skipping batch")
                continue
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / max(1, len(val_loader))

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

# === Post-Training Evaluation ===
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for Xb, yb in val_loader:
        Xb = Xb.to(DEVICE)
        logits = model(Xb)
        probs = torch.softmax(logits, dim=1)
        max_probs, raw_preds = probs.max(dim=1)
        # Apply confidence threshold → unsure=2
        preds = torch.where(max_probs < CONFIDENCE_THRESHOLD,
                            torch.tensor(2, device=DEVICE),
                            raw_preds)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.numpy())

# Confusion Matrix
cm = confusion_matrix(all_true, all_preds, labels=[0,1,2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["-1","1","0"])
disp.plot(cmap="Blues")
plt.title("Validation Confusion Matrix")
plt.show()

# Loss Plot
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.show()
