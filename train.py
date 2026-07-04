"""
Solar power prediction -- training script.

Fixes vs. the original repo:
  1. Real weather-station data (NREL TMY3) + physics-based PVWatts target,
     instead of `np.random.uniform` + a made-up formula.
  2. Chronological train/val/test split (no shuffling) -- the model is
     evaluated on days it has never seen, simulating real forecasting.
  3. Scaler is fit ONLY on the training set (no leakage from test data).
  4. Early stopping + LR scheduling instead of a fixed epoch count.
  5. Proper metrics reported on a held-out test set, not the training set.
  6. Results are exported to JSON for an interactive dashboard.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_pipeline import build_dataset

torch.manual_seed(42)
np.random.seed(42)

FEATURES = [
    "ghi", "dni", "dhi", "temp_air", "wind_speed", "relative_humidity",
    "cloud_cover", "solar_zenith", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
TARGET = "power_w"


class SolarNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def chronological_split(df, train_frac=0.7, val_frac=0.15):
    """Split by TIME, not randomly -- train on early days, test on later ones."""
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def main():
    df, meta = build_dataset()
    train_df, val_df, test_df = chronological_split(df)

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(train_df[FEATURES])
    y_train = y_scaler.fit_transform(train_df[[TARGET]])
    x_val = x_scaler.transform(val_df[FEATURES])
    y_val = y_scaler.transform(val_df[[TARGET]])
    x_test = x_scaler.transform(test_df[FEATURES])
    y_test_scaled = y_scaler.transform(test_df[[TARGET]])

    x_train_t, y_train_t = to_tensor(x_train), to_tensor(y_train)
    x_val_t, y_val_t = to_tensor(x_val), to_tensor(y_val)
    x_test_t = to_tensor(x_test)

    model = SolarNet(len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val_loss, patience_left, best_state = float("inf"), 20, None
    train_losses, val_losses = [], []

    max_epochs = 300
    batch_size = 64
    n_train = len(x_train_t)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_loss = epoch_loss / n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = 20
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(x_test_t).numpy()
    test_pred = y_scaler.inverse_transform(test_pred_scaled).flatten()
    test_actual = test_df[TARGET].values

    test_pred = np.clip(test_pred, 0, None)  # power can't be negative

    mae = mean_absolute_error(test_actual, test_pred)
    rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
    r2 = r2_score(test_actual, test_pred)

    print("\n=== Held-out TEST set performance (never seen in training) ===")
    print(f"MAE:  {mae:.2f} W")
    print(f"RMSE: {rmse:.2f} W")
    print(f"R^2:  {r2:.4f}")

    # ---- export everything the dashboard needs ----
    export = {
        "meta": {
            "station": meta["Name"].strip('"'),
            "state": meta["State"],
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
        },
        "metrics": {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)},
        "train_losses": [round(float(v), 5) for v in train_losses],
        "val_losses": [round(float(v), 5) for v in val_losses],
        "test": [
            {
                "timestamp": str(ts),
                "actual": round(float(a), 1),
                "predicted": round(float(p), 1),
                "ghi": round(float(g), 1),
            }
            for ts, a, p, g in zip(
                test_df["timestamp"], test_actual, test_pred, test_df["ghi"]
            )
        ],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    with open("results.json", "w") as f:
        json.dump(export, f)
    print("Saved results.json")


if __name__ == "__main__":
    main()
