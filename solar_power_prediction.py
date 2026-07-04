"""
Solar Power Prediction using Neural Network
Author: Anushka Kharatmol
Project for Renewable Energy Applications
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("=== Solar Power Prediction using Neural Network ===\n")

# ====================== 1. DATA GENERATION ======================
np.random.seed(42)
n_samples = 2000

temperature = np.random.uniform(15, 45, n_samples)
humidity = np.random.uniform(10, 90, n_samples)
irradiance = np.random.uniform(100, 1100, n_samples)
wind_speed = np.random.uniform(0, 15, n_samples)

# More realistic solar power formula
solar_power = (irradiance * 0.008 * 
              (1 - 0.005 * (temperature - 25)) * 
              (1 - 0.0035 * humidity) * 
              (1 + 0.015 * wind_speed)) + np.random.normal(0, 7, n_samples)

df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Irradiance': irradiance,
    'Wind_Speed': wind_speed,
    'Solar_Power': solar_power
})

print("Dataset Shape:", df.shape)
print(df.head())

# ====================== 2. DATA PREPROCESSING ======================
X = df.drop('Solar_Power', axis=1).values
y = df['Solar_Power'].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y_scaled)

# ====================== 3. NEURAL NETWORK ======================
class SolarPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SolarPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ====================== 4. TRAINING ======================
epochs = 800
losses = []

for epoch in range(epochs):
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("\nTraining Completed!")

# ====================== 5. EVALUATION ======================
model.eval()
with torch.no_grad():
    predictions_scaled = model(X_tensor)
    predictions = scaler_y.inverse_transform(predictions_scaled.numpy())
    actual = scaler_y.inverse_transform(y_tensor.numpy())

mae = mean_absolute_error(actual, predictions)
r2 = r2_score(actual, predictions)

print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f} kW")
print(f"R² Score: {r2:.4f} (Higher is better)")

# ====================== 6. VISUALIZATION ======================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(actual[:200], predictions[:200], alpha=0.6)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
plt.xlabel('Actual Solar Power (kW)')
plt.ylabel('Predicted Solar Power (kW)')
plt.title('Actual vs Predicted')
plt.grid(True)

plt.tight_layout()
plt.show()
