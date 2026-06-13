"""
Solar Power Prediction using Neural Network
Author: Your Name
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ====================== 1. DATA GENERATION ======================
np.random.seed(42)

n_samples = 1500  # Increased data size

# Generate realistic weather features
temperature = np.random.uniform(15, 45, n_samples)
humidity = np.random.uniform(10, 90, n_samples)
irradiance = np.random.uniform(100, 1100, n_samples)
wind_speed = np.random.uniform(0, 15, n_samples)

# Realistic Solar Power calculation
solar_power = (irradiance * 0.0078 * 
              (1 - 0.005 * (temperature - 25)) *      # Temperature penalty
              (1 - 0.003 * humidity) *                # Humidity penalty
              (1 + 0.012 * wind_speed)) + np.random.normal(0, 6, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Irradiance': irradiance,
    'Wind_Speed': wind_speed,
    'Solar_Power': solar_power
})

print("✅ Dataset Created!")
print(df.head())

# ====================== 2. DATA PREPROCESSING ======================
X = df.drop('Solar_Power', axis=1).values
y = df['Solar_Power'].values.reshape(-1, 1)

# Feature Scaling (Very Important for Neural Networks)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y_scaled)

print("✅ Data Scaled and Ready for Training!")

# ====================== 3. NEURAL NETWORK MODEL ======================
class SolarPowerPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SolarPowerPredictor()
print("✅ Neural Network Model Created!")

# ====================== 4. LOSS & OPTIMIZER ======================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ====================== 5. TRAINING ======================
epochs = 1200
losses = []

print("🚀 Training Started...\n")

for epoch in range(epochs):
    # Forward Pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("\n✅ Training Completed!")

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()

# ====================== 6. TESTING ======================
model.eval()

test_cases = torch.FloatTensor([
    [32, 35, 980, 9],    # Good sunny day
    [24, 78, 450, 11],   # Cloudy + Humid
    [40, 22, 1080, 4]    # Hot & Clear
])

with torch.no_grad():
    predictions = model(test_cases)

print("\n=== TEST PREDICTIONS ===")
conditions = ["Good Sunny Day", "Cloudy Humid Day", "Hot Clear Day"]

for i in range(3):
    print(f"{conditions[i]} → Predicted Solar Power: {predictions[i].item():.2f} kW")
