"""
Solar Power Prediction using Neural Network
Author: Anushka Kharatmol
Goal: Predict solar power using weather data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ====================== 1. DATA GENERATION ======================
np.random.seed(42)

n_samples = 1500

temperature = np.random.uniform(15, 45, n_samples)
humidity = np.random.uniform(10, 90, n_samples)
irradiance = np.random.uniform(100, 1100, n_samples)
wind_speed = np.random.uniform(0, 15, n_samples)

solar_power = (irradiance * 0.0078 * 
              (1 - 0.005 * (temperature - 25)) * 
              (1 - 0.003 * humidity) * 
              (1 + 0.012 * wind_speed)) + np.random.normal(0, 6, n_samples)

df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Irradiance': irradiance,
    'Wind_Speed': wind_speed,
    'Solar_Power': solar_power
})

print("✅ Dataset Created!")
print(df.head())

# ====================== 2. PREPARE DATA ======================
X = df.drop('Solar_Power', axis=1).values
y = df['Solar_Power'].values.reshape(-1, 1)

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# ====================== 3. NEURAL NETWORK ======================
class SolarPredictor(nn.Module):
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

model = SolarPredictor()
print("✅ Model Created!")

# ====================== 4. LOSS & OPTIMIZER ======================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ====================== 5. TRAINING ======================
epochs = 1000
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

print("\n✅ Training Completed!")

# Plot Loss
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# ====================== 6. TESTING ======================
model.eval()

test_input = torch.FloatTensor([
    [32, 35, 980, 9],
    [25, 78, 450, 11],
    [40, 22, 1080, 4]
])

with torch.no_grad():
    predictions = model(test_input)

print("\n=== TEST PREDICTIONS ===")
print("Good Sunny Day    →", predictions[0].item())
print("Cloudy Humid Day  →", predictions[1].item())
print("Hot Clear Day     →", predictions[2].item())
