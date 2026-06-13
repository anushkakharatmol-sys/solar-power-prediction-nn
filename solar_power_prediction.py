"""
Solar Power Prediction using Neural Network
Author: Anushka Kharatmol
Project: 2
Goal: Predict solar power generation using weather data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("=== Solar Power Prediction using Neural Network ===\n")

# ====================== 1. DATA GENERATION ======================
np.random.seed(42)                    # For reproducibility

n_samples = 1500

# Generate weather features
temperature = np.random.uniform(15, 45, n_samples)      # Temperature in °C
humidity = np.random.uniform(10, 90, n_samples)         # Humidity in %
irradiance = np.random.uniform(100, 1100, n_samples)    # Solar Irradiance (W/m²)
wind_speed = np.random.uniform(0, 15, n_samples)        # Wind Speed (km/h)

# Realistic Solar Power calculation
solar_power = (irradiance * 0.0078 * 
              (1 - 0.005 * (temperature - 25)) *        # Temperature effect
              (1 - 0.003 * humidity) *                  # Humidity effect
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

# ====================== 2. PREPARE DATA ======================
X = df.drop('Solar_Power', axis=1).values      # Input features (4 columns)
y = df['Solar_Power'].values.reshape(-1, 1)    # Target variable

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# ====================== 3. NEURAL NETWORK MODEL ======================
class SolarPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)     # Input layer: 4 features → 64 neurons
        self.fc2 = nn.Linear(64, 32)    # Hidden layer 1
        self.fc3 = nn.Linear(32, 16)    # Hidden layer 2
        self.fc4 = nn.Linear(16, 1)     # Output layer: 1 value (Power)
        
        self.relu = nn.ReLU()           # Activation function
        self.dropout = nn.Dropout(0.2)  # Prevent overfitting
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SolarPredictor()
print("✅ Neural Network Model Created!")

# ====================== 4. LOSS FUNCTION & OPTIMIZER ======================
criterion = nn.MSELoss()                                 # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam Optimizer

# ====================== 5. TRAINING ======================
epochs = 1000
losses = []

print("🚀 Training Started...\n")

for epoch in range(epochs):
    # Forward Pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backpropagation + Gradient Descent
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
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.show()

# ====================== 6. TESTING ======================
model.eval()   # Evaluation mode

test_input = torch.FloatTensor([
    [32, 35, 980, 9],     # Good Sunny Day
    [25, 78, 450, 11],    # Cloudy + Humid
    [40, 22, 1080, 4]     # Hot Clear Day
])

with torch.no_grad():
    predictions = model(test_input)

print("\n=== TEST PREDICTIONS ===")
print("Good Sunny Day   → Predicted Power:", round(predictions[0].item(), 2), "kW")
print("Cloudy Humid Day → Predicted Power:", round(predictions[1].item(), 2), "kW")
print("Hot Clear Day    → Predicted Power:", round(predictions[2].item(), 2), "kW")
