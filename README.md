# ⚡ Solar Power Prediction using Neural Network

This project uses a **Feedforward Neural Network** built with **PyTorch** to predict solar power generation based on weather conditions. 

## 🎯 Objective
To predict **Solar Power output (kW)** using four key weather parameters:
- Temperature (°C)
- Humidity (%)
- Solar Irradiance (W/m²)
- Wind Speed (km/h)

## 🛠 Technologies Used
- **Python**
- **PyTorch** (Deep Learning framework)
- NumPy, Pandas, Matplotlib

## 📊 Model Architecture
- Input Layer: 4 features
- Hidden Layers: 64 → 32 → 16 neurons
- Output Layer: 1 (Predicted Solar Power)
- Activation Function: ReLU
- Regularization: Dropout (0.2)
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam

## 🚀 How to Run
```bash
# 1. Install dependencies
pip install torch numpy pandas matplotlib

# 2. Run the project
python solar_power_prediction.py
📈 Results
The model learns to predict solar power generation based on weather data. Training loss decreases over epochs showing successful learning.
📌 Key Learnings

Building and training a neural network from scratch
Data preprocessing and feature scaling
Understanding Forward Pass, Backpropagation, and Gradient Descent
Practical application of Neural Networks in Renewable Energy

🔮 Future Improvements

Use real-world solar dataset (NREL or Kaggle)
Implement LSTM / GRU for time series forecasting
Build a web app using Streamlit or Flask
Deploy the model
