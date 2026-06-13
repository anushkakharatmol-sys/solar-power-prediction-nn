# Solar Power Prediction using Neural Network ⚡

This project uses a **Feedforward Neural Network** built with **PyTorch** to predict solar power generation based on weather conditions.

## Objective
Predict solar power output (kW) using four weather parameters:
- Temperature (°C)
- Humidity (%)
- Solar Irradiance (W/m²)
- Wind Speed (km/h)

## Technologies Used
- Python
- PyTorch
- NumPy, Pandas, Matplotlib

## How to Run
```bash
pip install torch numpy pandas matplotlib
python solar_power_prediction.py
Model Architecture

Input: 4 features
Hidden Layers: 64 → 32 → 16 neurons
Output: 1 (Predicted Solar Power)
ReLU Activation + Dropout (0.15)

Future Improvements

Use real solar dataset from NREL
Implement LSTM for time series forecasting
Build a web app
