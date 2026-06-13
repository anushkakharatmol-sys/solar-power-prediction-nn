# Solar Power Prediction using Neural Network

This project uses a **Feedforward Neural Network** built with PyTorch to predict solar power generation based on weather conditions.

##  Objective
Predict solar power output (kW) using:
- Temperature (°C)
- Humidity (%)
- Solar Irradiance (W/m²)
- Wind Speed (km/h)

##  Technologies Used
- Python
- PyTorch
- NumPy, Pandas, Matplotlib

##  Files
- `solar_power_prediction.py` → Main code with detailed comments

##  How to Run
```bash
pip install torch numpy pandas matplotlib
python solar_power_prediction.py
Model Architecture

4 Input Features
4 Hidden Layers (64 → 32 → 16 → 1)
ReLU activation + Dropout
MSE Loss + Adam Optimizer

Future Plans

Add real solar dataset
Implement LSTM for time series forecasting
Create a web app for predictions
