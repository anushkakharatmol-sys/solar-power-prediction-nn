# Solar Power Prediction using Neural Network

This project develops a Feedforward Neural Network using PyTorch to predict solar power generation based on meteorological parameters.

## Objective
To accurately predict solar power output (in kW) using the following weather features:
- Temperature (°C)
- Humidity (%)
- Solar Irradiance (W/m²)
- Wind Speed (km/h)

## Technologies Used
- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Model Architecture
- Input Layer: 4 features
- Hidden Layers: 128 → 64 → 32 neurons
- Output Layer: 1 (Predicted Solar Power)
- Activation Function: ReLU
- Regularization: Dropout (0.2)
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

## Results

**Training Loss Over Time**
![Training Loss](training_loss.png)

**Actual vs Predicted Values**
![Actual vs Predicted](actual_vs_predicted.png)

## Performance Metrics
- Mean Absolute Error (MAE)
- R² Score

## How to Run the Project
```bash
pip install -r requirements.txt
python solar_power_prediction.py

Engineering Application
This project demonstrates the practical application of Artificial Neural Networks in Renewable Energy systems and Smart Grids. It shows how data-driven models can be used for solar power forecasting.
Future Work

Integration with real-world solar datasets
Implementation of LSTM for time series forecasting
Development of a web-based prediction dashboard
