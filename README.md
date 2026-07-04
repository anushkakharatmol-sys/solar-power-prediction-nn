# Solar Power Prediction — Real Data Edition

Rebuilt version of the original repo, fixing the two biggest issues: fake
random data, and no train/test split.

## What's different from the original

| | Original | This version |
|---|---|---|
| Data | `np.random.uniform` + a made-up formula | Real NREL TMY3 weather-station data |
| Target | Hand-written formula | pvlib's PVWatts model (industry-standard) |
| Split | None (trained & tested on same rows) | Chronological 70/15/15 train/val/test |
| Scaling | Fit on all data | Fit on train only (no leakage) |
| Training | Fixed epoch count | Early stopping + LR scheduling |
| Results | Static plots | Interactive dashboard (`dashboard.html`) |

## Data source

[pvlib](https://pvlib-python.readthedocs.io) ships a real NREL **TMY3**
(Typical Meteorological Year) file for Sand Point, Alaska — genuine hourly
measurements of solar irradiance (GHI/DNI/DHI), temperature, wind, humidity,
and cloud cover.

The target (AC power output of a modeled 5kW rooftop system) is computed by
running that real weather through pvlib's solar-position → plane-of-array
irradiance → cell-temperature → **PVWatts** chain — the same model NREL and
utilities use for production estimates. So the network has to learn a real,
nonlinear physical relationship from raw weather inputs, not just copy a
formula it was trained on.

**Honest caveat:** this is a physics-model estimate, not a utility-metered
reading, so it won't capture things like snow cover, shading, or inverter
clipping in a real installation. Swap in metered data from your own system
if you have it, and the same pipeline still works.

## Files

- `data_pipeline.py` — loads the real TMY3 data, runs it through PVWatts, engineers features
- `train.py` — trains the PyTorch model with a proper chronological split, early stopping, exports `results.json`
- `dashboard.html` — **open this directly in a browser** for the interactive results (loss curve, day-by-day explorer, accuracy scatter)

## Running it yourself

```bash
pip install torch pvlib scikit-learn pandas numpy
python data_pipeline.py   # sanity-check the dataset
python train.py           # trains the model, prints test MAE/RMSE/R², writes results.json
```

Current run: **R² = 0.99, MAE ≈ 52W** on 1,314 held-out hours (54 full days)
the model never trained on.

## Extending it

- Point `data_pipeline.py` at your own inverter's CSV export instead of the bundled TMY3 file for real metered ground truth
- Swap in a sunnier NREL station or pull live data from PVGIS/NSRDC
- Add an LSTM/Transformer for multi-step-ahead forecasting instead of same-hour prediction
