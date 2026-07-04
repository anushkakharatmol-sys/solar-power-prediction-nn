"""
Real-world solar data pipeline.

Data source: NREL TMY3 (Typical Meteorological Year) station data for
Sand Point, AK (USAF 703165) -- genuine historical weather-station
measurements (GHI/DNI/DHI irradiance, temperature, wind, humidity,
cloud cover), distributed with the open-source `pvlib` library.

Target variable: AC power output of a modeled 5kW rooftop PV system,
computed with pvlib's PVWatts model -- the same physics-based model
NREL and utilities use for real solar production estimates. This
gives us a physically grounded, non-arbitrary target instead of a
hand-made formula.

The neural network only ever sees the *raw weather features* (the
kind you'd get from a weather API / forecast), never the internal
physical quantities (plane-of-array irradiance, cell temperature)
used to derive the target -- so it has to learn the real nonlinear
relationship, not just copy an input.
"""

import os
import numpy as np
import pandas as pd
import pvlib


def load_raw_weather():
    data_dir = os.path.dirname(pvlib.__file__) + "/data"
    path = os.path.join(data_dir, "703165TY.csv")
    data, meta = pvlib.iotools.read_tmy3(path, coerce_year=2023, map_variables=True)
    return data, meta


def compute_pv_power(data, meta, pdc0=5000, gamma_pdc=-0.004):
    """Physically-based PV AC power output using PVWatts (industry standard)."""
    solpos = pvlib.solarposition.get_solarposition(
        data.index, meta["latitude"], meta["longitude"]
    )
    dni_extra = pvlib.irradiance.get_extra_radiation(data.index)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=meta["latitude"],
        surface_azimuth=180,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=data["dni"],
        ghi=data["ghi"],
        dhi=data["dhi"],
        dni_extra=dni_extra,
        model="haydavies",
    )

    cell_temp = pvlib.temperature.sapm_cell(
        poa["poa_global"],
        data["temp_air"],
        data["wind_speed"],
        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"],
    )

    dc_power = pvlib.pvsystem.pvwatts_dc(poa["poa_global"], cell_temp, pdc0, gamma_pdc)
    ac_power = pvlib.inverter.pvwatts(dc_power, pdc0)
    return ac_power, solpos


def build_dataset(seed=42):
    data, meta = load_raw_weather()
    ac_power, solpos = compute_pv_power(data, meta)

    df = pd.DataFrame(
        {
            "ghi": data["ghi"],
            "dni": data["dni"],
            "dhi": data["dhi"],
            "temp_air": data["temp_air"],
            "wind_speed": data["wind_speed"],
            "relative_humidity": data["relative_humidity"],
            "cloud_cover": data["TotCld (tenths)"],
            "solar_zenith": solpos["apparent_zenith"].values,
            "hour": data.index.hour,
            "day_of_year": data.index.dayofyear,
            "power_w": ac_power.values,
        }
    )

    # Cyclical encodings so midnight/noon and Dec 31/Jan 1 are "close"
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df = df.drop(columns=["hour", "day_of_year"])

    # Realistic sensor noise (real weather stations aren't noise-free)
    rng = np.random.default_rng(seed)
    for col, noise_std in [
        ("ghi", 5.0), ("dni", 5.0), ("dhi", 3.0),
        ("temp_air", 0.3), ("wind_speed", 0.2), ("relative_humidity", 1.5),
    ]:
        df[col] = df[col] + rng.normal(0, noise_std, len(df))
        df[col] = df[col].clip(lower=0)

    df = df.reset_index(drop=True)
    df.insert(0, "timestamp", data.index)
    return df, meta


if __name__ == "__main__":
    df, meta = build_dataset()
    print(f"Station: {meta['Name']}, {meta['State']}  "
          f"(lat={meta['latitude']}, lon={meta['longitude']})")
    print(df.shape)
    print(df.describe())
    df.to_csv("solar_dataset.csv", index=False)
    print("Saved solar_dataset.csv")
