from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def predict_precip_mm(
    model_bundle: Dict[str, Any],
    ds,
    ds_climo,
    lat: float,
    lon: float,
    date_str: str,
) -> Dict[str, float]:
    """Shared inference helper for Flask-side or CLI usage."""
    clf = model_bundle["clf"]
    reg = model_bundle["reg"]
    X_cols = model_bundle["X_cols"]

    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise ValueError("Could not locate lat/lon coords in ds.")

    dt = pd.to_datetime(date_str)
    doy = int(dt.strftime("%j"))
    row = {
        "lat": float(ds[lat_name].sel({lat_name: lat}, method="nearest").values),
        "lon": float(ds[lon_name].sel({lon_name: lon}, method="nearest").values),
        "doy": doy,
        "doy_sin": np.sin(2 * np.pi * doy / 365.0),
        "doy_cos": np.cos(2 * np.pi * doy / 365.0),
    }

    cl = ds_climo.sel(
        doy=doy,
        **{lat_name: row["lat"], lon_name: row["lon"]},
        method="nearest",
    )
    for v in ds_climo.data_vars:
        row[v] = float(cl[v].values)

    X = pd.DataFrame([row])[X_cols]
    p = float(clf.predict_proba(X)[:, 1])
    amt = float(np.expm1(reg.predict(X)))

    return {
        "p_rain": max(0.0, min(1.0, p)),
        "amount_if_rain_mm": max(0.0, amt),
        "expected_mm": max(0.0, p * amt),
    }


