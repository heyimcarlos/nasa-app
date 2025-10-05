# power_data.py
# -----------------------------------------------------------------------------
# Shared utilities for downloading/caching NASA POWER "daily / regional" data,
# opening mosaics as xarray Datasets, building daily climatologies, and
# assembling a tidy training table for precipitation modeling.
#
# First run will download tiles and build mosaics; subsequent runs reuse the
# on-disk cache to avoid hitting the API and to load fast.
#
# Dependencies:
#   pip install requests numpy pandas xarray
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr

# -----------------
# Config / defaults
# -----------------

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/regional"

# File cache directories
BASE_DIR = Path(__file__).resolve().parent  # → nasa-app/data-modeling

CACHE = BASE_DIR / "power_cache"
CACHE.mkdir(parents=True, exist_ok=True)

TILES_DIR = CACHE / "tiles"
TILES_DIR.mkdir(parents=True, exist_ok=True)

MOSAICS_DIR = CACHE / "mosaics"
MOSAICS_DIR.mkdir(parents=True, exist_ok=True)

# Global target & parameters (shared across models)
TARGET_PARAM = "PRECTOTCORR"  # precipitation corrected (mm/day)

# Choose extra variables to summarize as climatology features
EXTRA_PARAMS = [
    # Moisture & temp
    "RH2M",  # Relative Humidity at 2 Meters (%)
    "T2M",  # Temperature at 2 Meters (°C)
    "T2M_MAX",  # Maximum Temperature at 2 Meters (°C)
    "T2M_MIN",  # Minimum Temperature at 2 Meters (°C)
    "T2MDEW",  # Dew Point Temperature at 2 Meters (°C)
    "QV2M",  # Specific Humidity at 2 Meters (kg/kg)
    # Pressure
    "PS",  # Surface Pressure (kPa)
    "SLP",  # Sea Level Pressure (kPa)
    # Wind
    "WS10M",  # Wind Speed at 10 Meters (m/s)
    # Radiation
    "ALLSKY_SFC_SW_DWN",  # All Sky Surface Shortwave Down
    # Water vapor column
    "PW",  # Water Vapor Column (cm)
    # Extended parameters - could be enabled later if necessary
    # # Clouds
    # "CLOUD_AMT",                # Cloud Amount (tenths)
    # # Longwave
    # "ALLSKY_SFC_LW_DWN",        # All Sky Surface Longwave Down
    # # Soil wetness
    "GWETTOP",  # Top Layer Soil Wetness (unitless)
    # "GWETROOT",                # Root Zone Soil Wetness (unitless)
    # "GWETPROF",                # Profile Soil Wetness (unitless)
    # # Top-of-atmosphere solar / geometry
    # "TOA_SW_DWN",              # Top of Atmosphere Shortwave Down (W/m^2)
    # "SZA",                     # Solar Zenith Angle (degrees)
    # # Aerosols
    # "AOD_55",                  # Aerosol Optical Depth at 550 nm (unitless)
]

# Handy bounding boxes

# Centered on GTA (~43.75N, -79.35W), expanded to meet the 2x2 min degree rule from the POWER API
BBOX_TORONTO = dict(
    latitude_min=42.75,  # ~Hamilton/Oakville (approx)
    latitude_max=44.75,  # ~Barrie/New Tecumseth (approx)
    longitude_min=-80.35,  # ~Milton/Campbellville (approx)
    longitude_max=-78.35,  # ~Whitby/Bowmanville (approx)
)

BBOX_ONTARIO = dict(
    latitude_min=41.6,  # ~Middle Island / Point Pelee (approx)
    latitude_max=56.9,  # ~Hudson Bay coast / Cape Henrietta Maria (approx)
    longitude_min=-95.2,  # ~Manitoba border (approx)
    longitude_max=-74.3,  # ~Ottawa River / near QC border (approx)
)

BBOX_CANADA = dict(
    latitude_min=41.0,  # ~Southern ON border / Point Pelee area (approx)
    latitude_max=83.5,  # ~High Arctic, Ellesmere Island (approx)
    longitude_min=-141.0,  # ~Yukon–Alaska border (approx)
    longitude_max=-52.5,  # ~Newfoundland & Labrador Atlantic coast (approx)
)

# -----------------
# Date helpers
# -----------------


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lag and rolling window features to the training data."""
    print("Adding time-series features (lags and rolling windows)...")

    df = df.sort_values(by=["lat", "lon", "time"]).reset_index(drop=True)

    # Define which features to create lags and rolling windows for
    features_to_process = [TARGET_PARAM, "T2M", "RH2M"]  # This is "PRECTOTCORR"

    grouped = df.groupby(["lat", "lon"])

    for col in features_to_process:
        if col in df.columns:
            # Create a "lag of 1" (yesterday's value)
            df[f"{col}_lag_1"] = grouped[col].shift(1)
            # Create a 7-day rolling average
            df[f"{col}_7_day_avg"] = (
                grouped[col]
                .rolling(window=7, min_periods=3)
                .mean()
                .reset_index(drop=True)
            )
        else:
            print(
                f"Warning: Column '{col}' not found for creating time-series features."
            )

    return df


def last_n_years_dates(n_years: int = 5) -> Tuple[str, str]:
    """Return (start_yyyymmdd, end_yyyymmdd) for the last `n_years`, ending yesterday (UTC).

    Handles leap-day edges robustly by moving the start date to Mar 1 when needed.
    """
    end = date.today() - timedelta(days=1)
    try:
        start = end.replace(year=end.year - n_years) + timedelta(days=1)
    except ValueError:
        start = end.replace(year=end.year - n_years, month=3, day=1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def year_slices(start_yyyymmdd: str, end_yyyymmdd: str) -> Iterable[Tuple[str, str]]:
    """Yield (year_start, year_end) YYYYMMDD pairs for each calendar year in [start, end].

    Useful to respect POWER limits while downloading long spans one year at a time.
    """
    start = pd.to_datetime(start_yyyymmdd, format="%Y%m%d").date()
    end = pd.to_datetime(end_yyyymmdd, format="%Y%m%d").date()
    for y in range(start.year, end.year + 1):
        ys = max(date(y, 1, 1), start)
        ye = min(date(y, 12, 31), end)
        yield ys.strftime("%Y%m%d"), ye.strftime("%Y%m%d")


# -----------------
# BBox tiling
# -----------------


def tile_bbox_no_overlap(
    bbox: Dict[str, float], max_span: float = 10.0, eps: float = 1e-6
):
    """Split a bbox into ≤ `max_span` degree tiles (lat & lon) using half-open intervals.

    POWER regional endpoint restricts requests to ≤10° in both latitude and longitude.
    This generator yields tiles that exactly cover the bbox without overlapping edges.
    """
    lat_min, lat_max = bbox["latitude_min"], bbox["latitude_max"]
    lon_min, lon_max = bbox["longitude_min"], bbox["longitude_max"]

    lat_edges = list(np.arange(lat_min, lat_max, max_span)) + [lat_max]
    lon_edges = list(np.arange(lon_min, lon_max, max_span)) + [lon_max]

    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            up_lat = (
                lat_edges[i + 1] if i == len(lat_edges) - 2 else lat_edges[i + 1] - eps
            )
            up_lon = (
                lon_edges[j + 1] if j == len(lon_edges) - 2 else lon_edges[j + 1] - eps
            )
            yield dict(
                latitude_min=float(round(lat_edges[i], 6)),
                latitude_max=float(round(up_lat, 6)),
                longitude_min=float(round(lon_edges[j], 6)),
                longitude_max=float(round(up_lon, 6)),
            )


# -----------------
# POWER download utils
# -----------------


def build_regional_url(
    param: str,
    bbox: Dict[str, float],
    start: str,
    end: str,
    fmt: str = "NETCDF",
    community: str = "AG",
    time_standard: str = "UTC",
) -> str:
    """Construct a POWER /temporal/daily/regional URL for ONE parameter and a bbox.

    Notes:
      - The regional endpoint supports ONE `parameters=` value per request.
      - Use `fmt="NETCDF"` for array-friendly ingestion; CSV is also available.
    """
    return (
        f"{POWER_BASE}"
        f"?latitude-min={bbox['latitude_min']}&latitude-max={bbox['latitude_max']}"
        f"&longitude-min={bbox['longitude_min']}&longitude-max={bbox['longitude_max']}"
        f"&parameters={param}&community={community}"
        f"&time-standard={time_standard}&start={start}&end={end}&format={fmt}"
    )


def _tile_fname(
    param: str, tile: Dict[str, float], ystart: str, yend: str, ext: str = ".nc"
) -> Path:
    """Create a deterministic local filename for a specific (param, tile, year-span)."""
    return TILES_DIR / (
        f"POWER_{param}_{ystart}_{yend}_"
        f"{tile['latitude_min']}_{tile['latitude_max']}_"
        f"{tile['longitude_min']}_{tile['longitude_max']}{ext}"
    )


def fetch_tile_year_netcdf(
    param: str,
    tile: Dict[str, float],
    ystart: str,
    yend: str,
    pause: float = 0.3,
    timeout: int = 180,
) -> Path:
    """Download one tile/year NetCDF and return the local cached path.

    Skips download if the file already exists. Raises with API error text on failure.
    The small `pause` helps avoid hammering the service during loops.
    """
    out = _tile_fname(param, tile, ystart, yend, ".nc")
    if out.exists():
        return out

    url = build_regional_url(param, tile, ystart, yend, fmt="NETCDF")
    r = requests.get(url, stream=True, timeout=timeout)
    if r.status_code >= 400:
        try:
            print("API error payload:\n", r.text[:2000])
        finally:
            r.raise_for_status()

    ct = (r.headers.get("Content-Type", "") or "").lower()
    if "netcdf" not in ct and "octet-stream" not in ct:
        raise RuntimeError(f"Expected NetCDF, got Content-Type={ct}. URL:\n{url}")

    with open(out, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    time.sleep(pause)
    return out


def collect_paths_for_param(
    param: str, bbox: Dict[str, float], start: str, end: str, tile_span: float = 10.0
) -> List[Path]:
    """Return the list of cached NetCDF paths for all (tiles × year-slices) for `param`.

    Ensures every required tile/year file exists locally by calling `fetch_tile_year_netcdf`.
    """
    tiles = list(tile_bbox_no_overlap(bbox, max_span=tile_span))
    paths: List[Path] = []
    for ys, ye in year_slices(start, end):
        for tile in tiles:
            paths.append(fetch_tile_year_netcdf(param, tile, ys, ye))
    return paths


# -----------------
# Open datasets & mosaic caching
# -----------------


def _engine() -> str:
    """Standardize on the SciPy backend for maximum portability."""
    return "scipy"


def open_mosaic(paths: List[Path]) -> xr.Dataset:
    """Open many NetCDF files and merge by coordinates into a single xarray Dataset."""
    return xr.open_mfdataset(
        [str(p) for p in paths],
        engine=_engine(),
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="outer",
    )


def ensure_unique_sorted_grid(ds: xr.Dataset, lat="lat", lon="lon") -> xr.Dataset:
    """Remove any duplicate lat/lon coordinates and return a dataset sorted by lat, lon.

    POWER tiles can share edge points; this ensures a clean, monotonic grid.
    """
    ds = ds.assign_coords(
        {
            lat: np.round(ds[lat].values.astype(float), 6),
            lon: np.round(ds[lon].values.astype(float), 6),
        }
    )
    lat_vals = pd.Index(ds[lat].values)
    lon_vals = pd.Index(ds[lon].values)
    ds = ds.isel({lat: ~lat_vals.duplicated(), lon: ~lon_vals.duplicated()})
    return ds.sortby([lat, lon])


def guess_lat_lon_coords(ds: xr.Dataset) -> Tuple[str, str]:
    """Heuristically find the latitude and longitude coordinate names in a Dataset.

    Returns a pair like ('lat','lon') or ('latitude','longitude'); raises if not found.
    """
    lat = next(
        (c for c in ("lat", "latitude", "LAT", "Latitude") if c in ds.coords), None
    )
    lon = next(
        (c for c in ("lon", "longitude", "LON", "Longitude") if c in ds.coords), None
    )
    if not lat or not lon:
        raise ValueError(f"Could not find lat/lon in coords: {list(ds.coords)}")
    return lat, lon


def _mosaic_key(param: str, bbox: Dict[str, float], start: str, end: str) -> str:
    """Build a stable string key that uniquely identifies a mosaic on disk."""
    return (
        f"{param}_{start}_{end}_"
        f"{bbox['latitude_min']}_{bbox['latitude_max']}_"
        f"{bbox['longitude_min']}_{bbox['longitude_max']}"
    )


def open_or_build_mosaic(
    param: str,
    bbox: Dict[str, float],
    start: str,
    end: str,
    tile_span: float = 10.0,
) -> xr.Dataset:
    """Open a single on-disk mosaic (NetCDF) for `param`, building it if missing.

    - If a NetCDF mosaic already exists, this is a fast open.
    - Otherwise, it collects tile/year files, stitches, writes to NetCDF, and reopens.
    """
    key = _mosaic_key(param, bbox, start, end)

    # Try cached NetCDF mosaic
    target = MOSAICS_DIR / f"{key}.nc"
    if target.exists():
        return xr.open_dataset(target, engine=_engine())

    # Build from tile cache
    paths = collect_paths_for_param(param, bbox, start, end, tile_span)
    ds = open_mosaic(paths)
    lat, lon = guess_lat_lon_coords(ds)
    ds = ensure_unique_sorted_grid(ds, lat, lon)

    # Write NetCDF once, then reopen
    ds.to_netcdf(target)
    return xr.open_dataset(target, engine=_engine())


# -----------------
# Public API
# -----------------


def get_target_dataset(
    target_param: str,
    bbox: Dict[str, float],
    start: str,
    end: str,
    tile_span: float = 10.0,
) -> xr.Dataset:
    """Download (if needed) and open the daily target dataset (e.g., PRECTOTCORR).

    Uses the on-disk mosaic cache for fast subsequent loads; returns an xarray Dataset.
    """
    ds = open_or_build_mosaic(target_param, bbox, start, end, tile_span)
    return ds


def get_climatology_dataset(
    params: List[str],
    bbox: Dict[str, float],
    start: str,
    end: str,
    align_to: Optional[xr.Dataset] = None,
    tile_span: float = 10.0,
) -> xr.Dataset:
    """Return a Dataset of daily climatologies for each parameter in `params`.

    - Computes mean over years after removing Feb 29 (365-day "doy" index).
    - If `align_to` is provided, regrids (nearest) to match its lat/lon grid.
    - Output variables are named `{PARAM}_clim` with coords (doy, lat, lon).
    """
    das = []
    for par in params:
        dsp = open_or_build_mosaic(par, bbox, start, end, tile_span)
        lat, lon = guess_lat_lon_coords(dsp)
        dsp = ensure_unique_sorted_grid(dsp, lat, lon)
        da = dsp[par]
        da = da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
        clim = (
            da.groupby("time.dayofyear")
            .mean("time", keep_attrs=True)
            .rename(dayofyear="doy")
        )
        clim.name = f"{par}_clim"
        das.append(clim)

    ds_climo = xr.merge(das, compat="override", join="outer")

    if align_to is not None:
        alat, alon = guess_lat_lon_coords(align_to)
        ds_climo = ds_climo.sel(
            doy=ds_climo["doy"],
            lat=align_to[alat],
            lon=align_to[alon],
            method="nearest",
        )

    return ds_climo


# def make_training_table(
#     ds: xr.Dataset, ds_climo: xr.Dataset, target_var: str = TARGET_PARAM
# ) -> pd.DataFrame:
#     """Convert target `ds` + climatology `ds_climo` into a tidy DataFrame for ML.
#
#     Adds:
#       - time-derived features: year, day-of-year (doy), sin/cos seasonal encoding
#       - targets: `rain_flag` (≥0.1 mm), and `y_log1p` for positive amounts
#       - merges daily climatology features on (doy, lat, lon)
#     """
#     lat, lon = guess_lat_lon_coords(ds)
#     dss = ds[[target_var]].sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
#     df_y = dss.to_dataframe().reset_index().rename(columns={lat: "lat", lon: "lon"})
#
#     df_y["year"] = df_y["time"].dt.year
#     df_y["doy"] = df_y["time"].dt.dayofyear
#     df_y["rain_flag"] = (df_y[target_var] >= 0.1).astype("int8")
#     df_y["y_log1p"] = np.log1p(df_y[target_var]).astype("float32")
#
#     df_x = ds_climo.to_dataframe().reset_index()
#     data = df_y.merge(df_x, on=["doy", "lat", "lon"], how="left")
#
#     theta = 2 * np.pi * data["doy"] / 365.0
#     data["doy_sin"] = np.sin(theta).astype("float32")
#     data["doy_cos"] = np.cos(theta).astype("float32")
#
#     data["lat"] = data["lat"].astype("float32")
#     data["lon"] = data["lon"].astype("float32")
#     data[target_var] = data[target_var].astype("float32")
#     return data


def make_training_table(
    ds: xr.Dataset, ds_climo: xr.Dataset, target_var: str = TARGET_PARAM
) -> pd.DataFrame:
    """Convert combined `ds` + climatology `ds_climo` into a tidy DataFrame for ML."""
    lat, lon = guess_lat_lon_coords(ds)

    # Process the entire dataset to include all raw daily values (T2M, RH2M, etc.)
    dss = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    df_y = dss.to_dataframe().reset_index().rename(columns={lat: "lat", lon: "lon"})

    # Create time features and targets
    df_y["year"] = df_y["time"].dt.year
    df_y["doy"] = df_y["time"].dt.dayofyear
    df_y["rain_flag"] = (df_y[target_var] >= 0.1).astype("int8")
    df_y["y_log1p"] = np.log1p(df_y[target_var]).astype("float32")

    # Merge climatology features
    df_x = ds_climo.to_dataframe().reset_index()
    data = df_y.merge(df_x, on=["doy", "lat", "lon"], how="left")

    # Add cyclical encoding for day of year
    theta = 2 * np.pi * data["doy"] / 365.0
    data["doy_sin"] = np.sin(theta).astype("float32")
    data["doy_cos"] = np.cos(theta).astype("float32")

    # Clean up and optimize types
    data = data.dropna(subset=[target_var])
    for col in data.columns:
        if data[col].dtype == "float64":
            data[col] = data[col].astype("float32")

    return data


def split_by_years(
    data: pd.DataFrame,
    train: float = 0.6,
    val: float = 0.2,
    test: float = 0.2,
    chronological: bool = True,
    require_full_years: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (train, val, test) by YEAR proportions (keeps temporal integrity).

    Args:
      data: tidy table from `make_training_table`.
      train/val/test: fractions that should sum to 1.0 (we normalize if slightly off).
      chronological: if True, uses earliest→latest for train/val/test; if False, shuffles years.
      require_full_years: if True, only uses years with exactly 365 days.

    Returns:
      (train_df, val_df, test_df)
    """
    frac = np.array([train, val, test], dtype=float)
    frac = frac / frac.sum()

    days_per_year = data.groupby("year")["time"].nunique().sort_index()
    years = days_per_year.index.tolist()
    if require_full_years:
        years = [y for y in years if days_per_year[y] == 365]
    assert len(years) >= 3, "Need at least 3 (full) years to split."

    if not chronological:
        rng = np.random.default_rng(42)
        rng.shuffle(years)

    n = len(years)
    n_train = max(1, int(round(frac[0] * n)))
    n_val = max(1, int(round(frac[1] * n)))
    # ensure all years used
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val

    y_train = years[:n_train]
    y_val = years[n_train : n_train + n_val]
    y_test = years[n_train + n_val :]

    tr = data[data["year"].isin(y_train)].reset_index(drop=True)
    va = data[data["year"].isin(y_val)].reset_index(drop=True)
    te = data[data["year"].isin(y_test)].reset_index(drop=True)
    return tr, va, te


# -----------------
# One-call dataset builder
# -----------------


@dataclass
class DatasetBundle:
    """Container returned by `get_dataset` so all models share identical inputs."""

    data: pd.DataFrame
    climo_cols: List[str]
    ds: xr.Dataset
    ds_climo: xr.Dataset
    bbox: Dict[str, float]
    start: str
    end: str
    target_param: str
    extra_params: List[str]


# def get_dataset(
#     years: int = 5,
#     bbox: Dict[str, float] = BBOX_TORONTO,
#     target_param: str = TARGET_PARAM,
#     extra_params: List[str] = EXTRA_PARAMS,
#     drop_allnull_climo: bool = True,
#     tile_span: float = 10.0,
# ) -> DatasetBundle:
#     """High-level entry point: build the shared dataset once for everyone.
#
#     Steps:
#       1) Pick date window of the last `years` ending yesterday (UTC).
#       2) Fetch/cache target and extra-parameter cubes for `bbox` (stitched mosaics).
#       3) Build daily climatology (365-day DOY) for `extra_params` aligned to the target grid.
#       4) Create tidy table with seasonal features + targets.
#       5) Optionally drop any climatology columns that are entirely NaN (tiny bboxes).
#
#     Returns:
#       DatasetBundle with:
#         - data: tidy DataFrame (same columns for all models)
#         - climo_cols: list of *_clim feature columns (post-drop)
#         - ds, ds_climo: xarray references (handy for inference/regridding)
#         - bbox, start, end, target_param, extra_params
#     """
#     start, end = last_n_years_dates(years)
#     print(f"Date window: {start} → {end} | bbox={bbox}")
#
#     ds = get_target_dataset(target_param, bbox, start, end, tile_span=tile_span)
#     ds_climo = get_climatology_dataset(
#         extra_params, bbox, start, end, align_to=ds, tile_span=tile_span
#     )
#
#     data = make_training_table(ds, ds_climo, target_var=target_param)
#
#     climo_cols = [c for c in data.columns if c.endswith("_clim")]
#     if drop_allnull_climo:
#         empty = [c for c in climo_cols if data[c].notna().sum() == 0]
#         if empty:
#             print("Dropping all-null climatology cols:", empty)
#             data = data.drop(columns=empty)
#             climo_cols = [c for c in climo_cols if c not in empty]
#
#     return DatasetBundle(
#         data=data,
#         climo_cols=climo_cols,
#         ds=ds,
#         ds_climo=ds_climo,
#         bbox=bbox,
#         start=start,
#         end=end,
#         target_param=target_param,
#         extra_params=list(extra_params),
#     )


def get_dataset(
    years: int = 5,
    bbox: Dict[str, float] = BBOX_TORONTO,
    target_param: str = TARGET_PARAM,
    extra_params: List[str] = EXTRA_PARAMS,
    drop_allnull_climo: bool = True,
    tile_span: float = 10.0,
) -> DatasetBundle:
    """High-level entry point that builds the complete dataset for modeling."""
    start, end = last_n_years_dates(years)
    print(f"Date window: {start} → {end} | bbox={bbox}")

    # Load ALL raw daily datasets (target + features)
    all_params = [target_param] + extra_params
    datasets = []
    for param in all_params:
        print(f"Loading mosaic for: {param}")
        datasets.append(open_or_build_mosaic(param, bbox, start, end, tile_span))

    # Merge into a single dataset containing all raw daily variables
    ds_combined = xr.merge(datasets, compat="override")

    # Build climatology from the extra parameters
    ds_climo = get_climatology_dataset(
        extra_params, bbox, start, end, align_to=ds_combined, tile_span=tile_span
    )

    # Create the base training table from the combined raw data
    data = make_training_table(ds_combined, ds_climo, target_var=target_param)

    # Add time-series features (lags and rolling windows) to the table
    data = add_time_series_features(data)

    # Clean up any empty climatology columns
    climo_cols = [c for c in data.columns if c.endswith("_clim")]
    if drop_allnull_climo:
        empty = [c for c in climo_cols if data[c].notna().sum() == 0]
        if empty:
            print("Dropping all-null climatology cols:", empty)
            data = data.drop(columns=empty)
            climo_cols = [c for c in climo_cols if c not in empty]

    return DatasetBundle(
        data=data,
        climo_cols=climo_cols,
        ds=ds_combined,
        ds_climo=ds_climo,
        bbox=bbox,
        start=start,
        end=end,
        target_param=target_param,
        extra_params=list(extra_params),
    )


__all__ = [
    "TARGET_PARAM",
    "EXTRA_PARAMS",
    "BBOX_TORONTO",
    "BBOX_ONTARIO",
    "BBOX_CANADA",
    "split_by_years",
    "get_dataset",
]
