"""
Feature Engineering Module for Traffic Demand Prediction
=========================================================
Handles all data transformations: geohash decoding, time parsing,
categorical encoding, missing value imputation, and target encoding.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import pygeohash as pgh
    GEOHASH_AVAILABLE = True
except ImportError:
    GEOHASH_AVAILABLE = False
    print("[WARNING] pygeohash not installed. Geohash lat/lon features will be skipped.")


# ─────────────────────────────────────────────
# 1. Geohash Decoding
# ─────────────────────────────────────────────

def decode_geohash(df: pd.DataFrame) -> pd.DataFrame:
    """Decode geohash string into lat, lon coordinates."""
    df = df.copy()
    if GEOHASH_AVAILABLE:
        try:
            decoded = df["geohash"].apply(
                lambda gh: pgh.decode(gh) if isinstance(gh, str) else (np.nan, np.nan)
            )
            df["lat"] = decoded.apply(lambda x: x[0])
            df["lon"] = decoded.apply(lambda x: x[1])
        except Exception as e:
            print(f"[WARNING] Geohash decode error: {e}. Using dummy lat/lon.")
            df["lat"] = 0.0
            df["lon"] = 0.0
    else:
        df["lat"] = 0.0
        df["lon"] = 0.0
    return df


def add_geohash_prefix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hierarchical geohash prefix groups for coarser spatial resolution."""
    df = df.copy()
    df["geohash_4"] = df["geohash"].apply(lambda x: x[:4] if isinstance(x, str) else "xxxx")
    df["geohash_5"] = df["geohash"].apply(lambda x: x[:5] if isinstance(x, str) else "xxxxx")
    return df


# ─────────────────────────────────────────────
# 2. Timestamp Parsing
# ─────────────────────────────────────────────

def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'H:MM' timestamp strings into hour, minute, and total minutes."""
    df = df.copy()

    def _parse(ts):
        try:
            parts = str(ts).split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            return h, m
        except Exception:
            return 0, 0

    parsed = df["timestamp"].apply(_parse)
    df["hour"] = parsed.apply(lambda x: x[0])
    df["minute"] = parsed.apply(lambda x: x[1])
    df["time_minutes"] = df["hour"] * 60 + df["minute"]

    # Cyclical encoding of time (captures periodicity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["time_sin"] = np.sin(2 * np.pi * df["time_minutes"] / 1440)
    df["time_cos"] = np.cos(2 * np.pi * df["time_minutes"] / 1440)

    return df


# ─────────────────────────────────────────────
# 3. Day Feature
# ─────────────────────────────────────────────

def add_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical day-of-week encoding."""
    df = df.copy()
    # Treat day as cyclical with period 7 (week)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)
    # Also keep raw day as a feature
    return df


# ─────────────────────────────────────────────
# 4. Categorical Encoding
# ─────────────────────────────────────────────

ROAD_TYPE_MAP = {
    "Residential": 0,
    "Street": 1,
    "Highway": 2,
}

WEATHER_MAP = {
    "Snowy": 0,
    "Rainy": 1,
    "Foggy": 2,
    "Sunny": 3,
}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features as numeric."""
    df = df.copy()

    # RoadType — ordinal (Residential < Street < Highway by typical demand)
    df["RoadType_enc"] = df["RoadType"].map(ROAD_TYPE_MAP).fillna(-1).astype(int)

    # Weather — ordinal proxy (Snowy=0 has lowest demand typically)
    df["Weather_enc"] = df["Weather"].map(WEATHER_MAP).fillna(-1).astype(int)

    # LargeVehicles → binary
    df["LargeVehicles_bin"] = (df["LargeVehicles"] == "Allowed").astype(int)

    # Landmarks → binary
    df["Landmarks_bin"] = (df["Landmarks"] == "Yes").astype(int)

    return df


# ─────────────────────────────────────────────
# 5. Missing Value Imputation
# ─────────────────────────────────────────────

def impute_missing_values(train: pd.DataFrame, test: pd.DataFrame):
    """
    Impute missing values for Temperature, RoadType_enc, Weather_enc.
    Uses group medians/modes from train set, applied to both.
    Returns (train_imputed, test_imputed, imputation_stats).
    """
    train = train.copy()
    test = test.copy()

    # ── Temperature imputation ──
    # First try: median per geohash_4 + hour bin
    train["hour_bin"] = (train["hour"] // 6).astype(int)  # 0=night,1=morning,2=afternoon,3=evening
    test["hour_bin"] = (test["hour"] // 6).astype(int)

    temp_median = train.groupby(["geohash_4", "hour_bin"])["Temperature"].median()

    def _fill_temp(row, median_map, global_median):
        if pd.isna(row["Temperature"]):
            key = (row["geohash_4"], row["hour_bin"])
            return median_map.get(key, global_median)
        return row["Temperature"]

    global_temp_median = train["Temperature"].median()
    temp_median_dict = temp_median.to_dict()

    train["Temperature"] = train.apply(
        lambda r: _fill_temp(r, temp_median_dict, global_temp_median), axis=1
    )
    test["Temperature"] = test.apply(
        lambda r: _fill_temp(r, temp_median_dict, global_temp_median), axis=1
    )

    # ── RoadType imputation ──
    road_mode_by_geo = train.groupby("geohash_4")["RoadType_enc"].agg(
        lambda x: x[x >= 0].mode()[0] if len(x[x >= 0]) > 0 else 0
    ).to_dict()
    global_road_mode = 0  # Residential

    def _fill_road(row, mode_map):
        if row["RoadType_enc"] == -1:
            return mode_map.get(row["geohash_4"], global_road_mode)
        return row["RoadType_enc"]

    train["RoadType_enc"] = train.apply(lambda r: _fill_road(r, road_mode_by_geo), axis=1)
    test["RoadType_enc"] = test.apply(lambda r: _fill_road(r, road_mode_by_geo), axis=1)

    # ── Weather imputation ──
    weather_mode_by_geo = train.groupby(["geohash_4", "hour_bin"])["Weather_enc"].agg(
        lambda x: x[x >= 0].mode()[0] if len(x[x >= 0]) > 0 else 2
    ).to_dict()
    global_weather_mode = 2  # Foggy as middle ground

    def _fill_weather(row, mode_map):
        if row["Weather_enc"] == -1:
            key = (row["geohash_4"], row["hour_bin"])
            return mode_map.get(key, global_weather_mode)
        return row["Weather_enc"]

    train["Weather_enc"] = train.apply(lambda r: _fill_weather(r, weather_mode_by_geo), axis=1)
    test["Weather_enc"] = test.apply(lambda r: _fill_weather(r, weather_mode_by_geo), axis=1)

    return train, test


# ─────────────────────────────────────────────
# 6. Target Encoding (geohash)
# ─────────────────────────────────────────────

class TargetEncoder:
    """
    Smoothed target mean encoder for high-cardinality categoricals.
    Applies cross-validation folds to avoid leakage on training data.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_maps_ = {}  # col -> {value: encoded_mean}

    def fit_transform(self, train: pd.DataFrame, target: pd.Series, cols: list, n_folds: int = 5) -> pd.DataFrame:
        """Fit encoder on train and return OOF encoded values."""
        from sklearn.model_selection import KFold

        train = train.copy()
        self.global_mean_ = target.mean()

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for col in cols:
            train[f"{col}_te"] = self.global_mean_

            # Cross-validated target encoding to avoid leakage
            oof_te = np.full(len(train), self.global_mean_)
            for tr_idx, val_idx in kf.split(train):
                tr_stats = target.iloc[tr_idx].groupby(train[col].iloc[tr_idx]).agg(["mean", "count"])
                # Smoothing formula: (count * group_mean + smoothing * global_mean) / (count + smoothing)
                smoothed = (tr_stats["count"] * tr_stats["mean"] + self.smoothing * self.global_mean_) / (tr_stats["count"] + self.smoothing)
                oof_te[val_idx] = train[col].iloc[val_idx].map(smoothed).fillna(self.global_mean_).values

            train[f"{col}_te"] = oof_te

            # Store full-train encoding for test transformation
            full_stats = target.groupby(train[col]).agg(["mean", "count"])
            self.encoding_maps_[col] = (
                (full_stats["count"] * full_stats["mean"] + self.smoothing * self.global_mean_)
                / (full_stats["count"] + self.smoothing)
            ).to_dict()

        return train

    def transform(self, test: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Apply learned encodings to test set."""
        test = test.copy()
        for col in cols:
            test[f"{col}_te"] = test[col].map(self.encoding_maps_[col]).fillna(self.global_mean_)
        return test


# ─────────────────────────────────────────────
# 7. Interaction & Derived Features
# ─────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features that capture demand patterns."""
    df = df.copy()

    # Peak hours flag (7-9 AM and 5-8 PM are typical rush hours)
    df["is_rush_hour"] = ((df["hour"].between(7, 9)) | (df["hour"].between(17, 20))).astype(int)
    df["is_night"] = (df["hour"].between(0, 5)).astype(int)
    df["is_daytime"] = (df["hour"].between(6, 22)).astype(int)

    # Highway × rush hour interaction (highways spike during rush)
    df["highway_rush"] = (df["RoadType_enc"] == 2).astype(int) * df["is_rush_hour"]

    # Temperature squared (non-linear effect)
    df["temp_squared"] = df["Temperature"] ** 2

    # Temperature × weather interaction
    df["temp_x_weather"] = df["Temperature"] * df["Weather_enc"]

    # Lanes × RoadType
    df["lanes_x_roadtype"] = df["NumberofLanes"] * df["RoadType_enc"]

    return df


# ─────────────────────────────────────────────
# 8. Full Pipeline
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Geographic
    "lat", "lon",
    # Geohash target encodings
    "geohash_te", "geohash_4_te", "geohash_5_te",
    # Temporal
    "day", "hour", "minute", "time_minutes",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "time_sin", "time_cos", "day_sin", "day_cos",
    "hour_bin",
    # Road/Location
    "RoadType_enc", "NumberofLanes", "LargeVehicles_bin", "Landmarks_bin",
    # Weather
    "Weather_enc", "Temperature",
    # Interaction features
    "is_rush_hour", "is_night", "is_daytime",
    "highway_rush", "temp_squared", "temp_x_weather",
    "lanes_x_roadtype",
    # Lag feature
    "lag_1_demand",
]


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame, target_col: str = "demand"):
    """
    Full feature engineering pipeline.
    Returns (X_train, y_train, X_test, feature_names, target_encoder).
    """
    print("[1/6] Decoding geohash and adding prefix features...")
    train = decode_geohash(train_raw)
    test = decode_geohash(test_raw)
    train = add_geohash_prefix_features(train)
    test = add_geohash_prefix_features(test)

    print("[2/6] Parsing timestamps...")
    train = parse_timestamp(train)
    test = parse_timestamp(test)

    print("[3/6] Adding day features...")
    train = add_day_features(train)
    test = add_day_features(test)

    print("[4/6] Encoding categorical variables...")
    train = encode_categoricals(train)
    test = encode_categoricals(test)

    print("[5/6] Imputing missing values...")
    train, test = impute_missing_values(train, test)

    print("[6/6] Target encoding and interaction features...")
    y = train[target_col]
    te = TargetEncoder(smoothing=10.0)
    te_cols = ["geohash", "geohash_4", "geohash_5"]
    train = te.fit_transform(train, y, cols=te_cols, n_folds=5)
    test = te.transform(test, cols=te_cols)

    train = add_interaction_features(train)
    test = add_interaction_features(test)

    print("[7/7] Adding Lag 1 Demand (Day 48)...")
    # Build a dictionary of demand for Day 48
    day_48_data = train_raw[train_raw["day"] == 48]
    lag_map = day_48_data.groupby(["geohash", "timestamp"])["demand"].mean().to_dict()

    def get_lag(row, is_train):
        if is_train and row["day"] == 48:
            return np.nan  # Prevent target leakage on day 48 itself
        return lag_map.get((row["geohash"], row["timestamp"]), np.nan)

    train["lag_1_demand"] = train.apply(lambda r: get_lag(r, is_train=True), axis=1)
    test["lag_1_demand"] = test.apply(lambda r: get_lag(r, is_train=False), axis=1)

    # Impute missing lag_1_demand with global mean or geohash_te
    global_mean_lag = train["lag_1_demand"].mean()
    train["lag_1_demand"] = train["lag_1_demand"].fillna(train["geohash_te"])
    test["lag_1_demand"] = test["lag_1_demand"].fillna(test["geohash_te"])

    # Select final features — keep only those present
    available_feats = [f for f in FEATURE_COLS if f in train.columns]
    missing_feats = [f for f in FEATURE_COLS if f not in train.columns]
    if missing_feats:
        print(f"  [INFO] Features not available (will be skipped): {missing_feats}")

    X_train = train[available_feats].copy()
    X_test = test[available_feats].copy()

    print(f"\n[OK] Features built: {len(available_feats)} features, {len(X_train)} train rows, {len(X_test)} test rows")
    return X_train, y, X_test, available_feats, te
