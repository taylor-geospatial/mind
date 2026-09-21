"""Map source columns to the CoordBench schema, retaining remaining columns."""

import numpy as np
import pandas as pd

CANONICAL_COLS = ("lon", "lat", "timestamp", "timestamp_end", "split", "id")


def build_canonical(
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    timestamp_col: str | None = None,
    timestamp_end_col: str | None = None,
    timestamp_unit: str = "ms",
    split_col: str | None = None,
    split_value: str | None = None,
    id_col: str | None = None,
    drop_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Map coordinates, timestamps, splits, and IDs; preserve other source columns.

    Timestamps use Unix milliseconds, with nulls for missing values. Set splits
    from either ``split_col`` or a constant ``split_value``. Exclude ``drop_cols``
    from the retained source columns.
    """
    if split_col is not None and split_value is not None:
        raise ValueError("split_col and split_value are mutually exclusive")

    out = pd.DataFrame()
    out["lon"] = df[lon_col].to_numpy(np.float64)
    out["lat"] = df[lat_col].to_numpy(np.float64)
    out["timestamp"] = (
        _to_unix_ms(df[timestamp_col], timestamp_unit)
        if timestamp_col
        else pd.array([pd.NA] * len(df), dtype="Int64")
    )
    out["timestamp_end"] = (
        _to_unix_ms(df[timestamp_end_col], timestamp_unit)
        if timestamp_end_col
        else pd.array([pd.NA] * len(df), dtype="Int64")
    )
    if split_col is not None:
        out["split"] = df[split_col].astype("string")
    elif split_value is not None:
        out["split"] = pd.array([split_value] * len(df), dtype="string")
    else:
        out["split"] = pd.array([pd.NA] * len(df), dtype="string")
    out["id"] = (
        df[id_col].astype("string") if id_col else pd.array([pd.NA] * len(df), dtype="string")
    )

    consumed = {lon_col, lat_col, timestamp_col, timestamp_end_col, split_col, id_col, *drop_cols}
    consumed.discard(None)
    passthrough_cols = [c for c in df.columns if c not in consumed]
    passthrough = df[passthrough_cols].reset_index(drop=True)
    return pd.concat([out.reset_index(drop=True), passthrough], axis=1)


def _to_unix_ms(series: pd.Series, unit: str) -> pd.Series:
    if unit == "ms":
        return series.astype("Int64")
    if unit == "year":
        # Convert Jan 1 UTC to milliseconds explicitly; pandas may default to microseconds.
        dt = pd.to_datetime(series, format="%Y", utc=True, errors="coerce")
        ms = dt.to_numpy("datetime64[ms]").view("int64")  # NaT -> INT64_MIN sentinel
        return pd.Series(ms, index=series.index).mask(dt.isna()).astype("Int64")
    raise ValueError(f"unknown timestamp_unit {unit!r}")


def validate_canonical(df: pd.DataFrame, config: str) -> None:
    """Check required columns and coordinate bounds."""
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{config}: missing canonical columns {missing}")
    if df["lon"].isna().any() or df["lat"].isna().any():
        raise ValueError(f"{config}: lon/lat must never be null")
    if not df["lon"].between(-180, 180).all():
        raise ValueError(f"{config}: lon out of [-180, 180] range")
    if not df["lat"].between(-90, 90).all():
        raise ValueError(f"{config}: lat out of [-90, 90] range")
