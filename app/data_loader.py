import logging
import os
import re
import zipfile
from typing import Any, Optional

import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

# Pattern: optional whitespace, numeric value, optional whitespace, unit letters
_WEIGHT_PATTERN = re.compile(r"^\s*([\d.]+)\s*([a-zA-Z]+)\s*$")

# Conversion factors relative to 1 gram
_WEIGHT_UNITS_TO_GRAMS = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "gm": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "mg": 0.001,
    "lb": 453.592,
    "lbs": 453.592,
    "pounds": 453.592,
    "oz": 28.3495,
    "ounces": 28.3495,
}
_WEIGHT_MISSING_SENTINEL = "999999999"

def _parse_weight_grams(raw: Optional[str]) -> float:
    # Parse a free-text weight string into grams. Returns NaN on failure.
    if not isinstance(raw, str) or not raw:
        return float("nan")

    raw = raw.strip()
    if raw == _WEIGHT_MISSING_SENTINEL:
        return float("nan")
    match = _WEIGHT_PATTERN.match(raw)
    if not match:
        return float("nan")

    val_str, unit_str = match.groups()
    unit = unit_str.lower()

    if unit not in _WEIGHT_UNITS_TO_GRAMS:
        return float("nan")
    try:
        return float(val_str) * _WEIGHT_UNITS_TO_GRAMS[unit]
    except ValueError:
        return float("nan")

def _cap_categorical_cardinality(series: pd.Series, max_values: int) -> pd.Series:
    # Keep the top `max_values` categories, setting the rest to 'OTHER' and missing to 'UNKNOWN'
    cleaned = series.fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")
    top_categories = set(cleaned.value_counts().nlargest(max_values).index)
    return cleaned.where(cleaned.isin(top_categories), other="OTHER")

def _ensure_data_file_present() -> str:
    # Ensure the JSON dataset exists locally, unzipping the archive if needed.
    if os.path.exists(settings.DATA_PATH):
        return settings.DATA_PATH
    if os.path.exists(settings.DATA_ZIP_PATH):
        logger.info("Extracting dataset from %s", settings.DATA_ZIP_PATH)
        target_dir = os.path.dirname(settings.DATA_ZIP_PATH) or "."
        with zipfile.ZipFile(settings.DATA_ZIP_PATH, "r") as zf:
            zf.extractall(target_dir)

        if os.path.exists(settings.DATA_PATH):
            return settings.DATA_PATH

    raise FileNotFoundError(
        f"Dataset file missing from '{settings.DATA_PATH}' and archive missing from '{settings.DATA_ZIP_PATH}'."
    )

def load_products() -> pd.DataFrame:
    # Load and sanitize product dataset from configured JSON source.
    path = _ensure_data_file_present()
    df = pd.read_json(path, lines=True)

    required_cols = {"uniq_id", "brand", "sales_price", "weight", "rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Numeric transformations (pd.to_numeric handles invalid strings natively)
    df["sales_price"] = pd.to_numeric(df["sales_price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["weight_grams"] = df["weight"].apply(_parse_weight_grams)

    # Categorical transformations
    df["brand"] = _cap_categorical_cardinality(df["brand"], settings.MAX_CATEGORY_VALUES)
    df["colour"] = (
        _cap_categorical_cardinality(df["colour"], settings.MAX_CATEGORY_VALUES)
        if "colour" in df.columns
        else "UNKNOWN"
    )

    # Clean unique identifier column
    df = df.dropna(subset=["uniq_id"])
    initial_count = len(df)
    df = df.drop_duplicates(subset=["uniq_id"], keep="first").reset_index(drop=True)

    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        logger.warning("Dropped %d duplicate product IDs", dropped_count)

    logger.info("Loaded %d products successfully from %s", len(df), path)
    return df