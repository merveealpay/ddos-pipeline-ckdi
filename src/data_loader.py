"""
data_loader.py
--------------
Loads the DDOS CICEV2023 Processed_Data directory.

Expected layout (one or more CSV files per traffic class)::

    Processed_Data/
        BENIGN.csv          (or any file whose Label column == "BENIGN")
        DDoS_SYN_Flood.csv
        DDoS_UDP_Flood.csv
        ...

Every CSV must have:
  - Numeric feature columns (float / int)
  - A 'Label' column (string) identifying the traffic class

If *data_dir* is ``None`` or does not exist the loader falls back to a
built-in synthetic dataset so that the pipeline can be exercised without
the real data files.
"""

from __future__ import annotations

import os
import glob
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Label value that represents normal / benign traffic
BENIGN_LABEL = "BENIGN"

# Columns to drop unconditionally (non-numeric meta-data that CIC tools add)
_DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP", "Source Port",
    "Destination Port", "Protocol", "Timestamp",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Optional[str]) -> pd.DataFrame:
    """Return a single DataFrame with all classes concatenated.

    Parameters
    ----------
    data_dir:
        Path to the Processed_Data directory.  Pass ``None`` to use the
        synthetic fallback.

    Returns
    -------
    pd.DataFrame
        Columns: numeric features  +  ``Label`` (str).
    """
    if data_dir and os.path.isdir(data_dir):
        df = _load_from_dir(data_dir)
    else:
        logger.warning(
            "data_dir=%r not found or not specified – using synthetic data.",
            data_dir,
        )
        df = _make_synthetic()

    df = _clean(df)
    logger.info(
        "Dataset loaded: %d rows, %d feature columns, classes: %s",
        len(df),
        df.shape[1] - 1,
        sorted(df["Label"].unique()),
    )
    return df


def split_by_label(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Split a dataset into the benign baseline and a dict of attack classes.

    Returns
    -------
    baseline : pd.DataFrame
        Rows with Label == BENIGN_LABEL.
    attacks : dict[str, pd.DataFrame]
        Keys are attack class names; values are the corresponding rows.
        The ``Label`` column is retained for reference.
    """
    baseline = df[df["Label"] == BENIGN_LABEL].copy()
    attacks: dict[str, pd.DataFrame] = {}
    for label, group in df[df["Label"] != BENIGN_LABEL].groupby("Label"):
        attacks[str(label)] = group.copy()
    return baseline, attacks


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all column names except 'Label'."""
    return [c for c in df.columns if c != "Label"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_from_dir(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "**", "*.csv")
    csv_files = glob.glob(pattern, recursive=True)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {data_dir!r}. "
            "Check that the path points to the Processed_Data directory."
        )
    chunks: list[pd.DataFrame] = []
    for path in sorted(csv_files):
        try:
            chunk = pd.read_csv(path, low_memory=False)
            # Normalise column names (strip whitespace)
            chunk.columns = [c.strip() for c in chunk.columns]
            if "Label" not in chunk.columns:
                logger.warning("Skipping %s: no 'Label' column.", path)
                continue
            chunks.append(chunk)
            logger.debug("Loaded %s (%d rows)", path, len(chunk))
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            logger.warning("Could not load %s: %s", path, exc)
    if not chunks:
        raise RuntimeError("Could not load any valid CSV from %r." % data_dir)
    return pd.concat(chunks, ignore_index=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop meta-data columns that are not features
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Keep only numeric feature columns + Label
    feature_cols = [c for c in df.columns if c != "Label"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows / columns that are entirely NaN
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")

    # Remove infinite values
    numeric = df.select_dtypes(include=[np.number])
    df = df[np.isfinite(numeric).all(axis=1)]

    # Ensure Label is string
    df["Label"] = df["Label"].astype(str).str.strip()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic fallback dataset
# ---------------------------------------------------------------------------

def _make_synthetic(
    n_benign: int = 500,
    n_per_attack: int = 300,
    n_features: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a simple synthetic dataset that mimics the real structure."""
    rng = np.random.default_rng(seed)
    attack_classes = [
        "DDoS_SYN_Flood",
        "DDoS_UDP_Flood",
        "DDoS_ACK_Flood",
        "DDoS_HTTP_Flood",
    ]
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]

    rows: list[pd.DataFrame] = []

    # Benign: mean ≈ 0, std ≈ 1
    benign_data = rng.normal(loc=0.0, scale=1.0, size=(n_benign, n_features))
    benign_df = pd.DataFrame(benign_data, columns=feature_names)
    benign_df["Label"] = BENIGN_LABEL
    rows.append(benign_df)

    # Each attack class: shifted mean so drift is detectable
    for i, cls in enumerate(attack_classes):
        shift = (i + 1) * 1.5
        scale = 1.0 + i * 0.3
        data = rng.normal(loc=shift, scale=scale, size=(n_per_attack, n_features))
        df_cls = pd.DataFrame(data, columns=feature_names)
        df_cls["Label"] = cls
        rows.append(df_cls)

    return pd.concat(rows, ignore_index=True)
