"""
Training-curve visualization from Ultralytics results.csv.

Why this exists: Ultralytics' built-in plot_results() computes its subplot
grid positionally and crashes with "index N is out of bounds" when the
loss/metric column count is ODD (observed with ultralytics 8.4.136 on the
BDD100K vehicle run: 13 loss/metric columns -> 2x6 grid -> overflow at the
13th column). This module is schema-agnostic: columns are grouped by NAME
(substring 'loss' / 'metric' / 'lr' prefix), never by position, and any
column count works.

Only numpy/pandas/matplotlib are used (already project dependencies).
matplotlib is forced to the headless 'Agg' backend so this is safe on
Kaggle and in CI.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg", force=True)  # headless-safe (Kaggle/CI), before pyplot

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CSV LOADING
# ============================================================

def _load_results_csv(csv_path: Path):
    """Read results.csv into a DataFrame with stripped column names.

    Ultralytics pads header names with whitespace; strip before use.
    Returns None if the file is missing or unreadable.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _group_columns(columns) -> Dict[str, List[str]]:
    """Group column names by semantics. Never positional.

    'epoch'/'time' are x-axis/bookkeeping and are excluded.
    """
    groups: Dict[str, List[str]] = {"loss": [], "metric": [], "lr": [], "other": []}
    for col in columns:
        c = str(col).strip().lower()
        if c in ("epoch", "time"):
            continue
        if "loss" in c:
            groups["loss"].append(col)
        elif "metric" in c:
            groups["metric"].append(col)
        elif c.startswith("lr"):
            groups["lr"].append(col)
        else:
            groups["other"].append(col)
    return {k: v for k, v in groups.items() if v}


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered rolling mean via convolution (no scipy dependency).

    Edge samples keep their raw values to avoid zero-padding artifacts.
    """
    s = np.asarray(values, dtype=float)
    if len(s) < window or window < 2:
        return s
    kernel = np.ones(window) / window
    smoothed = np.convolve(s, kernel, mode="same")
    half = window // 2
    smoothed[:half] = s[:half]
    smoothed[-(window - half):] = s[-(window - half):]
    return smoothed


# ============================================================
# SUMMARY + EPOCH TABLE (for JSON reports)
# ============================================================

def summarize_history(df) -> Dict[str, Any]:
    """Best/final values for the standard detection metrics, if present.

    Column names are canonicalized via .lower() so this keeps working if
    Ultralytics renames or adds columns.
    """
    summary: Dict[str, Any] = {}
    cols = {str(c).lower(): c for c in df.columns}

    if "epoch" in cols:
        try:
            summary["epochs_logged"] = int(df[cols["epoch"]].max())
        except Exception:
            pass

    key_metrics = {
        "metrics/map50-95(b)": "mAP50_95",
        "metrics/map50(b)": "mAP50",
        "metrics/precision(b)": "precision",
        "metrics/recall(b)": "recall",
    }
    for canon, name in key_metrics.items():
        if canon not in cols:
            continue
        series = df[cols[canon]].astype(float)
        if series.empty:
            continue
        best_idx = series.idxmax()
        summary[f"best_{name}"] = round(float(series.max()), 4)
        if "epoch" in cols:
            summary[f"best_{name}_epoch"] = int(df.loc[best_idx, cols["epoch"]])
        summary[f"final_{name}"] = round(float(series.iloc[-1]), 4)
    return summary


def epoch_table(df, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Per-epoch records for JSON embedding (compact, NaN -> null)."""
    table: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        record: Dict[str, Any] = {}
        for col in df.columns:
            value = row[col]
            if isinstance(value, (int, np.integer)):
                record[col] = int(value)
            elif isinstance(value, (float, np.floating)):
                f = float(value)
                record[col] = None if math.isnan(f) else round(f, 5)
            else:
                record[col] = value
        table.append(record)
    if max_rows is not None and len(table) > max_rows:
        table = table[-max_rows:]
    return table


# ============================================================
# MAIN ENTRY
# ============================================================

def plot_training_history(
    csv_path,
    out_path: Optional[Path] = None,
    logger=None,
    smooth_window: int = 5,
) -> Tuple[Optional[Path], List[Dict[str, Any]], Dict[str, Any]]:
    """Plot train/val curves from an Ultralytics results.csv.

    Schema-agnostic by construction: any number of loss/metric/lr columns,
    odd or even, known or unknown names.

    Args:
        csv_path: Path to results.csv (e.g. <exp>/train/results.csv).
        out_path: Where to save the PNG (default: <csv_dir>/curves.png).
        logger: Optional logger for status messages.
        smooth_window: Rolling-mean window for the smoothing overlay.

    Returns:
        (png_path_or_None, epoch_table, summary)
        png_path is None when the CSV is missing/unreadable/empty.
    """
    df = _load_results_csv(csv_path)
    if df is None or df.empty:
        if logger is not None:
            logger.warning(
                "results.csv not found or unreadable at %s; skipping curve plot", csv_path
            )
        return None, [], {}

    groups = _group_columns(df.columns)
    if not groups:
        if logger is not None:
            logger.warning("No plottable columns in %s", csv_path)
        return None, [], {}

    if out_path is None:
        out_path = Path(csv_path).parent / "curves.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_col = "epoch" if "epoch" in df.columns else str(df.columns[0])

    group_order = [g for g in ("loss", "metric", "lr", "other") if g in groups]
    max_cols = max(len(groups[g]) for g in group_order)

    fig, axes = plt.subplots(
        len(group_order),
        max_cols,
        figsize=(3.2 * max_cols, 2.8 * len(group_order)),
        squeeze=False,
        tight_layout=True,
    )

    for row_idx, gname in enumerate(group_order):
        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(groups[gname]):
                ax.axis("off")
                continue
            col = groups[gname][col_idx]
            pair = df[[x_col, col]].dropna()
            xs = pair[x_col].astype(float).to_numpy()
            ys = pair[col].astype(float).to_numpy()
            ax.plot(xs, ys, marker=".", markersize=3, linewidth=1.2, label="raw")
            if len(ys) >= smooth_window:
                ax.plot(
                    xs, _smooth(ys, smooth_window),
                    ":", linewidth=1.6, color="crimson", label="smooth",
                )
            ax.set_title(col, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel(gname, fontsize=8)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f"Training history ({x_col})", fontsize=10)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if logger is not None:
        logger.info("Training curves saved: %s", out_path)

    return out_path, epoch_table(df), summarize_history(df)
