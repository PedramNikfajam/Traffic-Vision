"""
Tests for training-curve visualization (src/viz.py).

Includes a regression test for the upstream Ultralytics plot_results() bug:
an ODD number of loss/metric columns must NOT crash here and must still
produce a valid PNG (upstream crashes with "index N out of bounds").
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TEST_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

pd = pytest.importorskip("pandas")
plt = pytest.importorskip("matplotlib.pyplot")

from src.viz import _group_columns, plot_training_history


# Standard detection results.csv schema: 10 loss/metric columns (even) -
# this is the case upstream plot_results() happens to survive.
COLUMNS = [
    "epoch",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "lr/pg0", "lr/pg1", "lr/pg2",
]
# 13 loss/metric columns (7 loss + 6 metric = ODD) - the schema shape that
# crashes upstream plot_results() in ultralytics 8.4.136 ("index 12 out of
# bounds for axis 0 with size 12"). Our grid arithmetic must survive this.
ODD_COLUMNS = [
    "epoch",
    "train/box_loss", "train/cls_loss", "train/dfl_loss", "train/aux_loss",
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
    "metrics/mAP50-95(B)", "metrics/mAP_s(B)", "metrics/mAP_m(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "lr/pg0", "lr/pg1", "lr/pg2",
]


def _synthetic_csv(tmp_path: Path, columns=None, n_rows=12):
    import numpy as np
    rng = np.random.default_rng(42)
    columns = columns or COLUMNS
    data = {}
    for c in columns:
        if c == "epoch":
            data[c] = list(range(1, n_rows + 1))
        elif "metric" in c:
            data[c] = list(np.round(rng.random(n_rows) * 0.9, 5))
        elif c.startswith("lr"):
            data[c] = list(np.round(rng.random(n_rows) * 0.01, 6))
        else:
            data[c] = list(np.round(rng.random(n_rows) * 2.0, 5))
    path = tmp_path / "results.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


class TestGroupColumns:
    def test_loss_metric_lr_split(self):
        groups = _group_columns(COLUMNS)
        assert set(groups.keys()) == {"loss", "metric", "lr"}
        assert groups["loss"] == [
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
        ]
        assert groups["metric"] == [
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        ]
        assert groups["lr"] == ["lr/pg0", "lr/pg1", "lr/pg2"]

    def test_epoch_excluded(self):
        groups = _group_columns(COLUMNS)
        # epoch is the x-axis; empty groups are filtered out entirely
        assert "other" not in groups


class TestPlotTrainingHistory:
    def test_odd_column_count_no_crash(self, tmp_path):
        """Regression: upstream crashes on 13 loss/metric columns; we must not."""
        csv_path = _synthetic_csv(tmp_path, columns=ODD_COLUMNS)
        out = tmp_path / "curves.png"
        png, table, summary = plot_training_history(csv_path, out_path=out)
        assert png is not None
        assert png.exists()
        assert png.stat().st_size > 0

    def test_standard_schema(self, tmp_path):
        csv_path = _synthetic_csv(tmp_path)
        out = tmp_path / "curves.png"
        png, table, summary = plot_training_history(csv_path, out_path=out)
        assert png == out
        assert len(table) == 12
        assert table[0]["epoch"] == 1

    def test_summary_values(self, tmp_path):
        csv_path = _synthetic_csv(tmp_path)
        _, _, summary = plot_training_history(csv_path)
        assert summary["epochs_logged"] == 12
        for key in ("best_mAP50", "best_mAP50_epoch", "final_mAP50",
                    "best_mAP50_95", "best_precision", "best_recall"):
            assert key in summary
        assert 0 <= summary["best_mAP50"] <= 0.9

    def test_missing_csv(self, tmp_path):
        png, table, summary = plot_training_history(tmp_path / "nope.csv")
        assert png is None
        assert table == []
        assert summary == {}

    def test_nan_cells_become_json_null(self, tmp_path):
        import numpy as np
        path = tmp_path / "results.csv"
        frame = pd.DataFrame({
            "epoch": [1, 2, 3],
            "train/box_loss": [1.0, np.nan, 0.5],
            "metrics/mAP50(B)": [0.1, 0.2, 0.3],
        })
        frame.to_csv(path, index=False)
        _, table, summary = plot_training_history(path)
        assert table[1]["train/box_loss"] is None  # NaN -> null (valid JSON)
        # json.dumps would raise on NaN; must pass
        json.dumps({"table": table, "summary": summary})

    def test_summary_json_serializable(self, tmp_path):
        csv_path = _synthetic_csv(tmp_path)
        _, table, summary = plot_training_history(csv_path)
        json.dumps({"table": table, "summary": summary})

    def test_default_out_path_next_to_csv(self, tmp_path):
        csv_path = _synthetic_csv(tmp_path)
        png, _, _ = plot_training_history(csv_path)
        assert png == tmp_path / "curves.png"
