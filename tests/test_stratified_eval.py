"""
Tests for stratified (day/night/weather) evaluation plumbing in 05_evaluate.py.

Runs WITHOUT the dataset: uses a fake model object and verifies group yaml
generation, image-list writing, min-image gating, and result aggregation.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TEST_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# The stratified path writes a per-group dataset.yaml, so it needs PyYAML. The
# function under test degrades gracefully without it (logs a warning and returns
# {}), which means these tests must SKIP rather than fail - otherwise a
# contributor without PyYAML sees three failures that are not real defects.
# This is exactly what CI did before pyyaml was added to its install list.
#
# A real import is used rather than importlib.util.find_spec: find_spec reports
# a BROKEN yaml module as present, and the ImportError then surfaces inside the
# test as a failure. CI additionally asserts PyYAML is importable, so a broken
# install fails the build instead of quietly skipping the coverage.
try:
    import yaml  # noqa: F401
    _HAS_YAML = True
except Exception:            # pragma: no cover - environment dependent
    _HAS_YAML = False

pytestmark = pytest.mark.skipif(
    not _HAS_YAML, reason="PyYAML is required by the stratified evaluation path")


class _FakeBox:
    """Minimal stand-in for ultralytics DetMetrics.box."""

    def __init__(self, map50=0.5, map5095=0.3):
        self.map50 = map50
        self.map5095 = map5095


class _FakeMetrics:
    """Mimics the results_dict surface of model.val()."""

    def __init__(self, map50=0.5, map5095=0.3):
        self.results_dict = {
            "metrics/precision(B)": 0.7,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": map50,
            "metrics/mAP50-95(B)": map5095,
        }
        self.box = None


class _FakeModel:
    """model.val() stand-in that returns per-group metrics."""

    def __init__(self, night_drop=0.1):
        self.night_drop = night_drop
        self.val_calls = []

    def val(self, data=None, **kwargs):
        self.val_calls.append(data)
        map50 = 0.65
        if data and "night" in Path(data).name or (data and "night" in str(data)):
            map50 -= self.night_drop
        return _FakeMetrics(map50=map50)


@pytest.fixture
def eval_env(tmp_path, monkeypatch):
    """Isolate OUTPUT paths and provide a base dataset.yaml."""
    import src.config as config

    monkeypatch.setattr(config, "OUTPUT", type(config.OUTPUT)(project_root=tmp_path))

    data_dir = tmp_path / "data" / "vehicle_detection"
    data_dir.mkdir(parents=True)
    (data_dir / "val" / "images").mkdir(parents=True)
    (data_dir / "val" / "labels").mkdir(parents=True)
    (data_dir / "dataset.yaml").write_text(
        "path: " + str(data_dir) + "\n"
        "train: train/images\n"
        "val: val/images\n"
        "nc: 5\n"
        "names: ['car', 'truck', 'bus', 'bike', 'motor']\n"
    )
    # Create a few image+label pairs referenced by groups
    for i in range(3):
        (data_dir / "val" / "images" / f"img_{i}.jpg").write_bytes(b"x")
        (data_dir / "val" / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return data_dir


def _load_module(monkeypatch=None):
    """Import 05_evaluate.py as a module (dotted name workaround)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p5_evaluate", _PROJECT_DIR / "scripts" / "05_evaluate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if monkeypatch is not None:
        import src.config as config
        # module bound OUTPUT at import time; rebind to current (patched) config
        monkeypatch.setattr(mod, "OUTPUT", config.OUTPUT)
    return mod


def test_stratified_min_image_gate(eval_env, monkeypatch):
    """Groups below min_images are skipped, others evaluated."""
    mod = _load_module(monkeypatch)
    fake = _FakeModel()
    args = SimpleNamespace(imgsz=640, conf=0.001, iou=0.6, device="cpu")

    groups = {
        "daytime": [f"img_{i}.jpg" for i in range(3)],
        "night": [f"img_{i}.jpg" for i in range(3)],
        "foggy": ["img_0.jpg"],  # below gate
    }
    out = mod.run_stratified_eval(
        fake, groups, eval_env / "dataset.yaml", args, None, min_images=2
    )

    assert "foggy" not in out
    assert set(out) == {"daytime", "night"}
    assert fake.val_calls and len(fake.val_calls) == 2


def test_stratified_yaml_and_list_written(eval_env, monkeypatch):
    """Per-group yaml + image list are written with correct content."""
    mod = _load_module(monkeypatch)
    args = SimpleNamespace(imgsz=640, conf=0.001, iou=0.6, device="cpu")
    groups = {"night": ["img_0.jpg", "img_1.jpg"]}

    out = mod.run_stratified_eval(
        _FakeModel(), groups, eval_env / "dataset.yaml", args, None, min_images=2
    )

    yaml_path = eval_env / "eval_splits" / "night" / "dataset.yaml"
    txt_path = eval_env / "eval_splits" / "night" / "val.txt"
    assert yaml_path.exists() and txt_path.exists()

    import yaml as pyyaml
    cfg = pyyaml.safe_load(yaml_path.read_text())
    assert cfg["nc"] == 5
    assert cfg["names"]["car"] if isinstance(cfg["names"], dict) else "car" in cfg["names"]

    lines = txt_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert all(l.endswith("img_0.jpg") or l.endswith("img_1.jpg") for l in lines)

    assert out["night"]["images"] == 2
    assert 0 < out["night"]["mAP50"] <= 1.0


def test_stratified_empty_groups(eval_env, monkeypatch):
    """No groups -> no eval, empty dict."""
    mod = _load_module(monkeypatch)
    args = SimpleNamespace(imgsz=640, conf=0.001, iou=0.6, device="cpu")
    out = mod.run_stratified_eval(
        _FakeModel(), {}, eval_env / "dataset.yaml", args, None
    )
    assert out == {}


def test_stratified_val_error_contained(eval_env, monkeypatch):
    """A val() exception for one group must not kill the whole run."""
    mod = _load_module(monkeypatch)

    class _Boom(_FakeModel):
        def val(self, data=None, **kwargs):
            if data and "night" in str(data):
                raise RuntimeError("boom")
            return _FakeMetrics()

    args = SimpleNamespace(imgsz=640, conf=0.001, iou=0.6, device="cpu")
    groups = {
        "daytime": [f"img_{i}.jpg" for i in range(3)],
        "night": [f"img_{i}.jpg" for i in range(3)],
    }
    out = mod.run_stratified_eval(
        _Boom(), groups, eval_env / "dataset.yaml", args, None, min_images=1
    )
    assert "daytime" in out and "night" not in out
