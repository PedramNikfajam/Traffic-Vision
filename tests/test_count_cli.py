"""
Tests for the Phase 7 CLI glue (scripts/07_count.py).

Focus on the parts that were actually wrong in production runs:
  - frame-height resolution order (a biased guess outranked a known fact)
  - per-video sweep bookkeeping (a stale loop variable leaked one video's
    table into every other video's, and into the sweep plot)
  - recursive video lookup (Kaggle nests videos several levels deep)
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TEST_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


def _load_p7(monkeypatch, tmp_path):
    import src.config as config
    spec = importlib.util.spec_from_file_location(
        "p7_count", _PROJECT_DIR / "scripts" / "07_count.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if monkeypatch is not None:
        monkeypatch.setattr(config, "OUTPUT",
                            type(config.OUTPUT)(project_root=tmp_path))
        monkeypatch.setattr(mod, "OUTPUT", config.OUTPUT)
    return mod


# ------------------------------------------------------------
# frame height resolution
# ------------------------------------------------------------

def test_height_prefers_opencv_then_known_constant(tmp_path, monkeypatch):
    mod = _load_p7(monkeypatch, tmp_path)
    (tmp_path / "a.mov").write_bytes(b"x")
    monkeypatch.setattr(mod, "probe_frame_size", lambda d, s: (1280.0, 720.0))
    h, src = mod.resolve_frame_size("a", str(tmp_path), tmp_path / "nope.json", 0.0)
    assert (h, src) == (720.0, "opencv")


def test_height_falls_back_to_bdd100k_constant_not_the_heuristic(tmp_path,
                                                                monkeypatch):
    """Regression: the biased bbox heuristic used to outrank the verified
    BDD100K constant and reported 440/640 px for 720 px videos."""
    from src.config import BDD100K_HEIGHT
    mod = _load_p7(monkeypatch, tmp_path)
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        return 440.0

    monkeypatch.setattr(mod, "frame_height_from_report", _boom)
    h, src = mod.resolve_frame_size("a", None, tmp_path / "r.json", 0.0)
    assert src == "bdd100k-constant"
    assert h == float(BDD100K_HEIGHT)
    assert not calls, "heuristic must not be consulted by default"


def test_height_heuristic_only_when_explicitly_allowed(tmp_path, monkeypatch):
    mod = _load_p7(monkeypatch, tmp_path)
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({"videos": [
        {"video": "a.mov", "tracks": [{"bbox_last": [0, 0, 10, 440]}]}]}),
        encoding="utf-8")
    h, src = mod.resolve_frame_size("a", None, rep, 0.0, allow_heuristic=True)
    assert (h, src) == (440.0, "bbox-heuristic")
    h2, src2 = mod.resolve_frame_size("a", None, rep, 0.0, allow_heuristic=False)
    assert src2 == "bdd100k-constant" and h2 == 720.0


def test_height_forced_overrides_all(tmp_path, monkeypatch):
    mod = _load_p7(monkeypatch, tmp_path)
    h, src = mod.resolve_frame_size("a", None, tmp_path / "nope.json", 480.0)
    assert (h, src) == (480.0, "forced")


# ------------------------------------------------------------
# recursive video lookup
# ------------------------------------------------------------

def test_video_path_for_searches_recursively(tmp_path, monkeypatch):
    """Kaggle nests videos deep: .../bdd100k_videos_train_00/bdd100k/videos/train"""
    mod = _load_p7(monkeypatch, tmp_path)
    nested = tmp_path / "bdd100k_videos_train_00" / "bdd100k" / "videos" / "train"
    nested.mkdir(parents=True)
    target = nested / "vidA.mov"
    target.write_bytes(b"x")
    found = mod.video_path_for(str(tmp_path), "vidA")
    assert found is not None and Path(found).name == "vidA.mov"


def test_video_path_for_direct_hit_and_misses(tmp_path, monkeypatch):
    mod = _load_p7(monkeypatch, tmp_path)
    (tmp_path / "b.mov").write_bytes(b"x")
    assert Path(mod.video_path_for(str(tmp_path), "b")).name == "b.mov"
    assert mod.video_path_for(str(tmp_path), "nope") is None
    assert mod.video_path_for(None, "b") is None
    assert mod.video_path_for(str(tmp_path / "nodir"), "b") is None


# ------------------------------------------------------------
# end-to-end: per-video sweep bookkeeping
# ------------------------------------------------------------

def _write_mot(path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _synthetic_mot(path, y0, dy, n_tracks=4, n_frames=400):
    rows = []
    for tid in range(1, n_tracks + 1):
        for f in range(1, n_frames + 1):
            cy = y0 + dy * f
            rows.append(f"{f},{tid},200.00,{cy - 20:.2f},80.00,40.00,0.9000,-1,-1,-1")
    _write_mot(path, rows)


def test_per_video_sweep_is_not_a_stale_copy(tmp_path, monkeypatch, capsys):
    """Regression: `entry["line_sweep"]` read a leaked pass-1 loop variable, so
    EVERY video stored the LAST video's sweep table and the merged sweep plot
    showed n_videos x that one table. Videos here have deliberately different
    optima so the error is visible."""
    mod = _load_p7(monkeypatch, tmp_path)
    mot = tmp_path / "predictions" / "tracking"
    mot.mkdir(parents=True)
    rep = tmp_path / "reports"
    rep.mkdir()

    # vidA crosses low lines, vidB only crosses high ones
    _synthetic_mot(mot / "vidA.txt", 300, 0.9)
    _synthetic_mot(mot / "vidB.txt", 600, 0.9)
    (rep / "phase6_tracking.json").write_text(json.dumps({"videos": [
        {"video": "vidA.mov", "tracks": [{"track_id": t, "cls": "car",
                                          "bbox_last": [0, 0, 50, 700]}
                                         for t in range(1, 5)]},
        {"video": "vidB.mov", "tracks": [{"track_id": t, "cls": "car",
                                          "bbox_last": [0, 0, 50, 700]}
                                         for t in range(1, 5)]},
    ]}), encoding="utf-8")

    argv = sys.argv
    sys.argv = ["07_count.py", "--mot-dir", str(mot), "--track-report",
                str(rep / "phase6_tracking.json"), "--line-y", "0.5",
                "--fps", "30", "--sweep", "0.4,0.5,0.6,0.7,0.8", "--no-plots"]
    try:
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 0

    data = json.loads((tmp_path / "reports" / "phase7_counting.json")
                      .read_text(encoding="utf-8"))
    by = {v["video"]: v for v in data["videos"]}

    # every video must carry ITS OWN sweep table
    assert by["vidA"]["line_sweep"] != by["vidB"]["line_sweep"], \
        "per-video sweep tables are identical - stale variable leaked"
    # and the stored table must match the per-video totals in the log
    for stem, v in by.items():
        stored = v["line_sweep"]
        for frac, cell in stored.items():
            assert 0 <= cell["total_counted"] <= v["n_tracks_seen"]
        assert set(stored) == {"0.4", "0.5", "0.6", "0.7", "0.8"}

    # one global line position applied to BOTH videos
    fracs = {v["line_y_fraction"] for v in data["videos"]}
    assert len(fracs) == 1, f"line chosen per video: {fracs}"
    assert data["line_position_chosen"] in {"0.4", "0.5", "0.6", "0.7", "0.8"}

    # the reported position totals must equal the sum over videos
    totals = data["line_position_totals"]
    for frac, total in totals.items():
        assert total == sum(v["line_sweep"][frac]["total_counted"]
                            for v in data["videos"]), \
            f"position totals disagree with per-video sweeps at {frac}"
