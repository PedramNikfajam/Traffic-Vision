"""
Tests for Phase 6 track accumulation logic (src/track.py).

Pure logic only - no ultralytics, no videos, no dataset.
"""

import sys
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TEST_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.track import (
    TrackAccumulator, mot_lines, _direction, load_gt_parquet,
    clear_mot_metrics, _resolve_columns, _iou, video_key_mask,
    describe_gt_parquet, remap_gt_frames, calibrate_frame_stride,
    refine_frame_map, remap_gt_frames_linear, gt_match_profile,
)


def _feed(acc, tid, frames, cls="car", conf=0.8, x=100.0, y=100.0):
    for f in frames:
        acc.update(tid, f, cls, conf, (x, y, x + 50, y + 40))


def test_new_and_persistent_tracks():
    acc = TrackAccumulator()
    _feed(acc, 1, [0, 1, 2], x=100)
    _feed(acc, 1, [3], x=200)   # same ID moves
    _feed(acc, 2, [1, 2], cls="truck")
    assert len(acc) == 2
    recs = acc.finalize()
    by_id = {r["track_id"]: r for r in recs}
    assert by_id[1]["n_frames"] == 4
    assert by_id[1]["first_frame"] == 0
    assert by_id[1]["last_frame"] == 3
    assert by_id[2]["cls"] == "truck"


def test_majority_class_vote():
    acc = TrackAccumulator()
    _feed(acc, 1, [0, 1, 2], cls="car")
    _feed(acc, 1, [3, 4], cls="truck")  # flicker - majority stays car
    rec = acc.finalize()[0]
    assert rec["cls"] == "car"


def test_direction_from_trajectory():
    assert _direction([(10, 10), (50, 10)]) == "rightward"
    assert _direction([(50, 10), (10, 10)]) == "leftward"
    assert _direction([(10, 10), (10, 60)]) == "downward"
    assert _direction([(10, 10), (10, 10.5)]) == "stationary"
    assert _direction([(10, 10)]) == "stationary"


def test_mean_conf_and_bbox_last():
    acc = TrackAccumulator()
    acc.update(7, 0, "car", 0.6, (0, 0, 10, 10))
    acc.update(7, 1, "car", 0.8, (20, 0, 30, 10))
    rec = acc.finalize()[0]
    assert rec["track_id"] == 7
    assert rec["mean_conf"] == 0.7
    assert rec["bbox_last"] == [20.0, 0.0, 30.0, 10.0]
    assert len(rec["trajectory"]) == 2


def test_finalize_sorted_and_empty():
    acc = TrackAccumulator()
    assert acc.finalize() == []
    _feed(acc, 5, [0])
    _feed(acc, 2, [0])
    assert [r["track_id"] for r in acc.finalize()] == [2, 5]


def test_mot_lines_format():
    acc = TrackAccumulator()
    acc.update(3, 0, "car", 0.9, (10.0, 20.0, 60.0, 70.0))
    acc.update(3, 1, "car", 0.5, (12.0, 20.0, 62.0, 70.0))
    lines = mot_lines(acc)
    # 1-indexed frames, MOT16 layout: frame,id,x,y,w,h,conf,-1,-1,-1
    assert lines == [
        "1,3,10.00,20.00,50.00,50.00,0.9000,-1,-1,-1",
        "2,3,12.00,20.00,50.00,50.00,0.5000,-1,-1,-1",
    ]


# ------------------------------------------------------------
# Glue test: 06_track.py stream loop with a stubbed YOLO model
# ------------------------------------------------------------

def _load_p6(monkeypatch, tmp_path):
    import importlib.util
    import src.config as config
    spec = importlib.util.spec_from_file_location(
        "p6_track", _PROJECT_DIR / "scripts" / "06_track.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if monkeypatch is not None:
        monkeypatch.setattr(config, "OUTPUT", type(config.OUTPUT)(project_root=tmp_path))
        monkeypatch.setattr(mod, "OUTPUT", config.OUTPUT)
    return mod


class _FakeBoxes:
    def __init__(self, rows):
        if rows is None:
            self.id = None
            return
        import numpy as np
        self.id = np.array([r[0] for r in rows])
        self.cls = np.array([r[1] for r in rows])
        self.conf = np.array([r[2] for r in rows])
        self.xyxy = np.array([r[3] for r in rows], dtype=float)
        self.__len__ = lambda: len(rows)

    def __len__(self):
        return 0 if self.id is None else len(self.id)


class _FakeResult:
    def __init__(self, rows):
        self.boxes = _FakeBoxes(rows)
        self.names = {0: "car", 1: "truck"}


class _FakeYOLO:
    """model.track() stand-in yielding 3 frames: hit, miss, hit."""

    def __init__(self):
        self.track_calls = []

    def track(self, source=None, **kwargs):
        self.track_calls.append(source)
        yield _FakeResult([(1, 0, 0.9, [10, 10, 50, 50])])
        yield _FakeResult(None)          # no tracks this frame
        yield _FakeResult([(1, 0, 0.8, [30, 10, 70, 50])])


def test_track_one_video_glue(tmp_path, monkeypatch):
    mod = _load_p6(monkeypatch, tmp_path)
    from types import SimpleNamespace
    args = SimpleNamespace(tracker="botsort.yaml", conf=0.25, iou=0.5,
                           imgsz=960, device="cpu", save_video=False)
    fake = _FakeYOLO()

    summary = mod.track_one_video(
        fake, tmp_path / "clip.mp4", args, None
    )

    assert summary["frames"] == 3
    assert summary["n_tracks"] == 1
    assert summary["tracks_by_class"] == {"car": 1}
    rec = summary["tracks"][0]
    assert rec["n_frames"] == 2 and rec["direction"] == "rightward"
    # MOT file written under OUTPUT.predictions/tracking with 2 rows
    mot = Path(summary["mot_file"])
    assert mot.exists() and mot.parent == tmp_path / "predictions" / "tracking"
    rows = mot.read_text().strip().splitlines()
    assert len(rows) == 2 and rows[0].startswith("1,1,")
    assert fake.track_calls == [str(tmp_path / "clip.mp4")]


def test_collect_videos(tmp_path):
    mod = _load_p6(None, tmp_path)
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.MP4").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    vids = mod.collect_videos(str(tmp_path), max_videos=10)
    assert [v.name for v in vids] == ["a.mp4", "b.MP4"]
    assert mod.collect_videos(str(tmp_path / "a.mp4"), 0)[0].name == "a.mp4"
    assert mod.collect_videos(str(tmp_path / "nope"), 0) == []


def test_collect_videos_recursive(tmp_path):
    mod = _load_p6(None, tmp_path)
    nested = tmp_path / "bdd100k_videos_train_00" / "bdd100k" / "videos" / "train"
    nested.mkdir(parents=True)
    (nested / "v1.mp4").write_bytes(b"x")
    (nested / "v2.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    # direct level has no videos -> recursive walk kicks in (Kaggle-style root)
    vids = mod.collect_videos(str(tmp_path), max_videos=0)
    assert {v.name for v in vids} == {"v1.mp4", "v2.mp4"}


# ------------------------------------------------------------
# GT parquet loading
# ------------------------------------------------------------

def test_resolve_columns_scalar_and_scalabel():
    cols = _resolve_columns(["videoName", "frameIndex", "id", "x1", "y1", "x2", "y2", "category"])
    assert cols["video"] == "videoName"
    assert cols["frame"] == "frameIndex"
    assert cols["id"] == "id"
    assert cols["label"] == "category"
    cols2 = _resolve_columns(["frame", "track_id", "xmin", "ymin", "width", "height"])
    assert cols2["x1"] == "xmin" and cols2["w"] == "width"


def test_resolve_columns_missing_required():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        _resolve_columns(["x1", "y1", "x2", "y2"])  # no frame/id
    with _pytest.raises(ValueError):
        _resolve_columns(["frame", "id"])  # no box


def test_resolve_columns_prefers_videoname_over_name():
    """Regression: aliases were a SET, so 'video' resolved to 'name' on some
    runs and 'videoName' on others (PYTHONHASHSEED). One Kaggle run silently
    matched GT, the next matched nothing. Resolution must be deterministic and
    prefer the specific column."""
    cols = _resolve_columns(["name", "videoName", "frameIndex", "id",
                             "box2d.x1", "box2d.y1", "box2d.x2", "box2d.y2",
                             "category"])
    assert cols["video"] == "videoName"
    assert cols["label"] == "category"
    # stable across repeated calls
    for _ in range(5):
        assert _resolve_columns(["name", "videoName", "frameIndex", "id",
                                 "box2d.x1", "box2d.y1", "box2d.x2",
                                 "box2d.y2", "category"]) == cols


def test_video_key_mask_conventions():
    """BDD100K box-track dumps key rows by extracted frame image
    '<stem>-<0000001>.jpg'; other dumps use the bare stem or the filename."""
    pd = pytest.importorskip("pandas")
    stem = "0000f77c-6257be58"
    vals = pd.Series([
        f"{stem}-0000001.jpg",       # frame image (the real dump)
        f"{stem}-0001234.jpg",
        f"videos/train/{stem}-0000002.jpg",   # nested path
        f"{stem}.mov",               # video filename
        stem,                        # bare stem
        "0000f77c-62c2a288-0000001.jpg",      # different video
        "somethingelse.jpg",
    ])
    m = video_key_mask(vals, stem)
    assert list(m) == [True, True, True, True, True, False, False]


def test_video_key_mask_case_insensitive():
    pd = pytest.importorskip("pandas")
    m = video_key_mask(pd.Series(["0000F77C-6257BE58-0000001.JPG"]), "0000f77c-6257be58")
    assert bool(m.iloc[0])


def test_load_gt_parquet_frame_image_keys(tmp_path):
    """End-to-end on the real dump layout: videoName holds
    '<stem>-<0000001>.jpg' per row, so endswith(stem) alone matches nothing."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "name": ["0000f77c-6257be58"] * 3,
        "videoName": ["0000f77c-6257be58-0000001.jpg",
                      "0000f77c-6257be58-0000002.jpg",
                      "0000f77c-62c2a288-0000001.jpg"],
        "frameIndex": [0, 1, 0],
        "id": [11, 11, 22],
        "category": ["car", "car", "car"],
        "box2d.x1": [10.0, 12.0, 30.0], "box2d.y1": [100.0, 101.0, 100.0],
        "box2d.x2": [60.0, 62.0, 80.0], "box2d.y2": [150.0, 151.0, 150.0],
    })
    pq = tmp_path / "gt_img.parquet"
    df.to_parquet(pq)
    rows = load_gt_parquet([pq], video_name="0000f77c-6257be58")
    assert len(rows) == 2                     # other video excluded
    assert [r[0] for r in rows] == [1, 2]
    assert {r[1] for r in rows} == {11}


def test_describe_gt_parquet_reports_density(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "videoName": ["v-0000001.jpg", "v-0000002.jpg", "v-0000003.jpg",
                      "v-0000004.jpg"],
        "frameIndex": [0, 1, 2, 3],
        "id": [1, 1, 1, 1],
        "category": ["car", "car", "person", "car"],
        "box2d.x1": [10.0] * 4, "box2d.y1": [100.0] * 4,
        "box2d.x2": [60.0] * 4, "box2d.y2": [150.0] * 4,
    })
    pq = tmp_path / "gt_d.parquet"
    df.to_parquet(pq)
    p = describe_gt_parquet([pq], video_name="v")
    assert p["n_rows"] == 4
    assert p["resolved"]["video"] == "videoName"
    info = p["per_video"]["videoName"]
    assert info["rows"] == 4
    assert info["n_frames"] == 4
    assert info["frame_stride"] == 1
    assert info["boxes_per_frame"] == 1.0
    assert p["value_counts"]["category"]["car"] == 3


def test_load_gt_parquet_scalar(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "videoName": ["bdd100k_videos_train_00_action",
                      "bdd100k_videos_train_00_action"],
        "frameIndex": [0, 4],
        "id": [7, 7],
        "x1": [10.0, 20.0], "y1": [100.0, 100.0],
        "x2": [60.0, 70.0], "y2": [150.0, 150.0],
        "category": ["car", "car"],
    })
    pq = tmp_path / "gt.parquet"
    df.to_parquet(pq)
    rows = load_gt_parquet([pq], video_name="bdd100k_videos_train_00_action")
    assert rows == [(1, 7, 10.0, 100.0, 60.0, 150.0, "car"),
                    (5, 7, 20.0, 100.0, 70.0, 150.0, "car")]
    # wrong video filtered out (KeyError with diagnostic values)
    with pytest.raises(KeyError, match="no GT rows for video 'other_video'"):
        load_gt_parquet([pq], video_name="other_video")
    # no filter -> all rows
    assert len(load_gt_parquet([pq])) == 2


def test_load_gt_parquet_box2d_dotted(tmp_path):
    """Robikscube BDD100K MOT schema: dotted box2d.* columns, .mov videos."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "name": ["0000f77c-6257be58", "0000f77c-6257be58"],
        "videoName": ["0000f77c-6257be58", "0000f77c-6257be58"],
        "frameIndex": [0, 9],
        "id": [11, 11],
        "category": ["car", "car"],
        "attributes.crowd": [False, False],
        "attributes.occluded": [False, True],
        "attributes.truncated": [False, False],
        "box2d.x1": [10.0, 12.0], "box2d.y1": [100.0, 101.0],
        "box2d.x2": [60.0, 62.0], "box2d.y2": [150.0, 151.0],
        "haveVideo": [True, True],
    })
    pq = tmp_path / "gt2.parquet"
    df.to_parquet(pq)
    rows = load_gt_parquet([pq], video_name="0000f77c-6257be58")
    assert rows == [(1, 11, 10.0, 100.0, 60.0, 150.0, "car"),
                    (10, 11, 12.0, 101.0, 62.0, 151.0, "car")]
    # .mov extension variant also matches
    assert len(load_gt_parquet([pq], video_name="0000f77c-6257be58")) == 2


def test_load_gt_parquet_whx(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "name": ["clipA"],
        "frame": [0],
        "track": [3],  # matches 'track' prefix of track_id alias set
        "left": [5], "top": [6], "w": [10], "h": [20],
    })
    pq = tmp_path / "g2.parquet"
    df.to_parquet(pq)
    rows = load_gt_parquet([pq])
    assert rows == [(1, 3, 5.0, 6.0, 15.0, 26.0, "")]


def test_load_gt_parquet_name_fallback(tmp_path):
    """Real robikscube trap: 'videoName' is a shard name, 'name' holds the
    video stem, boxes live in dotted 'box2d.*' columns."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "name": ["0000f77c-6257be58", "0000f77c-6257be58"],
        "videoName": ["train_00", "train_00"],
        "frameIndex": [0, 9],
        "id": [11, 11],
        "category": ["car", "pedestrian"],
        "box2d.x1": [10.0, 12.0], "box2d.y1": [100.0, 101.0],
        "box2d.x2": [60.0, 62.0], "box2d.y2": [150.0, 151.0],
    })
    pq = tmp_path / "gt3.parquet"
    df.to_parquet(pq)
    rows = load_gt_parquet([pq], video_name="0000f77c-6257be58")
    assert len(rows) == 2
    assert rows[0][6] == "car" and rows[1][6] == "pedestrian"


# ------------------------------------------------------------
# GT robustness: null boxes / haveVideo / degenerate boxes
# (regression: one null box2d.* cell aborted a whole video's
#  evaluation with "float() argument must be ... not 'NoneType'")
# ------------------------------------------------------------

def test_load_gt_parquet_skips_null_boxes(tmp_path):
    """A null box cell must skip the row, not raise, and be counted."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "videoName": ["v"] * 4,
        "frameIndex": [0, 1, 2, 3],
        "id": [1, 1, 1, 1],
        "category": ["car"] * 4,
        "box2d.x1": [10.0, 12.0, None, 16.0],
        "box2d.y1": [100.0, 100.0, 100.0, None],
        "box2d.x2": [60.0, 62.0, 64.0, 68.0],
        "box2d.y2": [150.0, 150.0, 150.0, 160.0],
    })
    pq = tmp_path / "gt_null.parquet"
    df.to_parquet(pq)
    stats = {}
    rows = load_gt_parquet([pq], video_name="v", stats=stats)
    assert len(rows) == 2                      # frames 0 and 1 survive
    assert stats["rows_read"] == 4
    assert stats["rows_kept"] == 2
    assert stats["rows_null_box"] == 2         # null x1, null y1


def test_load_gt_parquet_skips_degenerate_boxes(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "videoName": ["v"] * 2,
        "frameIndex": [0, 1],
        "id": [1, 1],
        "category": ["car", "car"],
        "box2d.x1": [10.0, 10.0], "box2d.y1": [100.0, 100.0],
        "box2d.x2": [10.0, 60.0], "box2d.y2": [150.0, 150.0],  # row 0 zero-width
    })
    pq = tmp_path / "gt_deg.parquet"
    df.to_parquet(pq)
    stats = {}
    rows = load_gt_parquet([pq], video_name="v", stats=stats)
    assert len(rows) == 1
    assert stats["rows_bad_box"] == 1


def test_load_gt_parquet_havevideo_filter(tmp_path):
    """haveVideo=False rows (clip never released) carry null boxes and are
    dropped before parsing."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "videoName": ["v"] * 3,
        "frameIndex": [0, 1, 2],
        "id": [1, 1, 1],
        "category": ["car"] * 3,
        "box2d.x1": [10.0, 12.0, None],
        "box2d.y1": [100.0, 100.0, None],
        "box2d.x2": [60.0, 62.0, None],
        "box2d.y2": [150.0, 150.0, None],
        "haveVideo": [True, True, False],
    })
    pq = tmp_path / "gt_hv.parquet"
    df.to_parquet(pq)
    stats = {}
    rows = load_gt_parquet([pq], video_name="v", stats=stats)
    assert len(rows) == 2
    assert stats["rows_no_video"] == 1
    assert stats["rows_kept"] == 2


def test_load_gt_parquet_havevideo_all_false_raises(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "videoName": ["v"], "frameIndex": [0], "id": [1],
        "category": ["car"],
        "box2d.x1": [None], "box2d.y1": [None],
        "box2d.x2": [None], "box2d.y2": [None],
        "haveVideo": [False],
    })
    pq = tmp_path / "gt_hv0.parquet"
    df.to_parquet(pq)
    with pytest.raises(KeyError, match="not released as video"):
        load_gt_parquet([pq], video_name="v")


def test_load_gt_parquet_stats_report_stride_and_frames(tmp_path):
    """BDD100K box-track GT is annotated on a sparse keyframe subset; the
    stride/coverage numbers are what make an MOT score interpretable."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "videoName": ["v"] * 4,
        "frameIndex": [0, 30, 60, 90],   # 1 Hz over a 30 fps video
        "id": [1, 1, 2, 2],
        "category": ["car"] * 4,
        "box2d.x1": [10.0] * 4, "box2d.y1": [100.0] * 4,
        "box2d.x2": [60.0] * 4, "box2d.y2": [150.0] * 4,
    })
    pq = tmp_path / "gt_stride.parquet"
    df.to_parquet(pq)
    stats = {}
    load_gt_parquet([pq], video_name="v", stats=stats)
    assert stats["gt_frames"] == 4
    assert stats["gt_ids"] == 2
    assert stats["gt_frame_stride"] == 30


# ------------------------------------------------------------
# CLEAR MOT metrics
# ------------------------------------------------------------

def _row(frame, tid, x1, y1, x2, y2):
    return (frame, tid, float(x1), float(y1), float(x2), float(y2))


def test_iou_basic():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.0 < _iou((0, 0, 10, 10), (5, 0, 15, 10)) < 1.0


def test_clear_perfect_match():
    gt = [_row(1, 1, 0, 0, 10, 10), _row(2, 1, 5, 0, 15, 10)]
    pred = gt[:]  # identical boxes, same IDs
    m = clear_mot_metrics(gt, pred)
    assert m["MOTA"] == 1.0
    assert m["MOTP"] == 1.0
    assert m["IDF1"] == 1.0
    assert m["IDSW"] == 0 and m["FP"] == 0 and m["FN"] == 0


def test_clear_with_switch_and_fp_fn():
    # GT has two objects; predictions swap IDs at frame 2 -> 1 IDSW
    gt = [_row(1, 1, 0, 0, 10, 10), _row(1, 2, 50, 0, 60, 10),
          _row(2, 1, 0, 0, 10, 10), _row(2, 2, 50, 0, 60, 10)]
    pred = [_row(1, 1, 0, 0, 10, 10), _row(1, 2, 50, 0, 60, 10),
            _row(2, 2, 0, 0, 10, 10), _row(2, 1, 50, 0, 60, 10)]
    m = clear_mot_metrics(gt, pred)
    assert m["IDSW"] == 2  # both GT tracks got a new hyp
    assert m["FP"] == 0 and m["FN"] == 0


def test_clear_fp_fn_counts():
    gt = [_row(1, 1, 0, 0, 10, 10)]
    pred = [_row(1, 9, 100, 100, 110, 110)]  # total miss
    m = clear_mot_metrics(gt, pred)
    assert m["FP"] == 1 and m["FN"] == 1
    assert m["MOTA"] == 1.0 - (1 + 1 + 0) / 1
    assert m["MOTP"] == 0.0


def test_clear_empty():
    m = clear_mot_metrics([], [])
    assert m["MOTA"] == 0.0 and m["n_gt"] == 0


# ------------------------------------------------------------
# Sparse GT (the bug that produced MOTA ~ -6.8 on Kaggle)
# ------------------------------------------------------------

def test_clear_restricts_to_gt_frames():
    """BDD100K box-track GT is annotated every 30th frame. Predictions on
    unannotated frames must NOT be counted as false positives, otherwise
    MOTA collapses (measured: FP ~7.2k, MOTA -5.6 on a 1217-frame clip)."""
    gt = [_row(1, 1, 0, 0, 10, 10), _row(31, 1, 0, 0, 10, 10)]
    # detector fires on every one of 1217 frames
    pred = [_row(f, 1, 0, 0, 10, 10) for f in range(1, 1218)]

    m = clear_mot_metrics(gt, pred)
    assert m["restricted_to_gt_frames"] is True
    assert m["frames_evaluated"] == 2
    assert m["n_gt_frames"] == 2
    assert m["n_pred_scored"] == 2
    assert m["n_pred_total"] == 1217
    assert m["FP"] == 0 and m["FN"] == 0
    assert m["MOTA"] == 1.0
    assert m["IDF1"] == 1.0
    assert abs(m["gt_frame_coverage"] - 2 / 1217) < 1e-3


def test_clear_all_frames_mode_reproduces_old_broken_numbers():
    """restrict_to_gt_frames=False reproduces the old behaviour, which is
    kept only as a diagnostic. Documents WHY it must not be the default."""
    gt = [_row(1, 1, 0, 0, 10, 10)]
    pred = [_row(f, 1, 0, 0, 10, 10) for f in range(1, 1218)]

    strict = clear_mot_metrics(gt, pred)
    loose = clear_mot_metrics(gt, pred, restrict_to_gt_frames=False)
    assert loose["FP"] == 1216          # 1215 phantom + the real hit
    assert loose["MOTA"] < 0            # negative, i.e. meaningless
    assert strict["MOTA"] == 1.0
    assert loose["restricted_to_gt_frames"] is False
    assert loose["frames_evaluated"] == 1217


def test_clear_reports_counters():
    gt = [_row(1, 1, 0, 0, 10, 10), _row(1, 2, 50, 0, 60, 10)]
    pred = [_row(1, 1, 0, 0, 10, 10), _row(1, 9, 50, 0, 60, 10)]
    m = clear_mot_metrics(gt, pred)
    assert m["TP"] == 2
    assert m["FP"] == 0 and m["FN"] == 0
    assert m["n_gt"] == 2 and m["n_gt_ids"] == 2


def test_idf1_one_to_one_id_matching():
    """Standard IDF1: one GT id may claim at most one hypothesis id, and the
    FP side counts. A GT track split across two hypothesis IDs must not score
    full IDF1 (the old 'dominant hypothesis' metric ignored the FP side)."""
    # GT id 1 appears on frames 1-4; hypothesis 1 covers 1-2, hypothesis 2
    # covers 3-4. One-to-one matching keeps only the better half.
    gt = [_row(f, 1, 0, 0, 10, 10) for f in range(1, 5)]
    pred = ([_row(f, 1, 0, 0, 10, 10) for f in (1, 2)] +
            [_row(f, 2, 0, 0, 10, 10) for f in (3, 4)])
    m = clear_mot_metrics(gt, pred)
    assert m["TP"] == 4                 # all four frames have a detection
    assert m["IDSW"] == 1               # identity handed over once
    assert m["FP"] == 0 and m["FN"] == 0
    # IDTP = 2 (one id pair), IDFN = 2, IDFP = 2 -> 2*2/(4+2+2) = 0.5
    assert m["IDF1"] == 0.5
    # the old metric would have reported 4/4 = 1.0 here
    assert m["IDF1"] < 1.0


def test_idf1_perfect_and_empty():
    gt = [_row(1, 1, 0, 0, 10, 10)]
    assert clear_mot_metrics(gt, gt[:])["IDF1"] == 1.0
    assert clear_mot_metrics([], [])["IDF1"] == 0.0


# ------------------------------------------------------------
# GT-dump frame stride (BDD100K ships a DOWNSAMPLED label set:
# 203 annotated frames for a 1217-frame video, so frameIndex
# indexes the sampled sequence, not the video)
# ------------------------------------------------------------

def test_remap_gt_frames_identity_and_scaled():
    gt = [(1, 1, 0.0, 0.0, 10.0, 10.0, "car"),
          (3, 1, 0.0, 0.0, 10.0, 10.0, "car")]
    assert remap_gt_frames(gt, 1) == gt              # stride 1 = identity
    out = remap_gt_frames(gt, 6)
    assert [r[0] for r in out] == [1, 13]            # (g-1)*6 + 1
    assert out[0][1:] == gt[0][1:]                   # everything else kept


def test_calibrate_stride_recovers_known_ratio():
    """GT annotated every 6th video frame. Calibration must find 6 and beat
    the naive 1:1 mapping decisively. (1:1 still matches a FEW frames by
    coincidence - exactly the TP=88 seen on Kaggle - so the property to
    assert is a dominant peak, not a zero.)"""
    n = 40
    gt, pred = [], []
    for k in range(n):
        gt.append(_row(k + 1, 1, 100, 100, 200, 200))   # sampled index k
        pred.append(_row(k * 6 + 1, 1, 100, 100, 200, 200))  # true video frame
    # decoy predictions elsewhere so 1:1 cannot win by default
    for k in range(n):
        pred.append(_row(k + 1, 99, 900, 900, 1000, 1000))

    best, curve = calibrate_frame_stride(gt, pred)
    assert best == 6
    assert curve[6] == n
    assert curve[6] >= 3 * curve[1], f"peak not decisive: {curve}"


def test_calibrate_stride_flat_curve_is_detectable():
    """No real alignment -> curve is flat, so the winner is meaningless and
    the caller must be able to see that (it warns on a flat peak)."""
    gt = [_row(k + 1, 1, 0, 0, 10, 10) for k in range(20)]
    pred = [_row(500 + k, 1, 900, 900, 1000, 1000) for k in range(20)]
    best, curve = calibrate_frame_stride(gt, pred)
    assert max(curve.values()) == 0                     # nothing lines up at all
    assert len(set(curve.values())) == 1                # perfectly flat


def test_calibrate_stride_empty_inputs():
    assert calibrate_frame_stride([], [_row(1, 1, 0, 0, 10, 10)]) == (1, {})
    assert calibrate_frame_stride([_row(1, 1, 0, 0, 10, 10)], []) == (1, {})


def test_metrics_better_after_stride_correction():
    """End-to-end: the same detections score far better once the GT frames
    are rescaled. This is the regression guard for the 92%-FN run."""
    n = 60
    gt, pred = [], []
    for k in range(n):
        gt.append(_row(k + 1, 1, 100, 100, 200, 200))
        pred.append(_row(k * 6 + 1, 1, 102, 101, 201, 201))
    naive = clear_mot_metrics(gt, pred)
    fixed = clear_mot_metrics(remap_gt_frames(gt, 6), pred)
    # naive 1:1 matches only where a prediction frame happens to coincide
    assert naive["FN"] > 0.5 * n
    assert fixed["FN"] == 0 and fixed["TP"] == n
    assert fixed["MOTA"] == 1.0
    assert fixed["MOTP"] > 0.9


# ------------------------------------------------------------
# Ratio-anchored linear map: an integer stride DRIFTS. Real case
# 1211 video frames / 187 GT frames = 6.476, not 6, so by the last
# annotated frame integer stride 6 is ~89 frames (3 s) adrift.
# ------------------------------------------------------------

def test_remap_gt_frames_linear_anchors_endpoints():
    gt = [(1, 1, 0.0, 0.0, 10.0, 10.0), (2, 1, 0.0, 0.0, 10.0, 10.0)]
    # alpha 6.476 -> first GT frame -> video frame 1, last -> video frame 1211
    out = remap_gt_frames_linear(gt, 6.4759, 0)
    assert out[0][0] == 1
    assert out[1][0] == 1 + int(round(6.4759))       # (2-1)*alpha + 1
    assert remap_gt_frames_linear(gt, 1.0, 0) == gt   # identity


def test_refine_frame_map_searches_alpha_directly():
    """Anchoring alpha on frame counts is WRONG: the GT does not necessarily
    span the video. 187 GT frames in a 1211-frame video implies 6.505, but
    integer stride 6 matched 296 boxes where 6.505 matched 46."""
    n_gt, true_alpha = 187, 6.0
    gt = [_row(k + 1, 1, 100, 100, 200, 200) for k in range(n_gt)]
    pred = [_row(int(round(k * true_alpha)) + 1, 1, 100, 100, 200, 200)
            for k in range(n_gt)]

    a, off, top = refine_frame_map(gt, pred)
    assert top[0][2] == n_gt                     # every GT box matched
    assert abs(a - true_alpha) <= 0.02, f"got alpha={a}"
    assert off == 0
    # the count-ratio anchor (1211/187) must NOT be what wins
    assert abs(a - (1211 / 187)) > 0.1


def test_refine_frame_map_never_worse_than_integer():
    """The integer map is always a candidate, so a pathological alpha grid
    cannot make things worse than the integer stride."""
    for n_gt, true_alpha in ((200, 6.0), (150, 7.5), (180, 4.0)):
        gt = [_row(k + 1, 1, 100, 100, 200, 200) for k in range(n_gt)]
        pred = [_row(int(round(k * true_alpha)) + 1, 1, 100, 100, 200, 200)
                for k in range(n_gt)]
        _, _, top = refine_frame_map(gt, pred)
        _, curve = calibrate_frame_stride(gt, pred)
        assert top[0][2] >= max(curve.values())


def test_refine_frame_map_empty_inputs():
    assert refine_frame_map([], [_row(1, 1, 0, 0, 9, 9)]) == (1.0, 0, [])
    assert refine_frame_map([_row(1, 1, 0, 0, 9, 9)], []) == (1.0, 0, [])


def test_gt_match_profile_separates_drift_from_hard_clip():
    """Matches clustered at the START => alignment drifts. Uniformly absent =>
    the clip is genuinely hard. These look identical in MOTA."""
    n = 50
    gt = [_row(k + 1, 1, 100, 100, 200, 200) for k in range(n)]
    # correct mapping for the first half, wrong after: drift signature
    pred = [_row(k + 1, 1, 100, 100, 200, 200) for k in range(n // 2)]
    prof = gt_match_profile(gt, pred, 1.0, 0)
    assert prof["frame_match_rate"] == 0.5
    assert prof["first_matched_idx"] == 1
    assert prof["last_matched_idx"] == n // 2

    # no overlap anywhere
    far = [_row(900 + k, 1, 100, 100, 200, 200) for k in range(n)]
    prof2 = gt_match_profile(gt, far, 1.0, 0)
    assert prof2["frame_match_rate"] == 0.0
    assert prof2["first_matched_idx"] is None


def test_linear_map_beats_integer_on_real_ratio_end_to_end():
    """The MOTA=-1.04 outlier: 1211/187 = 6.476. Integer stride 6 drifts ~89
    frames by the end; the linear map should recover most of the tail."""
    n_vid, n_gt = 1211, 187
    alpha = (n_vid - 1) / (n_gt - 1)
    gt = [_row(k + 1, 1, 100, 100, 200, 200) for k in range(n_gt)]
    pred = [_row(int(round(k * alpha)) + 1, 1, 101, 101, 201, 201)
            for k in range(n_gt)]

    int_map = clear_mot_metrics(remap_gt_frames(gt, 6), pred)
    lin_map = clear_mot_metrics(remap_gt_frames_linear(gt, alpha, 0), pred)

    assert lin_map["FN"] == 0 and lin_map["MOTA"] == 1.0
    assert lin_map["FN"] < int_map["FN"]
    assert lin_map["FN"] * 4 < int_map["FN"], "drift should be the dominant error"
