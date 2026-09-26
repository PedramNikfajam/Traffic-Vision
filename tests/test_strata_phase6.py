"""
Tests for Phase 6 stratification wiring (scripts/06_track.py) and for the
REAL day/night labels those tests depend on.

Two failure modes are guarded here, and neither shows up as a crash:

1. Someone reintroduces hand-written scene labels for the evaluation clips. The
   previous inventory was eyeballed and disagreed with BDD100K on 3 of 5
   (`0000f77c-cb820c98` was called a night highway; the dataset records
   `dawn/dusk` + `residential`). A stratified table built on those labels would
   have compared the wrong groups and reported 4 night clips when the truth is
   2 night / 2 dawn-dusk / 1 daytime. `test_real_manifest_*` below pins the real
   values, and it is skipped (not silently passed) when the manifest is absent.

2. Phase 6 grows a `stratified_mot_metrics` key that nothing computes, or
   computes it from records built before the labels were attached. The CLI tests
   drive `main()` with a stubbed tracker and assert the report actually
   contains the table, with the pooled arithmetic done from the per-video counts.

The real day/night *data* (10k val images, 30 GB of video) only exists on
Kaggle, so the numeric day/night comparison is verified by the run recorded in
PROGRESS.md, not here. What is verifiable offline is the join and the
arithmetic, and that is what these tests cover.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TEST_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.strata import load_video_metadata, video_key

# The 5 clips Phase 6/7 are evaluated on, and what BDD100K actually records for
# them. Read from reports/manifest_train.json on 2026-09-26.
REAL_CLIP_METADATA = {
    "0000f77c-6257be58": ("daytime", "city street", "clear"),
    "0000f77c-62c2a288": ("dawn/dusk", "highway", "clear"),
    "0000f77c-cb820c98": ("dawn/dusk", "residential", "clear"),
    "0001542f-5ce3cf52": ("night", "city street", "clear"),
    "0001542f-7c670be8": ("night", "highway", "clear"),
}

# Searched in order; the first that exists is used.
_MANIFEST_CANDIDATES = [
    _PROJECT_DIR.parent / "results" / "traffic_ai" / "reports" / "manifest_train.json",
    _PROJECT_DIR.parent / "_output__2" / "traffic_ai" / "reports" / "manifest_train.json",
    _PROJECT_DIR / "reports" / "manifest_train.json",
]


def _real_manifest() -> Path:
    for p in _MANIFEST_CANDIDATES:
        if p.is_file():
            return p
    pytest.skip("real Phase 2 manifest not available locally "
                "(it lives in the Kaggle working dir)")


@pytest.fixture(scope="module")
def real_meta():
    return load_video_metadata([_real_manifest()])


# ------------------------------------------------------------
# the real day/night labels, pinned
# ------------------------------------------------------------

@pytest.mark.parametrize("stem,expected", sorted(REAL_CLIP_METADATA.items()))
def test_real_manifest_metadata_for_each_evaluation_clip(real_meta, stem, expected):
    """Guards against hand-written scene labels creeping back in."""
    assert stem in real_meta, f"{stem} missing from the manifest - the join broke"
    tod, scene, weather = expected
    assert real_meta[stem]["timeofday"] == tod
    assert real_meta[stem]["scene"] == scene
    assert real_meta[stem]["weather"] == weather


def test_real_clip_strata_are_two_night_two_dawn_one_day(real_meta):
    """The headline correction: NOT '4 night + 1 day'."""
    from collections import Counter
    c = Counter(real_meta[s]["timeofday"] for s in REAL_CLIP_METADATA)
    assert c == {"dawn/dusk": 2, "daytime": 1, "night": 2}


def test_real_manifest_has_no_internal_conflicts(real_meta):
    for stem in REAL_CLIP_METADATA:
        assert real_meta[stem]["conflicts"] == {}, \
            f"{stem} has contradictory attributes in the manifest"


def test_real_clips_resolve_from_every_artefact_name(real_meta):
    """The same video is named .mov by Phase 6 and .txt by Phase 7; both must
    land on the same stratum or the two stages report different groups."""
    stem = "0000f77c-cb820c98"
    for name in (f"{stem}.mov", f"{stem}.txt", f"{stem}.jpg",
                 f"{stem}-0000001.jpg", f"/kaggle/x/y/{stem}.mov"):
        assert video_key(name) == stem
        assert real_meta[video_key(name)]["timeofday"] == "dawn/dusk"


# ------------------------------------------------------------
# Phase 6 CLI: the stratified block must actually be produced
# ------------------------------------------------------------

def _load_p6(monkeypatch, tmp_path):
    import src.config as config
    spec = importlib.util.spec_from_file_location(
        "p6_strata", _PROJECT_DIR / "scripts" / "06_track.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(config, "OUTPUT", type(config.OUTPUT)(project_root=tmp_path))
    monkeypatch.setattr(mod, "OUTPUT", config.OUTPUT)
    return mod


@pytest.fixture
def stub_ultralytics(monkeypatch):
    """Replace ultralytics.YOLO so main() can run without a checkpoint or GPU."""
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda path: types.SimpleNamespace(path=path, name="stub")
    fake.__version__ = "0.0.0-stub"
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


def _mot(tp, fp, fn, idsw, n_gt, mota=None):
    return {
        "TP": tp, "FP": fp, "FN": fn, "IDSW": idsw, "n_gt": n_gt,
        "MOTA": mota if mota is not None else round(1 - (fn + fp + idsw) / n_gt, 4),
        "MOTP": 0.8, "IDF1": 0.7, "gt_frame_coverage": 0.1667,
        "restricted_to_gt_frames": True, "gt_frame_stride_applied": 6,
        "n_gt_frames": 203, "n_pred_scored": 180,
    }


def _run_p6(mod, tmp_path, monkeypatch, mot_by_video, manifest_frames):
    """Drive 06_track.main() with a stubbed per-video tracker."""
    vids = tmp_path / "videos"
    vids.mkdir(exist_ok=True)
    for stem in mot_by_video:
        (vids / f"{stem}.mov").write_bytes(b"x")
    model = tmp_path / "best.pt"
    model.write_bytes(b"x")

    def _fake_track(model_obj, video_path, args, logger=None, gt_files=None):
        stem = Path(video_path).stem
        return {
            "video": Path(video_path).name,
            "frames": 1217, "n_tracks": 10,
            "tracks_by_class": {"car": 10},
            "tracks_by_direction": {"rightward": 10},
            "mot_file": f"predictions/tracking/{stem}.txt",
            "tracks": [{"track_id": i, "cls": "car"} for i in range(10)],
            "mot_metrics": mot_by_video[stem],
        }

    monkeypatch.setattr(mod, "track_one_video", _fake_track)
    argv = sys.argv
    sys.argv = ["06_track.py", "--model", str(model), "--videos", str(vids),
                "--max-videos", "0"]
    try:
        rc = mod.main()
    finally:
        sys.argv = argv
    data = json.loads((tmp_path / "reports" / "phase6_tracking.json")
                      .read_text(encoding="utf-8"))
    return rc, data


def _manifest(tmp_path, frames):
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "manifest_train.json").write_text(
        json.dumps([{"frame_name": f"{s}.jpg", "timeofday": t,
                     "scene": sc, "weather": "clear"}
                    for s, t, sc in frames]), encoding="utf-8")


def test_phase6_writes_stratified_mot_metrics(tmp_path, monkeypatch, stub_ultralytics):
    mod = _load_p6(monkeypatch, tmp_path)
    mot = {
        # daytime clip: clean, few FP
        "0000f77c-6257be58": _mot(80, 5, 20, 5, 100),
        # night clip: many FP
        "0001542f-7c670be8": _mot(70, 40, 30, 8, 100),
        # second night clip, a DIFFERENT size so pooled != mean
        "0001542f-5ce3cf52": _mot(210, 60, 90, 22, 300),
    }
    _manifest(tmp_path, [
        ("0000f77c-6257be58", "daytime", "city street"),
        ("0001542f-7c670be8", "night", "highway"),
        ("0001542f-5ce3cf52", "night", "city street"),
    ])
    rc, data = _run_p6(mod, tmp_path, monkeypatch, mot, None)
    assert rc == 0
    assert "stratified_mot_metrics" in data, \
        "Phase 6 produced no stratified table - the day/night comparison is dead"

    by_vid = {v["video"]: v for v in data["videos"]}
    assert by_vid["0000f77c-6257be58.mov"]["timeofday"] == "daytime"
    assert by_vid["0001542f-7c670be8.mov"]["scene"] == "highway"

    t = data["stratified_mot_metrics"]["timeofday"]
    assert t["daytime"]["n_videos_evaluated"] == 1
    assert t["night"]["n_videos_evaluated"] == 2
    # pooled night: TP=280 FP=100 FN=120 IDSW=30 n_gt=400
    #   MOTA = 1 - (120 + 100 + 30)/400 = 0.375
    assert t["night"]["MOTA_pooled"] == pytest.approx(0.375, abs=1e-3)
    assert t["night"]["n_gt"] == 400 and t["night"]["IDSW"] == 30
    # per-video mean reported alongside. It is a DIFFERENT number here (the two
    # clips have different n_gt), which is exactly why both are reported.
    assert t["night"]["MOTA_mean"] == pytest.approx(
        (mot["0001542f-7c670be8"]["MOTA"] + mot["0001542f-5ce3cf52"]["MOTA"]) / 2,
        abs=1e-3)
    assert t["night"]["MOTA_mean"] != t["night"]["MOTA_pooled"], \
        "pooled and mean MOTA must not be conflated"
    # every stratum here rests on fewer than MIN_VIDEOS_PER_STRATUM clips
    assert t["night"]["low_sample"] is True
    assert t["daytime"]["low_sample"] is True

    cov = data["stratified_coverage"]["timeofday"]
    assert cov["n_labelled"] == 3
    assert cov["n_videos_per_stratum"] == {"daytime": 1, "night": 2}

    checks = {c["name"]: c for c in data["integrity_checks"]}
    assert checks["videos_labelled_by_metadata"]["status"] == "PASS"
    assert checks["stratified_results_present"]["status"] == "PASS"


def test_phase6_flags_unlabelled_videos_rather_than_hiding_them(
        tmp_path, monkeypatch, stub_ultralytics):
    mod = _load_p6(monkeypatch, tmp_path)
    mot = {"0000f77c-6257be58": _mot(80, 5, 20, 5, 100),
           "mystery-clip": _mot(80, 5, 20, 5, 100)}
    _manifest(tmp_path, [("0000f77c-6257be58", "daytime", "city street")])
    rc, data = _run_p6(mod, tmp_path, monkeypatch, mot, None)

    t = data["stratified_mot_metrics"]["timeofday"]
    assert t["unlabelled"]["n_videos"] == 1
    assert t["unlabelled"]["videos"] == ["mystery-clip"]
    assert t["daytime"]["n_videos"] == 1        # NOT 2
    checks = {c["name"]: c for c in data["integrity_checks"]}
    assert checks["videos_labelled_by_metadata"]["status"] == "FAIL"
    assert "unlabelled" in checks["videos_labelled_by_metadata"]["detail"]
    assert rc == 0                              # warning severity only


def test_phase6_without_manifest_reports_an_unlabelled_bucket_and_says_so(
        tmp_path, monkeypatch, stub_ultralytics):
    """A table with one `unlabelled` row is reported rather than omitted: the
    videos exist, they just carry no stratum, and the check has to say why."""
    mod = _load_p6(monkeypatch, tmp_path)
    mot = {"0000f77c-6257be58": _mot(80, 5, 20, 5, 100)}
    rc, data = _run_p6(mod, tmp_path, monkeypatch, mot, None)

    t = data["stratified_mot_metrics"]["timeofday"]
    assert list(t) == ["unlabelled", "_excluded_without_metrics"]
    assert t["unlabelled"]["n_videos"] == 1
    assert t["unlabelled"]["videos"] == ["0000f77c-6257be58"]
    checks = {c["name"]: c for c in data["integrity_checks"]}
    assert checks["stratified_results_present"]["status"] == "FAIL"
    assert "no manifest" in checks["stratified_results_present"]["detail"]
    assert checks["videos_labelled_by_metadata"]["status"] == "FAIL"
    assert rc == 0                              # warning severity only


# ------------------------------------------------------------
# --fp-audit wiring
# ------------------------------------------------------------

def _run_p6_audit(tmp_path, monkeypatch, mod, gt_and_pred, manifest=True,
                  pass_flag=True):
    """Drive main() with --fp-audit, using real GT/pred rows per video."""
    from src.track import clear_mot_metrics, false_positive_audit

    vids = tmp_path / "videos"
    vids.mkdir(exist_ok=True)
    for stem in gt_and_pred:
        (vids / f"{stem}.mov").write_bytes(b"x")
    model = tmp_path / "best.pt"
    model.write_bytes(b"x")
    if manifest:
        _manifest(tmp_path, [(s, "night", "highway") for s in gt_and_pred])

    def _fake_track(model_obj, video_path, args, logger=None, gt_files=None):
        stem = Path(video_path).stem
        gt, pred = gt_and_pred[stem]
        m = clear_mot_metrics(gt, pred, collect_fp=True)
        audit = false_positive_audit(gt, m["fp_rows"], window=1)
        m.pop("fp_rows", None)
        return {
            "video": Path(video_path).name, "frames": 60, "n_tracks": 3,
            "tracks_by_class": {"car": 3}, "tracks_by_direction": {"rightward": 3},
            "mot_file": f"predictions/tracking/{stem}.txt",
            "tracks": [{"track_id": i, "cls": "car"} for i in range(3)],
            "mot_metrics": m, "fp_audit": audit,
        }

    monkeypatch.setattr(mod, "track_one_video", _fake_track)
    argv = sys.argv
    sys.argv = ["06_track.py", "--model", str(model), "--videos", str(vids),
                "--max-videos", "0"] + (["--fp-audit"] if pass_flag else [])
    try:
        rc = mod.main()
    finally:
        sys.argv = argv
    main_report = json.loads((tmp_path / "reports" / "phase6_tracking.json")
                             .read_text(encoding="utf-8"))
    audit_path = tmp_path / "reports" / "phase6_fp_audit.json"
    audit_report = json.loads(audit_path.read_text(encoding="utf-8")) \
        if audit_path.exists() else None
    return rc, main_report, audit_report


def test_phase6_fp_audit_aggregates_bands_per_stratum(tmp_path, monkeypatch,
                                                      stub_ultralytics):
    mod = _load_p6(monkeypatch, tmp_path)
    # one clip: exact TP + a gt_sampling FP; another: an unexplained FP
    rc, data, audit = _run_p6_audit(tmp_path, monkeypatch, mod, {
        "vidA": ([(1, 1, 0, 0, 100, 100), (7, 1, 200, 0, 300, 100)],
                 [(1, 1, 0, 0, 100, 100), (7, 2, 0, 0, 100, 100)]),
        "vidB": ([(1, 1, 0, 0, 100, 100)],
                 [(1, 1, 0, 0, 100, 100), (1, 2, 500, 500, 560, 560)]),
    })
    assert rc == 0
    s = data["fp_audit_summary"]
    assert s["n_videos_audited"] == 2
    assert s["n_fp"] == 2
    assert s["bands"] == {"localisation": 0, "gt_sampling": 1, "unexplained": 1}
    assert s["band_fractions"]["gt_sampling"] == 0.5
    assert s["window_annotated_samples"] == 1
    assert set(s["band_definitions"]) == {"localisation", "gt_sampling",
                                          "unexplained"}
    # per-video totals must add up to the overall totals
    assert sum(v["n_fp"] for v in s["per_video"].values()) == s["n_fp"]
    # the bulky records live in their own report
    assert set(audit["records"]) == {"vidA.mov", "vidB.mov"}
    assert audit["records"]["vidA.mov"][0]["band"] == "gt_sampling"
    checks = {c["name"]: c for c in data["integrity_checks"]}
    assert checks["fp_audit_computed"]["status"] == "PASS"
    # fp_rows must never leak into the main report's per-video metrics
    assert all("fp_rows" not in v["mot_metrics"] for v in data["videos"])


def test_phase6_fp_audit_absent_when_flag_not_given(tmp_path, monkeypatch,
                                                    stub_ultralytics):
    mod = _load_p6(monkeypatch, tmp_path)
    rc, data, audit = _run_p6_audit(tmp_path, monkeypatch, mod, {
        "vidA": ([(1, 1, 0, 0, 100, 100)], [(1, 1, 0, 0, 100, 100)]),
    }, pass_flag=False)
    # the stub always attaches an audit; without the flag nothing is reported
    assert "fp_audit_summary" not in data
    assert audit is None
    checks = {c["name"]: c for c in data["integrity_checks"]}
    assert checks["fp_audit_computed"]["status"] == "PASS"   # not required
