"""
Tests for scene-metadata propagation and stratified reporting (src/strata.py).

Stratifying by day/night is the project's headline experiment, and its input
is a 28 MB manifest keyed by BDD100K video id. Three things can go wrong and
all three produce plausible-looking numbers rather than errors:

  1. the key normalisation misses, so every video lands in 'unlabelled' and the
     day/night table silently compares nothing;
  2. a metadata value is invented or inferred instead of read from the dataset
     (the hand-written scene inventory disagreed with BDD100K on 3 of 5 clips);
  3. a stratum is reported without the number of videos behind it, so a
     1-clip bucket reads like a population result.
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

from src.strata import (
    METADATA_FIELDS, MIN_VIDEOS_PER_STRATUM, UNLABELLED, attach_metadata,
    coverage, load_video_metadata, resolve_manifest_paths, stratum_of,
    stratified_counting, stratified_mot_metrics, video_key,
)

DAYTIME = "daytime"
NIGHT = "night"
DAWN = "dawn/dusk"
HIGHWAY = "highway"
CLEAR = "clear"


# ------------------------------------------------------------
# video_key: the join between artefacts and the manifest
# ------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("0000f77c-6257be58.mov", "0000f77c-6257be58"),
    ("0000f77c-6257be58.txt", "0000f77c-6257be58"),
    ("0000f77c-6257be58.jpg", "0000f77c-6257be58"),
    # box-track GT frame images carry a zero-padded frame index
    ("0000f77c-6257be58-0000001.jpg", "0000f77c-6257be58"),
    ("0000f77c-6257be58-0000042.jpg", "0000f77c-6257be58"),
    # Kaggle nests videos several levels deep
    ("/kaggle/input/d/bdd100k_videos_train_00/bdd100k/videos/train/"
     "0000f77c-6257be58.mov", "0000f77c-6257be58"),
    ("C:\\data\\videos\\0000f77c-6257be58.mp4", "0000f77c-6257be58"),
    ("0000F77C-6257BE58.MOV", "0000f77c-6257be58"),
    ("  0000f77c-6257be58.mov  ", "0000f77c-6257be58"),
])
def test_video_key_normalises_every_artefact_name(name, expected):
    assert video_key(name) == expected


def test_video_key_never_truncates_a_numeric_hex_half():
    """A BDD100K id is two hex groups; the second can be all digits. Stripping
    a 'trailing frame index' must therefore require the EXTRA group, or real
    ids would collide with each other."""
    a = video_key("0000f77c-62575880.jpg")
    b = video_key("0000f77c-62575880-0000001.jpg")
    assert a == "0000f77c-62575880"
    assert b == "0000f77c-62575880"


def test_video_key_of_empty_is_empty():
    assert video_key("") == ""
    assert video_key(None) == ""


# ------------------------------------------------------------
# manifest loading
# ------------------------------------------------------------

def _frame(name, tod=DAYTIME, scene=HIGHWAY, weather=CLEAR):
    return {"frame_name": name, "timeofday": tod, "weather": weather,
            "scene": scene, "num_objects": 3}


def _write_manifest(path, frames, wrap=False):
    path.write_text(json.dumps({"frames": frames} if wrap else frames),
                    encoding="utf-8")
    return path


def test_load_metadata_from_bare_list_and_wrapped_form(tmp_path):
    a = _write_manifest(tmp_path / "manifest_train.json",
                        [_frame("vidA.jpg", tod=DAYTIME)])
    b = _write_manifest(tmp_path / "manifest_val.json",
                        [_frame("vidB.jpg", tod=NIGHT, scene="city street")],
                        wrap=True)
    meta = load_video_metadata([a, b])
    assert set(meta) == {"vida", "vidb"}
    assert meta["vida"]["timeofday"] == DAYTIME
    assert meta["vidb"]["timeofday"] == NIGHT
    assert meta["vidb"]["scene"] == "city street"
    assert meta["vida"]["n_images"] == 1
    assert meta["vida"]["conflicts"] == {}


def test_load_metadata_uses_manifest_video_id_not_the_frame_name(tmp_path):
    """The frame name is '<video-id>-<index>.jpg' in the box-track dump; a
    lookup keyed on the raw name would never match the video."""
    p = _write_manifest(tmp_path / "m.json",
                        [_frame("0000f77c-6257be58-0000001.jpg", tod=NIGHT)])
    assert set(load_video_metadata([p])) == {"0000f77c-6257be58"}


def test_load_metadata_takes_majority_and_preserves_conflicts(tmp_path):
    """A contradiction must be visible in the output, not silently resolved."""
    p = _write_manifest(tmp_path / "m.json", [
        _frame("0000f77c-6257be58.jpg", tod=DAYTIME),
        _frame("0000f77c-6257be58-0000002.jpg", tod=DAYTIME),
        _frame("0000f77c-6257be58-0000003.jpg", tod=NIGHT),
    ])
    meta = load_video_metadata([p])["0000f77c-6257be58"]
    assert meta["timeofday"] == DAYTIME           # majority wins
    assert meta["conflicts"]["timeofday"] == sorted([DAYTIME, NIGHT])
    assert meta["n_images"] == 3


def test_load_metadata_normalises_missing_values(tmp_path):
    p = _write_manifest(tmp_path / "m.json", [
        {"frame_name": "vidA.jpg", "timeofday": "undefined", "weather": None},
    ])
    meta = load_video_metadata([p])["vida"]
    assert meta["timeofday"] == "undefined"
    assert meta["weather"] == "undefined"
    # 'undefined' is not a stratum: stratum_of must fold it into 'unlabelled'
    assert stratum_of(meta, "timeofday") == UNLABELLED
    assert stratum_of(meta, "scene") == UNLABELLED   # field absent entirely


def test_load_metadata_tolerates_missing_and_broken_files(tmp_path):
    good = _write_manifest(tmp_path / "m.json", [_frame("vidA.jpg")])
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    assert set(load_video_metadata([good, bad, tmp_path / "nope.json"])) == {"vida"}
    assert load_video_metadata([]) == {}
    assert load_video_metadata(None) == {}


def test_load_metadata_reads_alternative_name_fields(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps([
        {"name": "vidA.jpg", "timeofday": NIGHT},
        {"source_image": "/x/y/vidB.jpg", "timeofday": DAYTIME},
    ]), encoding="utf-8")
    meta = load_video_metadata([p])
    assert meta["vida"]["timeofday"] == NIGHT
    assert meta["vidb"]["timeofday"] == DAYTIME


# ------------------------------------------------------------
# manifest path resolution
# ------------------------------------------------------------

def test_resolve_manifest_paths_auto_discovers_val_then_train(tmp_path):
    _write_manifest(tmp_path / "manifest_train.json", [_frame("a.jpg")])
    _write_manifest(tmp_path / "manifest_val.json", [_frame("b.jpg")])
    got = [p.name for p in resolve_manifest_paths(None, tmp_path)]
    assert got == ["manifest_val.json", "manifest_train.json"]


def test_resolve_manifest_paths_explicit_arg_wins_and_splits(tmp_path):
    a = _write_manifest(tmp_path / "a.json", [_frame("a.jpg")])
    b = _write_manifest(tmp_path / "b.json", [_frame("b.jpg")])
    got = resolve_manifest_paths(f"{a}, {b}", tmp_path)
    assert got == [a, b]


def test_resolve_manifest_paths_empty_when_nothing_there(tmp_path):
    assert resolve_manifest_paths(None, tmp_path) == []
    assert resolve_manifest_paths(None, None) == []


# ------------------------------------------------------------
# labelling records
# ------------------------------------------------------------

def test_attach_metadata_labels_and_counts(tmp_path):
    p = _write_manifest(tmp_path / "m.json", [
        _frame("vidA.mov", tod=DAYTIME), _frame("vidB.mov", tod=NIGHT)])
    meta = load_video_metadata([p])
    recs = [{"video": "vidA.mov"}, {"video": "vidB.mov"}, {"video": "vidC.mov"}]
    got = attach_metadata(recs, meta)
    assert got == {"labelled": 2, "unlabelled": 1}
    assert recs[0]["timeofday"] == DAYTIME
    assert recs[0]["video_id"] == "vida"
    assert recs[0]["metadata_source"] == "m.json"
    assert recs[2]["timeofday"] == UNLABELLED
    assert recs[2]["metadata_source"] is None
    for f in METADATA_FIELDS:
        assert f in recs[2], "unlabelled records must still carry every field"


def test_attach_metadata_records_conflicts_on_the_record(tmp_path):
    p = _write_manifest(tmp_path / "m.json", [
        _frame("0000f77c-6257be58.jpg", tod=DAYTIME),
        _frame("0000f77c-6257be58-0000002.jpg", tod=NIGHT)])
    recs = [{"video": "0000f77c-6257be58.mov"}]
    attach_metadata(recs, load_video_metadata([p]))
    assert recs[0]["metadata_conflicts"]["timeofday"] == sorted([DAYTIME, NIGHT])


def test_attach_metadata_handles_empty_input():
    assert attach_metadata([], {}) == {"labelled": 0, "unlabelled": 0}
    assert attach_metadata(None, {}) == {"labelled": 0, "unlabelled": 0}


# ------------------------------------------------------------
# stratified tracking metrics
# ------------------------------------------------------------

def _mot_rec(video, tp, fp, fn, idsw, n_gt, mota=None, motp=0.8, idf1=0.7):
    if mota is None:
        mota = 1 - (fn + fp + idsw) / n_gt if n_gt else None
    return {
        "video": video,
        "mot_metrics": {
            "TP": tp, "FP": fp, "FN": fn, "IDSW": idsw, "n_gt": n_gt,
            "MOTA": mota,
            "MOTP": motp, "IDF1": idf1, "gt_frame_coverage": 0.1667,
        },
    }


def test_stratified_mot_pools_counts_then_computes_mota():
    meta = {"vida": {"timeofday": DAYTIME}, "vidb": {"timeofday": NIGHT}}
    recs = [_mot_rec("vidA.mov", 80, 10, 10, 5, 100),
            _mot_rec("vidB.mov", 40, 10, 40, 5, 90)]
    out = stratified_mot_metrics(recs, meta, "timeofday")
    # day: 1 - (10 + 10 + 5)/100 = 0.75 ; night: 1 - (40 + 10 + 5)/90 = 0.3889
    assert out[DAYTIME]["MOTA_pooled"] == 0.75
    assert out[NIGHT]["MOTA_pooled"] == pytest.approx(0.3889, abs=1e-3)
    assert out[DAYTIME]["MOTA_mean"] == 0.75
    assert out[DAYTIME]["precision"] == pytest.approx(80 / 90, abs=1e-3)
    assert out[DAYTIME]["recall"] == pytest.approx(0.8, abs=1e-3)
    assert out[DAYTIME]["n_videos"] == 1 and out[DAYTIME]["n_videos_evaluated"] == 1
    assert out[DAYTIME]["low_sample"] is True
    assert out[DAYTIME]["videos"] == ["vida"]


def test_stratified_mot_counts_idsw_and_excludes_unscored_videos():
    meta = {"vida": {"timeofday": DAYTIME}, "vidb": {"timeofday": DAYTIME}}
    recs = [_mot_rec("vidA.mov", 80, 10, 10, 5, 100),
            _mot_rec("vidB.mov", 80, 10, 10, 7, 100),
            {"video": "vidC.mov"}]           # GT unavailable, and unlabelled
    out = stratified_mot_metrics(recs, meta, "timeofday")
    assert out[DAYTIME]["IDSW"] == 12
    assert out[DAYTIME]["n_gt"] == 200
    assert out[DAYTIME]["n_videos"] == 2
    assert out[DAYTIME]["n_videos_evaluated"] == 2
    # the unscored video is neither scored nor counted as evaluated
    assert out[UNLABELLED]["n_videos"] == 1
    assert out[UNLABELLED]["n_videos_evaluated"] == 0
    assert out["_excluded_without_metrics"] == 1


def test_stratified_mot_reports_a_stratum_with_no_metrics():
    meta = {"vida": {"timeofday": NIGHT}}
    out = stratified_mot_metrics([{"video": "vidA.mov"}], meta, "timeofday")
    assert out[NIGHT]["n_videos_evaluated"] == 0
    assert "no GT metrics" in out[NIGHT]["note"]


def test_stratified_mot_handles_zero_gt():
    meta = {"vida": {"timeofday": NIGHT}}
    out = stratified_mot_metrics([_mot_rec("vidA.mov", 0, 0, 0, 0, 0)], meta)
    assert out[NIGHT]["MOTA_pooled"] is None
    assert out[NIGHT]["recall"] is None


def test_stratified_mot_is_not_low_sample_at_three_videos():
    meta = {f"v{i}": {"timeofday": NIGHT} for i in range(3)}
    recs = [_mot_rec(f"v{i}.mov", 10, 1, 1, 0, 12) for i in range(3)]
    out = stratified_mot_metrics(recs, meta, "timeofday")
    assert out[NIGHT]["n_videos_evaluated"] == MIN_VIDEOS_PER_STRATUM
    assert out[NIGHT]["low_sample"] is False


def test_unlabelled_videos_form_their_own_bucket_not_a_missing_one():
    meta = {"vida": {"timeofday": DAYTIME}}
    recs = [_mot_rec("vidA.mov", 80, 10, 10, 5, 100),
            _mot_rec("vidZ.mov", 80, 10, 10, 5, 100)]
    out = stratified_mot_metrics(recs, meta, "timeofday")
    assert set(out) == {DAYTIME, UNLABELLED, "_excluded_without_metrics"}
    assert out[UNLABELLED]["videos"] == ["vidz"]


# ------------------------------------------------------------
# stratified counting
# ------------------------------------------------------------

def _count_rec(video, counted, minutes=0.667, gt=None, cls="car",
               direction="nearbound", tracks_seen=10):
    return {
        "video": video, "total_counted": counted, "n_tracks_seen": tracks_seen,
        "n_tracks_counted": counted, "by_class": {cls: counted},
        "by_direction": {direction: counted},
        "rates": {"observation_minutes": minutes,
                  "vehicles_per_minute": (counted / minutes) if minutes else None},
        "gt_counted": gt,
    }


def test_stratified_counting_sums_counts_and_uses_the_summed_window():
    meta = {"vida": {"timeofday": DAYTIME}, "vidb": {"timeofday": DAYTIME}}
    recs = [_count_rec("vidA.mov", 10, 0.5, gt=9), _count_rec("vidB.mov", 6, 0.5, gt=6)]
    out = stratified_counting(recs, meta, "timeofday")
    day = out[DAYTIME]
    assert day["n_videos"] == 2 and day["total_counted"] == 16
    assert day["observation_minutes"] == 1.0
    # rate comes from the summed window, not the mean of the two rates
    assert day["vehicles_per_minute"] == 16.0
    assert day["accuracy"]["MAE"] == pytest.approx(0.5, abs=1e-3)
    assert day["accuracy"]["gt_total"] == 15
    assert day["low_sample"] is True


def test_stratified_counting_merges_composition_across_videos():
    meta = {"vida": {"timeofday": NIGHT}, "vidb": {"timeofday": NIGHT}}
    recs = [_count_rec("vidA.mov", 4, cls="car", direction="farbound"),
            _count_rec("vidB.mov", 6, cls="truck", direction="nearbound")]
    night = stratified_counting(recs, meta, "timeofday")[NIGHT]
    assert night["by_class"] == {"car": 4, "truck": 6}
    assert night["by_direction"] == {"farbound": 4, "nearbound": 6}
    assert night["n_tracks_seen"] == 20


def test_stratified_counting_excludes_zero_gt_from_mape_like_the_overall():
    """Same rule as accuracy_metrics: a clip with no GT crossings has an
    undefined percentage error and must not be averaged in."""
    meta = {"vida": {"timeofday": NIGHT}, "vidb": {"timeofday": NIGHT}}
    recs = [_count_rec("vidA.mov", 4, gt=4), _count_rec("vidB.mov", 9, gt=0)]
    acc = stratified_counting(recs, meta, "timeofday")[NIGHT]["accuracy"]
    assert acc["n_videos_compared"] == 2
    assert acc["n_videos_excluded_from_mape"] == 1
    assert acc["MAPE_pct"] == 0.0


def test_stratified_counting_without_any_gt():
    meta = {"vida": {"timeofday": NIGHT}}
    out = stratified_counting([_count_rec("vidA.mov", 3)], meta, "timeofday")
    assert out[NIGHT]["total_counted"] == 3
    assert out[NIGHT]["accuracy"]["n_videos_compared"] == 0
    assert "note" in out[NIGHT]["accuracy"]


def test_stratified_counting_zero_minutes_gives_no_rate():
    meta = {"vida": {"timeofday": NIGHT}}
    out = stratified_counting([_count_rec("vidA.mov", 0, minutes=0.0)],
                              meta, "timeofday")
    assert out[NIGHT]["vehicles_per_minute"] is None
    assert out[NIGHT]["vehicles_per_5min"] is None


def test_stratified_counting_by_scene_uses_the_scene_field():
    meta = {"vida": {"timeofday": NIGHT, "scene": HIGHWAY},
            "vidb": {"timeofday": NIGHT, "scene": "city street"}}
    recs = [_count_rec("vidA.mov", 5), _count_rec("vidB.mov", 1)]
    out = stratified_counting(recs, meta, "scene")
    assert out[HIGHWAY]["total_counted"] == 5
    assert out["city street"]["total_counted"] == 1


# ------------------------------------------------------------
# coverage provenance
# ------------------------------------------------------------

def test_coverage_reports_videos_per_stratum():
    meta = {"vida": {"timeofday": DAYTIME}, "vidb": {"timeofday": DAYTIME},
            "vidc": {"timeofday": NIGHT}}
    recs = [{"video": "vidA.mov"}, {"video": "vidB.mov"},
            {"video": "vidC.mov"}, {"video": "vidQ.mov"}]
    cov = coverage(recs, "timeofday", meta)
    assert cov["n_videos"] == 4
    assert cov["n_labelled"] == 3
    assert cov["n_videos_per_stratum"] == {DAYTIME: 2, NIGHT: 1, UNLABELLED: 1}
    assert cov["videos_per_stratum"][DAYTIME] == ["vida", "vidb"]


# ------------------------------------------------------------
# end-to-end through the Phase 7 CLI
# ------------------------------------------------------------

def _load_p7(monkeypatch, tmp_path):
    import src.config as config
    spec = importlib.util.spec_from_file_location(
        "p7_strata", _PROJECT_DIR / "scripts" / "07_count.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(config, "OUTPUT", type(config.OUTPUT)(project_root=tmp_path))
    monkeypatch.setattr(mod, "OUTPUT", config.OUTPUT)
    return mod


def _mot(path, y0, dy, n_tracks=4, n_frames=400):
    rows = []
    for tid in range(1, n_tracks + 1):
        for f in range(1, n_frames + 1):
            cy = y0 + dy * f
            rows.append(f"{f},{tid},200.00,{cy - 20:.2f},80.00,40.00,0.9,-1,-1,-1")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_phase7_stratifies_with_the_real_manifest_keys(tmp_path, monkeypatch):
    """Two synthetic clips named after real BDD100K ids, labelled from a
    manifest by id - the exact join Phase 7 performs on Kaggle."""
    mod = _load_p7(monkeypatch, tmp_path)
    mot = tmp_path / "predictions" / "tracking"
    mot.mkdir(parents=True)
    rep = tmp_path / "reports"
    rep.mkdir()
    _mot(mot / "0000f77c-6257be58.txt", 300, 0.9)
    _mot(mot / "0001542f-5ce3cf52.txt", 600, 0.9)
    _write_manifest(rep / "manifest_train.json", [
        _frame("0000f77c-6257be58.jpg", tod=DAYTIME, scene=HIGHWAY),
        _frame("0001542f-5ce3cf52.jpg", tod=NIGHT, scene="city street"),
    ])

    argv = sys.argv
    sys.argv = ["07_count.py", "--mot-dir", str(mot), "--track-report",
                str(rep / "nope.json"), "--line-y", "0.5", "--fps", "30",
                "--no-plots"]
    try:
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 0

    data = json.loads((rep / "phase7_counting.json").read_text(encoding="utf-8"))
    by = {v["video"]: v for v in data["videos"]}
    assert by["0000f77c-6257be58"]["timeofday"] == DAYTIME
    assert by["0001542f-5ce3cf52"]["timeofday"] == NIGHT

    assert "stratified" in data, "no stratified block in the report"
    tod = data["stratified"]["timeofday"]
    assert set(tod) == {DAYTIME, NIGHT}
    assert tod[DAYTIME]["n_videos"] == 1 and tod[NIGHT]["n_videos"] == 1
    assert tod[DAYTIME]["low_sample"] is True
    assert data["stratified_coverage"]["timeofday"]["n_labelled"] == 2
    assert set(data["stratified"]["scene"]) == {HIGHWAY, "city street"}

    checks = {c["name"]: c for c in data["integrity_checks"]}
    assert checks["videos_labelled_by_metadata"]["status"] == "PASS"
    assert checks["stratified_results_present"]["status"] == "PASS"


def test_phase7_reports_unlabelled_videos_instead_of_dropping_them(
        tmp_path, monkeypatch):
    mod = _load_p7(monkeypatch, tmp_path)
    mot = tmp_path / "predictions" / "tracking"
    mot.mkdir(parents=True)
    rep = tmp_path / "reports"
    rep.mkdir()
    _mot(mot / "0000f77c-6257be58.txt", 300, 0.9)
    _mot(mot / "unknown-clip.txt", 600, 0.9)
    _write_manifest(rep / "manifest_train.json",
                    [_frame("0000f77c-6257be58.jpg", tod=DAYTIME)])

    argv = sys.argv
    sys.argv = ["07_count.py", "--mot-dir", str(mot), "--track-report",
                str(rep / "nope.json"), "--line-y", "0.5", "--fps", "30",
                "--no-plots"]
    try:
        rc = mod.main()
    finally:
        sys.argv = argv

    data = json.loads((rep / "phase7_counting.json").read_text(encoding="utf-8"))
    # the unknown clip is still counted in the totals ...
    assert len(data["videos"]) == 2
    # ... and it is visible as its own bucket, not silently in the daytime one
    assert UNLABELLED in data["stratified"]["timeofday"]
    assert data["stratified"]["timeofday"][UNLABELLED]["n_videos"] == 1
    assert data["stratified"]["timeofday"][DAYTIME]["n_videos"] == 1
    checks = {c["name"]: c for c in data["integrity_checks"]}
    assert checks["videos_labelled_by_metadata"]["status"] == "FAIL"
    assert "unlabelled" in checks["videos_labelled_by_metadata"]["detail"]
    # a warning-severity failure must not fail the stage
    assert rc == 0
    assert data["_meta"]["all_checks_passed"] is True


def test_phase7_without_a_manifest_says_stratification_is_impossible(
        tmp_path, monkeypatch):
    mod = _load_p7(monkeypatch, tmp_path)
    mot = tmp_path / "predictions" / "tracking"
    mot.mkdir(parents=True)
    _mot(mot / "vidA.txt", 300, 0.9)

    argv = sys.argv
    sys.argv = ["07_count.py", "--mot-dir", str(mot), "--track-report",
                str(tmp_path / "nope.json"), "--line-y", "0.5", "--fps", "30",
                "--no-plots"]
    try:
        rc = mod.main()
    finally:
        sys.argv = argv

    data = json.loads((tmp_path / "reports" / "phase7_counting.json")
                      .read_text(encoding="utf-8"))
    assert "stratified" not in data
    checks = {c["name"]: c for c in data["integrity_checks"]}
    assert checks["stratified_results_present"]["status"] == "FAIL"
    assert "NOT stratified" in checks["stratified_results_present"]["detail"]
    assert rc == 0
