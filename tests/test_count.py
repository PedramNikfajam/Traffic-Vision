"""
Tests for Phase 7 virtual-line counting (src/count.py).

Pure logic only - no ultralytics, no video decode, no dataset. MOT files are
written to tmp_path by the test itself.
"""

import sys
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TEST_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.count import (
    parse_mot_file, iter_mot_dir, tracks_from_mot_rows, detect_crossings,
    summarize, aggregate_videos, tracks_from_box_rows, accuracy_metrics,
)


# ------------------------------------------------------------
# MOT parsing
# ------------------------------------------------------------

def _write_mot(path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_parse_mot_file_full_layout(tmp_path):
    p = _write_mot(tmp_path / "a.txt", [
        "1,1,10.00,20.00,50.00,40.00,0.9000,-1,-1,-1",
        "2,1,12.00,20.00,50.00,40.00,0.8000,-1,-1,-1",
    ])
    rows = parse_mot_file(p)
    assert len(rows) == 2
    assert rows[0] == (1, 1, 10.0, 20.0, 60.0, 60.0, 0.9)
    assert rows[1][2] == 12.0


def test_parse_mot_file_tolerates_short_and_garbage(tmp_path):
    p = _write_mot(tmp_path / "b.txt", [
        "# a comment",
        "",
        "1,1,10,20,50,40",              # 6 columns, no conf
        "not,a,mot,row",                 # unparseable
        "2,1,10,20",                     # too few columns
        "3,1,10,20,0,40,0.5",            # zero width -> dropped
    ])
    rows = parse_mot_file(p)
    assert len(rows) == 1
    assert rows[0][0] == 1 and rows[0][6] == 1.0


def test_parse_mot_file_empty(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("", encoding="utf-8")
    assert parse_mot_file(p) == []


def test_iter_mot_dir(tmp_path):
    _write_mot(tmp_path / "v1.txt", ["1,1,0,0,10,10,0.9"])
    _write_mot(tmp_path / "v2.txt", ["1,1,0,0,10,10,0.9"])
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    got = iter_mot_dir(str(tmp_path))
    assert [stem for stem, _ in got] == ["v1", "v2"]
    assert [s for s, _ in iter_mot_dir(str(tmp_path), 1)] == ["v1"]


# ------------------------------------------------------------
# Crossing detection
# ------------------------------------------------------------

def _track(cls, pts):
    return {"cls": cls, "points": pts}


def test_crossing_downward_is_nearbound():
    """Centroid below -> above? no: moving DOWN the image = toward camera."""
    t = {1: _track("car", [(f, 100, 100 + 10 * f) for f in range(1, 11)])}
    ev = detect_crossings(t, line_y=120.0)
    assert len(ev) == 1
    assert ev[0]["direction"] == "nearbound"
    assert ev[0]["track_id"] == 1 and ev[0]["cls"] == "car"
    assert ev[0]["n_frames"] == 10


def test_crossing_upward_is_farbound():
    # y decreases 480 -> 200, so the centroid moves UP the image through y=300
    t = {1: _track("car", [(f, 100, 500 - 15 * f) for f in range(1, 21)])}
    ev = detect_crossings(t, line_y=300.0)
    assert len(ev) == 1
    assert ev[0]["direction"] == "farbound"


def test_counted_at_most_once_per_track():
    """A track that oscillates across the line must still count once."""
    pts = []
    y = 100
    for f in range(1, 21):
        pts.append((f, 100, y))
        y = 300 if y < 200 else 100          # cross repeatedly
    t = {1: _track("car", pts)}
    assert len(detect_crossings(t, line_y=200.0)) == 1


def test_no_crossing_when_stays_on_one_side():
    t = {1: _track("car", [(f, 100, 50 + f) for f in range(1, 11)])}
    assert detect_crossings(t, line_y=400.0) == []


def test_short_tracks_not_counted():
    """One noisy detection beside the line must not inject a count."""
    t = {1: _track("car", [(1, 100, 100), (2, 100, 300)])}
    assert detect_crossings(t, line_y=200.0, min_frames=3) == []
    assert len(detect_crossings(t, line_y=200.0, min_frames=2)) == 1


def test_min_frames_default_is_three():
    t = {1: _track("car", [(1, 100, 100), (2, 100, 300), (3, 100, 500)])}
    assert len(detect_crossings(t, line_y=200.0)) == 1


def test_out_of_order_points_are_sorted():
    t = {1: _track("car", [(3, 100, 500), (1, 100, 100), (2, 100, 300)])}
    ev = detect_crossings(t, line_y=200.0)
    assert len(ev) == 1 and ev[0]["direction"] == "nearbound"


def test_multiple_tracks_and_classes():
    tracks = {
        # y 120 -> 300: crosses 200 downward = nearbound
        1: _track("car", [(f, 100, 100 + 20 * f) for f in range(1, 11)]),
        # y 460 -> 100: crosses 200 upward = farbound
        2: _track("truck", [(f, 300, 500 - 40 * f) for f in range(1, 11)]),
        # y 51 -> 60: never reaches the line
        3: _track("car", [(f, 500, 50 + f) for f in range(1, 11)]),
    }
    ev = detect_crossings(tracks, line_y=200.0)
    assert len(ev) == 2
    assert {e["cls"] for e in ev} == {"car", "truck"}
    assert {e["direction"] for e in ev} == {"nearbound", "farbound"}
    assert ev[0]["frame"] <= ev[1]["frame"]        # sorted by frame


def test_no_hysteresis_band_is_correct():
    """A vehicle sampled ~6 frames apart moves much further than any sane
    band, so a band would let consecutive samples disagree about the side and
    manufacture crossings. Comparing consecutive samples directly is correct.
    A 20-frame gap straddling the line still yields exactly one crossing."""
    pts = [(1, 100, 50), (21, 100, 350)]
    assert len(detect_crossings({1: _track("car", pts)}, line_y=200.0,
                                min_frames=2)) == 1


# ------------------------------------------------------------
# net-displacement filter: separates vehicles that CROSSED from
# vehicles that LOITERED near the line
# ------------------------------------------------------------

def test_min_net_disp_rejects_loitering_track():
    """The production problem: tracks hover near the line, so a line placed
    inside the hover band maximises side flips and therefore the count."""
    # wobbles across y=200 repeatedly but ends up where it started
    pts = []
    y = 190
    for f in range(1, 61):
        pts.append((f, 100, y))
        y = 215 if y < 200 else 190
    loiter = {1: _track("car", pts)}
    assert len(detect_crossings(loiter, 200.0)) == 1          # counted if unfiltered
    assert detect_crossings(loiter, 200.0, min_net_disp=40.0) == []

    # a vehicle that genuinely drives through: large net travel
    trav = {2: _track("car", [(f, 100, 400 - 6 * f) for f in range(1, 61)])}
    ev = detect_crossings(trav, 200.0, min_net_disp=40.0)
    assert len(ev) == 1 and ev[0]["track_id"] == 2


def test_min_net_disp_is_direction_agnostic():
    up = {1: _track("car", [(f, 100, 100 + 8 * f) for f in range(1, 41)])}
    down = {2: _track("car", [(f, 100, 500 - 8 * f) for f in range(1, 41)])}
    assert len(detect_crossings(up, 300.0, min_net_disp=50.0)) == 1
    assert len(detect_crossings(down, 300.0, min_net_disp=50.0)) == 1


def test_min_net_disp_records_value():
    # y runs 110 (f=1) .. 200 (f=10); line at 150 is straddled between f=5
    # (exactly on it, side 0) and f=6 (y=160)
    t = {1: _track("car", [(f, 100, 100 + 10 * f) for f in range(1, 11)])}
    ev = detect_crossings(t, 150.0, min_net_disp=0.0,
                          require_opposite_ends=False)
    assert ev[0]["net_disp"] == 90.0
    assert ev[0]["y"] == 160.0            # centroid y at the crossing sample
    assert ev[0]["frame"] == 6
    assert ev[0]["direction"] == "nearbound"


# ------------------------------------------------------------
# opposite-ends rule: the DEFAULT, and the direction-neutral one
# ------------------------------------------------------------

def test_opposite_ends_rejects_loiterer_but_keeps_both_directions():
    """A loiterer crosses the line but starts and ends on the SAME side, so it
    is not a traversal. Real vehicles in either direction must still count -
    this is what makes the rule safe, unlike a net-displacement threshold."""
    # period 40 divides 60 exactly, so the track RETURNS to its starting side
    loiter_pts = [(f, 100, 190 if (f % 40) <= 20 else 215) for f in range(1, 61)]
    loiter = {1: _track("car", loiter_pts)}
    assert len(detect_crossings(loiter, 200.0, require_opposite_ends=False)) == 1
    assert detect_crossings(loiter, 200.0, require_opposite_ends=True) == []

    down = {2: _track("car", [(f, 100, 100 + 8 * f) for f in range(1, 41)])}
    up = {3: _track("car", [(f, 100, 700 - 15 * f) for f in range(1, 41)])}
    for tracks, direction in ((down, "nearbound"), (up, "farbound")):
        ev = detect_crossings(tracks, 400.0, require_opposite_ends=True)
        assert len(ev) == 1, f"{direction} traversal was rejected"
        assert ev[0]["direction"] == direction


def test_opposite_ends_is_symmetric_in_direction():
    """Direction neutrality is the whole point: mirror-image trajectories must
    be accepted equally."""
    down = [(f, 100, 100 + 5 * f) for f in range(1, 121)]        # 105 -> 700
    up = [(f, 100, 700 - 5 * f) for f in range(1, 121)]          # 695 -> 100
    for pts, direction in ((down, "nearbound"), (up, "farbound")):
        ev = detect_crossings({1: _track("car", pts)}, 400.0,
                              require_opposite_ends=True)
        assert len(ev) == 1
        assert ev[0]["direction"] == direction


def test_opposite_ends_rejects_track_starting_on_the_line():
    """Endpoints exactly on the line are ambiguous -> reject, do not guess."""
    pts = [(f, 100, 200 if f == 1 else 200 + 5 * (f - 1)) for f in range(1, 41)]
    assert detect_crossings({1: _track("car", pts)}, 200.0,
                            require_opposite_ends=True) == []


# ------------------------------------------------------------
# tracks_from_mot_rows
# ------------------------------------------------------------

def test_tracks_from_mot_rows_centroids_and_classes():
    rows = [
        (1, 7, 10.0, 20.0, 60.0, 60.0, 0.9),
        (2, 7, 12.0, 20.0, 62.0, 60.0, 0.8),
        (1, 8, 100.0, 100.0, 200.0, 160.0, 0.7),
    ]
    t = tracks_from_mot_rows(rows, {7: "car"})
    assert set(t) == {7, 8}
    assert t[7]["cls"] == "car" and t[8]["cls"] == "unknown"
    assert t[7]["points"][0] == (1, 35.0, 40.0)      # centroid of 10,60/20,60
    assert t[8]["points"][0] == (1, 150.0, 130.0)


# ------------------------------------------------------------
# Transportation metrics
# ------------------------------------------------------------

def test_summarize_counts_and_rates():
    events = [
        {"track_id": 1, "cls": "car", "direction": "nearbound", "frame": 1},
        {"track_id": 2, "cls": "car", "direction": "farbound", "frame": 2},
        {"track_id": 3, "cls": "truck", "direction": "nearbound", "frame": 3},
    ]
    s = summarize(events, n_tracks=5, n_frames=1800, fps=30.0)   # 60 s = 1 min
    assert s["total_counted"] == 3
    assert s["n_tracks_seen"] == 5
    assert s["n_tracks_counted"] == 3
    assert s["by_class"] == {"car": 2, "truck": 1}
    assert s["by_direction"] == {"farbound": 1, "nearbound": 2}
    assert s["class_composition"]["car"] == 0.6667
    assert s["count_rate"] == 0.6
    assert s["rates"]["observation_minutes"] == 1.0
    assert s["rates"]["vehicles_per_minute"] == 3.0
    assert s["rates"]["vehicles_per_5min"] == 15.0
    assert s["rates"]["vehicles_per_15min"] == 45.0


def test_summarize_rate_uses_whole_window_not_crossing_span():
    """A single crossing in a long video must not read as 1 veh/min."""
    events = [{"track_id": 1, "cls": "car", "direction": "nearbound",
               "frame": 100}]
    s = summarize(events, n_tracks=1, n_frames=36000, fps=30.0)  # 20 min
    assert s["rates"]["vehicles_per_minute"] == 0.05


def test_summarize_empty_is_safe():
    s = summarize([], n_tracks=0, n_frames=0, fps=30.0)
    assert s["total_counted"] == 0
    assert s["by_class"] == {} and s["class_composition"] == {}
    assert s["rates"]["vehicles_per_minute"] is None


def test_aggregate_videos_recomputes_rate_from_total_window():
    v1 = summarize([{"track_id": 1, "cls": "car", "direction": "nearbound",
                     "frame": 1}], 3, 1800, 30.0)      # 1 min, 1 count
    v2 = summarize([{"track_id": 2, "cls": "bus", "direction": "farbound",
                     "frame": 1}], 4, 5400, 30.0)      # 3 min, 1 count
    agg = aggregate_videos([v1, v2])
    assert agg["total_counted"] == 2
    assert agg["by_class"] == {"bus": 1, "car": 1}
    assert agg["by_direction"] == {"farbound": 1, "nearbound": 1}
    assert agg["rates"]["observation_minutes"] == 4.0
    # 2 counts / 4 min = 0.5/min, NOT the average of 1.0 and 0.333
    assert agg["rates"]["vehicles_per_minute"] == 0.5


def test_aggregate_distinct_ties():
    """Two videos, identical counts, must not be conflated."""
    ev = [{"track_id": 1, "cls": "car", "direction": "nearbound", "frame": 1}]
    v1 = summarize(ev, 2, 1800, 30.0)
    v2 = summarize(ev, 2, 1800, 30.0)
    agg = aggregate_videos([v1, v2])
    assert agg["total_counted"] == 2
    assert agg["n_tracks_seen"] == 4
    assert agg["n_tracks_counted"] == 2      # same track id in both videos


# ------------------------------------------------------------
# counting ACCURACY vs ground truth (skill §15)
# ------------------------------------------------------------

def test_tracks_from_box_rows_accepts_six_tuples():
    """GT rows are 6-tuples, MOT rows are 7 - the same crossing rule must work
    on both, otherwise accuracy compares two different things."""
    gt = [(1, 5, 10.0, 20.0, 60.0, 60.0), (2, 5, 12.0, 20.0, 62.0, 60.0)]
    t = tracks_from_box_rows(gt)
    assert set(t) == {5}
    assert t[5]["points"][0] == (1, 35.0, 40.0)
    assert t[5]["cls"] == "unknown"
    # a label in a 7th slot is picked up as the class
    t2 = tracks_from_box_rows([(1, 5, 0.0, 0.0, 10.0, 10.0, "truck")])
    assert t2[5]["cls"] == "truck"


def test_accuracy_metrics_known_values():
    vids = [{"gt_counted": 5, "total_counted": 3},
            {"gt_counted": 3, "total_counted": 3}]
    a = accuracy_metrics(vids)
    assert a["n_videos_compared"] == 2
    assert a["gt_total"] == 8 and a["pred_total"] == 6
    assert a["MAE"] == 1.0                              # (|-2| + |0|)/2
    assert a["RMSE"] == pytest.approx(1.4142, abs=1e-3)
    assert a["MAPE_pct"] == pytest.approx(20.0)         # (40% + 0%)/2
    assert a["counting_accuracy"] == pytest.approx(0.75)   # 1 - 1.0/4.0


def test_accuracy_metrics_excludes_zero_gt_from_mape():
    """A video where GT counted zero makes MAPE undefined; it must be excluded
    and the exclusion reported, not silently folded in."""
    vids = [{"gt_counted": 0, "total_counted": 4},
            {"gt_counted": 4, "total_counted": 4}]
    a = accuracy_metrics(vids)
    assert a["n_videos_compared"] == 2
    assert a["n_videos_excluded_from_mape"] == 1
    assert a["MAPE_pct"] == 0.0
    assert a["MAE"] == 2.0
    # mean GT is over ALL videos (4/2 = 2.0), so MAE/mean_gt = 1.0
    assert a["counting_accuracy"] == pytest.approx(0.0)


def test_accuracy_metrics_perfect_and_empty():
    vids = [{"gt_counted": 7, "total_counted": 7},
            {"gt_counted": 0, "total_counted": 0}]
    a = accuracy_metrics(vids)
    assert a["MAE"] == 0.0 and a["RMSE"] == 0.0
    assert a["counting_accuracy"] == 1.0
    assert accuracy_metrics([])["n_videos_compared"] == 0
    assert accuracy_metrics([{"total_counted": 3}])["n_videos_compared"] == 0


def test_accuracy_clipped_when_heavily_overcounting():
    """Over-counting beyond 100% is worse than useless but not 'negative
    accuracy'; the raw signed value is preserved too."""
    a = accuracy_metrics([{"gt_counted": 1, "total_counted": 10}])
    assert a["counting_accuracy"] == 0.0
    assert a["counting_accuracy_signed"] < 0
    assert a["MAE"] == 9.0
