"""
Tests for false-positive forensics (src/track.py).

False positives explode at night (94 in the daytime clip vs ~540 per clip at
night/dawn-dusk) while false negatives stay flat. That pattern has two
incompatible explanations - a detector that hallucinates in the dark, or a
correct detector being scored against a ~6x downsampled GT dump that never
sampled the frame it fired on - and they call for opposite fixes. These tests
pin the classifier that separates them, because getting it wrong would produce
a confident, wrong conclusion in the writeup.

Two ways it could silently go wrong, both guarded here:
  - re-deriving the FP set independently of the matcher, which disagrees with
    the reported FP count because greedy one-to-one matching leaves
    high-IoU-but-unmatched predictions as FPs;
  - measuring the neighbourhood in VIDEO FRAMES instead of ANNOTATED SAMPLES,
    which on a 6x-downsampled dump can never reach the neighbouring sample and
    therefore reports every FP as `unexplained`.
"""

import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.track import (
    FP_BANDS, clear_mot_metrics, false_positive_audit,
)

# A 6x-downsampled GT, like the real dump: annotated frames 6 apart.
GT_SPARSE = [
    (1, 1, 0, 0, 100, 100),
    (7, 1, 200, 0, 300, 100),
    (13, 1, 400, 0, 500, 100),
    (19, 1, 600, 0, 700, 100),
]


def _audit(gt, pred, window=1, **kw):
    """window belongs to the AUDIT, not the matcher."""
    m = clear_mot_metrics(gt, pred, collect_fp=True, **kw)
    return m, false_positive_audit(gt, m["fp_rows"], window=window)


# ------------------------------------------------------------
# the FP set must be the matcher's, not a re-derivation
# ------------------------------------------------------------

def test_collect_fp_rows_length_matches_reported_fp():
    gt = [(f, 1, 0, 0, 100, 100) for f in (1, 7, 13)]
    pred = [(1, 1, 0, 0, 100, 100), (7, 2, 500, 500, 550, 550),
            (13, 3, 600, 600, 650, 650), (13, 4, 700, 700, 750, 750)]
    m = clear_mot_metrics(gt, pred, collect_fp=True)
    assert len(m["fp_rows"]) == m["FP"] == 3
    assert "fp_rows" not in clear_mot_metrics(gt, pred)      # opt-in only


def test_greedy_competition_leaves_a_high_iou_prediction_as_fp():
    """Two predictions overlap one GT box; only the better one matches, and the
    loser is an FP DESPITE clearing the IoU threshold. Re-deriving FPs by
    thresholding IoU would wrongly count only 1 here."""
    gt = [(1, 1, 0, 0, 100, 100)]
    pred = [(1, 10, 0, 0, 100, 100),        # perfect match, wins
            (1, 11, 2, 0, 102, 100)]        # iou 0.96, loses -> FP
    m = clear_mot_metrics(gt, pred, collect_fp=True)
    assert m["TP"] == 1 and m["FP"] == 1
    assert len(m["fp_rows"]) == 1
    assert m["fp_rows"][0][1] == 11
    # and the audit must see that same single FP
    audit = false_positive_audit(gt, m["fp_rows"])
    assert audit["n_fp"] == 1


def test_pred_on_unannotated_frame_is_not_scored_at_all():
    """restrict_to_gt_frames drops predictions on frames the GT never sampled,
    so they are neither TP nor FP. This is the documented MOTChallenge
    convention and the reason FP is not ~7.2k."""
    gt = [(1, 1, 0, 0, 100, 100)]
    pred = [(1, 1, 0, 0, 100, 100)] + [(f, 9, 0, 0, 100, 100)
                                       for f in range(2, 30)]
    m = clear_mot_metrics(gt, pred, collect_fp=True)
    assert m["FP"] == 0 and m["n_pred_scored"] == 1 and m["n_pred_total"] == 29


# ------------------------------------------------------------
# the three bands
# ------------------------------------------------------------

def test_localisation_band_when_same_frame_overlap_is_subthreshold():
    gt = [(1, 1, 0, 0, 100, 100)]
    # 40..90 inside 0..100 -> IoU 0.25, below the 0.5 threshold
    pred = [(1, 1, 0, 0, 100, 100), (1, 2, 40, 40, 90, 90)]
    _m, audit = _audit(gt, pred)
    assert audit["bands"]["localisation"] == 1
    assert audit["bands"]["gt_sampling"] == 0
    assert audit["bands"]["unexplained"] == 0
    rec = audit["records"][0]
    assert 0.0 < rec["best_iou_same_frame"] < 0.5


def test_gt_sampling_band_when_a_neighbouring_sample_has_the_object():
    """The detector is right on frame 7 but the GT box there has already moved
    on; the object is annotated at frame 1. That is a GT sampling artifact."""
    gt = GT_SPARSE
    pred = [(1, 1, 0, 0, 100, 100),          # exact TP
            (7, 2, 0, 0, 100, 100),          # aligns with the frame-1 GT box
            (13, 3, 700, 700, 800, 800)]    # aligns with nothing
    _m, audit = _audit(gt, pred)
    assert audit["bands"]["gt_sampling"] == 1
    assert audit["bands"]["unexplained"] == 1
    rec = [r for r in audit["records"] if r["band"] == "gt_sampling"][0]
    assert rec["frame"] == 7
    assert rec["best_iou_same_frame"] == 0.0
    assert rec["best_iou_nearby_frames"] >= 0.5
    assert rec["nearest_gt_id"] == 1


def test_unexplained_band_when_nothing_matches_anywhere():
    gt = GT_SPARSE
    pred = [(13, 5, 0, 500, 60, 560)]
    _m, audit = _audit(gt, pred)
    assert audit["bands"] == {"localisation": 0, "gt_sampling": 0,
                              "unexplained": 1}


def test_window_counts_annotated_samples_not_video_frames():
    """Regression: with a raw-frame window of 1 on a 6x-downsampled dump, the
    nearest annotated neighbour is 6 frames away and unreachable, so EVERY FP
    would be reported `unexplained` and the conclusion would invert."""
    gt = GT_SPARSE                      # annotated at 1, 7, 13, 19
    pred = [(7, 2, 0, 0, 100, 100)]     # matches the frame-1 GT box
    _m, w0 = _audit(gt, pred, window=0)
    _m, w1 = _audit(gt, pred, window=1)
    assert w0["bands"]["gt_sampling"] == 0
    assert w1["bands"]["gt_sampling"] == 1
    assert w1["nearby_window_frames"] == 1


def test_window_two_reaches_two_samples_away():
    gt = [(1, 1, 0, 0, 100, 100), (7, 1, 200, 0, 300, 100),
          (13, 1, 400, 0, 500, 100), (19, 1, 0, 400, 100, 500)]
    pred = [(7, 2, 0, 400, 100, 500)]    # only the frame-19 box overlaps
    _m, w1 = _audit(gt, pred, window=1)
    _m, w2 = _audit(gt, pred, window=2)
    assert w1["bands"]["unexplained"] == 1
    assert w2["bands"]["gt_sampling"] == 1


# ------------------------------------------------------------
# bookkeeping
# ------------------------------------------------------------

def test_band_fractions_sum_to_one():
    gt = GT_SPARSE
    pred = [(7, 2, 0, 0, 100, 100), (13, 3, 700, 700, 800, 800),
            (13, 4, 395, 0, 455, 60)]
    _m, audit = _audit(gt, pred)
    assert sum(audit["bands"].values()) == audit["n_fp"] == 3
    # fractions are rounded per band, so they sum to 1 only to ~1e-3
    assert abs(sum(v for v in audit["band_fractions"].values() if v) - 1.0) < 1e-3
    assert set(audit["band_fractions"]) == set(FP_BANDS)


def test_area_bands_use_documented_pixel_thresholds():
    gt = [(1, 1, 0, 0, 10, 10)]
    pred = [(1, 1, 0, 0, 10, 10),
            (1, 2, 0, 0, 20, 20),        # 400 px  -> tiny  (<32x32)
            (1, 3, 0, 0, 60, 60),        # 3600    -> small (<96x96)
            (1, 4, 0, 0, 200, 200),      # 40000   -> medium(<256x256)
            (1, 5, 0, 0, 400, 400)]      # 160000  -> large
    _m, audit = _audit(gt, pred)
    assert audit["area_bands_px"] == {"tiny": 1, "small": 1, "medium": 1,
                                     "large": 1}
    assert sum(audit["area_bands_px"].values()) == audit["n_fp"] == 4


def test_audit_of_zero_fp_is_well_formed():
    gt = [(1, 1, 0, 0, 100, 100)]
    pred = [(1, 1, 0, 0, 100, 100)]
    _m, audit = _audit(gt, pred)
    assert audit["n_fp"] == 0
    assert audit["bands"] == {b: 0 for b in FP_BANDS}
    assert all(v is None for v in audit["band_fractions"].values())
    assert audit["records"] == []


def test_every_band_is_documented():
    _m, audit = _audit(GT_SPARSE, [(13, 5, 0, 500, 60, 560)])
    for b in FP_BANDS:
        assert b in audit["band_definitions"]
        assert audit["band_definitions"][b].strip()
    assert "ANNOTATED" in audit["band_definitions"]["gt_sampling"] or \
        "annotated sample" in audit["band_definitions"]["gt_sampling"]


def test_audit_accepts_gt_rows_carrying_a_label():
    """GT rows may be 7-tuples with a category label; the audit must not care."""
    gt = [(1, 1, 0, 0, 100, 100, "car"), (7, 1, 0, 0, 100, 100, "car")]
    # exact same-frame match -> a TP, so there is nothing to classify
    m, audit = _audit(gt, [(7, 2, 0, 0, 100, 100)])
    assert m["FP"] == 0 and audit["n_fp"] == 0
    # and an FP is still classified correctly with labelled GT rows
    _m, audit = _audit(gt, [(7, 2, 500, 500, 560, 560)])
    assert audit["bands"]["unexplained"] == 1
