"""
Virtual-line vehicle counting for Phase 7.

Pure, video-IO-free logic. Reads the MOT-format files Phase 6 already wrote
(`predictions/tracking/<video>.txt`), so counting needs no GPU, no re-tracking,
and is reproducible from stored artifacts (skill §19: independently testable).

Counting rules (skill §13):
  - a vehicle is counted on a CROSSING EVENT, never on per-frame presence
  - each track ID is counted at most once (duplicate-count prevention)
  - counting is direction-aware and per-class
  - counts are derived from persistent track IDs, not raw detections

Direction semantics (skill §14): BDD100K dashcam frames are not calibrated, so
NO geographic direction (northbound/southbound, inbound/outbound) is fabricated.
Directions are screen-relative and named for what they physically are:
  - 'farbound' : centroid crossed the line moving UP the image (away from camera)
  - 'nearbound': centroid crossed the line moving DOWN the image (toward camera)
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A track must exist this many frames before it can be counted, so a single
# noisy detection next to the line cannot inject a count. No hysteresis band:
# see _side_of() for why a band is actively harmful on a down-sampled signal.
MIN_TRACK_FRAMES = 3


# ============================================================
# MOT FILE I/O
# ============================================================

def parse_mot_file(path: Any) -> List[Tuple[int, int, float, float, float, float, float]]:
    """Parse a MOT16 file -> [(frame, id, x1, y1, x2, y2, conf)].

    Accepts the 10-column layout written by `mot_lines()`
    (frame,id,x,y,w,h,conf,-1,-1,-1) and tolerates the common 6/7-column
    variants. Non-positive w/h are dropped rather than producing an
    inside-out box that would corrupt the centroid.
    """
    rows: List[Tuple[int, int, float, float, float, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split(",")
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x, y = float(parts[2]), float(parts[3])
                w, h = float(parts[4]), float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            rows.append((frame, tid, x, y, x + w, y + h, conf))
    return rows


def iter_mot_dir(directory: Any, max_videos: int = 0) -> List[Tuple[str, Any]]:
    """List (video_stem, path) for MOT files in `directory`, sorted, capped."""
    files = sorted(f for f in os.listdir(directory)
                   if f.lower().endswith(".txt") and not f.startswith("."))
    if max_videos > 0:
        files = files[:max_videos]
    return [(os.path.splitext(f)[0], os.path.join(directory, f)) for f in files]


# ============================================================
# CROSSING DETECTION
# ============================================================

def _side_of(cy: float, line_y: float) -> int:
    """-1 above the line, +1 below it (or 0 within EPS of the line).

    A dead band is deliberately absent. A band looks safer but is actively
    harmful here: a vehicle whose annotated frames are ~6 frames apart moves
    far more than the band width per sample, so the side can flip between
    consecutive samples and manufacture a crossing that never happened. At
    30 fps the true displacement per frame is small, so comparing consecutive
    samples directly is both simpler and more correct.
    """
    if cy < line_y:
        return -1
    if cy > line_y:
        return 1
    return 0


def _class_of(track: Dict[str, Any]) -> str:
    return str(track.get("cls", "unknown"))


def detect_crossings(
    tracks: Dict[int, Dict[str, Any]],
    line_y: float,
    min_frames: int = MIN_TRACK_FRAMES,
    min_net_disp: float = 0.0,
    require_opposite_ends: bool = True,
) -> List[Dict[str, Any]]:
    """Count line crossings from per-track centroid trajectories.

    `tracks` maps track_id -> {"cls": str, "points": [(frame, cx, cy), ...]}.
    Points may be in any order; they are sorted by frame internally. A crossing
    is recorded when the side of the centroid flips between consecutive
    annotated samples, and each track contributes at most one crossing.

    require_opposite_ends (default True) counts a track only when its FIRST and
    LAST centroids lie on OPPOSITE sides of the line. Most tracks in dashcam
    footage HOVER near a line rather than traversing it, and a loitering track
    produces side flips, so a line placed inside the hover band maximises the
    count - which made the total swing ~40x across line positions. Requiring
    opposite ends is the standard remedy and, unlike a net-displacement
    threshold, is exactly SYMMETRIC: it cannot favour farbound over nearbound.
    A measured net-displacement filter was tried and rejected - at 5% of frame
    height it removed 93% of farbound crossings but only 56% of nearbound,
    because receding vehicles produce short tracks. It is kept only for
    experimentation, off by default.

    `min_net_disp` (pixels, 0 = off) additionally requires net vertical travel.
    Direction-biased; see above.

    Returns one record per crossing:
        {track_id, cls, direction, frame, x, y, line_y, n_frames, net_disp}
    """
    events: List[Dict[str, Any]] = []
    for tid, track in tracks.items():
        pts = sorted(track.get("points", []), key=lambda p: p[0])
        if len(pts) < min_frames:
            continue
        net = abs(pts[-1][2] - pts[0][2])
        if min_net_disp > 0.0 and net < min_net_disp:
            continue
        first_side = _side_of(pts[0][2], line_y)
        last_side = _side_of(pts[-1][2], line_y)
        if require_opposite_ends:
            # Must genuinely pass through: entered on one side, left on the
            # other. Both endpoints must also be OFF the line itself.
            if first_side == 0 or last_side == 0 or first_side == last_side:
                continue
        prev_side = first_side
        for frame, cx, cy in pts[1:]:
            side = _side_of(cy, line_y)
            if side != 0 and prev_side != 0 and side != prev_side:
                direction = "farbound" if side < prev_side else "nearbound"
                events.append({
                    "track_id": int(tid),
                    "cls": _class_of(track),
                    "direction": direction,
                    "frame": int(frame),
                    "x": round(float(cx), 1),
                    "y": round(float(cy), 1),
                    "line_y": round(float(line_y), 1),
                    "n_frames": len(pts),
                    "net_disp": round(net, 1),
                })
                break                      # one count per track
            if side != 0:
                prev_side = side
    events.sort(key=lambda e: e["frame"])
    return events


def tracks_from_mot_rows(
    rows: Sequence[Tuple[int, int, float, float, float, float, float]],
    class_of: Optional[Dict[int, str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Group MOT rows into per-track centroid series.

    `class_of` optionally supplies track_id -> class name (e.g. read from the
    Phase 6 report); otherwise the class is recorded as 'unknown'.
    """
    acc: Dict[int, Dict[str, Any]] = {}
    for frame, tid, x1, y1, x2, y2, _conf in rows:
        t = acc.get(tid)
        if t is None:
            t = {"cls": (class_of or {}).get(tid, "unknown"), "points": []}
            acc[tid] = t
        t["points"].append((frame, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return acc


def tracks_from_box_rows(
    rows: Sequence[Tuple],
    class_of: Optional[Dict[int, str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Group (frame, tid, x1, y1, x2, y2[, label]) rows into centroid series.

    Unlike `tracks_from_mot_rows` this takes 6-tuples, so the SAME crossing
    rule can be applied to ground-truth boxes as to predictions - which is the
    only way counting accuracy can be measured rather than asserted.
    """
    acc: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        frame, tid, x1, y1, x2, y2 = r[0], r[1], r[2], r[3], r[4], r[5]
        label = r[6] if len(r) > 6 else None
        t = acc.get(tid)
        if t is None:
            t = {"cls": (class_of or {}).get(tid) or label or "unknown",
                 "points": []}
            acc[tid] = t
        t["points"].append((int(frame), (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return acc


def accuracy_metrics(
    per_video: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Counting accuracy of predictions against ground truth (skill §15).

    Reports MAE, RMSE, MAPE and counting accuracy. MAPE is undefined where the
    GT count is zero, so those videos are EXCLUDED and the exclusion count is
    reported - silently including them would produce an infinite or meaningless
    percentage.

    counting_accuracy = 1 - MAE / mean(gt), i.e. 1.0 is perfect and 0.0 means
    the error equals the typical true count. It is clipped to [0, 1] for
    reporting because over-counting by more than 100% is worse than useless but
    not meaningfully "negative accuracy"; the raw signed figure is kept too.
    """
    pairs = [(v["gt_counted"], v["total_counted"]) for v in per_video
             if v.get("gt_counted") is not None]
    if not pairs:
        return {"n_videos_compared": 0,
                "note": "no ground-truth counts available"}
    errs = [p - g for g, p in pairs]
    abs_errs = [abs(e) for e in errs]
    gt_total = sum(g for g, _ in pairs)
    mae = sum(abs_errs) / len(abs_errs)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    nonzero = [(abs(p - g) / g, g, p) for g, p in pairs if g > 0]
    mape = (sum(m for m, _, _ in nonzero) / len(nonzero) * 100.0) if nonzero \
        else None
    mean_gt = gt_total / len(pairs)
    signed = 1.0 - mae / mean_gt if mean_gt > 0 else None
    return {
        "n_videos_compared": len(pairs),
        "n_videos_excluded_from_mape": len(pairs) - len(nonzero),
        "gt_total": gt_total,
        "pred_total": sum(p for _, p in pairs),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE_pct": round(mape, 2) if mape is not None else None,
        "counting_accuracy": round(max(0.0, min(1.0, signed)), 4) if signed is not None else None,
        "counting_accuracy_signed": round(signed, 4) if signed is not None else None,
    }


# ============================================================
# TRANSPORTATION METRICS (skill §15)
# ============================================================

def summarize(
    events: Sequence[Dict[str, Any]],
    n_tracks: int,
    n_frames: int,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Aggregate crossing events into traffic-flow statistics.

    `n_frames`/`fps` define the observation window. Rates are reported over the
    WHOLE window, not just the span between the first and last crossing, so a
    quiet period still counts against the rate instead of inflating it.
    """
    by_class: Dict[str, int] = {}
    by_direction: Dict[str, int] = {}
    for e in events:
        by_class[e["cls"]] = by_class.get(e["cls"], 0) + 1
        by_direction[e["direction"]] = by_direction.get(e["direction"], 0) + 1

    total = len(events)
    minutes = (n_frames / fps) / 60.0 if fps > 0 and n_frames > 0 else 0.0
    rates: Dict[str, Any] = {"observation_minutes": round(minutes, 3)}
    if minutes > 0:
        rates["vehicles_per_minute"] = round(total / minutes, 2)
        rates["vehicles_per_5min"] = round(total / minutes * 5.0, 2)
        rates["vehicles_per_15min"] = round(total / minutes * 15.0, 2)
    else:
        rates["vehicles_per_minute"] = None
        rates["vehicles_per_5min"] = None
        rates["vehicles_per_15min"] = None

    composition = {k: round(v / total, 4) for k, v in sorted(by_class.items())} \
        if total else {}

    return {
        "total_counted": total,
        "n_tracks_seen": n_tracks,
        "n_tracks_counted": len({e["track_id"] for e in events}),
        "count_rate": round(len({e["track_id"] for e in events}) / n_tracks, 4)
        if n_tracks else 0.0,
        "by_class": dict(sorted(by_class.items())),
        "by_direction": dict(sorted(by_direction.items())),
        "class_composition": composition,
        "directions": list(by_direction.keys()),
        "rates": rates,
    }


def aggregate_videos(per_video: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum per-video counts. Rates are recomputed from the summed window so
    the fleet total is not an average of averages."""
    total = sum(v["total_counted"] for v in per_video)
    tracks = sum(v["n_tracks_seen"] for v in per_video)
    by_class: Dict[str, int] = {}
    by_direction: Dict[str, int] = {}
    minutes = 0.0
    for v in per_video:
        for k, n in v["by_class"].items():
            by_class[k] = by_class.get(k, 0) + n
        for k, n in v["by_direction"].items():
            by_direction[k] = by_direction.get(k, 0) + n
        minutes += v["rates"]["observation_minutes"] or 0.0
    return {
        "total_counted": total,
        "n_tracks_seen": tracks,
        "n_tracks_counted": sum(v["n_tracks_counted"] for v in per_video),
        "by_class": dict(sorted(by_class.items())),
        "by_direction": dict(sorted(by_direction.items())),
        "class_composition": {k: round(n / total, 4) for k, n in sorted(by_class.items())}
        if total else {},
        "rates": {
            "observation_minutes": round(minutes, 3),
            "vehicles_per_minute": round(total / minutes, 2) if minutes > 0 else None,
            "vehicles_per_5min": round(total / minutes * 5.0, 2) if minutes > 0 else None,
            "vehicles_per_15min": round(total / minutes * 15.0, 2) if minutes > 0 else None,
        },
    }
