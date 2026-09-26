"""
Visual verification for Phase 7 counting (skill §20: visualization).

Three complementary views, because "58 crossings over 5 videos" is not
something anyone can sanity-check numerically:

  1. render_sample_frames()  - real video frames with the counting line
     drawn, track boxes coloured by whether they were counted, and a running
     per-direction tally in the header. This is the view that answers "is the
     line in a sensible place, and are these the right vehicles?".
  2. plot_crossing_timelines() - centroid y vs frame for every track, with the
     line and the crossing points marked and coloured by direction. This is the
     view that answers "why did this video count zero?" - tracks that never
     approach the line are visible immediately.
  3. plot_line_sweep() - total counts vs line position, so the placement is
     chosen from data instead of guessed.

Frame rendering uses OpenCV (already a project dependency) and only ever
WRITES images - it never opens a window, so it is safe on Kaggle and headless.
Every function degrades to a logged warning rather than failing the stage.
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# BGR colours (OpenCV is BGR, matplotlib is RGB - do not mix them up)
_C_LINE = (40, 40, 235)        # red: the counting line
_C_COUNTED = (90, 200, 90)     # green: track crossed the line
_C_UNCOUNTED = (170, 170, 170)  # grey: track never crossed
_C_FAR = (220, 130, 40)        # blue-ish: farbound crossing marker
_C_NEAR = (60, 160, 240)       # orange-ish: nearbound crossing marker
_C_TEXT = (255, 255, 255)


def _load_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _put(img, text, org, scale=0.5, color=_C_TEXT, thickness=1):
    cv2 = _load_cv2()
    if cv2 is None:
        return
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _dashed_line(img, y, color, thickness=3, dash=18, gap=12):
    cv2 = _load_cv2()
    if cv2 is None:
        return
    h, w = img.shape[:2]
    x = 0
    while x < w:
        cv2.line(img, (x, y), (min(x + dash, w), y), color, thickness)
        x += dash + gap


def _evenly_spaced(total: int, n: int) -> List[int]:
    if total <= 0 or n <= 0:
        return []
    n = min(n, total)
    if n == 1:
        return [total // 2]
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


def _direction_of(by_frame, rows, tid: int, frame: int) -> str:
    """Direction of a track's crossing at `frame`, from centroid motion.

    Recomputed from the MOT rows rather than trusted from a caller, so the
    on-frame tally cannot drift out of step with the detector's own answer.
    """
    pts = sorted((f, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
                 for f, t, x1, y1, x2, y2, _c in rows if t == tid)
    for i, (f, _cx, cy) in enumerate(pts):
        if f >= frame and i > 0:
            return "farbound" if cy < pts[i - 1][2] else "nearbound"
    return "nearbound"


def tracks_by_frame(
    rows: Sequence[Tuple[int, int, float, float, float, float, float]],
) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """frame -> [(track_id, x1, y1, x2, y2)]"""
    out: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    for frame, tid, x1, y1, x2, y2, _c in rows:
        out.setdefault(frame, []).append((tid, x1, y1, x2, y2))
    return out


def render_sample_frames(
    video_path: Any,
    rows: Sequence[Tuple[int, int, float, float, float, float, float]],
    line_y: float,
    counted_ids: Sequence[int],
    out_path: Any,
    n_frames: int = 4,
    tile_width: int = 460,
    class_of: Optional[Dict[int, str]] = None,
    crossing_frames: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """Montage of `n_frames` real frames with the counting line and boxes.

    Green box = that track was counted (it crossed). Grey = it never crossed.
    The header carries the tally of crossings up to THAT frame, derived from
    `crossing_frames` - showing the whole-video total under a "so far" label
    would misrepresent the counter. Pass crossing_frames when available.
    Returns the output path, or None if the video could not be read.
    """
    cv2 = _load_cv2()
    if cv2 is None:
        return None
    if not os.path.exists(str(video_path)):
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    picks = _evenly_spaced(total, n_frames)
    by_frame = tracks_by_frame(rows)
    counted = set(int(i) for i in counted_ids)
    class_of = class_of or {}
    xf = {int(k): int(v) for k, v in (crossing_frames or {}).items()}

    tiles = []
    for pick in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pick)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        _dashed_line(frame, int(round(line_y)), _C_LINE, 3)
        _put(frame, f"counting line y={int(round(line_y))} of {h}",
             (8, 24), 0.55, _C_LINE, 2)

        n_far = n_near = 0
        for tid, cf in xf.items():
            if cf <= pick + 1:
                d = _direction_of(by_frame, rows, tid, cf)
                if d == "farbound":
                    n_far += 1
                else:
                    n_near += 1
        for tid, x1, y1, x2, y2 in by_frame.get(pick + 1, []):
            col = _C_COUNTED if tid in counted else _C_UNCOUNTED
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 3, col, -1)
            name = class_of.get(tid, "")
            _put(frame, f"{tid} {name}".strip(),
                 (int(x1), max(14, int(y1) - 6)), 0.45, col, 1)

        # text kept well clear of the bottom edge so it survives downscaling
        _put(frame, f"frame {pick + 1}/{total}   crossings by here: "
                    f"{n_far + n_near}  (far {n_far} / near {n_near})",
             (8, h - 46), 0.5, _C_TEXT, 1)
        _put(frame, f"green = counted ({len(counted)} total)   "
                    f"grey = never crossed the line",
             (8, h - 22), 0.45, _C_COUNTED, 1)

        if tile_width and w > tile_width:
            scale = tile_width / float(w)
            frame = cv2.resize(frame, (tile_width, int(h * scale)))
        tiles.append(frame)
    cap.release()

    return _write_montage(cv2, tiles, out_path)


def render_box_only_montage(
    rows: Sequence[Tuple[int, int, float, float, float, float, float]],
    line_y: float,
    frame_size: Tuple[float, float],
    counted_ids: Sequence[int],
    out_path: Any,
    n_frames: int = 4,
    tile_width: int = 460,
    class_of: Optional[Dict[int, str]] = None,
    crossing_frames: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """Same view as render_sample_frames but drawn from MOT boxes on a blank
    canvas - NO video file required.

    This exists because the frame montage is the fastest way to judge whether
    counting is right, and it should not be blocked on having the videos
    mounted. Boxes, the line, ids, classes and the running tally are all
    preserved; only the background pixels are missing. Prefer the real-frame
    version when the video is available (`--videos`).
    """
    cv2 = _load_cv2()
    if cv2 is None or not rows:
        return None
    W, H = int(frame_size[0]), int(frame_size[1])
    by_frame = tracks_by_frame(rows)
    counted = set(int(i) for i in counted_ids)
    class_of = class_of or {}
    xf = {int(k): int(v) for k, v in (crossing_frames or {}).items()}
    n_frames_total = max(r[0] for r in rows)
    picks = _evenly_spaced(n_frames_total, n_frames)

    tiles = []
    for pick in picks:
        import numpy as np
        frame = np.full((H, W, 3), 38, np.uint8)
        _dashed_line(frame, int(round(line_y)), _C_LINE, 3)
        _put(frame, f"counting line y={int(round(line_y))} of {H}  "
                    f"[geometry view - no video decoded]",
             (8, 24), 0.55, _C_LINE, 2)
        n_far = n_near = 0
        for tid, cf in xf.items():
            if cf <= pick + 1:
                if _direction_of(by_frame, rows, tid, cf) == "farbound":
                    n_far += 1
                else:
                    n_near += 1
        for tid, x1, y1, x2, y2 in by_frame.get(pick + 1, []):
            col = _C_COUNTED if tid in counted else _C_UNCOUNTED
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 3, col, -1)
            _put(frame, f"{tid} {class_of.get(tid, '')}".strip(),
                 (int(x1), max(14, int(y1) - 6)), 0.45, col, 1)
        _put(frame, f"frame {pick + 1}/{n_frames_total}   crossings by here: "
                    f"{n_far + n_near}  (far {n_far} / near {n_near})",
             (8, H - 46), 0.5, _C_TEXT, 1)
        _put(frame, f"green = counted ({len(counted)} total)   "
                    f"grey = never crossed the line",
             (8, H - 22), 0.45, _C_COUNTED, 1)
        if tile_width and W > tile_width:
            s = tile_width / float(W)
            frame = cv2.resize(frame, (tile_width, int(H * s)))
        tiles.append(frame)

    return _write_montage(cv2, tiles, out_path)


def _write_montage(cv2, tiles, out_path) -> Optional[str]:
    if not tiles:
        return None
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    padded = [cv2.resize(t, (tw, th)) if (t.shape[0] != th or t.shape[1] != tw)
              else t for t in tiles]
    if len(padded) == 1:
        montage = padded[0]
    elif len(padded) == 2:
        montage = cv2.hconcat(padded)
    elif len(padded) == 4:
        montage = cv2.vconcat([cv2.hconcat(padded[:2]),
                               cv2.hconcat(padded[2:4])])
    else:
        montage = cv2.hconcat(padded[:3])
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, montage)
    return out_path


def plot_crossing_timelines(
    tracks: Dict[int, Dict[str, Any]],
    line_y: float,
    frame_height: float,
    out_path: Any,
    events: Optional[Sequence[Dict[str, Any]]] = None,
    max_tracks: int = 60,
    title: str = "",
) -> Optional[str]:
    """Centroid y vs frame for every track, with the line and crossings marked.

    This is the diagnostic view: a track that never approaches the line is a
    track that cannot be counted, and it is obvious here rather than hidden
    inside an aggregate.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not tracks:
        return None

    ev_by_track: Dict[int, Dict[str, Any]] = {}
    for e in events or []:
        ev_by_track.setdefault(int(e["track_id"]), e)

    def cy_at(tid: int, frame: int) -> float:
        pts = sorted(tracks[tid].get("points", []))
        if not pts:
            return line_y
        return min(pts, key=lambda p: abs(p[0] - frame))[2]

    fig, ax = plt.subplots(figsize=(11, 6))
    shown = 0
    for tid, t in sorted(tracks.items()):
        if shown >= max_tracks:
            break
        pts = sorted(t.get("points", []))
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[2] for p in pts]
        counted = tid in ev_by_track
        ax.plot(xs, ys, linewidth=1.1, alpha=0.75,
                color="#2e7d32" if counted else "#9e9e9e",
                label=("counted" if counted else "not counted") if shown == 0
                else None)
        if counted:
            e = ev_by_track[tid]
            ax.plot([e["frame"]], [cy_at(tid, e["frame"])], marker="o",
                    markersize=7,
                    color="#1565c0" if e["direction"] == "farbound" else "#ef6c00",
                    markeredgecolor="black", markeredgewidth=0.6)
        shown += 1

    ax.axhline(line_y, color="#d32f2f", linewidth=2.2, linestyle="--",
               label=f"counting line y={line_y:.0f}")
    if events:
        ax.plot([], [], marker="o", linestyle="none", color="#1565c0",
                label="farbound crossing (up the image)")
        ax.plot([], [], marker="o", linestyle="none", color="#ef6c00",
                label="nearbound crossing (down the image)")
    ax.set_xlabel("video frame")
    ax.set_ylabel("track centroid y (px)")
    # Never clip the data: the assumed frame height can be wrong, and a fixed
    # ylim at that value silently HIDES every track below it. A 440 px
    # assumption on a 720 px frame concealed a third of the image.
    observed = [p[2] for t in tracks.values() for p in t.get("points", [])]
    y_max = max([frame_height] + observed) * 1.04 if observed else frame_height
    ax.set_ylim(y_max, 0)                      # image coords: y grows downward
    if abs(y_max - frame_height) > 1.0:
        ax.axhline(frame_height, color="#455a64", linewidth=1.0, linestyle=":")
        ax.text(0.995, frame_height, f"assumed frame bottom {frame_height:.0f}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7, color="#455a64")
    ax.set_title(title or "Track centroid trajectories vs counting line")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_line_sweep(
    sweep: Dict[str, Dict[str, Any]],
    out_path: Any,
    title: str = "Count sensitivity to counting-line position",
) -> Optional[str]:
    """Total / farbound / nearbound counts for each candidate line position."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not sweep:
        return None

    keys = sorted(sweep, key=lambda k: float(k))
    totals = [sweep[k]["total_counted"] for k in keys]
    far = [sweep[k]["by_direction"].get("farbound", 0) for k in keys]
    near = [sweep[k]["by_direction"].get("nearbound", 0) for k in keys]
    zero = [sweep[k].get("n_videos_zero", 0) for k in keys]
    xs = [float(k) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, totals, marker="o", color="#1565c0", label="total crossings")
    ax.plot(xs, far, marker="^", color="#6a1b9a", label="farbound")
    ax.plot(xs, near, marker="s", color="#ef6c00", label="nearbound")
    ax.set_xlabel("counting line position (fraction of frame height)")
    ax.set_ylabel("crossing events (all videos)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.bar(xs, zero, width=0.012, color="#d32f2f", alpha=0.35)
    ax2.set_ylabel("videos with ZERO crossings", color="#b71c1c")
    ax2.set_ylim(0, max(1, max(zero) + 1))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    fig.tight_layout()
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_count_by_stratum(
    per_video: List[Dict[str, Any]],
    field: str,
    out_path: Any,
    title: str = "Crossing events by BDD100K scene attribute",
) -> Optional[str]:
    """Predicted vs ground-truth crossings per stratum, e.g. by time of day.

    The `n=` in each bar label is the number of 40 s clips behind that bar:
    these strata rest on very few videos, and the plot has to say so rather
    than let a 1-clip bucket look like a population result.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for v in per_video or []:
        groups.setdefault(str(v.get(field) or "unlabelled"), []).append(v)
    if len(groups) < 2:
        return None                      # a single bucket is not a comparison

    labels = sorted(groups)
    xs = list(range(len(labels)))
    pred = [sum(int(v.get("total_counted", 0)) for v in groups[k]) for k in labels]
    with_gt = [k for k in labels
               if any(v.get("gt_counted") is not None for v in groups[k])]
    width = 0.38 if with_gt else 0.6

    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(labels)), 4.8))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}\n(n={len(groups[k])})" for k in labels])
    if with_gt:
        gt = [sum(int(v.get("gt_counted") or 0) for v in groups[k]) for k in with_gt]
        pred_at = [pred[labels.index(k)] for k in with_gt]
        ax.bar([x - width / 2 for x in range(len(with_gt))], gt, width=width,
               label="ground truth", color="#455a64")
        ax.bar([x + width / 2 for x in range(len(with_gt))], pred_at,
               width=width, label="predicted", color="#1565c0")
        for x, p in enumerate(pred_at):
            ax.text(x + width / 2, p, str(p), ha="center", va="bottom")
        for x, g in enumerate(gt):
            ax.text(x - width / 2, g, str(g), ha="center", va="bottom")
        ax.legend()
    else:
        ax.bar(xs, pred, width=width, color="#1565c0", label="predicted")
        for x, p in zip(xs, pred):
            ax.text(x, p, str(p), ha="center", va="bottom")
    ax.set_ylabel("crossing events")
    ax.set_title(f"{title} ({field})\n"
                 "small n per stratum - indicative, not a population estimate")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
