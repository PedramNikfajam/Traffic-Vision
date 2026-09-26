"""
Track accumulation logic for Phase 6 (multi-object tracking).

Pure, video-IO-free logic: aggregates per-frame track detections into
persistent track records and emits MOT-format rows. Kept independent from
ultralytics/cv2 so it is unit-testable without the dataset or a GPU.

A track record maintains (skill §12):
    - track ID, object class (majority vote over frames), confidence
    - bounding-box history / centroid trajectory
    - first frame, last frame, direction (screen-relative)
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


class TrackAccumulator:
    """Aggregate per-frame tracked detections into per-track records.

    Usage (per video):
        acc = TrackAccumulator()
        for frame_idx, boxes in enumerate(stream):
            for tid, cls, conf, xyxy in boxes:
                acc.update(tid, frame_idx, cls, conf, xyxy)
        records = acc.finalize()
    """

    def __init__(self, max_trajectory_points: int = 5000):
        self._tracks: Dict[int, Dict] = {}
        # Per-frame rows in MOT layout (frame_idx 0-based, xyxy): filled in
        # update() order so mot_lines() preserves temporal order.
        self.mot_rows: List[Tuple[int, int, float, float, float, float, float]] = []
        # BDD100K videos are 40 s @ 30 fps = 1200 frames; 5000 pts is a
        # generous ceiling that keeps JSON bounded for long inputs.
        self.max_trajectory_points = max_trajectory_points

    def update(
        self,
        track_id: int,
        frame_idx: int,
        cls_name: str,
        conf: float,
        xyxy: Tuple[float, float, float, float],
    ) -> None:
        t = self._tracks.get(track_id)
        if t is None:
            t = {
                "track_id": int(track_id),
                "cls_votes": Counter(),
                "confs": [],
                "first_frame": frame_idx,
                "last_frame": frame_idx,
                "trajectory": [],
                "last_xyxy": xyxy,
            }
            self._tracks[track_id] = t

        x1, y1, x2, y2 = xyxy
        t["cls_votes"][cls_name] += 1
        t["confs"].append(float(conf))
        t["last_frame"] = max(t["last_frame"], frame_idx)
        t["last_xyxy"] = tuple(xyxy)
        if len(t["trajectory"]) < self.max_trajectory_points:
            t["trajectory"].append(
                (round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1))
            )
        self.mot_rows.append(
            (frame_idx, int(track_id), x1, y1, x2 - x1, y2 - y1, float(conf))
        )

    def __len__(self) -> int:
        return len(self._tracks)

    def finalize(self) -> List[Dict]:
        """Return sorted track records (by track_id)."""
        records = []
        for tid in sorted(self._tracks):
            t = self._tracks[tid]
            traj = t["trajectory"]
            x1, y1, x2, y2 = t["last_xyxy"]
            records.append({
                "track_id": t["track_id"],
                "cls": t["cls_votes"].most_common(1)[0][0],
                "first_frame": t["first_frame"],
                "last_frame": t["last_frame"],
                "n_frames": len(t["confs"]),
                "mean_conf": round(sum(t["confs"]) / len(t["confs"]), 4),
                "bbox_last": [round(v, 1) for v in (x1, y1, x2, y2)],
                "trajectory": traj,
                "displacement": _displacement(traj),
                "direction": _direction(traj),
            })
        return records


def _displacement(trajectory: List[Tuple[float, float]]) -> List[float]:
    """Net centroid displacement [dx, dy] over the trajectory."""
    if len(trajectory) < 2:
        return [0.0, 0.0]
    x0, y0 = trajectory[0]
    x1, y1 = trajectory[-1]
    return [round(x1 - x0, 1), round(y1 - y0, 1)]


def _direction(trajectory: List[Tuple[float, float]]) -> str:
    """Screen-relative direction (dashcam geometry is not calibrated, so no
    geographic labels are fabricated). Dominant-axis displacement."""
    if len(trajectory) < 2:
        return "stationary"
    dx, dy = _displacement(trajectory)
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "rightward" if dx > 0 else "leftward"
    return "downward" if dy > 0 else "upward"


def mot_lines(accumulator: TrackAccumulator) -> List[str]:
    """Convert accumulated per-frame rows to MOT-format lines.

    MOT16 layout: frame, id, x, y, w, h, conf, -1, -1, -1
    (x, y = top-left; frame is 1-indexed).
    """
    lines = []
    for frame, tid, x, y, w, h, conf in accumulator.mot_rows:
        lines.append(
            f"{frame + 1},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"
        )
    return lines


# ============================================================
# GROUND TRUTH (BDD100K box-track parquet)
# ============================================================

# Column aliases seen across BDD100K parquet dumps, resolved case-insensitively.
#
# ORDER IS SIGNIFICANT - these are tuples, not sets. Resolution takes the first
# alias present in the file, so priority must be explicit: 'videoName' outranks
# the 'name' fallback column. This was previously a set, and set iteration order
# for str is randomized per process (PYTHONHASHSEED), so the SAME code resolved
# 'video' to a different column on different runs - one Kaggle run silently
# matched GT and the next found nothing. Never use a set here.
_GT_ALIASES = {
    "video": ("videoname", "video_name", "video", "videoid", "name"),
    "frame": ("frameindex", "frame_index", "frame", "frameidx", "index",
              "video_frame"),
    "id": ("id", "track_id", "instance_id", "obj_id", "object_id", "track"),
    "x1": ("box2d.x1", "x1", "xmin", "left"),
    "y1": ("box2d.y1", "y1", "ymin", "top"),
    "x2": ("box2d.x2", "x2", "xmax", "right"),
    "y2": ("box2d.y2", "y2", "ymax", "bottom"),
    "w": ("w", "width"),
    "h": ("h", "height"),
    "label": ("category", "label", "category_name", "class"),
}

# Optional column flagging rows whose clip was never released as video. Those
# rows carry null box2d.* cells; parsing them as boxes aborts the whole video.
_GT_HAVEVIDEO = ("havevideo", "have_video")

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# BDD100K box-track dumps key every row by the EXTRACTED FRAME IMAGE named
# '<video-stem>-<7-digit frame>.jpg' (observed: 0000f77c-6257be58-0000001.jpg),
# not by the bare video stem. Other dumps use the bare stem or the video
# filename. All three conventions must resolve to the same video.
_GT_FRAME_IMAGE_RE = (
    r".*[/\\]?(?P<stem>.+?)(?:-\d+)?\.(?:jpg|jpeg|png|bmp|webp)$"
)


def video_key_mask(values: Any, video_name: str) -> Any:
    """Boolean mask of rows whose video key belongs to `video_name`.

    Accepts, case-insensitively:
        <stem>-<0000001>.jpg   BDD100K box-track frame-image dump
        <stem>.mov / .mp4      video filename
        <stem>                 bare stem, no extension
    Paths are tolerated ('videos/train/<stem>-0000001.jpg').
    """
    stem = re.escape(video_name)
    with_ext = (
        r".*[/\\]?" + stem + r"(?:-\d+)?\.(?:" +
        "|".join(e[1:] for e in _VIDEO_EXTS) + r"|jpg|jpeg|png|bmp|webp)$"
    )
    s = values.astype(str)
    return s.str.contains(with_ext, case=False, regex=True, na=False) | \
        s.str.endswith(video_name)


def _cell_float(v: Any) -> Any:
    """Coerce a parquet cell to a finite float.

    Dicts pass through unchanged (scalabel-style nested box, resolved by the
    caller). Null / non-numeric / NaN / inf become None so the caller skips the
    row instead of raising `float() argument must be ... not 'NoneType'`.
    """
    if v is None or isinstance(v, dict):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _truthy(v: Any) -> bool:
    """Interpret a parquet cell as a boolean, tolerating str/bool/NA."""
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y", "t")
    try:
        return bool(v)
    except (TypeError, ValueError):
        return False


def _resolve_columns(columns: List[str]) -> Dict[str, str]:
    """Map canonical names (video/frame/id/x1/...) to actual parquet columns.

    Canonical -> None means the column is absent (optional ones only).
    """
    lower = {c.lower(): c for c in columns}
    resolved: Dict[str, str] = {}
    for canon, aliases in _GT_ALIASES.items():
        resolved[canon] = None
        for a in aliases:
            if a in lower and lower[a] not in resolved.values():
                resolved[canon] = lower[a]
                break
    for needed in ("frame", "id"):
        if resolved.get(needed) is None:
            raise ValueError(
                f"GT parquet missing required column '{needed}' (have: {columns})"
            )
    have_box = (resolved.get("x1") and resolved.get("y1")
                and (resolved.get("x2") or resolved.get("w"))
                and (resolved.get("y2") or resolved.get("h")))
    if not have_box:
        raise ValueError(
            f"GT parquet missing box columns (have: {columns})"
        )
    return resolved


def load_gt_parquet(
    paths: List[Any],
    video_name: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> List[Tuple]:
    """Load BDD100K box-track GT -> [(frame_1idx, id, x1, y1, x2, y2, label)].

    Handles scalar corner coordinates, w/h pairs, and scalabel-style nested
    dicts. Rows that cannot yield a valid box are SKIPPED rather than fatal:
    the released parquet contains entries for clips that were never released
    as video (haveVideo=False) whose box2d.* cells are null, and a single such
    row previously aborted the entire video's evaluation with
    `float() argument must be a string or a real number, not 'NoneType'`.

    If `stats` is a dict it is filled with load diagnostics - rows read, rows
    kept, rows skipped and why, distinct annotated frames, and the annotation
    stride (GCD of frame gaps). BDD100K box-track GT is annotated on a sparse
    keyframe subset, so the stride/coverage numbers are what make an MOT score
    interpretable; they are surfaced in the report instead of being implicit.
    """
    import pandas as pd

    st: Dict[str, Any] = {} if stats is None else stats
    st.update({
        "rows_read": 0, "rows_no_video": 0, "rows_null_box": 0,
        "rows_bad_box": 0, "rows_kept": 0,
    })

    rows: List[Tuple] = []
    for path in paths:
        df = pd.read_parquet(path)
        lower_cols = {str(c).lower(): str(c) for c in df.columns}
        cols = _resolve_columns([str(c) for c in df.columns])

        if video_name is not None and cols.get("video"):
            v = df[cols["video"]].astype(str)
            mask = video_key_mask(v, video_name)
            # 'name' column often holds the video stem even when 'videoName'
            # is a shard/folder name - try it as a fallback
            if not mask.any() and "name" in lower_cols and \
                    lower_cols["name"] != cols["video"]:
                mask = video_key_mask(df[lower_cols["name"]].astype(str),
                                      video_name)
            if not mask.any():
                raise KeyError(
                    f"no GT rows for video '{video_name}' in '{path}' "
                    f"(videoName values look like: {sorted(set(v))[:3]} of "
                    f"{len(v)} rows)"
                )
            df = df[mask]

        # haveVideo=False rows describe clips that were never released as
        # video; their boxes are null. Drop them before parsing.
        hv_col = next((lower_cols[c] for c in _GT_HAVEVIDEO if c in lower_cols), None)
        if hv_col is not None:
            before = len(df)
            df = df[df[hv_col].map(_truthy)]
            st["rows_no_video"] += before - len(df)
            if len(df) == 0:
                raise KeyError(
                    f"all {before} GT rows for '{video_name}' have "
                    f"{hv_col}=False (clip not released as video)"
                )

        st["rows_read"] += len(df)
        c_x1, c_y1 = cols.get("x1"), cols.get("y1")
        c_x2, c_y2 = cols.get("x2"), cols.get("y2")
        c_w, c_h = cols.get("w"), cols.get("h")
        c_frame, c_id, c_label = cols["frame"], cols["id"], cols.get("label")

        for row in df.itertuples(index=False):
            d = dict(zip(df.columns, row))

            x1 = _cell_float(d.get(c_x1))
            y1 = _cell_float(d.get(c_y1))
            if isinstance(x1, dict):
                # scalabel-style nested box: {'x1':..,'y1':..,'x2':..,'y2':..}
                b = x1
                x1 = _cell_float(b.get("x1"))
                y1 = _cell_float(b.get("y1"))
                x2 = _cell_float(b.get("x2"))
                y2 = _cell_float(b.get("y2"))
            elif c_x2 and c_y2:
                x2 = _cell_float(d.get(c_x2))
                y2 = _cell_float(d.get(c_y2))
            else:
                w = _cell_float(d.get(c_w))
                h = _cell_float(d.get(c_h))
                x2 = x1 + w if (x1 is not None and w is not None) else None
                y2 = y1 + h if (y1 is not None and h is not None) else None

            if any(v is None for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                st["rows_null_box" if any(v is None for v in (x1, y1, x2, y2))
                   else "rows_bad_box"] += 1
                continue

            f_raw = _cell_float(d.get(c_frame))
            id_raw = _cell_float(d.get(c_id))
            if f_raw is None or id_raw is None:
                st["rows_null_box"] += 1
                continue

            label = d.get(c_label) if c_label else ""
            if isinstance(label, dict):
                label = label.get("name", "")
            rows.append((int(f_raw) + 1, int(id_raw), x1, y1, x2, y2,
                         str(label).strip()))
            st["rows_kept"] += 1

    frames = sorted({r[0] for r in rows})
    st["gt_frames"] = len(frames)
    st["gt_ids"] = len({r[1] for r in rows})
    stride = 0
    for a, b in zip(frames, frames[1:]):
        stride = math.gcd(stride, b - a)
    st["gt_frame_stride"] = stride or 1
    return rows


def describe_gt_parquet(
    paths: List[Any],
    video_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Structural dump of a GT parquet - facts, not guesses.

    Reports: column names + dtypes, the resolved alias mapping, sample values
    per column, value counts for low-cardinality columns, and - for every
    column - how many rows match `video_name` plus the frame density of that
    subset (n_frames, min/max, stride).

    Needed because the BDD100K box-track dump keys rows by extracted frame
    image ('<stem>-<0000001>.jpg') rather than by video stem, and two dumps of
    the same dataset disagree on which column holds the key. Guessing produced
    a silently empty match on one run and a 16x-too-small one on the next.
    Only the first shard is inspected.
    """
    import pandas as pd

    out: Dict[str, Any] = {}
    df = pd.read_parquet(paths[0])
    names = [str(c) for c in df.columns]
    out["shard"] = str(paths[0])
    out["n_rows"] = int(len(df))
    out["columns"] = {str(c): str(df[c].dtype) for c in df.columns}
    try:
        out["resolved"] = _resolve_columns(names)
    except ValueError as e:
        out["resolved"] = None
        out["resolved_error"] = str(e)

    out["samples"] = {}
    for c in df.columns:
        vals = df[str(c)].dropna().astype(str)
        try:
            out["samples"][str(c)] = sorted(set(vals))[:3]
        except TypeError:                      # unhashable (dict cells)
            out["samples"][str(c)] = [str(v)[:60] for v in vals.head(3)]

    out["value_counts"] = {}
    for c in df.columns:
        try:
            vc = df[str(c)].astype(str).value_counts()
        except TypeError:
            continue
        if 1 < len(vc) <= 25:
            out["value_counts"][str(c)] = {k: int(v) for k, v in vc.items()}

    if video_name:
        resolved = out.get("resolved") or {}
        frame_col = resolved.get("frame")
        per: Dict[str, Any] = {}
        for c in df.columns:
            m = video_key_mask(df[str(c)].astype(str), video_name)
            if not m.any():
                continue
            sub = df[m]
            entry: Dict[str, Any] = {"rows": int(len(sub))}
            if frame_col and frame_col in sub.columns:
                fr = sorted({int(v) for v in
                             pd.to_numeric(sub[frame_col], errors="coerce").dropna()})
                entry["n_frames"] = len(fr)
                entry["frame_min"] = fr[0] if fr else None
                entry["frame_max"] = fr[-1] if fr else None
                stride = 0
                for a, b in zip(fr, fr[1:]):
                    stride = math.gcd(stride, b - a)
                entry["frame_stride"] = stride or 1
                entry["boxes_per_frame"] = (round(len(sub) / len(fr), 2)
                                            if fr else None)
            per[str(c)] = entry
        out["per_video"] = per
        out["video_name"] = video_name
    return out


def remap_gt_frames(gt: List[Tuple], stride: int) -> List[Tuple]:
    """Map GT frame indices from the dump's sampled index space into video
    frame space: `(g - 1) * stride + 1` (frames are 1-indexed, so stride=1 is
    the identity).

    The BDD100K box-track dump is a DOWNSAMPLED label set: a 1217-frame video
    ships 203 annotated frames, so `frameIndex` runs 0..202 over the SAMPLED
    sequence, not over the video. Comparing GT frame g against video frame g
    misaligns every frame by up to 5*(n-1) frames and produced 92% FN with
    MOTP 0.67 on coincidental overlaps.
    """
    if stride <= 1:
        return list(gt)
    return [((r[0] - 1) * stride + 1,) + tuple(r[1:]) for r in gt]


# Candidate GT-dump -> video frame strides. 6 is the observed BDD100K ratio
# (~5 Hz labels on 30 fps video); the rest let the data decide rather than
# hard-coding it.
_STRIDE_CANDIDATES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 24, 30)


def calibrate_frame_stride(
    gt: List[Tuple],
    pred: List[Tuple],
    strides: Optional[Tuple[int, ...]] = None,
    iou_thr: float = 0.5,
) -> Tuple[int, Dict[int, int]]:
    """Coarse search: find the integer GT->video frame stride with the most
    GT/prediction agreement.

    Returns `(best_stride, {stride: n_matched_gt_boxes})`. The full curve is
    returned so the choice stays auditable: a decisive peak (one stride far
    above the rest) is evidence of a real sampling ratio, a flat curve means
    the GT is not frame-aligned to the video at all and NO stride should be
    trusted. Integer strides are only an approximation - refine with
    `refine_frame_map`.
    """
    if not gt or not pred:
        return 1, {}
    candidates = strides or _STRIDE_CANDIDATES
    pr_by_frame: Dict[int, List[Tuple]] = {}
    for r in pred:
        pr_by_frame.setdefault(r[0], []).append(r)

    curve: Dict[int, int] = {}
    for s in candidates:
        curve[s] = _count_frame_hits(gt, pr_by_frame,
                                     lambda g, s=s: (g - 1) * s + 1, iou_thr)
    best = max(curve, key=lambda k: (curve[k], -k))
    return best, curve


def remap_gt_frames_linear(gt: List[Tuple], alpha: float,
                           offset: int = 0) -> List[Tuple]:
    """Map GT sampled index -> video frame as `round((g-1)*alpha) + offset + 1`.

    An integer stride only approximates the GT dump's true sampling ratio, and
    the error grows with frame index. 1211 video frames over 187 GT frames is
    6.476, not 6, so by the last annotated frame the two are ~89 frames (3 s)
    apart - that drift is what drove one clip to MOTA -1.04 while the
    near-exact 5.995 clips reached 0.75.
    """
    if alpha == 1 and offset == 0:
        return list(gt)
    a = float(alpha)
    return [(int(round((r[0] - 1) * a)) + offset + 1,) + tuple(r[1:])
            for r in gt]


def _count_frame_hits(gt, pr_by_frame, frame_of, iou_thr) -> int:
    hits = 0
    for r in gt:
        for pr in pr_by_frame.get(frame_of(r[0]), ()):
            if _iou(r[2:6], pr[2:6]) >= iou_thr:
                hits += 1
                break
    return hits


def refine_frame_map(
    gt: List[Tuple],
    pred: List[Tuple],
    strides: Optional[Tuple[int, ...]] = None,
    alpha_span: float = 0.6,
    alpha_step: float = 0.02,
    offsets: Tuple[int, ...] = (-2, -1, 0, 1, 2),
    iou_thr: float = 0.5,
) -> Tuple[float, int, List[Tuple[float, int, int]]]:
    """Search the GT->video frame map `f = round((g-1)*alpha) + offset + 1`
    directly, instead of assuming one.

    Anchoring alpha on the frame counts looked reasonable but is WRONG: the GT
    does not necessarily span the whole video. For 0000f77c-62c2a288 there are
    187 GT frames in a 1211-frame video, so (n_video-1)/(n_gt-1) = 6.505, but
    integer stride 6 matches 296 GT boxes where alpha 6.505 matched only 46 -
    the labels cover roughly the first 1116 frames, not all 1211.

    So alpha is searched on a fine grid centred on the best integer stride, and
    the integer map is always one of the candidates - this can therefore never
    score worse than the integer stride. Returns
    `(best_alpha, best_offset, top_candidates)` where top_candidates is
    `[(alpha, offset, hits), ...]` sorted by hits, for audit.
    """
    if not gt or not pred:
        return 1.0, 0, []
    s0, _curve = calibrate_frame_stride(gt, pred, strides, iou_thr)
    pr_by_frame: Dict[int, List[Tuple]] = {}
    for r in pred:
        pr_by_frame.setdefault(r[0], []).append(r)

    cands: List[Tuple[float, int, int]] = [
        (float(s0), 0, _count_frame_hits(
            gt, pr_by_frame, lambda g, s=s0: (g - 1) * s + 1, iou_thr))
    ]
    steps = int(round(alpha_span / alpha_step))
    for i in range(-steps, steps + 1):
        a = round(s0 + i * alpha_step, 4)
        for off in offsets:
            cands.append((
                a, off,
                _count_frame_hits(
                    gt, pr_by_frame,
                    lambda g, a=a, off=off:
                        int(round((g - 1) * a)) + off + 1,
                    iou_thr),
            ))
    cands.sort(key=lambda c: (-c[2], abs(c[0] - s0), abs(c[1])))
    best_a, best_off, _ = cands[0]
    return best_a, best_off, cands[:5]


def gt_match_profile(
    gt: List[Tuple],
    pred: List[Tuple],
    alpha: float,
    offset: int,
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    """Which GT frames produced a match, and where they sit in the sequence.

    Separates the two failure modes that look identical in MOTA:
      - alignment drifts -> matches cluster at the START of the sequence
      - the clip is simply hard -> matches are uniformly absent
    """
    pr_by_frame: Dict[int, List[Tuple]] = {}
    for r in pred:
        pr_by_frame.setdefault(r[0], []).append(r)
    a = float(alpha)
    matched, total = [], 0
    for r in gt:
        f = int(round((r[0] - 1) * a)) + offset + 1
        if any(_iou(r[2:6], pr[2:6]) >= iou_thr for pr in pr_by_frame.get(f, ())):
            matched.append(r[0])
    for r in gt:
        total += 1
    return {
        "gt_frames": total,
        "frames_with_match": len(matched),
        "frame_match_rate": round(len(matched) / total, 4) if total else 0.0,
        "first_matched_idx": matched[0] if matched else None,
        "last_matched_idx": matched[-1] if matched else None,
    }


# ============================================================
# TRACKING METRICS (CLEAR-style: MOTA/IDF1/MOTP/IDSW)
# ============================================================

def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def clear_mot_metrics(
    gt: List[Tuple],
    pred: List[Tuple],
    iou_thr: float = 0.5,
    restrict_to_gt_frames: bool = True,
    collect_fp: bool = False,
) -> Dict[str, Any]:
    """CLEAR MOT evaluation for one video.

    gt/pred rows: (frame_1indexed, track_id, x1, y1, x2, y2[, label]).
    Greedy per-frame IoU matching with continuity preference for the same
    hyp-GT pair (standard CLEAR convention).

    restrict_to_gt_frames=True (default, MOTChallenge convention) scores ONLY
    frames that carry ground truth. This is essential for BDD100K box-track
    GT, which is annotated on a sparse keyframe subset rather than every video
    frame: scoring 30 fps predictions against a ~1 Hz label set counts every
    detection on an unannotated frame as a false positive. Measured on
    baseline_s960 (1217-frame clips, ~6.1 detections/frame, n_gt=1126) that
    alone produced FP ~7.2k and MOTA ~ -5.6 before any real tracking error.
    Coverage fields are reported either way so the sparsity stays visible.

    ponytail: simplified CLEAR - per-video greedy matching, not the official
    py-motmetrics implementation (no global bipartite optimization of
    detections). IDF1 does use the standard one-to-one ID matching.
    Upgrade path: swap in `motmetrics` for publishable numbers.

    collect_fp=True adds `fp_rows`, the exact predictions left unmatched. They
    cannot be re-derived downstream: matching is greedy one-to-one, so a
    prediction well above iou_thr is still an FP when a higher-IoU competitor
    took its GT box first. `false_positive_audit` consumes these rows so its
    FP set is the same set the FP count came from.
    """
    gt_frames = {r[0] for r in gt}
    pred_frames = {r[0] for r in pred}
    if restrict_to_gt_frames:
        frames = sorted(gt_frames)
        scored_pred = [r for r in pred if r[0] in gt_frames]
    else:
        frames = sorted(gt_frames | pred_frames)
        scored_pred = list(pred)

    gt_by_frame: Dict[int, List[Tuple]] = {}
    for r in gt:
        gt_by_frame.setdefault(r[0], []).append(r)
    pr_by_frame: Dict[int, List[Tuple]] = {}
    for r in scored_pred:
        pr_by_frame.setdefault(r[0], []).append(r)

    total_gt = total_fp = total_fn = idsw = 0
    iou_sum, iou_n = 0.0, 0
    fp_rows: List[Tuple] = []
    # matched pairs history: gt_id -> last matched hyp_id
    last_match: Dict[Any, Any] = {}
    # identity co-occurrence: (gt_id, hyp_id) -> frames matched together
    id_hits: Dict[Tuple[Any, Any], int] = Counter()

    for f in frames:
        g = gt_by_frame.get(f, [])
        p = pr_by_frame.get(f, [])
        total_gt += len(g)

        pairs = []
        for gi, gr in enumerate(g):
            for pi, pr in enumerate(p):
                iou = _iou(gr[2:6], pr[2:6])
                if iou >= iou_thr:
                    # prefer keeping previous frame's pairing
                    prior = 1 if last_match.get(gr[1]) == pr[1] else 0
                    pairs.append((-prior, -iou, gi, pi))
        pairs.sort()
        used_g, used_p = set(), set()
        for neg_prior, neg_iou, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            gr, pr = g[gi], p[pi]
            if last_match.get(gr[1]) is not None and last_match[gr[1]] != pr[1]:
                idsw += 1
            last_match[gr[1]] = pr[1]
            id_hits[(gr[1], pr[1])] += 1
            iou_sum += -neg_iou
            iou_n += 1
        total_fp += len(p) - len(used_p)
        total_fn += len(g) - len(used_g)
        if collect_fp:
            fp_rows.extend(p[i] for i in range(len(p)) if i not in used_p)

    fn = total_fn
    fp = total_fp
    mota = 1.0 - (fn + fp + idsw) / total_gt if total_gt else 0.0
    motp = iou_sum / iou_n if iou_n else 0.0

    # IDF1 (Ristani et al. 2016): match GT ids to hypothesis ids ONE-TO-ONE to
    # maximise IDTP = frames where that pair co-occurs, then
    # IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN). Greedy over the IDTP matrix
    # (exact Hungarian would differ only marginally here). The previous
    # "dominant hypothesis per GT id" variant ignored the FP side entirely and
    # scored 0.07 on runs where tracking was in fact reasonable.
    used_g, used_h, idtp = set(), set(), 0
    for (g_id, h_id), n in sorted(id_hits.items(), key=lambda kv: (-kv[1], kv[0])):
        if g_id in used_g or h_id in used_h:
            continue
        used_g.add(g_id)
        used_h.add(h_id)
        idtp += n
    idfn = total_gt - idtp
    idfp = len(scored_pred) - idtp
    denom = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / denom if denom else 0.0

    out = {
        "MOTA": round(mota, 4),
        "MOTP": round(motp, 4),
        "IDF1": round(idf1, 4),
        "IDSW": int(idsw),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(iou_n),
        "n_gt": int(total_gt),
        "n_gt_ids": len({r[1] for r in gt}),
        "n_gt_frames": len(gt_frames),
        "n_pred_scored": len(scored_pred),
        "n_pred_total": len(pred),
        "frames_evaluated": len(frames),
        "gt_frame_coverage": round(len(gt_frames) / len(pred_frames), 4) if pred_frames else None,
        "restricted_to_gt_frames": bool(restrict_to_gt_frames),
    }
    if collect_fp:
        # 6-tuples, the same shape `false_positive_audit` documents.
        out["fp_rows"] = [tuple(r[:6]) for r in fp_rows]
    return out


# ============================================================
# FALSE-POSITIVE FORENSICS
# ============================================================

# How a false positive relates to the ground truth. The point of the split is
# that "FP" is not one thing, and the three causes have opposite fixes:
#
#   localisation   - the detector found the right object and the box is a bit
#                    off (best IoU in the SAME frame is close to the threshold).
#                    Fix: box regression / higher IoU threshold, not new data.
#   gt_sampling    - the box lines up with a GT object annotated in a NEIGHBOURING
#                    keyframe. The object IS labelled; the ~6x downsampled dump
#                    just did not sample the frame the detector fired on. Fix:
#                    score against a temporal window, not exact frames.
#   unexplained    - no GT overlap in the window. Either a real object the dump
#                    never annotated, or a hallucination. Only pixels can
#                    separate those two, so this band is the one to look at.
FP_BANDS = ("localisation", "gt_sampling", "unexplained")


def false_positive_audit(
    gt: List[Tuple],
    fp_rows: List[Tuple],
    iou_thr: float = 0.5,
    window: int = 1,
) -> Dict[str, Any]:
    """Classify every false positive by HOW it missed, not just count it.

    Motivation, measured on baseline_s960: false positives jump from 94 in the
    daytime clip to ~540 per clip at night/dawn-dusk while FALSE NEGATIVES stay
    flat. FN flat + FP multiplied is the signature of a scoring artifact (a
    correct detection on a frame the sparse GT dump never annotated) as much as
    it is the signature of a hallucinating detector, and the two are opposites:
    one means the model is fine and the metric is unfair, the other means a real
    night-robustness defect. Counting alone cannot tell them apart, so this
    records the best IoU of each FP against the same frame and against the
    nearest `window` annotated frames.

    `fp_rows` MUST be the rows that `clear_mot_metrics(collect_fp=True)`
    returned. They cannot be re-derived here: matching is greedy one-to-one, so
    a prediction well above the IoU threshold is still an FP when a
    higher-IoU competitor took its GT box first, and any independent pass would
    silently disagree with the reported FP count.

    gt rows: (frame_1indexed, track_id, x1, y1, x2, y2[, label]).
    fp rows: (frame, track_id, x1, y1, x2, y2).

    `window` counts ANNOTATED SAMPLES, not video frames. The BDD100K dump is
    ~6x downsampled, so the nearest annotated neighbours of a given frame are
    ~6 raw frames away; a raw-frame window of 1 would never reach them and
    every FP would be reported as `unexplained`.
    """
    gt_by_frame: Dict[int, List[Tuple]] = {}
    for r in gt:
        gt_by_frame.setdefault(r[0], []).append(r)
    sample_frames = sorted(gt_by_frame)
    pos_of = {f: i for i, f in enumerate(sample_frames)}

    def _best_iou(box: Tuple[float, float, float, float],
                  rows: List[Tuple]) -> Tuple[float, Optional[Any]]:
        best, best_id = 0.0, None
        for r in rows:
            v = _iou(box, r[2:6])
            if v > best:
                best, best_id = v, r[1]
        return best, best_id

    bands: Dict[str, int] = {b: 0 for b in FP_BANDS}
    records: List[Dict[str, Any]] = []
    size_bands: Dict[str, int] = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
    for pr in fp_rows:
        f = int(pr[0])
        here = gt_by_frame.get(f, [])
        near: List[Tuple] = []
        pos = pos_of.get(f)
        if pos is not None:
            lo, hi = max(0, pos - max(0, window)), min(len(sample_frames),
                                                        pos + max(0, window) + 1)
            for k in range(lo, hi):
                if k != pos:
                    near.extend(gt_by_frame[sample_frames[k]])
        iou_same, _ = _best_iou(pr[2:6], here)
        iou_near, near_id = _best_iou(pr[2:6], near)
        if iou_same > 0.0:
            band = "localisation"
        elif iou_near >= iou_thr:
            band = "gt_sampling"
        else:
            band = "unexplained"
        bands[band] += 1
        area = max(0.0, (pr[4] - pr[2])) * max(0.0, (pr[5] - pr[3]))
        if area < 32 * 32:
            size_bands["tiny"] += 1
        elif area < 96 * 96:
            size_bands["small"] += 1
        elif area < 256 * 256:
            size_bands["medium"] += 1
        else:
            size_bands["large"] += 1
        records.append({
            "frame": f,
            "track_id": pr[1],
            "box": [round(float(v), 1) for v in pr[2:6]],
            "area_px": int(area),
            "best_iou_same_frame": round(iou_same, 4),
            "best_iou_nearby_frames": round(iou_near, 4),
            "nearest_gt_id": near_id,
            "band": band,
        })

    n_fp = len(fp_rows)
    return {
        "n_fp": n_fp,
        "bands": bands,
        "band_fractions": {b: (round(bands[b] / n_fp, 4) if n_fp else None)
                           for b in FP_BANDS},
        "area_bands_px": size_bands,
        "iou_thr": iou_thr,
        "nearby_window_frames": window,
        "band_definitions": {
            "localisation": f"best IoU in the same frame is > 0 but < {iou_thr} - "
                            "detector found the object, the box is misaligned",
            "gt_sampling": f"no same-frame overlap, but a GT box within "
                           f"+/-{window} annotated sample(s) matches - the object "
                           "IS labelled, the downsampled dump just missed this frame",
            "unexplained": "no GT overlap in the window - a real unannotated object "
                           "or a hallucination; only pixels separate those",
        },
        "records": records,
    }

