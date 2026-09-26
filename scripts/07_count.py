#!/usr/bin/env python3
"""
Phase 7: Virtual-Line Vehicle Counting

Counts vehicles crossing a configurable horizontal line, using the persistent
track IDs produced by Phase 6. Reads the MOT-format files Phase 6 already wrote
(`predictions/tracking/<video>.txt`), so this stage needs no GPU and does not
re-run tracking.

Counting rules (skill §13):
  - a vehicle is counted on a CROSSING EVENT, never on per-frame presence
  - each track ID is counted at most once (duplicate-count prevention)
  - counting is direction-aware and per-class

Direction semantics (skill §14): BDD100K dashcam frames are NOT calibrated, so no
geographic direction is fabricated. Directions are screen-relative:
  - 'farbound' : crossed moving UP the image (away from the camera)
  - 'nearbound': crossed moving DOWN the image (toward the camera)

Inputs:
    - predictions/tracking/*.txt   (from 06_track.py, or --mot-dir)
    - reports/phase6_tracking.json (optional, for per-track class names)

Outputs:
    - reports/phase7_counting.json
    - plots/counting/*.png        (per-class and per-direction bars)

Run (after 06_track.py):
    python traffic_ai/scripts/07_count.py --line-y 0.6
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.config import OUTPUT
from src.log import (
    setup_logger, log_section, log_pass, log_fail, log_warn,
    Report, collect_environment,
)
from src.count import (
    parse_mot_file, iter_mot_dir, tracks_from_mot_rows, detect_crossings,
    summarize, aggregate_videos, MIN_TRACK_FRAMES, tracks_from_box_rows,
    accuracy_metrics,
)
from src.viz_count import (
    render_sample_frames, render_box_only_montage, plot_crossing_timelines,
    plot_line_sweep, plot_count_by_stratum,
)
from src.config import BDD100K_HEIGHT, BDD100K_WIDTH, GT_LABEL_MAP
from src.track import load_gt_parquet, remap_gt_frames_linear
from src.strata import (
    METADATA_FIELDS, attach_metadata, coverage, load_video_metadata,
    resolve_manifest_paths, stratified_counting, UNLABELLED,
)


def parse_args():
    p = argparse.ArgumentParser(description="Count vehicles crossing a virtual line")
    p.add_argument("--line-y", type=float, default=0.6,
                   help="Counting line as a fraction of frame height "
                        "(0=top, 1=bottom). Default 0.6.")
    p.add_argument("--sweep", type=str, default=None,
                   help="Comma-separated line fractions to evaluate instead of "
                        "--line-y, e.g. '0.4,0.5,0.6,0.7'. Counting needs no "
                        "GPU, so line placement can be chosen from data rather "
                        "than guessed. Reports totals for every position.")
    p.add_argument("--frame-height", type=float, default=0.0,
                   help="Video height in px. 0 = auto (OpenCV probe, then the "
                        "BDD100K constant). Guessing this misplaces the line.")
    p.add_argument("--videos", type=str, default=None,
                   help="Video directory, used only to probe true frame size "
                        "with OpenCV and to render annotated frame samples. "
                        "Optional - without it a geometry-only montage is "
                        "drawn from the MOT boxes instead.")
    p.add_argument("--allow-height-heuristic", action="store_true",
                   help="Permit inferring frame height from the largest box "
                        "bottom edge. Biased (reported 440-697 px for 720 px "
                        "videos); off by so the verified BDD100K constant wins.")
    p.add_argument("--auto-line", action="store_true",
                   help="Place the line at the median centroid y of all "
                        "observed tracks (overridden by --sweep/--line-y).")
    p.add_argument("--mot-dir", type=str, default=None,
                   help="Directory of MOT files. Default predictions/tracking.")
    p.add_argument("--track-report", type=str, default=None,
                   help="Phase 6 report JSON, for per-track class names. "
                        "Default reports/phase6_tracking.json if present.")
    p.add_argument("--min-frames", type=int, default=MIN_TRACK_FRAMES,
                   help="Minimum annotated frames before a track can be counted.")
    p.add_argument("--min-net-disp-frac", type=float, default=0.0,
                   help="EXPERIMENTAL, direction-biased, off by default. "
                        "Require a track's net vertical travel to reach this "
                        "fraction of frame height. Measured on BDD100K: at 0.05 "
                        "it removed 93%% of farbound crossings but only 56%% of "
                        "nearbound, because receding vehicles yield short "
                        "tracks. Prefer the default opposite-ends rule.")
    p.add_argument("--loose-crossing", action="store_true",
                   help="Count any side-flip, even if the track starts and "
                        "ends on the SAME side of the line. This is what makes "
                        "the total swing ~40x across line positions, because "
                        "tracks that merely hover across the line get counted. "
                        "Use only to reproduce the unfiltered baseline.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Video frame rate, used only for vehicles/minute.")
    p.add_argument("--max-videos", type=int, default=0,
                   help="Cap videos processed (0 = all).")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG output.")
    p.add_argument("--viz-frames", type=int, default=4,
                   help="Sample frames per video in the annotated montage.")
    p.add_argument("--gt", type=str, default=None,
                   help="GT parquet file/dir. Enables COUNTING ACCURACY "
                        "(MAE/RMSE/MAPE) by running the identical crossing rule "
                        "over GT tracks. Requires 06_track.py to have run with "
                        "--gt so the GT->video frame map is available.")
    p.add_argument("--select-line", type=str, default="max-count",
                   choices=["max-count", "gt-mae"],
                   help="How to choose the line position from the sweep. "
                        "'max-count' (default) is BIASED: the count peaks "
                        "where tracks hover across the line rather than "
                        "traverse it (measured +170%% error). 'gt-mae' places "
                        "the line where predicted counts best match GT, among "
                        "positions that see at least --min-gt-coverage of the "
                        "traffic - minimising error alone is degenerate, "
                        "because a line where little traffic passes trivially "
                        "has a small error. Selection is on the evaluation "
                        "set, so the reported accuracy is optimistic and must "
                        "be stated.")
    p.add_argument("--min-gt-coverage", type=float, default=0.5,
                   help="With --select-line gt-mae, only consider line "
                        "positions whose GT crossing count reaches this "
                        "fraction of the maximum. Prevents selecting a quiet "
                        "line that misses most traffic. Default 0.5.")
    p.add_argument("--manifest", type=str, default=None,
                   help="Phase 2 manifest JSON(s), comma-separated, used to "
                        "label each video with its BDD100K timeofday / "
                        "weather / scene so counts can be stratified "
                        "(skill §3, §10, §17F). Default: "
                        "reports/manifest_val.json + manifest_train.json if "
                        "present. Unlabelled videos are reported in the "
                        "'unlabelled' stratum, never dropped.")
    return p.parse_args()


def load_class_map(report_path: Path) -> Dict[int, str]:
    """track_id -> class name, from the Phase 6 report. {} when unavailable."""
    if not report_path or not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log_warn(logging.getLogger("phase7"),
                 f"  could not read track report {report_path}: {e}")
        return {}
    out: Dict[int, str] = {}
    for vid in data.get("videos", []):
        for t in vid.get("tracks", []):
            if "track_id" in t and "cls" in t:
                out[int(t["track_id"])] = str(t["cls"])
    return out


def frame_height_from_report(report_path: Path, video: str) -> Optional[float]:
    """DEPRECATED heuristic, kept only as a last-resort fallback.

    Inferring height from the largest `bbox_last` bottom edge is biased: it
    returned 440-697 px for 720 px videos and put the counting line in the wrong
    place, so two clips counted zero crossings. Prefer the OpenCV probe, then
    the BDD100K constant. Never call this first.
    """
    if not report_path or not report_path.exists():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for vid in data.get("videos", []):
        if vid.get("video") != video and \
                Path(str(vid.get("video", ""))).stem != video:
            continue
        ymax = 0.0
        for t in vid.get("tracks", []):
            bbox = t.get("bbox_last") or []
            if len(bbox) == 4:
                ymax = max(ymax, float(bbox[3]))
        return ymax or None
    return None


def probe_frame_size(video_dir: Any, stem: str) -> Optional[Tuple[float, float]]:
    """True (width, height) of a video via OpenCV, or None if unavailable.

    Preferred source of truth for the counting line. Guessing the height is
    not safe: an earlier version inferred it from the largest `bbox_last`
    bottom edge, which gave 440-697 px for 720 px videos and put the line
    in the wrong place (two clips then counted zero crossings).
    """
    try:
        import cv2
    except ImportError:
        return None
    if not video_dir:
        return None
    for ext in (".mov", ".mp4", ".avi", ".mkv", ".webm"):
        p = Path(video_dir) / f"{stem}{ext}"
        if not p.exists():
            continue
        cap = cv2.VideoCapture(str(p))
        try:
            if cap.isOpened():
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                if h and w:
                    return float(w), float(h)
        finally:
            cap.release()
    return None


def resolve_frame_size(stem: str, videos_arg: Optional[str],
                       report_path: Path, forced: float,
                       allow_heuristic: bool = False) -> Tuple[float, str]:
    """True frame height for a video, with a documented fallback chain.

    opencv probe -> --frame-height -> BDD100K constant (720, a verified fact)
    -> bbox heuristic (only with --allow-height-heuristic).

    The bbox heuristic is a GUESS and the BDD100K constant is a KNOWN FACT, so
    the fact must outrank the guess. An earlier chain had them the other way
    round and, with no --videos given, reported 440/640 px for 720 px videos -
    which misplaced the line and clipped the timeline plots.
    """
    if videos_arg:
        got = probe_frame_size(videos_arg, stem)
        if got:
            return got[1], "opencv"
    if forced and forced > 0:
        return float(forced), "forced"
    if allow_heuristic and report_path and report_path.exists():
        h = frame_height_from_report(report_path, stem)
        if h:
            return h, "bbox-heuristic"
    return float(BDD100K_HEIGHT), "bdd100k-constant"


_VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv", ".webm")


def video_path_for(video_dir: Any, stem: str) -> Optional[Path]:
    """Locate a video by stem. Searches RECURSIVELY: on Kaggle the videos sit
    several levels down (e.g. .../bdd100k_videos_train_00/bdd100k/videos/train),
    so a top-level-only lookup silently found nothing and every montage fell
    back to the geometry view."""
    if not video_dir:
        return None
    root = Path(video_dir)
    if not root.is_dir():
        return None
    for ext in _VIDEO_EXTS:
        direct = root / f"{stem}{ext}"
        if direct.exists():
            return direct
    for ext in _VIDEO_EXTS:
        for p in root.rglob(f"{stem}{ext}"):
            return p
    # last resort: any file whose stem matches
    for p in root.rglob(f"{stem}.*"):
        if p.suffix.lower() in _VIDEO_EXTS:
            return p
    return None


def load_frame_maps(report_path: Path) -> Dict[str, Dict[str, Any]]:
    """video stem -> GT->video frame map, as calibrated by Phase 6.

    Counting accuracy compares a predicted crossing count against a GT crossing
    count, and the GT dump is DOWNSAMPLED (~5 Hz on a 30 fps video). Phase 6
    already solved that mapping per video; reusing it here means both sides of
    the comparison live in the same video-frame space. Re-calibrating
    independently would risk comparing counts from two different alignments.
    """
    if not report_path or not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for vid in data.get("videos", []):
        m = vid.get("mot_metrics") or {}
        if "gt_frame_alpha" not in m:
            continue
        stem = Path(str(vid.get("video", ""))).stem
        out[stem] = {
            "alpha": float(m["gt_frame_alpha"]),
            "offset": int(m.get("gt_frame_offset", 0)),
            "mode": m.get("gt_frame_map_mode"),
            "stride": m.get("gt_frame_stride_applied"),
        }
    return out


def draw_plots(per_video: List[Dict[str, Any]], agg: Dict[str, Any],
               out_dir: Path) -> List[str]:
    """Per-class and per-direction bar charts. Best-effort: never fatal."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as e:
        log_warn(logging.getLogger("phase7"), f"  plots skipped ({e})")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for title, key, fname in (
            ("Counted vehicles per class", "by_class", "count_by_class.png"),
            ("Counted vehicles per direction", "by_direction",
             "count_by_direction.png")):
        data = agg.get(key) or {}
        if not data:
            continue
        labels = list(data.keys())
        values = [data[k] for k in labels]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values, color="#3b6ea5")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                    ha="center", va="bottom")
        ax.set_title(f"{title} (all videos, n={agg['total_counted']})")
        ax.set_ylabel("crossing events")
        if key == "by_direction":
            ax.set_title(title + "\nscreen-relative; frames are not calibrated")
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    # per-video predicted vs GT
    if any(v.get("gt_counted") is not None for v in per_video):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        pair = [v for v in per_video if v.get("gt_counted") is not None]
        labels = [v["video"] for v in pair]
        xs = range(len(pair))
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.5))
        ax.bar([x - 0.2 for x in xs], [v["gt_counted"] for v in pair], width=0.4,
               label="ground truth", color="#455a64")
        ax.bar([x + 0.2 for x in xs], [v["total_counted"] for v in pair],
               width=0.4, label="predicted", color="#1565c0")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("crossing events")
        a = accuracy_metrics(per_video)
        ax.set_title("Counting accuracy per video"
                     + (f"\nMAE={a.get('MAE')} RMSE={a.get('RMSE')} "
                        f"MAPE={a.get('MAPE_pct')}% accuracy={a.get('counting_accuracy')}"
                        if a.get("n_videos_compared") else ""))
        ax.legend()
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        p = out_dir / "counting_accuracy.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    # per-video totals
    if per_video:
        labels = [v["video"] for v in per_video]
        values = [v["total_counted"] for v in per_video]
        fig, ax = plt.subplots(figsize=(max(7, len(labels)), 4))
        ax.bar(labels, values, color="#5a9367")
        ax.set_title("Crossing events per video")
        ax.set_ylabel("crossing events")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        path = out_dir / "count_per_video.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    # day/night stratification (skill §10, §17E/§17F)
    p = plot_count_by_stratum(per_video, "timeofday",
                              out_dir / "count_by_timeofday.png")
    if p:
        written.append(p)
    return written


def main() -> int:
    args = parse_args()
    OUTPUT.ensure_all()
    logger = setup_logger("phase7", log_file=OUTPUT.logs / "phase7.log")
    log_section(logger, "PHASE 7: VIRTUAL-LINE VEHICLE COUNTING")

    if not 0.0 < args.line_y < 1.0:
        log_fail(logger, f"--line-y must be a fraction in (0, 1), got {args.line_y}")
        return 1

    mot_dir = Path(args.mot_dir) if args.mot_dir else OUTPUT.predictions / "tracking"
    if not mot_dir.is_dir():
        log_fail(logger, f"MOT directory not found: {mot_dir} "
                          f"- run 06_track.py first")
        return 1

    entries = iter_mot_dir(mot_dir, args.max_videos)
    if not entries:
        log_fail(logger, f"No .txt MOT files in {mot_dir}")
        return 1

    track_report = Path(args.track_report) if args.track_report else \
        OUTPUT.reports / "phase6_tracking.json"
    class_map = load_class_map(track_report)
    frame_maps = load_frame_maps(track_report)
    args.gt_files = None
    if args.gt:
        gp = Path(args.gt)
        args.gt_files = [gp] if gp.is_file() else sorted(gp.glob("*.parquet"))
        if not args.gt_files:
            log_warn(logger, f"  no .parquet under {args.gt} - counting "
                             f"accuracy will be skipped")
            args.gt_files = None
    logger.info("MOT files: %d in %s", len(entries), mot_dir)
    logger.info("Track classes: %s from %s",
                "loaded" if class_map else "UNAVAILABLE (will report 'unknown')",
                track_report)
    logger.info("Counting accuracy: %s",
                f"GT frame maps for {len(frame_maps)} video(s) from Phase 6"
                if (args.gt_files and frame_maps)
                else "UNAVAILABLE (need --gt AND 06_track.py run with --gt)")
    logger.info("Counting line: %.1f%% of frame height | min_frames=%d | fps=%.1f",
                100.0 * args.line_y, args.min_frames, args.fps)

    # BDD100K scene metadata per video, from the Phase 2 manifests. Day/night
    # stratification is the research question (skill §3, §10, §17F), and the
    # labels come from the dataset: a hand-written scene inventory for these
    # clips disagreed with the metadata on 3 of 5.
    manifest_paths = resolve_manifest_paths(args.manifest, OUTPUT.reports)
    video_meta = load_video_metadata(manifest_paths)
    logger.info("Scene metadata: %d video(s) from %s", len(video_meta),
                ", ".join(p.name for p in manifest_paths) or "NO MANIFEST FOUND")

    report = Report(stage="07_counting",
                    description=f"Virtual-line counting on {len(entries)} video(s)")
    report.add("environment", collect_environment())
    report.add("config", {
        "line_y_fraction": args.line_y,
        "min_frames": args.min_frames,
        "fps": args.fps,
        "mot_dir": str(mot_dir),
        "track_report": str(track_report),
        "direction_semantics": "screen-relative (frames are not calibrated; "
                               "no geographic direction is claimed)",
    })

    per_video: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    sweep_cache: Dict[str, Dict[str, Any]] = {}
    viz_dir = OUTPUT.plots / "counting"
    for stem, path in entries:
        rows = parse_mot_file(path)
        if not rows:
            log_warn(logger, f"  {stem}: no usable MOT rows, skipped")
            skipped.append({
                "video": stem, "total_counted": 0, "n_tracks_seen": 0,
                "n_tracks_counted": 0, "by_class": {}, "by_direction": {},
                "class_composition": {}, "count_rate": 0.0, "directions": [],
                "rates": {"observation_minutes": 0.0,
                          "vehicles_per_minute": None,
                          "vehicles_per_5min": None,
                          "vehicles_per_15min": None},
                "error": "no usable MOT rows",
            })
            continue

        n_frames = max(r[0] for r in rows)
        height, hsrc = resolve_frame_size(stem, args.videos, track_report,
                                          args.frame_height,
                                          args.allow_height_heuristic)
        tracks = tracks_from_mot_rows(rows, class_map)

        # --sweep evaluates several line positions at once (cheap: no GPU).
        fractions = ([float(x) for x in args.sweep.split(",") if x.strip()]
                     if args.sweep else [args.line_y])
        if args.auto_line and not args.sweep:
            all_y = sorted(p[2] for t in tracks.values() for p in t["points"])
            fractions = [round(all_y[len(all_y) // 2] / height, 4)] if all_y \
                else fractions
        results: Dict[str, Dict[str, Any]] = {}
        events_by_frac: Dict[str, List[Dict[str, Any]]] = {}
        min_disp = args.min_net_disp_frac * height
        for frac in fractions:
            line_y = frac * height
            ev = detect_crossings(tracks, line_y, args.min_frames, min_disp,
                                  require_opposite_ends=not args.loose_crossing)
            results[f"{frac:g}"] = summarize(ev, len(tracks), n_frames, args.fps)
            events_by_frac[f"{frac:g}"] = ev

        logger.info(
            "  %s: %d tracks | height=%.0f (%s) | opposite_ends=%s "
            "min_net_disp=%.0fpx | %s",
            stem, len(tracks), height, hsrc,
            not args.loose_crossing, min_disp,
            {k: v["total_counted"] for k, v in sorted(
                results.items(), key=lambda kv: float(kv[0]))},
        )
        # GT counts at EVERY swept position, so the line can be selected by
        # agreement with GT rather than by maximum count. See select_line.
        gt_by_frac: Dict[str, Any] = {}
        if args.gt_files and stem in frame_maps:
            fm = frame_maps[stem]
            try:
                gt_rows = load_gt_parquet(args.gt_files, video_name=stem)
                gt_rows = [r for r in gt_rows if r[6].lower() in GT_LABEL_MAP]
                gt_rows = remap_gt_frames_linear(gt_rows, fm["alpha"],
                                                fm["offset"])
                gt_tracks = tracks_from_box_rows(gt_rows)
                for frac in fractions:
                    ly = frac * height
                    gev = detect_crossings(
                        gt_tracks, ly, args.min_frames,
                        require_opposite_ends=not args.loose_crossing)
                    gt_by_frac[f"{frac:g}"] = {
                        "count": len(gev),
                        "by_direction": {
                            d: sum(1 for e in gev if e["direction"] == d)
                            for d in ("farbound", "nearbound")},
                    }
                logger.info("    GT crossings per position: %s",
                            {k: v["count"] for k, v in sorted(
                                gt_by_frac.items(), key=lambda kv: float(kv[0]))})
            except Exception as e:
                log_warn(logger, f"    GT count unavailable for {stem}: {e}")
        sweep_cache[stem] = {
            "rows": rows, "tracks": tracks, "height": height, "hsrc": hsrc,
            "n_frames": n_frames, "results": results,
            "events_by_frac": events_by_frac, "fractions": fractions,
            "gt_by_frac": gt_by_frac, "frame_map": frame_maps.get(stem),
        }
        continue

    # ---- pick ONE line position for the whole set -------------------------
    # Per-video best-case line placement would inflate the total: every clip
    # would be counted at its own optimum. A traffic-flow figure is only
    # meaningful if the same line is applied to every video.
    all_fracs = sorted({f for c in sweep_cache.values() for f in c["fractions"]},
                       key=float)
    totals = {f"{f:g}": sum(c["results"].get(f"{f:g}", {}).get("total_counted", 0)
                            for c in sweep_cache.values())
              for f in all_fracs}
    logger.info("  Predicted totals per position: %s",
                {k: totals[k] for k in sorted(totals, key=float)})

    # Robustness: the naive max/min spread is useless because the sweep edges
    # legitimately count ~0 (nobody is up there), and a min of 0 previously made
    # the sensitivity check PASS. Report a PLATEAU instead: a real counting line
    # has a flat top with falloff either side; a hover band gives one spike.
    tvals = sorted(totals.values())
    best = tvals[-1] if tvals else 0
    plateau = [v for v in tvals if best > 0 and v >= 0.8 * best]
    plateau_frac = round(len(plateau) / len(tvals), 3) if tvals else 0.0
    nonzero = [v for v in tvals if v > 0]
    spread = round(best / nonzero[0], 2) if len(nonzero) > 1 else None

    # ---- how to CHOOSE the line -------------------------------------------
    # `max-count` is the default but it is DEMONSTRABLY biased: the count peaks
    # where tracks hover across the line instead of traversing it. Measured on
    # 0000f77c-6257be58: 12 / 27 / 11 at lines 0.45 / 0.50 / 0.55 while GT says
    # 10 - the argmax sits in a hover band and inflates the count 2.5x, giving
    # +170% error. With GT available, select the position that MINIMISES
    # counting error. That is selection on the evaluation set, so it must be
    # reported as such, but one parameter over 5 videos is mild.
    gt_select: Dict[str, Dict[str, Any]] = {}
    for f in all_fracs:
        k = f"{f:g}"
        errs, gts, preds = [], [], []
        for c in sweep_cache.values():
            g = (c.get("gt_by_frac") or {}).get(k, {}).get("count")
            if g is None:
                continue
            p_ = c["results"][k]["total_counted"]
            errs.append(p_ - g); gts.append(g); preds.append(p_)
        if errs:
            gt_select[k] = {
                "MAE": round(sum(abs(e) for e in errs) / len(errs), 3),
                "RMSE": round(math.sqrt(sum(e * e for e in errs) / len(errs)), 3),
                "gt_total": sum(gts), "pred_total": sum(preds), "n": len(errs),
            }

    if gt_select:
        logger.info("  Counting error vs GT per position (MAE): %s",
                    {k: gt_select[k]["MAE"] for k in sorted(gt_select, key=float)})

    if args.select_line == "gt-mae" and gt_select:
        # Minimising MAE alone is DEGENERATE: a line positioned where little
        # traffic passes makes BOTH the GT and the prediction small, so the
        # absolute error is small. Measured: line 0.65 has the lowest MAE
        # (1.0) but sees only 25% of the GT crossings, while 0.55 has MAE 1.8
        # at 75% coverage. So constrain the search to positions that actually
        # SEE most of the traffic, then minimise error within that set.
        max_gt = max(v["gt_total"] for v in gt_select.values())
        floor = args.min_gt_coverage * max_gt
        eligible = {k: v for k, v in gt_select.items() if v["gt_total"] >= floor}
        if not eligible:
            eligible = gt_select
            log_warn(logger, f"  no line position reaches {args.min_gt_coverage:.0%} "
                             f"of the max GT volume ({max_gt}); searching all")
        chosen = min(eligible,
                     key=lambda k: (eligible[k]["MAE"],
                                    -eligible[k]["gt_total"],
                                    abs(float(k) - args.line_y)))
        logger.info("  GT volume per position (pred/GT/MAE/coverage): %s", {
            k: (v["pred_total"], v["gt_total"], v["MAE"],
                f"{100.0 * v['gt_total'] / max_gt:.0f}%")
            for k, v in sorted(gt_select.items(), key=lambda kv: float(kv[0]))})
        logger.info("  Eligible at >=%.0f%% GT coverage: %s -> chose %s",
                    100.0 * args.min_gt_coverage,
                    sorted(eligible, key=float), chosen)
        log_warn(logger, "  line position SELECTED ON GROUND TRUTH, so the "
                         "reported accuracy is optimistic - state this in the "
                         "writeup")
    else:
        best_n = max(totals.values())
        tied = [k for k, v in totals.items() if v == best_n]
        chosen = min(tied, key=lambda k: abs(float(k) - args.line_y))
        if len(tied) > 1:
            log_warn(logger, f"  line sweep TIE at {best_n} total crossings for "
                             f"{sorted(tied, key=float)} - chose {chosen} "
                             f"(closest to default {args.line_y})")
        if args.select_line == "gt-mae":
            log_warn(logger, "  --select-line gt-mae requested but no GT counts "
                             "were available; fell back to max-count")

    report.add("line_position_totals", totals)
    report.add("line_position_chosen", chosen)
    report.add("line_plateau_fraction", plateau_frac)
    report.add("line_position_spread_ratio", spread)
    report.add("line_position_has_zeros", len(nonzero) < len(tvals))
    report.add("line_selection_rule", args.select_line)
    if gt_select:
        report.add("line_position_accuracy_vs_gt", gt_select)

    logger.info("  Using line %s | plateau(>=80%% of best)=%.0f%% of positions, "
                "spread(max/min over non-zero)=%s, positions with zero=%d",
                chosen, 100.0 * plateau_frac, spread, len(tvals) - len(nonzero))
    if len(nonzero) < len(tvals):
        log_warn(logger, f"  {len(tvals) - len(nonzero)} swept line position(s) "
                         f"count ZERO while the best counts {best} - the count "
                         f"is strongly position-dependent")
    if plateau_frac < 0.25 and args.select_line == "max-count":
        log_warn(
            logger,
            f"  Only {100.0 * plateau_frac:.0f}% of swept positions come within "
            f"20% of the best count ({best}). A robust counting line should have "
            f"a PLATEAU; a narrow spike means the line sits inside a band that "
            f"tracks merely hover in. Re-run with --select-line gt-mae and --gt "
            f"to place the line by agreement with ground truth instead.",
        )

    per_video = []
    for stem, c in sweep_cache.items():
        rows, tracks, height, hsrc = (c["rows"], c["tracks"], c["height"],
                                      c["hsrc"])
        n_frames = c["n_frames"]
        stats = c["results"][chosen]
        line_y = float(chosen) * height
        events = c["events_by_frac"][chosen]

        logger.info(
            "  %s: line y=%.0f of %.0f (%.0f%%) -> %d counted | by_class=%s "
            "| by_direction=%s | %.2f veh/min",
            stem, line_y, height, 100.0 * float(chosen),
            stats["total_counted"], stats["by_class"], stats["by_direction"],
            stats["rates"]["vehicles_per_minute"] or 0.0,
        )
        if len(c["fractions"]) > 1:
            logger.info("    per-video sweep: %s",
                        {k: v["total_counted"] for k, v in sorted(
                            c["results"].items(), key=lambda kv: float(kv[0]))})

        entry: Dict[str, Any] = {
            "video": stem,
            "n_frames": n_frames,
            "frame_height": height,
            "frame_height_source": hsrc,
            "line_y_fraction": float(chosen),
            "line_y_px": round(line_y, 1),
            "mot_rows": len(rows),
            **stats,
            "events": events,
        }
        if len(c["fractions"]) > 1:
            entry["line_sweep"] = {
                k: {"total_counted": v["total_counted"],
                    "by_direction": v["by_direction"],
                    "by_class": v["by_class"]}
                for k, v in c["results"].items()
            }
        gsel = (c.get("gt_by_frac") or {}).get(chosen)
        if gsel is not None:
            entry["gt_counted"] = gsel["count"]
            entry["gt_by_direction"] = gsel["by_direction"]
            entry["gt_frame_map"] = c.get("frame_map")


        # ---- visual verification -------------------------------------------
        if not args.no_plots:
            counted_ids = [e["track_id"] for e in events]
            xf = {e["track_id"]: e["frame"] for e in events}
            out_png = None
            vp = video_path_for(args.videos, stem) if args.videos else None
            if args.videos and vp is None:
                log_warn(logger, f"    no video file found for {stem} under "
                                 f"{args.videos} (searched recursively) - "
                                 f"using geometry view")
            if vp is not None:
                out_png = render_sample_frames(
                    vp, rows, line_y, counted_ids,
                    viz_dir / f"{stem}_counting_line.jpg",
                    n_frames=args.viz_frames, class_of=class_map,
                    crossing_frames=xf)
                if out_png:
                    logger.info("    frame sample (real video): %s", out_png)
                else:
                    log_warn(logger, f"    could not decode {vp.name}, "
                                     f"falling back to geometry view")
            # Always produce a view, even with no video mounted: the montage is
            # the fastest way to judge the count, so it must not be blocked on
            # having the videos available.
            if out_png is None:
                out_png = render_box_only_montage(
                    rows, line_y, (BDD100K_WIDTH, height), counted_ids,
                    viz_dir / f"{stem}_counting_line.jpg",
                    n_frames=args.viz_frames, class_of=class_map,
                    crossing_frames=xf)
                if out_png:
                    logger.info("    geometry view (boxes only, no video "
                                "decoded): %s", out_png)
            if out_png:
                entry["viz_frames"] = out_png
            out_tl = plot_crossing_timelines(
                tracks, line_y, height, viz_dir / f"{stem}_timelines.png",
                events=events, title=f"{stem}: centroid y vs frame "
                                     f"(height={height:.0f}px, line at y={line_y:.0f})")
            if out_tl:
                entry["viz_timelines"] = out_tl

        # Counting accuracy: gt_counted was already resolved in pass 2 from the
        # cached per-position GT counts, so the 2.89M-row parquet is read once
        # per video rather than twice. The GT crossing count is itself
        # APPROXIMATE because the GT is sampled at ~5 Hz and a vehicle can
        # cross between two annotated samples; the opposite-ends rule is used
        # on both sides precisely because it only inspects the first and last
        # side, which is robust to that sampling.
        if args.gt_files and stem not in frame_maps:
            log_warn(logger, f"    no Phase 6 frame map for {stem} - run "
                             f"06_track.py with --gt so counting accuracy can "
                             f"be evaluated")

        per_video.append(entry)

    per_video = skipped + per_video
    # Label every video with its BDD100K scene attributes (timeofday / weather /
    # scene) before any aggregate is computed, so the stratified tables below
    # are built from the same records as the overall totals.
    lab = attach_metadata(per_video, video_meta)
    agg = aggregate_videos(per_video)
    report.add("videos", per_video)
    report.add("summary", agg)
    report.add("video_metadata", {
        "manifests": [str(p) for p in manifest_paths],
        "n_videos_labelled_available": len(video_meta),
        "fields": list(METADATA_FIELDS),
    })

    log_section(logger, "SUMMARY")
    logger.info("Videos processed: %d", len(per_video))
    logger.info("Tracks seen: %d | counted once: %d",
                agg["n_tracks_seen"], agg["n_tracks_counted"])
    logger.info("Total crossing events: %d", agg["total_counted"])
    logger.info("By class: %s", agg["by_class"])
    logger.info("By direction (screen-relative): %s", agg["by_direction"])
    r = agg["rates"]
    logger.info("Rate: %.2f veh/min over %.2f min (%.1f / 5min, %.1f / 15min)",
                r["vehicles_per_minute"] or 0.0, r["observation_minutes"],
                r["vehicles_per_5min"] or 0.0, r["vehicles_per_15min"] or 0.0)
    logger.info("Per-video rate (quote THESE, not the pooled figure - the "
                "pooled window includes clips with no flow at all):")
    for v in per_video:
        logger.info("    %-24s %3d counted  %6.2f veh/min  %s",
                    v["video"], v["total_counted"],
                    (v.get("rates", {}).get("vehicles_per_minute") or 0.0),
                    f"GT={v['gt_counted']}" if v.get("gt_counted") is not None
                    else "GT=n/a")

    # ---- counting accuracy vs ground truth (skill §15) --------------------
    acc = accuracy_metrics(per_video)
    if acc.get("n_videos_compared"):
        report.add("counting_accuracy", acc)
        log_section(logger, "COUNTING ACCURACY vs GROUND TRUTH")
        logger.info("Videos compared: %d (%d excluded from MAPE: zero GT count)",
                    acc["n_videos_compared"],
                    acc["n_videos_excluded_from_mape"])
        logger.info("GT total crossings: %d | predicted: %d",
                    acc["gt_total"], acc["pred_total"])
        logger.info("MAE=%.2f  RMSE=%.2f  MAPE=%s%%  counting_accuracy=%.3f",
                    acc["MAE"], acc["RMSE"],
                    f"{acc['MAPE_pct']:.1f}" if acc["MAPE_pct"] is not None else "n/a",
                    acc["counting_accuracy"] or 0.0)
        logger.info("Per-video (predicted vs GT):")
        for v in per_video:
            if v.get("gt_counted") is None:
                continue
            g, p = v["gt_counted"], v["total_counted"]
            logger.info("    %-24s pred=%3d gt=%3d  err=%+4d  (%s)",
                        v["video"], p, g, p - g,
                        f"{100.0*(p-g)/g:+.1f}%" if g else "GT=0")
        if acc["counting_accuracy"] is not None and acc["counting_accuracy"] < 0.5:
            log_warn(logger, f"  counting accuracy {acc['counting_accuracy']:.2f} "
                             f"is below 0.5 - counts are not yet trustworthy as "
                             f"traffic volumes; report MAE/RMSE alongside")
    else:
        logger.info("Counting accuracy: not available (no GT frame maps)")

    # ---- stratified by scene metadata (skill §3, §10, §17E/§17F) ----------
    # The headline research question is whether counting survives night and
    # dusk, so it gets its own table instead of one pooled figure. Per-stratum
    # accuracy reuses accuracy_metrics unchanged, and every stratum reports how
    # many 40 s clips it rests on.
    strat_fields = ("timeofday", "scene", "weather")
    stratified = {f: stratified_counting(per_video, video_meta, f)
                  for f in strat_fields}
    if any(len(stratified[f]) > 1 or UNLABELLED not in stratified[f]
           for f in strat_fields):
        report.add("stratified", stratified)
        report.add("stratified_coverage",
                   {f: coverage(per_video, f, video_meta) for f in strat_fields})
        log_section(logger, "COUNTS STRATIFIED BY SCENE METADATA")
        for f in strat_fields:
            table = stratified[f]
            # A single unlabelled bucket is not a stratification.
            if len(table) <= 1 and UNLABELLED in table:
                continue
            logger.info("  by %s:", f)
            for value, e in table.items():
                a = e.get("accuracy") or {}
                mape = a.get("MAPE_pct")
                logger.info(
                    "    %-16s n=%d  counted=%3d  %6.2f veh/min  GT=%s  "
                    "MAE=%s  MAPE=%s  dirs=%s%s",
                    value, e["n_videos"], e["total_counted"],
                    e["vehicles_per_minute"] or 0.0,
                    a.get("gt_total", "n/a"), a.get("MAE", "n/a"),
                    f"{mape:.1f}%" if mape is not None else "n/a",
                    e["by_direction"] or "-",
                    "  [LOW SAMPLE - anecdote, not an effect]"
                    if e.get("low_sample") else "",
                )
        if lab["unlabelled"]:
            log_warn(logger,
                     f"  {lab['unlabelled']} video(s) carry no BDD100K scene "
                     f"metadata - they appear in the '{UNLABELLED}' stratum and "
                     f"are EXCLUDED from every day/night comparison")
        if not video_meta:
            log_warn(logger, "  no Phase 2 manifest found - stratified counting "
                             "is impossible. Re-run Phase 2 (or pass "
                             "--manifest) before quoting day/night numbers")
    if not args.no_plots:
        written = draw_plots(per_video, agg, OUTPUT.plots / "counting")
        for v in per_video:
            written += [v[k] for k in ("viz_frames", "viz_timelines") if k in v]
        sweeps = [v["line_sweep"] for v in per_video if "line_sweep" in v]
        if sweeps:
            merged: Dict[str, Dict[str, Any]] = {}
            for s in sweeps:
                for frac, v in s.items():
                    m = merged.setdefault(frac, {"total_counted": 0,
                                                "by_direction": {},
                                                "n_videos_zero": 0})
                    m["total_counted"] += v["total_counted"]
                    for d, n in v["by_direction"].items():
                        m["by_direction"][d] = m["by_direction"].get(d, 0) + n
                    if v["total_counted"] == 0:
                        m["n_videos_zero"] += 1
            p = plot_line_sweep(merged, OUTPUT.plots / "counting" / "line_sweep.png")
            if p:
                written.append(p)
                report.add("line_sweep", merged)
                logger.info("Line sweep: %s",
                            {k: v["total_counted"] for k, v in
                             sorted(merged.items(), key=lambda kv: float(kv[0]))})
        if written:
            report.add("plots", written)
            logger.info("Plots: %d written to %s", len(written),
                        OUTPUT.plots / "counting")

    zero_videos = [v["video"] for v in per_video
                   if v.get("n_tracks_seen", 0) > 0 and v["total_counted"] == 0]
    report.add_check(
        "frame_height_resolved",
        all(v.get("frame_height_source") in ("opencv", "forced",
                                             "bdd100k-constant")
            for v in per_video if "frame_height_source" in v),
        "frame heights from " + ", ".join(sorted(
            {v.get("frame_height_source", "?") for v in per_video})),
        severity="warning",
    )
    report.add_check(
        "all_videos_counted",
        not zero_videos,
        "every video produced crossings" if not zero_videos
        else f"{len(zero_videos)} video(s) counted ZERO: {zero_videos} - "
             f"check the timeline plots and the line sweep",
        severity="warning",
    )
    report.add_check(
        "counting_accuracy_computed",
        acc.get("n_videos_compared", 0) > 0,
        f"MAE={acc.get('MAE')} RMSE={acc.get('RMSE')} "
        f"MAPE={acc.get('MAPE_pct')}% accuracy={acc.get('counting_accuracy')} "
        f"over {acc.get('n_videos_compared')} video(s)"
        if acc.get("n_videos_compared") else
        "not computed - pass --gt and ensure 06_track.py ran with --gt",
        severity="warning",
    )
    report.add_check(
        "counting_accuracy_acceptable",
        (acc.get("counting_accuracy") or 0.0) >= 0.5,
        f"counting accuracy {acc.get('counting_accuracy')}"
        + ("" if (acc.get("counting_accuracy") or 0.0) >= 0.5
           else " - below 0.5, counts are not yet trustworthy as volumes"),
        severity="warning",
    )
    report.add_check(
        "count_not_line_sensitive",
        report.data.get("line_plateau_fraction", 0.0) >= 0.25
        and not report.data.get("line_position_has_zeros", False),
        f"plateau = {100.0 * report.data.get('line_plateau_fraction', 0.0):.0f}% "
        f"of swept positions within 20% of best; spread(max/min over non-zero) "
        f"= {report.data.get('line_position_spread_ratio')}; positions "
        f"counting zero = {len(entries) and sum(1 for v in totals.values() if v == 0)}"
        + ("" if report.data.get("line_plateau_fraction", 0.0) >= 0.25
           else " - too narrow to report as a traffic volume"),
        severity="warning",
    )
    report.add_check(
        "mot_inputs_found",
        bool(entries),
        f"{len(entries)} MOT file(s) in {mot_dir}",
    )
    report.add_check(
        "counts_computed",
        agg["total_counted"] > 0,
        f"{agg['total_counted']} crossing events over "
        f"{agg['n_tracks_counted']} tracks",
    )
    report.add_check(
        "no_duplicate_counts",
        all(v["total_counted"] == len({e["track_id"] for e in v.get("events", [])})
            for v in per_video),
        "each track contributes at most one crossing event",
    )
    report.add_check(
        "class_names_resolved",
        bool(class_map),
        "per-track classes loaded from the Phase 6 report" if class_map
        else "Phase 6 report not found - classes reported as 'unknown'",
        severity="warning",
    )
    report.add_check(
        "videos_labelled_by_metadata",
        lab["unlabelled"] == 0,
        f"{lab['labelled']}/{lab['labelled'] + lab['unlabelled']} video(s) "
        f"labelled with BDD100K timeofday/scene from "
        f"{', '.join(p.name for p in manifest_paths) or 'NO MANIFEST'}"
        + ("" if lab["unlabelled"] == 0 else
           f" - {lab['unlabelled']} unlabelled video(s) sit in the "
           f"'{UNLABELLED}' stratum and are excluded from day/night comparisons"),
        severity="warning",
    )
    report.add_check(
        "stratified_results_present",
        bool(video_meta) and "stratified" in report.data,
        f"counts stratified by {', '.join(strat_fields)}"
        if ("stratified" in report.data and video_meta) else
        "NOT stratified - no Phase 2 manifest found, so the day/night "
        "comparison (skill §10, §17E/§17F) cannot be made",
        severity="warning",
    )
    report.add_check(
        "both_directions_present",
        len(agg["by_direction"]) == 2,
        f"directions observed: {sorted(agg['by_direction'])}"
        + ("" if len(agg["by_direction"]) == 2
           else " - only one direction means the line is badly placed"),
        severity="warning",
    )

    out = report.save(OUTPUT.reports / "phase7_counting.json")
    logger.info("Report: %s", out)
    report.print_summary(logger)

    if report.all_checks_passed:
        log_pass(logger, "Phase 7 counting complete")
        return 0
    log_fail(logger, "Phase 7 counting failed its integrity checks")
    return 1


if __name__ == "__main__":
    sys.exit(main())
