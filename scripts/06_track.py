#!/usr/bin/env python3
"""
Phase 6: Multi-Object Tracking

Tracks vehicles across video frames using ultralytics model.track()
(BoT-SORT or ByteTrack trackers, persistent IDs). Produces:
    - MOT-format track file per video (predictions/tracking/<video>.txt)
    - Track summary JSON (per-track class/trajectory/direction)
    - Optional annotated video with boxes + track IDs (videos/)

Tracker choice (skill §12): BoT-SORT (default) and ByteTrack both ship with
ultralytics and integrate cleanly with the YOLO detector. DeepSORT is not
integrated with ultralytics and would require an extra ReID dependency.

Video source: robikscube/driving-video-with-object-tracking on Kaggle
(BDD100K MOT subset: 100 driving videos + box-track ground-truth parquet).
If that GT parquet is provided via --gt, MOTA/IDF1/IDSW/MOTP are computed
per video (skill §16).

Inputs:
    - Trained model checkpoint (best.pt)
    - Video file(s) or directory (--videos)
    - Optional GT parquet file/dir (--gt)

Outputs:
    - predictions/tracking/<video_name>.txt   (MOT format)
    - reports/phase6_tracking.json            (summary)
    - videos/<video_name>.mp4                 (with --save-video)

Run on Kaggle:
    python traffic_ai/scripts/06_track.py \
        --model experiments/<exp>/train/weights/best.pt \
        --videos /kaggle/input/driving-video-with-object-tracking \
        --gt /kaggle/input/driving-video-with-object-tracking \
        --max-videos 10 --save-video
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.config import CLASSES, OUTPUT, GT_LABEL_MAP
from src.log import (
    setup_logger, log_section, log_pass, log_fail, log_warn,
    Report, collect_environment,
)
from src.track import (
    TrackAccumulator, mot_lines, load_gt_parquet, clear_mot_metrics,
    describe_gt_parquet, calibrate_frame_stride, remap_gt_frames,
    refine_frame_map, remap_gt_frames_linear, gt_match_profile,
    false_positive_audit, FP_BANDS,
)
from src.strata import (
    METADATA_FIELDS, UNLABELLED, attach_metadata, coverage, load_video_metadata,
    mean_metric, resolve_manifest_paths, stratified_mot_metrics,
)

# BDD100K MOT GT category -> our taxonomy. Shared with Phase 7 via
# src/config.py (GT_LABEL_MAP) so counting accuracy compares like with like:
# if the two stages filtered GT differently, the accuracy figure would be
# meaningless.


def parse_args():
    parser = argparse.ArgumentParser(description="Track vehicles in videos")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint (best.pt)")
    parser.add_argument("--videos", type=str, required=True,
                        help="Video file or directory (searched recursively)")
    parser.add_argument("--gt", type=str, default=None,
                        help="Optional: parquet file/dir with BDD100K box-track "
                             "ground truth for tracking metrics (MOTA/IDF1/IDSW)")
    parser.add_argument("--tracker", type=str, default="botsort.yaml",
                        choices=["botsort.yaml", "bytetrack.yaml"],
                        help="Tracker config (ultralytics built-in)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Inference image size")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device")
    parser.add_argument("--max-videos", type=int, default=10,
                        help="Max videos to process (0 = all)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated video with track IDs")
    parser.add_argument("--all-frames", action="store_true",
                        help="Score every video frame instead of only the "
                             "frames carrying GT. BDD100K box-track GT is "
                             "annotated on a sparse keyframe subset, so this "
                             "counts detections on unannotated frames as "
                             "false positives and is NOT comparable to "
                             "MOTChallenge numbers. Diagnostics only.")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Phase 2 manifest JSON(s), comma-separated, used to "
                             "label each video with its BDD100K timeofday / "
                             "weather / scene so results can be stratified "
                             "(skill §3, §10, §17F). Default: "
                             "reports/manifest_val.json + manifest_train.json if "
                             "present. Videos that cannot be labelled are "
                             "reported in the 'unlabelled' stratum, never "
                             "dropped.")
    parser.add_argument("--fp-audit", action="store_true",
                        help="Classify every false positive by HOW it missed: "
                             "localisation (box slightly off), gt_sampling (the "
                             "object IS annotated in a neighbouring keyframe, the "
                             "~6x downsampled dump just missed this frame), or "
                             "unexplained (no GT overlap at all - a real "
                             "unannotated object or a hallucination). Needed "
                             "before claiming a night-robustness result: FP "
                             "explodes at night while FN stays flat, and only "
                             "this split says whether the model or the GT is at "
                             "fault. Writes reports/phase6_fp_audit.json.")
    parser.add_argument("--fp-audit-window", type=int, default=1,
                        help="Neighbourhood for --fp-audit, in ANNOTATED SAMPLES "
                             "(not video frames; the dump is ~6x downsampled). "
                             "1 = the immediately preceding/following annotated "
                             "frame. Default 1.")
    return parser.parse_args()


def collect_videos(videos_arg: str, max_videos: int) -> List[Path]:
    """Resolve --videos to a list of video files (sorted, capped).

    Direct search first, then a recursive walk (Kaggle inputs nest videos
    several levels deep, e.g. .../bdd100k_videos_train_00/bdd100k/videos/train).
    """
    p = Path(videos_arg)
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    if p.is_file():
        return [p]
    if p.is_dir():
        vids = sorted(f for f in p.iterdir()
                      if f.suffix.lower() in exts and not f.name.startswith("."))
        if not vids:
            vids = sorted(f for f in p.rglob("*")
                          if f.is_file() and f.suffix.lower() in exts
                          and not f.name.startswith("."))
        if max_videos > 0:
            vids = vids[:max_videos]
        return vids
    return []


def parse_track_results(result: Any) -> List[Dict[str, Any]]:
    """Extract (track_id, cls_name, conf, xyxy) rows from one ultralytics
    Results object. Returns [] when no boxes or IDs present."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []
    ids = [int(v) for v in boxes.id.tolist()]
    confs = [float(v) for v in boxes.conf.tolist()]
    clss = [int(v) for v in boxes.cls.tolist()]
    xyxy = boxes.xyxy.tolist()
    id2name = result.names if isinstance(result.names, dict) else \
        {i: n for i, n in enumerate(result.names)}
    rows = []
    for tid, ci, cf, bb in zip(ids, clss, confs, xyxy):
        rows.append({
            "track_id": int(tid),
            "cls": id2name.get(int(ci), str(ci)),
            "conf": float(cf),
            "xyxy": [float(v) for v in bb],
        })
    return rows


def track_one_video(model, video_path: Path, args,
                    logger: logging.Logger = None,
                    gt_files: List[Path] = None) -> Dict[str, Any]:
    """Track a single video; accumulate tracks and write MOT + summary.

    When gt_files (BDD100K box-track parquet shards) are provided, CLEAR MOT
    metrics (MOTA/IDF1/IDSW) are computed against the vehicle-class GT.
    """
    logger = logger or logging.getLogger("phase6")  # null-safe for tests
    log_section(logger, f"Tracking: {video_path.name}")
    acc = TrackAccumulator()
    n_frames = 0

    stream = model.track(
        source=str(video_path),
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        persist=True,
        verbose=False,
        save=args.save_video,
        project=str(OUTPUT.videos.parent),
        name="videos",
        exist_ok=True,
    )

    for result in stream:
        n_frames += 1
        for row in parse_track_results(result):
            acc.update(
                row["track_id"], n_frames - 1, row["cls"], row["conf"],
                tuple(row["xyxy"]),
            )

    records = acc.finalize()

    # Per-class and direction counts
    cls_counts = Counter(r["cls"] for r in records)
    dir_counts = Counter(r["direction"] for r in records)

    # Write MOT file
    mot_dir = OUTPUT.predictions / "tracking"
    mot_dir.mkdir(parents=True, exist_ok=True)
    mot_path = mot_dir / f"{video_path.stem}.txt"
    with open(mot_path, "w", encoding="utf-8") as f:
        f.write("\n".join(mot_lines(acc)) + "\n")

    logger.info(
        "  %s: %d frames, %d tracks, classes=%s",
        video_path.name, n_frames, len(records), dict(cls_counts),
    )

    summary = {
        "video": video_path.name,
        "frames": n_frames,
        "n_tracks": len(records),
        "tracks_by_class": dict(cls_counts),
        "tracks_by_direction": dict(dir_counts),
        "mot_file": str(mot_path),
        "tracks": records,
    }

    # Optional GT evaluation (CLEAR MOT: MOTA/IDF1/IDSW) - vehicle classes only
    if gt_files:
        gt_stats: Dict[str, Any] = {}
        stride = 1
        try:
            gt_rows = load_gt_parquet(gt_files, video_name=video_path.stem,
                                      stats=gt_stats)
            # keep only our taxonomy (case-insensitive); drop pedestrians etc.
            n_all = len(gt_rows)
            gt_vehicle = [r[:6] for r in gt_rows
                          if r[6].lower() in GT_LABEL_MAP]
            # Record which GT categories were excluded so the class-filter
            # decision is auditable rather than silent. The full dump also
            # contains pedestrian / rider / trailer / 'other vehicle' / train,
            # which have no matching detector class - counting them would
            # manufacture unmatchable false negatives.
            dropped_cats = sorted({r[6] for r in gt_rows
                                   if r[6].lower() not in GT_LABEL_MAP})
            pred_vehicle = [
                (f + 1, tid, x, y, x + w, y + h)
                for (f, tid, x, y, w, h, _c) in acc.mot_rows
            ]
            if gt_vehicle:
                # The GT dump is downsampled: frameIndex indexes the sampled
                # sequence, not the video. Search the frame map directly
                # (alpha + offset). The integer stride is always a candidate,
                # so this can never do worse than it.
                alpha, off, top = refine_frame_map(gt_vehicle, pred_vehicle)
                profile = gt_match_profile(gt_vehicle, pred_vehicle,
                                           alpha, off)
                stride, curve = calibrate_frame_stride(gt_vehicle,
                                                      pred_vehicle)
                int_hits = curve.get(stride, 0)
                lin_hits = top[0][2] if top else 0
                if lin_hits > int_hits:
                    gt_vehicle = remap_gt_frames_linear(gt_vehicle, alpha, off)
                    map_mode = "linear"
                else:
                    gt_vehicle = remap_gt_frames(gt_vehicle, stride)
                    map_mode = "integer"
                map_desc = (f"linear alpha={alpha} offset={off}" if map_mode == "linear"
                            else f"integer stride={stride}")
                ranked = sorted(curve.items(), key=lambda kv: -kv[1])[:4]
                logger.info("  GT/video frame map: %s | integer curve(top)=%s "
                            "| linear hits=%d vs integer hits=%d",
                            map_desc, ranked, lin_hits, int_hits)
                logger.info("  GT frame map candidates (alpha,offset,hits): %s",
                            [(a, o, h) for a, o, h in top])
                logger.info("  GT frame match profile: %s", profile)
                metrics = clear_mot_metrics(
                    gt_vehicle, pred_vehicle,
                    restrict_to_gt_frames=not getattr(args, "all_frames", False),
                    collect_fp=bool(getattr(args, "fp_audit", False)),
                )
                if getattr(args, "fp_audit", False):
                    # Same remapped rows the metrics were computed from, and the
                    # exact FP set behind the reported FP count.
                    audit = false_positive_audit(
                        gt_vehicle, metrics.get("fp_rows", []),
                        window=int(getattr(args, "fp_audit_window", 1)))
                    metrics.pop("fp_rows", None)
                    summary["fp_audit"] = audit
                metrics["gt_frame_map_mode"] = map_mode
                metrics["gt_frame_stride_applied"] = int(stride)
                metrics["gt_frame_alpha"] = round(float(alpha), 4)
                metrics["gt_frame_offset"] = int(off)
                metrics["gt_match_profile"] = profile
                metrics["gt_map_candidates"] = [
                    [a, o, h] for a, o, h in top]
                metrics["gt_stride_curve"] = {str(k): v for k, v in curve.items()}
                summary["gt_load_stats"] = dict(gt_stats)
                summary["gt_classes_kept"] = len(gt_vehicle)
                summary["gt_classes_dropped"] = n_all - len(gt_vehicle)
                summary["gt_categories_excluded"] = dropped_cats
                summary["mot_metrics"] = metrics
                logger.info(
                    "  GT: %d/%d rows kept (stride=%d, annotated %d frames, "
                    "skipped null=%d no_video=%d bad=%d), %d vehicle boxes "
                    "after class filter",
                    gt_stats.get("rows_kept", 0), n_all,
                    gt_stats.get("gt_frame_stride", 1),
                    gt_stats.get("gt_frames", 0),
                    gt_stats.get("rows_null_box", 0),
                    gt_stats.get("rows_no_video", 0),
                    gt_stats.get("rows_bad_box", 0),
                    len(gt_vehicle),
                )
                logger.info(
                    "  MOT metrics: MOTA=%.3f IDF1=%.3f MOTP=%.3f IDSW=%d "
                    "TP=%d FP=%d FN=%d (n_gt=%d over %d annotated frames, "
                    "%.1f%% of predicted frames, %d preds scored of %d, "
                    "map=%s)",
                    metrics["MOTA"], metrics["IDF1"], metrics["MOTP"],
                    metrics["IDSW"], metrics["TP"], metrics["FP"],
                    metrics["FN"], metrics["n_gt"], metrics["n_gt_frames"],
                    100.0 * (metrics["gt_frame_coverage"] or 0.0),
                    metrics["n_pred_scored"], metrics["n_pred_total"],
                    map_desc,
                )
                if dropped_cats:
                    log_warn(
                        logger,
                        f"  GT categories EXCLUDED (no matching detector "
                        f"class, excluded from n_gt so they cannot "
                        f"manufacture false negatives): {dropped_cats}",
                    )
                cov = metrics["gt_frame_coverage"] or 0.0
                ranked_hits = sorted(curve.values(), reverse=True)
                flat = len(ranked_hits) > 1 and ranked_hits[0] == ranked_hits[1]
                if map_mode == "linear":
                    log_warn(
                        logger,
                        f"  GT frame map is NOT an integer stride: "
                        f"{map_desc}. The GT dump is a downsampled label set, "
                        f"not frame-aligned to the video.",
                    )
                if flat:
                    log_warn(logger, "  stride curve is FLAT - calibration is "
                                     "not decisive, treat these metrics as "
                                     "unreliable")
                rate = profile.get("frame_match_rate", 0.0)
                if rate < 0.5:
                    log_warn(
                        logger,
                        f"  only {100.0 * rate:.1f}% of annotated GT frames "
                        f"produced ANY match (matches span GT index "
                        f"{profile.get('first_matched_idx')}-"
                        f"{profile.get('last_matched_idx')} of "
                        f"{profile.get('gt_frames')}). A linear frame map "
                        f"cannot align this clip - treat its metrics as "
                        f"suspect rather than as tracker quality.",
                    )
                if cov < 0.5:
                    log_warn(
                        logger,
                        f"  GT covers only {100.0 * cov:.1f}% of predicted "
                        f"frames. MOT metrics are computed on annotated frames "
                        f"ONLY. IDF1 measures re-identification across "
                        f"unannotated gaps and will read low for any "
                        f"motion-only tracker.",
                    )
            else:
                cats = sorted({r[6] for r in gt_rows})
                log_warn(logger, f"  No vehicle GT rows for '{video_path.stem}' "
                                 f"- {len(gt_rows)} GT rows, categories: {cats}")
        except ImportError as e:
            log_warn(logger, f"  GT evaluation skipped (missing dep): {e}")
        except Exception as e:
            log_warn(logger, f"  GT evaluation failed for '{video_path.stem}': {e}")
            summary["gt_error"] = str(e)

    return summary


def main() -> int:
    args = parse_args()
    OUTPUT.ensure_all()
    logger = setup_logger("phase6", log_file=OUTPUT.logs / "phase6.log")
    log_section(logger, "PHASE 6: MULTI-OBJECT TRACKING")

    model_path = Path(args.model)
    if not model_path.exists():
        log_fail(logger, f"Model not found: {model_path}")
        return 1

    videos = collect_videos(args.videos, args.max_videos)
    if not videos:
        log_fail(logger, f"No videos found at: {args.videos}")
        return 1

    logger.info("Model: %s", model_path)
    logger.info("Videos: %d (tracker=%s, imgsz=%d, conf=%.2f)",
                len(videos), args.tracker, args.imgsz, args.conf)

    # BDD100K scene metadata (timeofday / weather / scene) per video, taken from
    # the Phase 2 manifests. Stratifying tracking by time of day is the point of
    # the experiment (skill §10, §17F), and the labels come from the dataset
    # rather than from eyeballing frames - a hand-written scene list disagreed
    # with the metadata on 3 of the 5 evaluation clips.
    manifest_paths = resolve_manifest_paths(args.manifest, OUTPUT.reports)
    video_meta = load_video_metadata(manifest_paths)
    logger.info("Scene metadata: %d video(s) from %s",
                len(video_meta),
                ", ".join(p.name for p in manifest_paths) or "NO MANIFEST FOUND")

    # Ground-truth parquet shards (optional) for CLEAR MOT metrics
    gt_files = None
    gt_probe = None
    if args.gt:
        gp = Path(args.gt)
        gt_files = [gp] if gp.is_file() else sorted(gp.glob("*.parquet"))
        if gt_files:
            logger.info("GT parquet shards: %d (for MOTA/IDF1 evaluation)",
                        len(gt_files))
            # Structure probe: this dump keys rows by extracted frame image
            # ('<stem>-<0000001>.jpg'), and two dumps of the same dataset
            # disagree on which column holds the video key. Print the facts
            # once so an empty match is diagnosable without guesswork.
            try:
                gt_probe = describe_gt_parquet(gt_files,
                                               video_name=videos[0].stem)
                logger.info("GT schema probe (%s): %d rows",
                            gt_probe.get("shard"), gt_probe.get("n_rows", 0))
                logger.info("  resolved aliases: %s", gt_probe.get("resolved"))
                if gt_probe.get("resolved_error"):
                    log_warn(logger, f"  alias resolution: "
                                     f"{gt_probe['resolved_error']}")
                per = gt_probe.get("per_video") or {}
                for col, info in per.items():
                    logger.info("  MATCH via %-22s %s", col, info)
                if not per:
                    log_warn(logger, f"  no column matched "
                                     f"'{videos[0].stem}' - samples below")
                for col, vals in (gt_probe.get("samples") or {}).items():
                    logger.info("  sample %-24s %s", col, vals)
                for col, vc in (gt_probe.get("value_counts") or {}).items():
                    logger.info("  values %-24s %s", col, vc)
            except Exception as e:
                log_warn(logger, f"  GT schema probe failed: {e}")
        else:
            log_warn(logger, f"No .parquet files under {gp} - skipping MOT metrics")

    try:
        from ultralytics import YOLO
    except ImportError:
        log_fail(logger, "ultralytics not installed")
        return 1
    model = YOLO(str(model_path))

    report = Report(
        stage="06_tracking",
        description=f"Tracking with {args.tracker} on {len(videos)} video(s)",
    )
    report.add("environment", collect_environment())
    report.add("model_path", str(model_path))
    report.add("tracker", args.tracker)
    report.add("gt_source", args.gt)
    if gt_probe is not None:
        report.add("gt_schema_probe", gt_probe)
    report.add("track_config", {
        "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
        "max_videos": args.max_videos,
    })
    report.add("video_metadata", {
        "manifests": [str(p) for p in manifest_paths],
        "n_videos_labelled_available": len(video_meta),
        "fields": list(METADATA_FIELDS),
        "note": "BDD100K scene attributes keyed by video id; a BDD100K "
                "detection image id is its source video id, so the Phase 2 "
                "manifest labels the tracking videos directly.",
    })

    results, failed = [], []
    for vp in videos:
        try:
            results.append(track_one_video(model, vp, args, logger, gt_files))
        except Exception as e:
            log_warn(logger, f"Tracking failed for '{vp.name}': {e}")
            failed.append({"video": vp.name, "error": str(e)})

    if not results:
        log_fail(logger, "All videos failed - nothing to report")
        report.add("videos", failed)
        report.save(OUTPUT.reports / "phase6_tracking.json")
        return 1

    # Aggregate
    total_tracks = sum(r["n_tracks"] for r in results)
    total_cls = Counter()
    for r in results:
        total_cls.update(r["tracks_by_class"])

    # Label every video with its BDD100K scene attributes BEFORE the report is
    # assembled, so the per-video entries and every aggregate below are built
    # from the same labelled records.
    lab = attach_metadata(results, video_meta)
    logger.info("Scene metadata resolved for %d/%d video(s)%s",
                lab["labelled"], lab["labelled"] + lab["unlabelled"],
                "" if not lab["unlabelled"] else
                f"; {lab['unlabelled']} unlabelled -> 'unlabelled' stratum")
    for r in results:
        logger.info("  %-24s %-10s %-14s %s", r["video"], r.get("timeofday"),
                    r.get("scene"), r.get("weather"))

    report.add("videos", results)
    summary = {
        "n_videos_ok": len(results),
        "n_videos_failed": len(failed),
        "total_tracks": total_tracks,
        "tracks_by_class": dict(total_cls),
    }
    # Mean CLEAR MOT metrics across evaluated videos.
    # Rate-like keys are averaged; count keys are summed. Booleans/None are
    # excluded so a mixed set of videos cannot corrupt the aggregate.
    mot_evals = [r["mot_metrics"] for r in results if "mot_metrics" in r]
    if mot_evals:
        rate_keys = ("MOTA", "MOTP", "IDF1")
        count_keys = ("IDSW", "TP", "FP", "FN", "n_gt", "n_gt_frames",
                      "n_pred_scored")
        n = len(mot_evals)
        # .get, not []: this aggregation runs AFTER every video has been
        # tracked on the GPU, so a single metrics dict missing one count key
        # must not throw away the whole run. Absent counts contribute 0 and
        # absent rates are dropped from the mean (see _mean in src/strata.py).
        mean = {k: mean_metric(mot_evals, k) for k in rate_keys}
        mean.update({k: int(sum(int(m.get(k, 0) or 0) for m in mot_evals))
                     for k in count_keys})
        cov = [m.get("gt_frame_coverage") for m in mot_evals
               if m.get("gt_frame_coverage") is not None]
        mean["gt_frame_coverage"] = round(sum(cov) / len(cov), 4) if cov else None
        mean["n_videos_evaluated"] = n
        mean["n_videos_skipped"] = len(results) - n
        mean["restricted_to_gt_frames"] = bool(
            mot_evals[0]["restricted_to_gt_frames"])
        # Strides should agree across videos; a mixed set means the GT dump
        # does not share one sampling ratio and the scores are not comparable.
        strides_used = sorted({m.get("gt_frame_stride_applied", 1)
                               for m in mot_evals})
        mean["gt_frame_stride_applied"] = (strides_used[0]
                                           if len(strides_used) == 1
                                           else strides_used)
        summary["mot_metrics_mean"] = mean
        summary["n_videos_with_mot_metrics"] = n
    report.add("summary", summary)

    # ---- stratified by scene metadata (skill §3, §10, §17F) ---------------
    # Night/dusk performance is the research question, so it gets its own table
    # rather than being averaged into one overall number.
    if mot_evals:
        stratified = {f: stratified_mot_metrics(results, video_meta, f)
                      for f in ("timeofday", "scene")}
        report.add("stratified_mot_metrics", stratified)
        report.add("stratified_coverage",
                   {f: coverage(results, f, video_meta)
                    for f in ("timeofday", "scene")})
        log_section(logger, "TRACKING STRATIFIED BY SCENE METADATA")
        for field in ("timeofday", "scene"):
            table = stratified[field]
            logger.info("  by %s:", field)
            for value, e in table.items():
                if value.startswith("_"):
                    continue
                if not e.get("n_videos_evaluated"):
                    logger.info("    %-14s %d video(s), no GT metrics",
                                value, e["n_videos"])
                    continue
                logger.info(
                    "    %-14s n=%d  MOTA_pooled=%s  MOTA_mean=%s  "
                    "MOTP=%s  IDF1=%s  TP=%d FP=%d FN=%d IDSW=%d%s",
                    value, e["n_videos_evaluated"], e["MOTA_pooled"],
                    e["MOTA_mean"], e["MOTP_mean"], e["IDF1_mean"],
                    e["TP"], e["FP"], e["FN"], e["IDSW"],
                    "  [LOW SAMPLE - anecdote, not an effect]"
                    if e.get("low_sample") else "",
                )
        if lab["unlabelled"]:
            log_warn(logger,
                     f"  {lab['unlabelled']} video(s) carry no BDD100K scene "
                     f"metadata - they appear in the 'unlabelled' stratum and "
                     f"are EXCLUDED from every day/night comparison")
        if not video_meta:
            log_warn(logger, "  no Phase 2 manifest found - stratified results "
                             "are impossible. Re-run Phase 2 (or pass "
                             "--manifest) before quoting day/night numbers")

    # ---- false-positive forensics (--fp-audit) -----------------------------
    # Reported overall AND per stratum, because the whole question is whether
    # the night FP explosion is the model or the sparse GT.
    if getattr(args, "fp_audit", False):
        audited = [r for r in results if r.get("fp_audit")]
        if not audited:
            log_warn(logger, "  --fp-audit requested but no GT metrics were "
                             "computed - pass --gt as well")
        else:
            tot = {b: sum(r["fp_audit"]["bands"][b] for r in audited)
                   for b in FP_BANDS}
            n_fp = sum(tot.values())
            report.add("fp_audit_summary", {
                "n_videos_audited": len(audited),
                "n_fp": n_fp,
                "bands": tot,
                "band_fractions": {b: (round(tot[b] / n_fp, 4) if n_fp else None)
                                   for b in FP_BANDS},
                "window_annotated_samples": args.fp_audit_window,
                "band_definitions": audited[0]["fp_audit"]["band_definitions"],
                "per_video": {r["video"]: {k: v for k, v in r["fp_audit"].items()
                                           if k != "records"} for r in audited},
            })
            log_section(logger, "FALSE-POSITIVE FORENSICS")
            logger.info("  %d FP over %d video(s) | bands: %s", n_fp, len(audited),
                        {b: f"{tot[b]} ({100.0*tot[b]/n_fp:.0f}%)"
                         for b in FP_BANDS} if n_fp else {})
            for field in ("timeofday", "scene"):
                logger.info("  by %s:", field)
                groups: Dict[str, List[Dict[str, Any]]] = {}
                for r in audited:
                    groups.setdefault(str(r.get(field) or UNLABELLED),
                                      []).append(r)
                for value, recs in sorted(groups.items()):
                    b = {k: sum(x["fp_audit"]["bands"][k] for x in recs)
                         for k in FP_BANDS}
                    m = sum(b.values())
                    logger.info(
                        "    %-14s n=%d  FP=%-5d  %s",
                        value, len(recs), m,
                        {k: f"{100.0*b[k]/m:.0f}%" for k in FP_BANDS} if m
                        else "no FP")
            logger.info("  Interpretation: a large 'gt_sampling' share means the "
                        "model found objects the ~6x downsampled GT did not "
                        "sample on that frame - a metric artifact, not a model "
                        "defect. Only 'unexplained' needs pixel inspection.")

    log_section(logger, "SUMMARY")
    logger.info("Videos OK: %d / %d", len(results), len(videos))
    logger.info("Total tracks: %d", total_tracks)
    logger.info("Tracks by class: %s", dict(total_cls))
    if "mot_metrics_mean" in summary:
        mm_ = summary["mot_metrics_mean"]
        logger.info("MOT metrics (mean of %d video(s), %d skipped): "
                    "MOTA=%s IDF1=%s MOTP=%s IDSW=%s",
                    mm_["n_videos_evaluated"], mm_["n_videos_skipped"],
                    mm_["MOTA"], mm_["IDF1"], mm_["MOTP"], mm_["IDSW"])
        logger.info("  totals: TP=%d FP=%d FN=%d n_gt=%d over %d annotated "
                    "frames (%.1f%% GT frame coverage, restrict=%s)",
                    mm_["TP"], mm_["FP"], mm_["FN"], mm_["n_gt"],
                    mm_["n_gt_frames"], 100.0 * (mm_["gt_frame_coverage"] or 0.0),
                    mm_["restricted_to_gt_frames"])

    report.add_check(
        "tracks_generated",
        total_tracks > 0,
        f"{total_tracks} tracks across {len(results)} video(s)",
    )
    report.add_check(
        "mot_metrics_computed",
        bool(mot_evals),
        f"{len(mot_evals)}/{len(results)} video(s) evaluated against "
        f"BDD100K box-track GT",
        severity="warning",
    )
    # Stratified results are only meaningful if every video carries a dataset
    # label. An unlabelled video is a hole in the day/night table, so it is
    # surfaced rather than averaged away.
    report.add_check(
        "videos_labelled_by_metadata",
        lab["unlabelled"] == 0,
        f"{lab['labelled']}/{lab['labelled'] + lab['unlabelled']} video(s) "
        f"labelled with BDD100K timeofday/scene from "
        f"{', '.join(p.name for p in manifest_paths) or 'NO MANIFEST'}"
        + ("" if lab["unlabelled"] == 0 else
           f" - {lab['unlabelled']} unlabelled video(s) sit in the "
           f"'unlabelled' stratum and are excluded from day/night comparisons"),
        severity="warning",
    )
    report.add_check(
        "stratified_results_present",
        "stratified_mot_metrics" in report.data and bool(video_meta),
        f"MOT metrics stratified by timeofday, scene (skill §10/§17F)"
        if ("stratified_mot_metrics" in report.data and video_meta) else
        "NOT stratified - " + ("no manifest found, so videos carry no "
                               "timeofday/scene label"
                               if not video_meta else
                               "no GT metrics to stratify (re-run with --gt)"),
        severity="warning",
    )
    # A sparse GT label set silently produced MOTA ~ -6.8 before this check
    # existed: scoring 30 fps predictions against ~1 Hz GT turns every
    # detection on an unannotated frame into a false positive. Flag low
    # coverage explicitly so the number is never read as tracker quality.
    if mot_evals:
        cov = summary["mot_metrics_mean"]["gt_frame_coverage"] or 0.0
        report.add_check(
            "gt_frame_coverage_sufficient",
            cov >= 0.5,
            f"GT annotates {100.0 * cov:.1f}% of predicted frames; MOT "
            f"metrics computed on annotated frames only "
            f"(restricted={summary['mot_metrics_mean']['restricted_to_gt_frames']}). "
            f"Low coverage means MOTP/TP are the trustworthy numbers; IDF1 "
            f"measures re-identification across unannotated gaps.",
            severity="warning",
        )
    gt_errs = [r for r in results if r.get("gt_error")]
    report.add_check(
        "gt_eval_no_errors",
        not gt_errs,
        "all videos evaluated against GT" if not gt_errs
        else f"{len(gt_errs)} video(s) failed GT eval: "
             f"{[r['video'] for r in gt_errs]}",
        severity="warning",
    )
    if mot_evals:
        st = summary["mot_metrics_mean"]["gt_frame_stride_applied"]
        report.add_check(
            "gt_frame_stride_consistent",
            isinstance(st, int),
            f"GT->video frame stride {st} applied to all evaluated videos"
            if isinstance(st, int) else
            f"INCONSISTENT GT strides across videos: {st} - scores are not "
            f"comparable and must not be averaged",
            severity="warning",
        )
    report.add_check(
        "fp_audit_computed",
        (not getattr(args, "fp_audit", False)) or "fp_audit_summary" in report.data,
        (f"{(report.data.get('fp_audit_summary') or {}).get('n_fp', 0)} FP "
         f"classified into {len(FP_BANDS)} bands")
        if "fp_audit_summary" in report.data else
        "not computed - --fp-audit needs --gt so real FP rows exist",
        severity="warning",
    )
    out = report.save(OUTPUT.reports / "phase6_tracking.json")
    # The per-FP records are bulky (one row per false positive) and are only
    # needed for pixel inspection, so they get their own file rather than
    # inflating the main report.
    if "fp_audit_summary" in report.data:
        recs = Report(stage="06_fp_audit",
                      description="Per-false-positive records from --fp-audit")
        recs.add("summary", report.data["fp_audit_summary"])
        recs.add("records", {r["video"]: r["fp_audit"]["records"]
                             for r in results if r.get("fp_audit")})
        rec = recs.save(OUTPUT.reports / "phase6_fp_audit.json")
        logger.info("FP audit records: %s", rec)
    logger.info("Report: %s", out)

    if report.all_checks_passed:
        log_pass(logger, "Phase 6 tracking complete")
        return 0
    log_warn(logger, "Phase 6 completed with no tracks - review detection conf threshold")
    return 1


if __name__ == "__main__":
    sys.exit(main())
