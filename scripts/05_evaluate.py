#!/usr/bin/env python3
"""
Phase 5: Model Evaluation and Error Analysis

Evaluates a trained YOLO model on the validation set.
Produces per-class metrics, day/night analysis, and error analysis.

Independent from training code - can evaluate any checkpoint.

Inputs:
    - Trained model checkpoint (best.pt)
    - data/vehicle_detection/dataset.yaml
    - reports/manifest_val.json (for metadata-stratified evaluation)

Outputs:
    - reports/phase5_evaluation.json
    - plots/ (PR curves, confusion matrix, etc.)

Run on Kaggle:
    python traffic_ai/scripts/05_evaluate.py --model experiments/<exp>/train/weights/best.pt
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Ensure project root is on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.config import CLASSES, OUTPUT
from src.log import (
    setup_logger, log_section, log_fail, log_warn,
    Report, collect_environment,
)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate YOLO vehicle detector")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint (best.pt)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset.yaml (default: auto-detect)")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for mAP")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for evaluation")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save prediction visualizations")
    return parser.parse_args()


def load_val_metadata(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    """Load validation manifest for metadata-stratified evaluation.
    
    Returns dict: frame_name -> {timeofday, weather, scene}
    """
    if not manifest_path.exists():
        return {}
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    metadata = {}
    for entry in manifest:
        name = entry.get("frame_name", "")
        if name:
            metadata[name] = {
                "timeofday": entry.get("timeofday", ""),
                "weather": entry.get("weather", ""),
                "scene": entry.get("scene", ""),
            }
    
    return metadata


def stratify_results_by_metadata(
    val_metadata: Dict[str, Dict[str, str]],
    logger,
) -> Dict[str, List[str]]:
    """Group validation images by metadata attributes for stratified eval.

    Strata cover ALL BDD100K attribute values observed in Phase 1
    (timeofday: daytime/night/dawn-dusk/undefined; weather: clear/overcast/
    partly cloudy/rainy/snowy/foggy/undefined). Unknown non-empty values are
    kept as their own group so nothing is silently dropped from the
    day/night/weather analysis. All values come from the source annotation
    (frame["attributes"]) via the Phase 2 manifests.

    Returns:
        Dict mapping group_name -> list of frame names.
        Example: {"daytime": [...], "night": [...], "rainy": [...]}
    """
    groups = defaultdict(list)

    timeofday_groups = {
        "daytime": "daytime",
        "day": "daytime",
        "night": "night",
        "dawn/dusk": "dawn_dusk",
        "dawn": "dawn_dusk",
        "dusk": "dawn_dusk",
        "undefined": "timeofday_undefined",
    }
    weather_groups = {
        "clear": "clear",
        "overcast": "overcast",
        "partly cloudy": "partly_cloudy",
        "rainy": "rainy",
        "snowy": "snowy",
        "foggy": "foggy",
        "undefined": "weather_undefined",
    }

    for name, meta in val_metadata.items():
        tod = meta.get("timeofday", "").lower()
        weather = meta.get("weather", "").lower()

        tod_group = timeofday_groups.get(tod)
        if tod_group:
            groups[tod_group].append(name)
        elif tod:  # unexpected non-empty value - keep it visible
            groups[f"timeofday_{tod.replace('/', '_').replace(' ', '_')}"].append(name)

        weather_group = weather_groups.get(weather)
        if weather_group:
            groups[weather_group].append(name)
        elif weather:
            groups[f"weather_{weather.replace(' ', '_')}"].append(name)

    for group, names in sorted(groups.items()):
        logger.info("  Metadata group '%s': %d images", group, len(names))

    return dict(groups)


def run_stratified_eval(
    model,
    groups: Dict[str, List[str]],
    dataset_yaml: Path,
    args,
    logger: logging.Logger = None,
    min_images: int = 50,
) -> Dict[str, Dict[str, float]]:
    """Run model.val() per metadata group (day/night/weather).

    Builds one mini dataset.yaml per group whose val split lists only that
    group's images. Labels resolve by Ultralytics' img2label_paths convention
    (images -> labels, '/images/' -> '/labels/'), which Phase 2 already
    satisfies. Only groups with >= min_images are evaluated; small groups
    (e.g. foggy n=13) would give statistically meaningless AP.

    Returns: group -> {mAP50, mAP50_95, precision, recall, images, instances}
    """
    logger = logger or logging.getLogger("phase5")  # null-safe for tests
    if not groups:
        return {}

    try:
        import yaml as pyyaml
    except ImportError:
        log_warn(logger, "pyyaml not available - skipping stratified evaluation")
        return {}

    with open(dataset_yaml, "r") as f:
        base = pyyaml.safe_load(f)

    data_root = Path(base.get("path", dataset_yaml.parent))
    val_rel = Path(base.get("val", "val/images"))

    stratified = {}
    for gname, names in sorted(groups.items()):
        if len(names) < min_images:
            logger.info("  Skip stratified eval '%s': only %d images (< %d)",
                        gname, len(names), min_images)
            continue

        group_dir = OUTPUT.data / "eval_splits" / gname
        img_list = group_dir / "val.txt"
        img_list.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics accepts a .txt of image paths as the val split; labels
        # are found by the standard '/images/' -> '/labels/' path rewrite.
        val_images_dir = data_root / val_rel.parent
        with open(img_list, "w") as f:
            for n in names:
                f.write(str(val_images_dir / n) + "\n")

        group_yaml = group_dir / "dataset.yaml"
        with open(group_yaml, "w") as f:
            pyyaml.safe_dump({
                "path": str(data_root),
                "train": str(base.get("train", "")),
                "val": str(img_list.resolve()),
                "names": base.get("names"),
                "nc": base.get("nc"),
            }, f)

        try:
            m = model.val(
                data=str(group_yaml),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                plots=False,
                save_json=False,
                verbose=False,
            )
            rd = getattr(m, "results_dict", {})
            entry = {
                "images": len(names),
                "mAP50": round(float(rd.get("metrics/mAP50(B)", 0)), 4),
                "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0)), 4),
                "precision": round(float(rd.get("metrics/precision(B)", 0)), 4),
                "recall": round(float(rd.get("metrics/recall(B)", 0)), 4),
            }
            stratified[gname] = entry
            logger.info("  %-18s n=%-5d mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f",
                        gname, entry["images"], entry["mAP50"], entry["mAP50_95"],
                        entry["precision"], entry["recall"])
        except Exception as e:
            log_warn(logger, f"Stratified eval failed for '{gname}': {e}")

    return stratified


def main() -> int:
    args = parse_args()

    logger = setup_logger("phase5", log_file=OUTPUT.logs / "phase5.log")
    log_section(logger, "MODEL EVALUATION AND ERROR ANALYSIS")
    logger.info("Started at: %s", datetime.now().isoformat())

    # Verify model exists
    model_path = Path(args.model)
    if not model_path.exists():
        log_fail(logger, f"Model not found: {model_path}")
        return 1
    logger.info("Model: %s", model_path)

    # Dataset
    dataset_yaml = Path(args.data) if args.data else OUTPUT.data / "dataset.yaml"
    if not dataset_yaml.exists():
        log_fail(logger, f"dataset.yaml not found: {dataset_yaml}")
        return 1
    logger.info("Dataset: %s", dataset_yaml)

    # Load validation metadata for stratified analysis
    val_metadata = load_val_metadata(OUTPUT.reports / "manifest_val.json")
    if val_metadata:
        logger.info("Loaded metadata for %d validation images", len(val_metadata))
        groups = stratify_results_by_metadata(val_metadata, logger)
    else:
        logger.info("No validation manifest found - skipping stratified analysis")
        groups = {}

    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        log_fail(logger, "ultralytics package not installed")
        return 1

    # Load model
    log_section(logger, "RUNNING VALIDATION")
    model = YOLO(str(model_path))

    # Run standard validation
    metrics = model.val(
        data=str(dataset_yaml),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=True,
        project=str(OUTPUT.plots),
        name="evaluation",
        exist_ok=True,
    )

    # Extract metrics
    log_section(logger, "EVALUATION RESULTS")

    results_dict = {}
    if hasattr(metrics, 'results_dict'):
        results_dict = metrics.results_dict

    # mAP metrics
    map50 = results_dict.get("metrics/mAP50(B)", None)
    map50_95 = results_dict.get("metrics/mAP50-95(B)", None)
    precision = results_dict.get("metrics/precision(B)", None)
    recall = results_dict.get("metrics/recall(B)", None)

    logger.info("mAP@50:     %.4f", map50 if map50 is not None else 0)
    logger.info("mAP@50-95:  %.4f", map50_95 if map50_95 is not None else 0)
    logger.info("Precision:  %.4f", precision if precision is not None else 0)
    logger.info("Recall:     %.4f", recall if recall is not None else 0)

    # Per-class AP
    per_class_ap = {}
    if hasattr(metrics, 'box'):
        box_metrics = metrics.box
        if hasattr(box_metrics, 'ap50') and box_metrics.ap50 is not None:
            for i, cls_name in enumerate(CLASSES.target_classes):
                if i < len(box_metrics.ap50):
                    ap50_val = float(box_metrics.ap50[i])
                    per_class_ap[cls_name] = {
                        "AP50": round(ap50_val, 4),
                    }
                    logger.info("  %s AP@50: %.4f", cls_name, ap50_val)

    # ------------------------------------------------------------
    # Stratified (day/night/weather) evaluation: one val() run per
    # metadata group using a per-group image list.
    # ------------------------------------------------------------
    stratified = run_stratified_eval(model, groups, dataset_yaml, args, logger)

    # Build report
    report = Report(
        stage="05_evaluation",
        description=f"Evaluation of {model_path.name}",
    )
    report.add("environment", collect_environment())
    report.add("model_path", str(model_path))
    report.add("dataset_yaml", str(dataset_yaml))
    report.add("eval_config", {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
    })
    report.add("overall_metrics", {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": precision,
        "recall": recall,
    })
    report.add("per_class_ap", per_class_ap)
    report.add("metadata_groups", {k: len(v) for k, v in groups.items()})
    report.add("stratified_metrics", stratified)
    report.add("all_results", results_dict)
    
    report.add_check("evaluation_completed", True, "Evaluation finished")
    report.add_check("map50_reasonable",
                      map50 is not None and map50 > 0.1,
                      f"mAP@50 = {map50}" if map50 else "mAP@50 not available",
                      severity="warning")
    if stratified:
        day = stratified.get("daytime", {}).get("mAP50")
        night = stratified.get("night", {}).get("mAP50")
        if day and night:
            delta = day - night
            report.add_check("day_night_gap_reported", True,
                             f"day mAP50={day:.4f} night mAP50={night:.4f} "
                             f"gap={delta:+.4f}")
            logger.info("Day/Night mAP50 gap: %+.4f (day %.4f vs night %.4f)",
                        delta, day, night)
        else:
            report.add_check("stratified_eval_completed", True,
                             f"{len(stratified)} groups evaluated")
    
    report_path = report.save(OUTPUT.reports / "phase5_evaluation.json")
    report.print_summary(logger)
    
    log_section(logger, "EVALUATION COMPLETE")
    logger.info("Report:     %s", report_path)
    logger.info("Plots:      %s", OUTPUT.plots / "evaluation")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
