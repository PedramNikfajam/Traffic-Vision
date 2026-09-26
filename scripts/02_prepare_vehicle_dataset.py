#!/usr/bin/env python3
"""
Phase 2: BDD100K Vehicle Detection Dataset Preparation

Converts BDD100K annotations to YOLO format for vehicle detection training.
Uses centralized configuration from src.config.

Inputs:
    - BDD100K annotation JSON files (train + val)
    - BDD100K image directories (train + val)

Outputs:
    - data/vehicle_detection/train/images/   (symlinks)
    - data/vehicle_detection/train/labels/   (YOLO .txt files)
    - data/vehicle_detection/val/images/     (symlinks)
    - data/vehicle_detection/val/labels/     (YOLO .txt files)
    - data/vehicle_detection/dataset.yaml
    - reports/phase2_preparation.json
    - reports/manifest_train.json
    - reports/manifest_val.json

Run on Kaggle:
    python traffic_ai/scripts/02_prepare_vehicle_dataset.py
"""

import json
import os
import shutil
import sys
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Ensure project root is on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.config import CLASSES, DATASET_PATHS, OUTPUT, IMAGE_EXTENSIONS, SEED, ClassConfig
from src.dataset.convert import (
    AnnotationAccounting, ConvertedFrame,
    convert_frame, fingerprint_file, fingerprint_config,
)
from src.log import setup_logger, log_section, log_pass, log_fail, Report, collect_environment


# ============================================================
# IMAGE UTILITIES
# ============================================================

def get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
    """Get image dimensions without loading full image into memory."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def build_image_index(img_root: Path) -> Dict[str, Path]:
    """Build filename -> path index. Detects duplicates.
    
    Returns dict mapping filename to its first found path.
    Raises ValueError if non-identical duplicates are found.
    """
    index: Dict[str, List[Path]] = {}
    for root, _, files in os.walk(img_root):
        for fname in files:
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                fpath = Path(root) / fname
                if fname not in index:
                    index[fname] = [fpath]
                else:
                    index[fname].append(fpath)
    
    # Check for non-identical duplicates
    duplicates = {k: v for k, v in index.items() if len(v) > 1}
    if duplicates:
        for name, paths in duplicates.items():
            # Verify byte-identical using MD5
            hashes = set()
            for p in paths:
                h = hashlib.md5(p.read_bytes()).hexdigest()
                hashes.add(h)
            if len(hashes) > 1:
                raise ValueError(
                    f"Non-identical duplicate images found for '{name}': "
                    f"{[str(p) for p in paths]}"
                )
    
    # Return single-path index
    return {k: v[0] for k, v in index.items()}


# ============================================================
# ANNOTATION LOADING
# ============================================================

def load_bdd100k_annotations(ann_path: Path) -> List[dict]:
    """Load BDD100K annotation JSON (list-of-frames format)."""
    with open(ann_path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "frames" in data:
        return data["frames"]
    else:
        raise ValueError(f"Unknown annotation format in {ann_path}")


# ============================================================
# SPLIT PROCESSING
# ============================================================

def process_split(
    ann_path: Path,
    img_root: Path,
    split: str,
    class_config: ClassConfig,
    logger,
) -> Tuple[List[ConvertedFrame], AnnotationAccounting, Dict[str, Any]]:
    """Process a single split (train or val).
    
    Returns:
        (converted_frames, accounting, split_metadata)
    """
    log_section(logger, f"PROCESSING {split.upper()} SPLIT")
    
    # Load annotations
    logger.info("Loading annotations from %s", ann_path)
    frames = load_bdd100k_annotations(ann_path)
    logger.info("Loaded %d frames", len(frames))
    
    # Build image index
    logger.info("Building image index for %s...", img_root)
    img_index = build_image_index(img_root)
    logger.info("Found %d unique images", len(img_index))
    
    # Process frames
    accounting = AnnotationAccounting()
    converted: List[ConvertedFrame] = []
    seen_names: Set[str] = set()
    
    missing_images = 0
    duplicate_frames = 0
    unreadable_images = 0
    
    for frame in sorted(frames, key=lambda f: f.get("name", "")):  # Deterministic order
        frame_name = frame.get("name", "")
        if not frame_name:
            continue
        
        # Duplicate frame check
        if frame_name in seen_names:
            duplicate_frames += 1
            continue
        seen_names.add(frame_name)
        
        # Find image
        img_path = img_index.get(frame_name)
        if img_path is None:
            missing_images += 1
            # Still count annotations for accounting
            for label in frame.get("labels", []) or []:
                accounting.source_total += 1
                cat = label.get("category", "")
                has_box = isinstance(label.get("box2d"), dict) and len(label.get("box2d", {})) > 0
                disposition = class_config.classify_annotation(cat, has_box)
                if disposition == "non_bbox_class":
                    accounting.non_bbox_class += 1
                elif disposition == "excluded_class":
                    accounting.excluded_class += 1
                    accounting.excluded_by_class[cat] = accounting.excluded_by_class.get(cat, 0) + 1
                elif disposition == "target_bbox":
                    accounting.invalid_bbox += 1
                    accounting.invalid_reasons["missing_image"] = accounting.invalid_reasons.get("missing_image", 0) + 1
                elif disposition == "target_no_bbox":
                    accounting.target_no_bbox += 1
                elif disposition == "unknown_class":
                    accounting.unknown_class += 1
            continue
        
        # Get image dimensions
        dims = get_image_dimensions(img_path)
        if dims is None:
            unreadable_images += 1
            continue
        img_w, img_h = dims
        
        # Convert frame
        cf = convert_frame(
            frame=frame,
            img_w=img_w,
            img_h=img_h,
            split=split,
            source_image_path=str(img_path),
            class_config=class_config,
            accounting=accounting,
        )
        converted.append(cf)
    
    # Sort deterministically
    converted.sort(key=lambda c: c.frame_name)
    
    # Compute statistics
    frames_with_objects = sum(1 for c in converted if c.has_objects)
    empty_frames = len(converted) - frames_with_objects
    
    logger.info("Converted: %d frames (%d with objects, %d empty)",
                len(converted), frames_with_objects, empty_frames)
    logger.info("Missing images: %d, Duplicate frames: %d, Unreadable: %d",
                missing_images, duplicate_frames, unreadable_images)
    
    # Per-class summary
    class_counter = Counter()
    image_class_counter = Counter()
    for cf in converted:
        classes_seen = set()
        for l in cf.labels:
            class_counter[l.class_name] += 1
            classes_seen.add(l.class_name)
        for c in classes_seen:
            image_class_counter[c] += 1
    
    for cls in class_config.target_classes:
        obj_count = class_counter.get(cls, 0)
        img_count = image_class_counter.get(cls, 0)
        logger.info("  %s: %d objects in %d images", cls, obj_count, img_count)
    
    # Verify accounting
    passed, msg = accounting.verify()
    if passed:
        log_pass(logger, msg)
    else:
        log_fail(logger, msg)
        raise RuntimeError(f"Annotation accounting failed for {split}: {msg}")
    
    split_metadata = {
        "source_frames": len(frames),
        "unique_frames": len(seen_names),
        "converted_frames": len(converted),
        "frames_with_objects": frames_with_objects,
        "empty_frames": empty_frames,
        "missing_images": missing_images,
        "duplicate_frames": duplicate_frames,
        "unreadable_images": unreadable_images,
        "source_images_found": len(img_index),
        "object_counts_by_class": dict(class_counter),
        "image_counts_by_class": dict(image_class_counter),
    }
    
    return converted, accounting, split_metadata


# ============================================================
# OUTPUT WRITING
# ============================================================

def write_yolo_dataset(
    train_frames: List[ConvertedFrame],
    val_frames: List[ConvertedFrame],
    output_dir: Path,
    class_config: ClassConfig,
    logger,
) -> Dict[str, int]:
    """Write YOLO format dataset (labels + image symlinks + dataset.yaml).
    
    Returns counts dict.
    """
    log_section(logger, "WRITING YOLO DATASET")
    
    counts = {}
    
    for split, frames in [("train", train_frames), ("val", val_frames)]:
        labels_dir = output_dir / split / "labels"
        images_dir = output_dir / split / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        written = 0
        for cf in frames:
            # Write label file
            stem = Path(cf.frame_name).stem
            label_path = labels_dir / f"{stem}.txt"
            with open(label_path, "w") as f:
                for label in cf.labels:
                    f.write(label.to_yolo_line() + "\n")
            
            # Create symlink to source image
            img_dest = images_dir / cf.frame_name
            if img_dest.is_symlink() or img_dest.exists():
                img_dest.unlink()
            try:
                img_dest.symlink_to(cf.source_image_path)
            except OSError:
                # Fallback: copy on Windows or if symlinks not supported
                shutil.copy2(cf.source_image_path, img_dest)
            
            written += 1
        
        counts[f"{split}_labels"] = written
        counts[f"{split}_images"] = written
        logger.info("Wrote %d label files and image links for %s", written, split)
    
    # Write dataset.yaml (Ultralytics format)
    yaml_content = (
        f"# BDD100K Vehicle Detection Dataset\n"
        f"# Generated by Phase 2 pipeline at {datetime.now().isoformat()}\n"
        f"# DO NOT EDIT - regenerate with 02_prepare_vehicle_dataset.py\n"
        f"\n"
        f"path: {output_dir}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"nc: {class_config.num_classes}\n"
        f"names: {list(class_config.target_classes)}\n"
    )
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    logger.info("Wrote dataset.yaml: %s", yaml_path)
    
    return counts


def write_manifests(
    frames: List[ConvertedFrame],
    manifest_path: Path,
    logger,
) -> None:
    """Write JSON manifest for provenance tracking."""
    manifest = [cf.to_manifest_dict() for cf in frames]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    logger.info("Wrote manifest: %s (%d entries)", manifest_path, len(manifest))


# ============================================================
# INTEGRITY CHECKS
# ============================================================

def check_split_leakage(
    train_frames: List[ConvertedFrame],
    val_frames: List[ConvertedFrame],
    logger,
) -> Tuple[bool, int]:
    """Check for train/val split leakage by filename.
    
    Returns:
        (passed, overlap_count)
    """
    train_names = {cf.frame_name for cf in train_frames}
    val_names = {cf.frame_name for cf in val_frames}
    overlap = train_names & val_names
    
    if overlap:
        log_fail(logger, f"SPLIT LEAKAGE: {len(overlap)} filenames in both train and val")
        for name in sorted(overlap)[:10]:
            logger.error("  Leaked: %s", name)
        return False, len(overlap)
    
    log_pass(logger, f"No split leakage: train={len(train_names)}, val={len(val_names)}, overlap=0")
    return True, 0


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> int:
    # Setup
    logger = setup_logger("phase2", log_file=OUTPUT.logs / "phase2.log")
    log_section(logger, "BDD100K VEHICLE DETECTION DATASET PREPARATION")
    logger.info("Started at: %s", datetime.now().isoformat())
    
    # Verify source paths
    errors = DATASET_PATHS.verify()
    if errors:
        for e in errors:
            logger.error(e)
        logger.error("Source dataset not found. This script must run on Kaggle.")
        return 1
    log_pass(logger, "All source paths verified")
    
    # Clean previous outputs (idempotent)
    data_dir = OUTPUT.data
    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            p = data_dir / split / subdir
            if p.exists():
                shutil.rmtree(p)
                logger.info("Cleaned: %s", p)
    
    # Create output dirs
    OUTPUT.ensure_all()
    
    # Initialize report
    report = Report(
        stage="02_dataset_preparation",
        description="Convert BDD100K annotations to YOLO format for vehicle detection",
    )
    report.add("environment", collect_environment())
    report.add("config", {
        "target_classes": list(CLASSES.target_classes),
        "class_to_id": CLASSES.class_to_id,
        "excluded_categories": list(CLASSES.excluded_categories),
        "non_bbox_categories": list(CLASSES.non_bbox_categories),
        "seed": SEED,
    })
    
    # Fingerprint source data
    logger.info("Computing source data fingerprints...")
    fingerprints = {
        "train_annotations": fingerprint_file(Path(DATASET_PATHS.train_annotations)),
        "val_annotations": fingerprint_file(Path(DATASET_PATHS.val_annotations)),
        "class_config": fingerprint_config(CLASSES),
    }
    report.add("fingerprints", fingerprints)
    logger.info("Fingerprints: %s", fingerprints)
    
    # Process splits
    train_frames, train_accounting, train_meta = process_split(
        ann_path=Path(DATASET_PATHS.train_annotations),
        img_root=Path(DATASET_PATHS.train_images),
        split="train",
        class_config=CLASSES,
        logger=logger,
    )
    
    val_frames, val_accounting, val_meta = process_split(
        ann_path=Path(DATASET_PATHS.val_annotations),
        img_root=Path(DATASET_PATHS.val_images),
        split="val",
        class_config=CLASSES,
        logger=logger,
    )
    
    # Cross-split leakage check
    leakage_passed, leakage_count = check_split_leakage(train_frames, val_frames, logger)
    report.add_check("split_leakage", leakage_passed, 
                      f"{leakage_count} overlapping filenames")
    if not leakage_passed:
        logger.error("CRITICAL: Split leakage detected. Aborting.")
        return 1
    
    # Accounting checks
    train_ok, train_msg = train_accounting.verify()
    val_ok, val_msg = val_accounting.verify()
    report.add_check("train_accounting", train_ok, train_msg)
    report.add_check("val_accounting", val_ok, val_msg)
    
    # Write YOLO dataset
    write_counts = write_yolo_dataset(
        train_frames, val_frames, data_dir, CLASSES, logger
    )
    
    # Write manifests
    write_manifests(train_frames, OUTPUT.reports / "manifest_train.json", logger)
    write_manifests(val_frames, OUTPUT.reports / "manifest_val.json", logger)
    
    # Metadata extraction
    timeofday_counter = Counter()
    weather_counter = Counter()
    scene_counter = Counter()
    for cf in train_frames + val_frames:
        tod = cf.frame_attributes.get("timeofday", "")
        if tod:
            timeofday_counter[tod] += 1
        w = cf.frame_attributes.get("weather", "")
        if w:
            weather_counter[w] += 1
        s = cf.frame_attributes.get("scene", "")
        if s:
            scene_counter[s] += 1
    
    # Add statistics to report
    report.add("source_paths", {
        "train_annotations": DATASET_PATHS.train_annotations,
        "val_annotations": DATASET_PATHS.val_annotations,
        "train_images": DATASET_PATHS.train_images,
        "val_images": DATASET_PATHS.val_images,
    })
    report.add("output_paths", {
        "data_root": str(data_dir),
        "dataset_yaml": str(data_dir / "dataset.yaml"),
    })
    report.add("train_statistics", train_meta)
    report.add("val_statistics", val_meta)
    report.add("train_accounting", train_accounting.to_dict())
    report.add("val_accounting", val_accounting.to_dict())
    
    # Combined statistics
    combined_objects = train_accounting.target_bbox + val_accounting.target_bbox
    combined_by_class = Counter(train_accounting.target_by_class)
    combined_by_class += Counter(val_accounting.target_by_class)
    
    report.add("combined_statistics", {
        "total_train_frames": len(train_frames),
        "total_val_frames": len(val_frames),
        "total_objects": combined_objects,
        "objects_by_class": dict(combined_by_class),
        "class_percentages": {
            c: round(n / combined_objects * 100, 2) if combined_objects > 0 else 0
            for c, n in combined_by_class.items()
        },
    })
    
    report.add("metadata_distribution", {
        "timeofday": dict(timeofday_counter),
        "weather": dict(weather_counter),
        "scene": dict(scene_counter),
    })
    
    # Final summary checks
    report.add_check("train_has_objects", 
                      train_meta["frames_with_objects"] > 0,
                      f"{train_meta['frames_with_objects']} train frames with objects")
    report.add_check("val_has_objects",
                      val_meta["frames_with_objects"] > 0,
                      f"{val_meta['frames_with_objects']} val frames with objects")
    report.add_check("all_classes_present",
                      len(combined_by_class) == CLASSES.num_classes,
                      f"Found {len(combined_by_class)}/{CLASSES.num_classes} classes")
    
    # Save report
    report_path = report.save(OUTPUT.reports / "phase2_preparation.json")
    report.print_summary(logger)
    
    # Final output
    log_section(logger, "PHASE 2 COMPLETE")
    logger.info("Dataset:      %s", data_dir)
    logger.info("dataset.yaml: %s", data_dir / "dataset.yaml")
    logger.info("Report:       %s", report_path)
    logger.info("Train frames: %d (%d with objects)", 
                len(train_frames), train_meta["frames_with_objects"])
    logger.info("Val frames:   %d (%d with objects)",
                len(val_frames), val_meta["frames_with_objects"])
    logger.info("Total objects: %d", combined_objects)
    
    if not report.all_checks_passed:
        logger.error("Some checks failed - review the report")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
