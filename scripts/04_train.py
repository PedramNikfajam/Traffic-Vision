#!/usr/bin/env python3
"""
Phase 4: Vehicle Detection Model Training

Fine-tunes a YOLO model on the prepared BDD100K vehicle detection dataset.
Uses Ultralytics YOLO with centralized configuration.

Prerequisites:
    - Phase 2 (dataset preparation) completed
    - Phase 3 (dataset validation) passed
    - ultralytics package installed

Inputs:
    - data/vehicle_detection/dataset.yaml
    - Pretrained YOLO weights (auto-downloaded)

Outputs:
    - experiments/<experiment_id>/  (training artifacts)
    - reports/phase4_training.json

Run on Kaggle:
    python traffic_ai/scripts/04_train.py
    python traffic_ai/scripts/04_train.py --model yolov8s.pt --epochs 30 --name quick_test
    python traffic_ai/scripts/04_train.py --resume /path/to/last.pt

Usage notes:
    - Default model: yolov8m.pt (medium, good accuracy/speed for T4)
    - Single GPU recommended for reproducibility
    - Mixed precision enabled by default
    - Checkpoints saved every 5 epochs + best model
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

# Ensure project root is on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.config import OUTPUT, ExperimentConfig, TrainConfig, MODEL_PROGRESSION
from src.log import (
    setup_logger, log_section, log_pass, log_fail, log_warn,
    Report, collect_environment,
)


def parse_args():
    """Parse command-line arguments for training flexibility."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Train YOLO vehicle detector on BDD100K"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model weights (e.g., yolov8n.pt, yolov8m.pt)")
    parser.add_argument("--name", type=str, default="baseline",
                        help="Experiment name")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch size per GPU")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Input image size")
    parser.add_argument("--device", type=str, default=None,
                        help="CUDA device (0, 0,1, or cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--cache", type=str, default=None,
                        choices=["ram", "disk", "false"],
                        help="Data caching strategy")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(MODEL_PROGRESSION.keys()),
                        help="Use a preset model configuration (nano/small/medium/large/xlarge)")
    return parser.parse_args()


def build_train_config(args) -> TrainConfig:
    """Build training config from defaults + command-line overrides."""
    config = TrainConfig()
    
    # Apply preset if specified
    if args.preset:
        preset = MODEL_PROGRESSION[args.preset]
        config.model = preset["weights"]
        config.batch_size = preset["batch_size_t4"]
    
    # Apply individual overrides
    if args.model is not None:
        config.model = args.model
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch is not None:
        config.batch_size = args.batch
    if args.imgsz is not None:
        config.imgsz = args.imgsz
    if args.device is not None:
        config.device = args.device
    if args.cache is not None:
        config.cache = False if args.cache == "false" else args.cache
    if args.patience is not None:
        config.patience = args.patience
    
    return config


def resolve_experiment_dir(checkpoint: Path, output_experiments: Path) -> Path:
    """Find the experiment dir a checkpoint belongs to.

    Handles both layouts:
        <exp>/train/weights/last.pt  -> <exp>
        <exp>/weights/last.pt        -> <exp>
    Falls back to <output_experiments>/resumed_<timestamp> if the checkpoint
    is not inside any recognized experiment directory.

    Used by --resume: Ultralytics continues training inside the ORIGINAL
    experiment dir (args stored in the checkpoint), so reporting and the
    best.pt lookup must point there, not at a newly created dir.
    """
    checkpoint = checkpoint.resolve()
    for parent in checkpoint.parents:
        if parent.name == "weights":
            if parent.parent.name == "train":
                return parent.parent.parent  # <exp>/train/weights/<file> -> <exp>
            return parent.parent             # <exp>/weights/<file> -> <exp>
    return output_experiments / f"resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> int:
    args = parse_args()

    logger = setup_logger("phase4", log_file=OUTPUT.logs / "phase4.log")
    log_section(logger, "VEHICLE DETECTION MODEL TRAINING")
    logger.info("Started at: %s", datetime.now().isoformat())

    # Build config
    train_config = build_train_config(args)
    experiment_name = args.name

    logger.info("Model:     %s", train_config.model)
    logger.info("Epochs:    %d", train_config.epochs)
    logger.info("Batch:     %d", train_config.batch_size)
    logger.info("Image:     %d", train_config.imgsz)
    logger.info("Device:    %s", train_config.device)
    logger.info("AMP:       %s", train_config.amp)
    logger.info("Cache:     %s", train_config.cache)
    logger.info("Seed:      %d", train_config.seed)

    # Verify dataset exists
    dataset_yaml = OUTPUT.data / "dataset.yaml"
    if not dataset_yaml.exists():
        log_fail(logger, f"dataset.yaml not found at {dataset_yaml}")
        logger.error("Run Phase 2 and Phase 3 first.")
        return 1

    # Check Phase 3 validation passed
    validation_report = OUTPUT.reports / "phase3_validation.json"
    if validation_report.exists():
        with open(validation_report, "r") as f:
            val_data = json.load(f)
        if not val_data.get("_meta", {}).get("all_checks_passed", False):
            log_fail(logger, "Phase 3 validation did not pass. Fix dataset first.")
            return 1
        log_pass(logger, "Phase 3 validation passed")
    else:
        log_warn(logger, "Phase 3 validation report not found - proceeding anyway")

    # Resolve experiment directory.
    # With --resume, Ultralytics continues inside the ORIGINAL experiment dir
    # (stored in the checkpoint args), so anchor the whole run there.
    if args.resume:
        ckpt = Path(args.resume)
        if not ckpt.exists():
            log_fail(logger, f"Resume checkpoint not found: {ckpt}")
            return 1
        exp_dir = resolve_experiment_dir(ckpt, OUTPUT.experiments)
        exp_id = exp_dir.name
        logger.info("Resuming from: %s", ckpt)
        log_pass(logger, f"Anchored to original experiment dir: {exp_dir}")
    else:
        exp_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = OUTPUT.experiments / exp_id
        logger.info("Experiment: %s", exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Directory:  %s", exp_dir)

    # Save experiment config (keep the original on resume - the original
    # run's hyperparameters describe this experiment, not the resume args)
    exp_config = ExperimentConfig(
        name=experiment_name,
        description=f"YOLO vehicle detection with {train_config.model}",
        train=train_config,
    )
    config_path = exp_dir / "config.json"
    if args.resume and config_path.exists():
        logger.info("Keeping existing experiment config (resume): %s", config_path)
    else:
        exp_config.save(config_path)
    
    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        log_fail(logger, "ultralytics package not installed")
        logger.error("Install with: pip install ultralytics")
        return 1
    
    # Initialize model
    log_section(logger, "INITIALIZING MODEL")

    if args.resume:
        logger.info("Loading checkpoint: %s", args.resume)
        model = YOLO(args.resume)
    else:
        logger.info("Loading pretrained: %s", train_config.model)
        model = YOLO(train_config.model)
    
    # Build training arguments
    train_args = train_config.to_ultralytics_args()
    train_args["data"] = str(dataset_yaml)
    train_args["project"] = str(exp_dir)
    train_args["name"] = "train"
    train_args["exist_ok"] = True
    train_args["plots"] = True
    train_args["save_json"] = True
    
    if args.resume:
        train_args["resume"] = True
    
    # Log all training args
    logger.info("Training arguments:")
    for k, v in sorted(train_args.items()):
        logger.info("  %s: %s", k, v)
    
    # Train
    log_section(logger, "STARTING TRAINING")
    
    try:
        results = model.train(**train_args)
    except Exception as e:
        log_fail(logger, f"Training failed: {e}")
        logger.exception("Training error details:")
        return 1
    
    # Extract results
    log_section(logger, "TRAINING COMPLETE")
    
    # Find best model
    best_model_path = exp_dir / "train" / "weights" / "best.pt"
    last_model_path = exp_dir / "train" / "weights" / "last.pt"
    
    if best_model_path.exists():
        log_pass(logger, f"Best model: {best_model_path}")
    else:
        log_warn(logger, "Best model not found at expected path")
    
    if last_model_path.exists():
        logger.info("Last model: %s", last_model_path)
    
    # Generate training report
    report = Report(
        stage="04_training",
        description=f"YOLO training: {train_config.model}, {train_config.epochs} epochs",
    )
    report.add("environment", collect_environment())
    report.add("experiment_id", exp_id)
    report.add("experiment_dir", str(exp_dir))
    report.add("config", exp_config.to_dict())
    report.add("dataset_yaml", str(dataset_yaml))
    report.add("model", {
        "weights": "resumed_checkpoint" if args.resume else train_config.model,
        "resumed_from": args.resume,
        "best_checkpoint": str(best_model_path) if best_model_path.exists() else None,
        "last_checkpoint": str(last_model_path) if last_model_path.exists() else None,
    })
    
    # Parse training metrics if available
    try:
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            report.add("final_metrics", metrics)
            logger.info("Final metrics:")
            for k, v in metrics.items():
                logger.info("  %s: %s", k, v)
    except Exception as e:
        logger.warning("Could not extract training metrics: %s", e)

    # Training curves + per-epoch history from results.csv.
    # Schema-agnostic replacement for the upstream results.png, which crashes
    # on odd loss/metric column counts (ultralytics 8.4.136 "index 12 out of
    # bounds"). Lazy import keeps --help fast.
    try:
        from src.viz import plot_training_history
        curves_png, epoch_history, curve_summary = plot_training_history(
            exp_dir / "train" / "results.csv",
            out_path=exp_dir / "train" / "curves.png",
            logger=logger,
        )
        if curves_png is not None:
            report.add("training_curves", str(curves_png))
            report.add("curve_summary", curve_summary)
            report.add("epoch_history", epoch_history)
        report.add_check(
            "training_curves_generated",
            curves_png is not None,
            str(curves_png) if curves_png else "results.csv missing/unreadable",
            severity="warning",
        )
    except Exception as e:
        log_warn(logger, f"Training-curve plotting failed: {e}")

    report.add_check("training_completed", True, "Training finished without errors")
    report.add_check("best_model_saved", best_model_path.exists(),
                      str(best_model_path))
    
    report_path = report.save(OUTPUT.reports / "phase4_training.json")
    report.print_summary(logger)
    
    logger.info("Training report: %s", report_path)
    logger.info("To evaluate: python scripts/05_evaluate.py --model %s", best_model_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
