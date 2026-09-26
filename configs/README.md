# Configuration

All project configuration lives in a single module:

```
src/config.py
```

This is the **single source of truth** for:

- Dataset paths (`DatasetPaths`) - BDD100K source locations and expected counts
- Class taxonomy (`ClassConfig`) - target classes, BDD100K category mapping, exclusions
- Output paths (`OutputPaths`) - data, reports, experiments, weights, plots, logs
- Training hyperparameters (`TrainConfig`) - model, image size, batch, LR, augmentation
- Evaluation settings (`EvalConfig`) - confidence/IoU thresholds
- Experiment metadata (`ExperimentConfig`) - serialization for reproducibility
- Model progression presets (`MODEL_PROGRESSION`) - nano → small → medium → large → xlarge

## Why not YAML files?

Earlier versions had `classes.yaml`, `detector.yaml`, etc., but **no script ever loaded
them**, so they silently drifted out of sync with the code (they still claimed
`motorcycle`/`bicycle` while the real BDD100K categories are `bike`/`motor`).
A Python module cannot drift: if you import it, you get exactly what runs.

## Modifying configuration

Edit `src/config.py` directly, or override per-run via script CLI flags:

```bash
python scripts/04_train.py --model yolov8s.pt --epochs 30 --batch 24 --name exp_small
```

CLI overrides are applied on top of the defaults in `TrainConfig`.
