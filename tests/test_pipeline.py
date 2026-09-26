"""
End-to-end pipeline smoke tests using a synthetic mini-dataset.

Creates a tiny fake BDD100K-like dataset (real small images + annotation JSON),
then runs the Phase 2 conversion and Phase 3 validation functions against it.
NO real BDD100K data required.

Run:
    python -m pytest tests/test_pipeline.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TEST_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.config import CLASSES


# ============================================================
# FIXTURES: synthetic BDD100K-like dataset
# ============================================================

def _make_fake_image(path: Path, width: int = 1280, height: int = 720) -> None:
    """Create a real (tiny) image file so PIL can read dimensions."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(100, 150, 200)).save(path)


@pytest.fixture
def mini_dataset(tmp_path: Path) -> dict:
    """Build a synthetic BDD100K dataset structure.

    Layout:
        tmp_path/
            images/train/  img_000.jpg .. img_004.jpg
            images/val/    img_100.jpg .. img_101.jpg
            labels/bdd100k_labels_images_train.json
            labels/bdd100k_labels_images_val.json

    Train annotations include: valid car, excluded person, non-bbox lane,
    excluded train (rail), target with invalid bbox, target with out-of-image
    bbox, unknown category.
    """
    img_root = tmp_path / "images"
    ann_root = tmp_path / "labels"
    ann_root.mkdir(parents=True, exist_ok=True)

    # --- images ---
    for i in range(5):
        _make_fake_image(img_root / "train" / f"img_{i:03d}.jpg")
    for i in range(2):
        _make_fake_image(img_root / "val" / f"img_{100 + i:03d}.jpg")
    # LEAK: img_002.jpg exists in BOTH train and val image directories
    _make_fake_image(img_root / "val" / "img_002.jpg")

    def lbl(category, box=None):
        d = {"category": category}
        if box is not None:
            d["box2d"] = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
        return d

    # --- train annotations ---
    train_frames = [
        {   # 2 targets + 1 excluded + 1 non-bbox
            "name": "img_000.jpg",
            "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "city street"},
            "labels": [
                lbl("car", (100, 200, 300, 400)),
                lbl("truck", (500, 100, 800, 500)),
                lbl("person", (10, 100, 50, 300)),
                lbl("lane"),
            ],
        },
        {   # 1 target + excluded rail 'train' + unknown class
            "name": "img_001.jpg",
            "attributes": {"timeofday": "night", "weather": "rainy", "scene": "highway"},
            "labels": [
                lbl("bike", (200, 300, 260, 380)),
                lbl("train", (0, 0, 400, 300)),
                lbl("alien", (0, 0, 10, 10)),
            ],
        },
        {   # invalid bboxes: zero-area + outside-image
            "name": "img_002.jpg",
            "attributes": {"timeofday": "daytime", "weather": "snowy", "scene": "parking lot"},
            "labels": [
                lbl("motor", (100, 100, 100, 200)),    # zero width
                lbl("car", (1200, 600, 1400, 900)),     # outside image
                lbl("bus", (300, 300, 500, 450)),
            ],
        },
        {   # empty frame (negative sample)
            "name": "img_003.jpg",
            "attributes": {"timeofday": "dawn/dusk", "weather": "clear", "scene": "residential"},
            "labels": [],
        },
        {   # duplicate frame name of img_000 (should be deduped)
            "name": "img_000.jpg",
            "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "city street"},
            "labels": [lbl("car", (100, 200, 300, 400))],
        },
        {   # frame whose image does not exist
            "name": "img_999.jpg",
            "attributes": {"timeofday": "night", "weather": "clear", "scene": "highway"},
            "labels": [lbl("car", (100, 200, 300, 400))],
        },
    ]

    # --- val annotations (1 leaked name + 2 clean) ---
    val_frames = [
        {
            "name": "img_100.jpg",
            "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"},
            "labels": [lbl("car", (100, 200, 300, 400))],
        },
        {
            "name": "img_101.jpg",
            "attributes": {"timeofday": "night", "weather": "clear", "scene": "highway"},
            "labels": [lbl("motor", (400, 400, 460, 480))],
        },
        {
            "name": "img_002.jpg",  # LEAKED: also exists in train
            "attributes": {"timeofday": "daytime", "weather": "clear", "scene": "highway"},
            "labels": [lbl("car", (100, 200, 300, 400))],
        },
    ]

    with open(ann_root / "bdd100k_labels_images_train.json", "w") as f:
        json.dump(train_frames, f)
    with open(ann_root / "bdd100k_labels_images_val.json", "w") as f:
        json.dump(val_frames, f)

    return {
        "root": tmp_path,
        "train_images": img_root / "train",
        "val_images": img_root / "val",
        "train_ann": ann_root / "bdd100k_labels_images_train.json",
        "val_ann": ann_root / "bdd100k_labels_images_val.json",
    }


# ============================================================
# PHASE 2 END-TO-END
# ============================================================

class TestPhase2EndToEnd:
    """Run the Phase 2 conversion against the synthetic dataset."""

    def _get_script(self):
        """Import the phase 2 script module by file path."""
        import importlib.util
        script_path = _PROJECT_DIR / "scripts" / "02_prepare_vehicle_dataset.py"
        spec = importlib.util.spec_from_file_location("phase2_script", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_train_conversion(self, mini_dataset):
        mod = self._get_script()
        from src.log import setup_logger
        logger = setup_logger("test_phase2")

        frames, acc, meta = mod.process_split(
            ann_path=mini_dataset["train_ann"],
            img_root=mini_dataset["train_images"],
            split="train",
            class_config=CLASSES,
            logger=logger,
        )

        # 4 unique frames converted (dup img_000 deduped; img_999 has no image)
        assert meta["converted_frames"] == 4
        assert meta["duplicate_frames"] == 1
        assert meta["missing_images"] == 1

        # Accounting: count each label across unique frames
        #   img_000: car + truck + person(excluded) + lane(non-bbox) = 4
        #   img_001: bike + train(excluded rail) + alien(unknown)   = 3
        #   img_002: motor(zero-area, invalid) + car(outside, invalid) + bus = 3
        #   img_003: 0
        #   img_999 (missing image): car -> invalid_bbox(missing_image) = 1
        # Total source = 11
        assert acc.source_total == 11
        assert acc.target_bbox == 4      # car, truck, bike, bus
        assert acc.excluded_class == 2   # person, train
        assert acc.non_bbox_class == 1   # lane
        assert acc.unknown_class == 1    # alien
        assert acc.invalid_bbox == 3     # zero-area motor, outside car, missing-img car
        assert acc.target_no_bbox == 0
        assert acc.excluded_class == 2
        assert acc.non_bbox_class == 1
        assert acc.unknown_class == 1
        assert acc.invalid_bbox == 3

        # Reconciliation must hold
        passed, msg = acc.verify()
        assert passed is True, msg

        # Only valid labels converted
        assert len(frames) == 4
        by_name = {f.frame_name: f for f in frames}
        assert by_name["img_000.jpg"].num_objects == 2
        assert by_name["img_002.jpg"].num_objects == 1  # only bus survived
        assert by_name["img_003.jpg"].num_objects == 0

        # Metadata preserved
        assert by_name["img_001.jpg"].frame_attributes["timeofday"] == "night"

    def test_val_conversion(self, mini_dataset):
        mod = self._get_script()
        from src.log import setup_logger
        logger = setup_logger("test_phase2v")

        frames, acc, meta = mod.process_split(
            ann_path=mini_dataset["val_ann"],
            img_root=mini_dataset["val_images"],
            split="val",
            class_config=CLASSES,
            logger=logger,
        )

        # img_100 + img_101 + leaked img_002 converted (its image exists in val dir)
        assert meta["converted_frames"] == 3
        assert meta["missing_images"] == 0
        assert acc.source_total == 3
        assert acc.target_bbox == 3

    def test_leakage_detection(self, mini_dataset):
        """The synthetic dataset deliberately leaks img_002.jpg train<->val."""
        mod = self._get_script()
        from src.log import setup_logger
        logger = setup_logger("test_phase2l")

        train_frames, _, _ = mod.process_split(
            mini_dataset["train_ann"], mini_dataset["train_images"],
            "train", CLASSES, logger,
        )
        val_frames, _, _ = mod.process_split(
            mini_dataset["val_ann"], mini_dataset["val_images"],
            "val", CLASSES, logger,
        )

        passed, count = mod.check_split_leakage(train_frames, val_frames, logger)
        # img_002.jpg exists in both converted sets -> leak detected
        assert passed is False
        assert count == 1

    def test_full_write_and_yaml(self, mini_dataset, tmp_path):
        """Write YOLO dataset and verify all output files."""
        mod = self._get_script()
        from src.log import setup_logger
        logger = setup_logger("test_phase2w")

        train_frames, _, _ = mod.process_split(
            mini_dataset["train_ann"], mini_dataset["train_images"],
            "train", CLASSES, logger,
        )
        val_frames, _, _ = mod.process_split(
            mini_dataset["val_ann"], mini_dataset["val_images"],
            "val", CLASSES, logger,
        )

        out_dir = tmp_path / "output" / "vehicle_detection"
        counts = mod.write_yolo_dataset(train_frames, val_frames, out_dir, CLASSES, logger)

        assert counts["train_labels"] == 4
        assert counts["val_labels"] == 3

        # Labels exist and parse
        lbl_dir = out_dir / "train" / "labels"
        lines = (lbl_dir / "img_000.txt").read_text().strip().split("\n")
        assert len(lines) == 2
        first = lines[0].split()
        assert len(first) == 5
        assert first[0] == "0"  # car
        cx, cy, w, h = map(float, first[1:])
        assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1

        # dataset.yaml valid
        yaml_text = (out_dir / "dataset.yaml").read_text()
        assert "nc: 5" in yaml_text
        assert "train: train/images" in yaml_text
        assert "val: val/images" in yaml_text

        # Images were copied/linked and are readable
        img_path = out_dir / "train" / "images" / "img_000.jpg"
        assert img_path.exists()
        from PIL import Image
        with Image.open(img_path) as im:
            assert im.size == (1280, 720)


# ============================================================
# PHASE 3 END-TO-END
# ============================================================

class TestPhase3EndToEnd:
    """Run Phase 3 validation against output produced by Phase 2."""

    def _prepare(self, mini_dataset, tmp_path):
        import importlib.util
        from src.log import setup_logger

        p2_path = _PROJECT_DIR / "scripts" / "02_prepare_vehicle_dataset.py"
        spec = importlib.util.spec_from_file_location("phase2_script", p2_path)
        p2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p2)

        logger = setup_logger("test_phase3prep")
        train_frames, _, _ = p2.process_split(
            mini_dataset["train_ann"], mini_dataset["train_images"],
            "train", CLASSES, logger,
        )
        val_frames, _, _ = p2.process_split(
            mini_dataset["val_ann"], mini_dataset["val_images"],
            "val", CLASSES, logger,
        )
        out_dir = tmp_path / "vehicle_detection"
        p2.write_yolo_dataset(train_frames, val_frames, out_dir, CLASSES, logger)
        return out_dir

    def _get_phase3(self):
        import importlib.util
        p3_path = _PROJECT_DIR / "scripts" / "03_validate_dataset.py"
        spec = importlib.util.spec_from_file_location("phase3_script", p3_path)
        p3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p3)
        return p3

    def test_validation_detects_all_issues(self, mini_dataset, tmp_path):
        from src.log import setup_logger

        out_dir = self._prepare(mini_dataset, tmp_path)
        p3 = self._get_phase3()
        logger = setup_logger("test_phase3")

        # Split validation should PASS on structural integrity
        train_ok, train_stats = p3.validate_split(out_dir, "train", CLASSES.num_classes, logger)
        val_ok, val_stats = p3.validate_split(out_dir, "val", CLASSES.num_classes, logger)
        assert train_ok is True
        assert val_ok is True

        assert train_stats["image_count"] == 4
        assert train_stats["total_labels"] == 4
        assert train_stats["malformed_lines"] == 0
        assert train_stats["invalid_class_ids"] == 0
        assert train_stats["invalid_coords"] == 0

        # Cross-split leakage: img_002.jpg was written to both output dirs
        leak_ok, leak_count = p3.check_cross_split_leakage(out_dir, logger)
        assert leak_ok is False
        assert leak_count == 1

    def test_clean_dataset_passes(self, mini_dataset, tmp_path):
        """Removing the leaked val frame yields a fully PASSING dataset."""
        from src.log import setup_logger

        # Remove leaked val frame annotation before conversion
        with open(mini_dataset["val_ann"], "r") as f:
            val_frames = json.load(f)
        val_frames = [fr for fr in val_frames if fr["name"] != "img_002.jpg"]
        with open(mini_dataset["val_ann"], "w") as f:
            json.dump(val_frames, f)

        out_dir = self._prepare(mini_dataset, tmp_path)
        p3 = self._get_phase3()
        logger = setup_logger("test_phase3clean")

        yaml_ok, _ = p3.validate_dataset_yaml(out_dir, logger)
        train_ok, _ = p3.validate_split(out_dir, "train", CLASSES.num_classes, logger)
        val_ok, _ = p3.validate_split(out_dir, "val", CLASSES.num_classes, logger)
        leak_ok, _ = p3.check_cross_split_leakage(out_dir, logger)

        assert yaml_ok and train_ok and val_ok and leak_ok

    def test_corrupted_label_detected(self, mini_dataset, tmp_path):
        """Malformed label lines must be detected."""
        from src.log import setup_logger

        out_dir = self._prepare(mini_dataset, tmp_path)
        # Corrupt a label file
        label = out_dir / "train" / "labels" / "img_000.txt"
        label.write_text("0 0.5 not_a_number 0.3 0.2\n")

        p3 = self._get_phase3()
        logger = setup_logger("test_phase3bad")
        ok, stats = p3.validate_split(out_dir, "train", CLASSES.num_classes, logger)

        assert ok is False
        assert stats["malformed_lines"] == 1

    def test_duplicate_label_lines_detected(self, mini_dataset, tmp_path):
        """Exact-duplicate label lines are counted and excluded from class counts."""
        from src.log import setup_logger

        out_dir = self._prepare(mini_dataset, tmp_path)
        # img_000.txt has 2 label lines (car + truck). Append exact duplicate
        # of the car line -> 1 duplicate.
        label = out_dir / "train" / "labels" / "img_000.txt"
        original_lines = label.read_text().strip().split("\n")
        car_line = next(l for l in original_lines if l.startswith("0 "))
        label.write_text("\n".join(original_lines + [car_line]) + "\n")

        p3 = self._get_phase3()
        logger = setup_logger("test_phase3dup")
        ok, stats = p3.validate_split(out_dir, "train", CLASSES.num_classes, logger)

        # Duplicates are a WARNING, not a failure (training dedups silently)
        assert ok is True
        assert stats["duplicate_label_lines"] == 1
        # Class counts exclude the duplicate: car=1 (not 2), truck=1
        assert stats["class_distribution"]["car"] == 1
        assert stats["class_distribution"]["truck"] == 1

    def test_duplicate_line_helper(self):
        """Unit-level check of count_duplicate_label_lines."""
        p3 = self._get_phase3()
        lines = ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1", "0 0.5 0.5 0.1 0.1"]
        assert p3.count_duplicate_label_lines(lines) == 1
        assert p3.count_duplicate_label_lines(["0 0.5 0.5 0.1 0.1"]) == 0
        assert p3.count_duplicate_label_lines(
            ["0 0.5 0.5 0.1 0.1"] * 3
        ) == 2
        assert p3.count_duplicate_label_lines([]) == 0


# ============================================================
# PHASE 4 HELPERS
# ============================================================

class TestResumePathResolution:
    """Test resolve_experiment_dir from 04_train.py."""

    def _get_phase4(self):
        import importlib.util
        p4_path = _PROJECT_DIR / "scripts" / "04_train.py"
        spec = importlib.util.spec_from_file_location("phase4_script", p4_path)
        p4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p4)
        return p4

    def test_train_layout(self, tmp_path):
        """<exp>/train/weights/last.pt -> <exp> (standard Ultralytics layout)."""
        p4 = self._get_phase4()
        ckpt = tmp_path / "baseline_s_20260101_000000" / "train" / "weights" / "last.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()

        result = p4.resolve_experiment_dir(ckpt, tmp_path)
        assert result == tmp_path / "baseline_s_20260101_000000"

    def test_flat_weights_layout(self, tmp_path):
        """<exp>/weights/last.pt -> <exp>."""
        p4 = self._get_phase4()
        ckpt = tmp_path / "some_exp" / "weights" / "best.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()

        result = p4.resolve_experiment_dir(ckpt, tmp_path)
        assert result == tmp_path / "some_exp"

    def test_unknown_location_fallback(self, tmp_path):
        """Checkpoint outside any weights/ dir -> resumed_<timestamp> dir."""
        p4 = self._get_phase4()
        ckpt = tmp_path / "somewhere_else" / "model.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()

        result = p4.resolve_experiment_dir(ckpt, tmp_path)
        assert result.parent == tmp_path
        assert result.name.startswith("resumed_")
