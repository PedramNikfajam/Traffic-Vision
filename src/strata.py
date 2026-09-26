"""
Scene-metadata propagation, so results can be stratified (skill §3, §10, §17).

BDD100K labels every frame with scene attributes (`timeofday`, `weather`,
`scene`) and Phase 2 copies them verbatim into
`reports/manifest_{train,val}.json`. A BDD100K detection image id IS the id of
the video it was sampled from (measured on the real manifest: 69,863 labelled
train frames with 69,863 DISTINCT ids, i.e. exactly one frame per id), so a
tracking video's attributes are recoverable from the manifest by id.

This is why nothing here infers day/night from a folder name, from a file
name, or by eye: the earlier hand-written scene inventory for the 5 evaluation
clips disagreed with the dataset metadata on 3 of 5 (it called
`0000f77c-cb820c98` a night highway; BDD100K records `dawn/dusk` + `residential`),
and a stratified analysis built on eyeballed labels would be worthless.

Everything in this module is pure and unit-tested: id normalisation, manifest
parsing, and per-stratum aggregation of tracking (Phase 6) and counting
(Phase 7) results. No I/O beyond reading the manifest JSON.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .count import accuracy_metrics

# Metadata fields carried through from the BDD100K manifest.
METADATA_FIELDS = ("timeofday", "weather", "scene")

# A stratum value used when the manifest has no (or unusable) entry for a video.
# It is a real, reported bucket - never silently dropped from the totals.
UNLABELLED = "unlabelled"

# BDD100K spells a missing attribute "undefined" in some label files.
UNDEFINED_LABEL = "undefined"
_UNDEFINED = {"", UNDEFINED_LABEL, "none", "null", "nan"}

# Extensions seen on the artefacts that name a video: the raw clips, the MOT
# files Phase 6 writes, and the manifest/GT frame images. A BDD100K id never
# contains a dot, so stripping these is unambiguous.
_MEDIA_EXTS = (".mov", ".mp4", ".avi", ".mkv", ".webm", ".jpg", ".jpeg",
               ".png", ".txt")

# Frame names in the box-track GT dump are '<video-id>-<0000001>.jpg': one extra
# dash-separated group, zero-padded and >= 6 digits. A real id has exactly two
# groups, so this rule cannot fire on an id.
_FRAME_INDEX_MIN_DIGITS = 6


def video_key(name: Any) -> str:
    """Normalise any artefact name to its BDD100K video id.

    Accepts what the pipeline actually has to reconcile:
        '0000f77c-6257be58.mov'          (Phase 6 report, videos on disk)
        '0000f77c-6257be58.txt'          (MOT file name)
        '0000f77c-6257be58.jpg'          (Phase 2 manifest frame)
        '0000f77c-6257be58-0000001.jpg'  (box-track GT frame image)
        '.../videos/train/0000f77c-6257be58.mov'
    Directory parts, a media extension and a trailing zero-padded frame index
    are removed. Case is folded so a Windows-authored path still matches.
    """
    stem = Path(str(name).strip().replace("\\", "/")).name
    if not stem or stem.lower() in (".", "none"):
        return ""
    low = stem.lower()
    for ext in _MEDIA_EXTS:
        if low.endswith(ext):
            stem = stem[: -len(ext)]
            break
    parts = stem.split("-")
    if len(parts) > 2 and parts[-1].isdigit() and len(parts[-1]) >= _FRAME_INDEX_MIN_DIGITS:
        parts = parts[:-1]
    return "-".join(parts).lower()


def _clean(value: Any) -> str:
    """Normalise one metadata value; empty/undefined become 'undefined'."""
    s = str(value).strip().lower() if value is not None else ""
    return UNDEFINED_LABEL if s in _UNDEFINED else s


def _entries_from_manifest(data: Any) -> List[Dict[str, Any]]:
    """Accept either a bare list of frame records or {'frames': [...]}."""
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if isinstance(data, dict):
        for key in ("frames", "images", "records", "entries"):
            v = data.get(key)
            if isinstance(v, list):
                return [e for e in v if isinstance(e, dict)]
    return []


def _name_of(entry: Dict[str, Any]) -> Optional[str]:
    for k in ("frame_name", "name", "image", "image_name", "file_name",
              "source_image"):
        v = entry.get(k)
        if v:
            return str(v)
    return None


def load_video_metadata(
    manifest_paths: Sequence[Any],
) -> Dict[str, Dict[str, Any]]:
    """video id -> {timeofday, weather, scene, n_images, conflicts, source}.

    Reads Phase 2 manifests (one JSON list of per-frame records each). A video
    may legitimately appear in several manifests or several times inside one;
    values are taken by majority and any disagreement is PRESERVED under
    `conflicts` rather than being dropped, because a silently-resolved
    contradiction is exactly the kind of thing that invalidates a stratified
    result.

    A 28 MB manifest expands to a few hundred MB in memory, so this is called
    once per stage and the result is reused, never per video.
    """
    acc: Dict[str, Dict[str, Any]] = {}
    for path in manifest_paths or []:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        for entry in _entries_from_manifest(data):
            name = _name_of(entry)
            if not name:
                continue
            key = video_key(name)
            if not key:
                continue
            slot = acc.setdefault(key, {"counts": {f: Counter() for f in METADATA_FIELDS},
                                        "n_images": 0, "sources": []})
            slot["n_images"] += 1
            if p.name not in slot["sources"]:
                slot["sources"].append(p.name)
            for f in METADATA_FIELDS:
                if f in entry:
                    slot["counts"][f][_clean(entry[f])] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for key, slot in acc.items():
        record: Dict[str, Any] = {
            "n_images": slot["n_images"],
            "source": "+".join(slot["sources"]),
        }
        conflicts: Dict[str, List[str]] = {}
        for f in METADATA_FIELDS:
            counts: Counter = slot["counts"].get(f) or Counter()
            if not counts:
                record[f] = UNLABELLED
                continue
            # Counter.most_common is insertion-ordered for ties, and insertion
            # order follows the manifest, so the result is deterministic.
            value, _n = counts.most_common(1)[0]
            record[f] = value
            if len(counts) > 1:
                conflicts[f] = sorted(counts)
        record["conflicts"] = conflicts
        out[key] = record
    return out


def resolve_manifest_paths(
    arg: Optional[str],
    reports_dir: Any,
) -> List[Path]:
    """Manifest paths from --manifest, else the Phase 2 manifests in reports/.

    Auto-discovery matters on Kaggle: the manifests come back with the rest of
    /kaggle/working from the output archive, so stratification works with no
    extra flag. A path that does not exist is returned as given so the caller
    can report it instead of silently evaluating nothing.
    """
    if arg:
        out: List[Path] = []
        for part in str(arg).split(","):
            part = part.strip()
            if part:
                out.append(Path(part))
        return out
    if not reports_dir:
        return []
    d = Path(reports_dir)
    found = [d / f"manifest_{split}.json" for split in ("val", "train")]
    return [p for p in found if p.is_file()]


def stratum_of(meta: Optional[Dict[str, Any]], field: str) -> str:
    """The stratum a video belongs to, or 'unlabelled'."""
    if not meta:
        return UNLABELLED
    value = meta.get(field)
    if not value or value == UNDEFINED_LABEL:
        return UNLABELLED
    return str(value)


def attach_metadata(
    records: Iterable[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
    key_field: str = "video",
    fields: Sequence[str] = METADATA_FIELDS,
) -> Dict[str, int]:
    """Copy metadata onto per-video report records IN PLACE.

    Returns {'labelled': n, 'unlabelled': n} so the caller can raise an
    integrity check when a video could not be labelled - an unlabelled video
    must never be quietly missing from a stratified table.
    """
    labelled = unlabelled = 0
    for rec in records or []:
        key = video_key(rec.get(key_field, ""))
        meta = metadata.get(key)
        if meta:
            labelled += 1
            rec["video_id"] = key
            rec["metadata_source"] = meta.get("source")
            for f in fields:
                rec[f] = meta.get(f, UNLABELLED)
            if meta.get("conflicts"):
                rec["metadata_conflicts"] = meta["conflicts"]
        else:
            unlabelled += 1
            rec["video_id"] = key
            rec["metadata_source"] = None
            for f in fields:
                rec[f] = UNLABELLED
    return {"labelled": labelled, "unlabelled": unlabelled}


def _group(records: Sequence[Dict[str, Any]], field: str,
           metadata: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records or []:
        key = video_key(rec.get("video", ""))
        groups.setdefault(stratum_of(metadata.get(key), field), []).append(rec)
    return groups


def coverage(records: Sequence[Dict[str, Any]], field: str,
             metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Provenance for a stratified table: how many videos landed in each bucket."""
    groups = _group(records, field, metadata)
    return {
        "n_videos": len(records or []),
        "n_labelled": sum(1 for r in (records or [])
                          if stratum_of(metadata.get(video_key(r.get("video", ""))),
                                        field) != UNLABELLED),
        "n_videos_per_stratum": {k: len(v) for k, v in sorted(groups.items())},
        "videos_per_stratum": {
            k: [video_key(r.get("video", "")) for r in v]
            for k, v in sorted(groups.items())
        },
    }


# A stratum holding fewer videos than this cannot support a comparison; it is
# still reported, but flagged so it is not read as an effect.
MIN_VIDEOS_PER_STRATUM = 3


def mean_metric(evals: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    """Mean of one metric, ignoring absent/None entries.

    A video can be scored for MOTA but not for a rate the implementation could
    not compute (e.g. an empty GT). Coercing those to 0 would quietly drag the
    stratum mean down, so they are excluded - same rule Phase 6 uses for its
    overall mean.
    """
    vals = [float(m[key]) for m in evals
            if m.get(key) is not None and not isinstance(m.get(key), bool)]
    return round(sum(vals) / len(vals), 4) if vals else None


def stratified_mot_metrics(
    records: Sequence[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
    field: str = "timeofday",
) -> Dict[str, Any]:
    """Pooled CLEAR MOT metrics per stratum (skill §10, §16, §17F).

    Two aggregates are reported per stratum, because they answer different
    questions and disagree in general:
      - *_pooled : counts summed over the stratum's videos, then
        MOTA = 1 - (FN + FP + IDSW)/n_gt. This is the figure to quote, since
        it is a single ratio over every matched object rather than an average
        of per-video ratios.
      - *_mean   : unweighted mean of the per-video metrics, which is what the
        existing overall summary reports and what a reader will compare
        against. Kept so the two are visibly consistent.
    Videos with no MOT metrics (GT unavailable) are excluded and counted.
    """
    out: Dict[str, Any] = {}
    n_without = 0
    for value, recs in sorted(_group(records, field, metadata).items()):
        evals = [r["mot_metrics"] for r in recs if r.get("mot_metrics")]
        n_without += len(recs) - len(evals)
        if not evals:
            out[value] = {"n_videos": len(recs), "n_videos_evaluated": 0,
                          "videos": [video_key(r.get("video", "")) for r in recs],
                          "note": "no GT metrics for this stratum"}
            continue
        tp = sum(int(m.get("TP", 0)) for m in evals)
        fp = sum(int(m.get("FP", 0)) for m in evals)
        fn = sum(int(m.get("FN", 0)) for m in evals)
        idsw = sum(int(m.get("IDSW", 0)) for m in evals)
        n_gt = sum(int(m.get("n_gt", 0)) for m in evals)
        n = len(evals)
        cov = [m.get("gt_frame_coverage") for m in evals
               if m.get("gt_frame_coverage") is not None]
        out[value] = {
            "n_videos": len(recs),
            "n_videos_evaluated": n,
            "videos": [video_key(r.get("video", "")) for r in recs],
            "MOTA_pooled": round(1.0 - (fn + fp + idsw) / n_gt, 4) if n_gt else None,
            "MOTA_mean": mean_metric(evals, "MOTA"),
            "MOTP_mean": mean_metric(evals, "MOTP"),
            "IDF1_mean": mean_metric(evals, "IDF1"),
            "IDSW": idsw, "TP": tp, "FP": fp, "FN": fn, "n_gt": n_gt,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "recall": round(tp / n_gt, 4) if n_gt else None,
            "gt_frame_coverage_mean": round(sum(cov) / len(cov), 4) if cov else None,
            "low_sample": n < MIN_VIDEOS_PER_STRATUM,
        }
    out["_excluded_without_metrics"] = n_without
    return out


def stratified_counting(
    records: Sequence[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
    field: str = "timeofday",
) -> Dict[str, Any]:
    """Crossing counts AND counting accuracy per stratum (skill §10, §15, §17F).

    Accuracy per stratum reuses `accuracy_metrics` unchanged, so a stratum is
    scored by exactly the same rule as the overall figure - including the rule
    that videos with zero GT crossings are excluded from MAPE. Strata are also
    flagged `low_sample`, because with a handful of 40 s clips per bucket a
    difference in MAE is an anecdote, not an effect.
    """
    out: Dict[str, Any] = {}
    for value, recs in sorted(_group(records, field, metadata).items()):
        total = sum(int(r.get("total_counted", 0)) for r in recs)
        minutes = sum(float((r.get("rates") or {}).get("observation_minutes") or 0.0)
                      for r in recs)
        by_class: Dict[str, int] = {}
        by_direction: Dict[str, int] = {}
        for r in recs:
            for k, n in (r.get("by_class") or {}).items():
                by_class[k] = by_class.get(k, 0) + int(n)
            for k, n in (r.get("by_direction") or {}).items():
                by_direction[k] = by_direction.get(k, 0) + int(n)
        entry: Dict[str, Any] = {
            "n_videos": len(recs),
            "videos": [video_key(r.get("video", "")) for r in recs],
            "total_counted": total,
            "n_tracks_seen": sum(int(r.get("n_tracks_seen", 0)) for r in recs),
            "n_tracks_counted": sum(int(r.get("n_tracks_counted", 0)) for r in recs),
            "by_class": dict(sorted(by_class.items())),
            "by_direction": dict(sorted(by_direction.items())),
            "observation_minutes": round(minutes, 3),
            # Rates come from the summed window, not an average of averages.
            "vehicles_per_minute": round(total / minutes, 2) if minutes > 0 else None,
            "vehicles_per_5min": round(total / minutes * 5.0, 2) if minutes > 0 else None,
            "vehicles_per_15min": round(total / minutes * 15.0, 2) if minutes > 0 else None,
            "low_sample": len(recs) < MIN_VIDEOS_PER_STRATUM,
        }
        acc = accuracy_metrics(recs)
        if acc.get("n_videos_compared"):
            entry["accuracy"] = acc
        else:
            entry["accuracy"] = {"n_videos_compared": 0,
                                 "note": "no GT counts for this stratum"}
        out[value] = entry
    return out
