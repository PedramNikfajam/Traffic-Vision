# ============================================================================
# CELL: verify the run, then package the results for download
# Run this LAST, after 06_track.py --fp-audit and 07_count.py in this session.
# It refuses to package a run that is missing the analysis you want, so you
# never download a stale archive and only discover it afterwards.
# ============================================================================
import json, os, shutil, glob
from pathlib import Path
from IPython.display import FileLink, FileLinks

W   = "/kaggle/working/traffic_ai"
OUT = "/kaggle/working/traffic_ai_results"
J   = lambda n: Path(f"{W}/reports/{n}")

def load(n):
    p = J(n)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

p5, p6, p7 = load("phase5_evaluation.json"), load("phase6_tracking.json"), load("phase7_counting.json")
audit = load("phase6_fp_audit.json")

# ---- 1. is this run the one you want? -------------------------------------
print("=" * 74)
print("RUN VERIFICATION")
print("=" * 74)
checks = []

# Phase 5: the stratified key must EXIST and be non-empty
sm = (p5 or {}).get("stratified_metrics") or {}
checks.append(("phase5 stratified_metrics", bool(sm),
               f"{len(sm)} groups: {sorted(sm)[:6]}" if sm
               else "MISSING - re-run 05_evaluate.py --imgsz 960"))

# A byte-identical overall mAP to the archived report means the STALE file was
# restored from _output_.zip and Phase 5 never actually ran.
STALE_MAP50 = 0.6454747170428357
if p5:
    got = (p5.get("overall_metrics") or {}).get("mAP50")
    checks.append(("phase5 not the stale archive", got != STALE_MAP50,
                   f"mAP50={got}" if got != STALE_MAP50
                   else f"mAP50={got} is identical to the 2026-09-06 archive"))

# Phase 6: stratification + FP forensics
st = (p6 or {}).get("stratified_mot_metrics", {}).get("timeofday", {})
checks.append(("phase6 stratified_mot_metrics", bool(st),
               ", ".join(f"{k}(n={v.get('n_videos_evaluated')})"
                         for k, v in st.items() if not k.startswith("_")) or "MISSING"))
fs = (p6 or {}).get("fp_audit_summary")
checks.append(("phase6 fp_audit_summary", bool(fs),
               f"{fs['n_fp']} FP {fs['bands']}" if fs else "MISSING - re-run 06 with --fp-audit"))
checks.append(("phase6_fp_audit.json on disk", audit is not None,
               "present" if audit else "MISSING"))

# Phase 7: the chosen line and the accuracy block
chosen = (p7 or {}).get("line_position_chosen")
acc = (p7 or {}).get("counting_accuracy") or {}
checks.append(("phase7 line chosen", chosen is not None, f"line={chosen}"))
checks.append(("phase7 counting accuracy", bool(acc.get("n_videos_compared")),
               f"MAE={acc.get('MAE')} MAPE={acc.get('MAPE_pct')}% "
               f"accuracy={acc.get('counting_accuracy')}" if acc else "MISSING"))
st7 = (p7 or {}).get("stratified", {}).get("timeofday", {})
checks.append(("phase7 stratified counts", bool(st7),
               ", ".join(f"{k}(n={v['n_videos']})" for k, v in st7.items()) or "MISSING"))

for name, ok, detail in checks:
    print(f"  {'OK   ' if ok else 'FAIL '}  {name:<32} {detail}")
missing = [n for n, ok, _ in checks if not ok]
if missing:
    print(f"\n{len(missing)} check(s) failed: {missing}")
    print("The archive below will be marked INCOMPLETE. Fix the stage(s) above")
    print("and re-run this cell, or expect placeholders in the notebook.")

# ---- 2. package ----------------------------------------------------------
print("\n" + "=" * 74)
print("PACKAGING")
print("=" * 74)
shutil.rmtree(OUT, ignore_errors=True)
(Path(OUT) / "reports").mkdir(parents=True, exist_ok=True)

# Reports. manifest_*.json is 31 MB of BDD100K metadata: excluded on purpose -
# it is already gitignored and you have a local copy.
skipped = []
for p in sorted(Path(f"{W}/reports").glob("*.json")):
    if p.name.startswith("phase"):
        shutil.copy2(p, f"{OUT}/reports/{p.name}")
    else:
        skipped.append(f"{p.name} ({p.stat().st_size/1024/1024:.1f} MB)")

# Figures: annotated frames, timelines, sweeps, PR curves, confusion matrix.
n_plots = 0
if Path(f"{W}/plots").is_dir():
    shutil.copytree(f"{W}/plots", f"{OUT}/plots", dirs_exist_ok=True)
    n_plots = sum(1 for _ in Path(f"{OUT}/plots").rglob("*") if _.is_file())

# Run logs: small, and they are the primary record of what actually happened.
if Path(f"{W}/logs").is_dir():
    shutil.copytree(f"{W}/logs", f"{OUT}/logs", dirs_exist_ok=True)

# A machine-readable flag so the notebook/README can tell a complete run apart.
(Path(OUT) / "RUN_COMPLETE.json").write_text(json.dumps({
    "run_date": "2026-09-26",
    "all_checks_passed": not missing,
    "failed_checks": missing,
    "line_position_chosen": chosen,
    "counting_accuracy": acc,
    "stratified_mot": {k: v.get("MOTA_pooled") for k, v in st.items()
                       if not k.startswith("_")},
    "fp_audit_bands": (fs or {}).get("bands"),
}, indent=2), encoding="utf-8")

for label, items in (("skipped (intentionally)", skipped),):
    if items:
        print(f"  {label}: {', '.join(items)}")

total = sum(f.stat().st_size for f in Path(OUT).rglob("*") if f.is_file())
print(f"  reports : {len(list((Path(OUT)/'reports').glob('*.json')))} files")
print(f"  plots   : {n_plots} files")
print(f"  total   : {total/1024/1024:.1f} MB")

shutil.make_archive(OUT, "zip", OUT)
print(f"\n  archive : /kaggle/working/traffic_ai_results.zip "
      f"({Path(OUT + '.zip').stat().st_size/1024/1024:.1f} MB)")

# ---- 3. download ---------------------------------------------------------
print("\n" + "=" * 74)
print("DOWNLOAD  (click the link; if it does not work, use the file browser on")
print("the right-hand panel: /kaggle/working -> traffic_ai_results.zip)")
print("=" * 74)
display(FileLink(OUT + ".zip"))
print("\nOr download the folder item by item:")
display(FileLinks(OUT + "/reports"))
