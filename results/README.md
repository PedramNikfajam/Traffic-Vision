# results

Derived artefacts from the last pipeline run. **No BDD100K imagery, video or
label file is committed here** — only metrics, plots, and frames with the
project's own annotations drawn on them.

Regenerate by re-running the pipeline; nothing in this folder is hand-edited.

```
reports/
  phase2_preparation.json   annotation accounting, class counts, source totals
  phase3_validation.json    dataset integrity checks
  phase5_evaluation.json    detection metrics (+ stratified_metrics when the
                            grouped val() run has been executed)
  phase6_tracking.json      per-video tracks, CLEAR MOT metrics, stratified
                            MOTA by timeofday/scene, fp_audit_summary
  phase6_fp_audit.json      one record per false positive: box, best IoU to the
                            same and to neighbouring annotated frames, band
  phase7_counting.json      crossing events, line selection, counting accuracy,
                            stratified counts
plots/
  evaluation/               PR curves, confusion matrix, val-batch previews
  counting/                 annotated frames, crossing timelines, line sweep,
                            per-class / per-direction / per-timeofday bars
```

## Reading the FP audit

`reports/phase6_fp_audit.json` is the evidence behind the project's main error
analysis. Each false positive is classified by *how* it missed:

| band | definition | what it means for the fix |
|---|---|---|
| `localisation` | same-frame GT overlap, IoU below threshold | box regression, not new data |
| `gt_sampling` | matches a GT box in a neighbouring annotated sample | the metric is unfair; score on a temporal window |
| `unexplained` | no GT overlap within ±1 annotated sample | a real unannotated object *or* a hallucination — only pixels separate these |

Cross-reference `area_bands_px`: on this dataset the `unexplained` band is
almost entirely `tiny` (<32×32 px), which is what identifies the failure mode as
small-blob detection on dark backgrounds rather than phantom vehicles at night.

## A note on freshness

`phase5_evaluation.json` may lack `stratified_metrics`, and `phase6_tracking.json`
lacks `stratified_mot_metrics` / `fp_audit_summary`, if the corresponding stage
predates those features. The showcase notebook reports that explicitly rather
than substituting a remembered value.
