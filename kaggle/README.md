# notebooks

## `kaggle_showcase.ipynb`

The published notebook: every result in this project, rendered from the committed
JSON reports and figures, with the real annotated video frames. It reads
`results/traffic_ai/`, so it executes in seconds and always displays exactly
what the last pipeline run produced.

Sections: dataset composition → detection metrics → day/night detection split →
tracking with CLEAR MOT → **false-positive forensics** → traffic counting →
annotated frames and trajectories → what the results do and do not support →
reproduction commands.

It degrades honestly. If a stage has not been run, the cell prints what is
missing instead of showing a remembered number, so a stale notebook is obvious
rather than misleading.

## `build_showcase.py`

The notebook is **generated**, not hand-edited:

```bash
python kaggle/build_showcase.py
```

Edit the cells in the builder, never in the `.ipynb`. That way a number can
never drift between the report it came from and the notebook that displays it.

To also embed the outputs (so GitHub renders the charts without executing):

```bash
pip install nbclient nbformat ipykernel
jupyter nbconvert --to notebook --execute \
  --inplace kaggle/kaggle_showcase.ipynb
```

## Publishing to Kaggle

1. Upload this repository as a Kaggle dataset (the notebook locates it at
   `/kaggle/input/datasets/PedramNikfajam/Traffic-Vision` and falls back to
   `/kaggle/input/traffic-vision`).
2. Create the notebook, add the dataset as an input, and import
   `kaggle_showcase.ipynb` — or paste the cells in order.
3. Run All. No GPU is required to *display* results; a GPU is only needed to
   re-run the pipeline (section 8 of the notebook has the commands).
