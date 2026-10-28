# Stereo acquisition and reconstruction

Run commands from this directory. The calibration and reconstruction entry point is
`python scripts/stereo_calibration_workflow/stereo_pipeline.py --help`.

- `scripts/` — camera capture and earlier calibration/rectification utilities. Run these from the project root; for example, `python scripts/capture/capture_stereo_pair.py --help`.
- `data/calibration_images/stereo_calibration_images/` — checkerboard capture pairs and metadata. Pair 020 is retained in `excluded/` and is not included by the pipeline.
- `data/captures/` — recorded stereo scenes, including `stereo_capture_pair_close/`.
- `data/samples/` — standalone camera images.
- `test/calibration/` and `test/results/` — calibration and reconstruction runs.
- `test/right_intrinsics_experiment/` — the experiment using right-camera intrinsics for both cameras; results are grouped by depth and block size. These calibrations and point clouds have not been independently validated for metric accuracy.
- `output/legacy/` — earlier calibration and disparity outputs; `output/debug_previews/` contains debug images.
- `scripts/stereo_calibration_workflow/README.md` — workflow instructions for the adjacent pipeline.
- `archives/` — original zip archives. Archives and generated reports retain their original historical paths.

The checkerboard capture metadata records 9 × 6 inner corners and 80 mm squares. The workflow document contains illustrative board sizes; use the dimensions of the actual target. Preserve captures and use a new output directory for new capture sessions.

For a new stereo checkerboard capture with a post-capture calibration review:

```bash
python scripts/capture/capture_stereo_calibration.py \
  --output data/calibration_images/session_new \
  --cols 9 --rows 6 --square-mm 80 --count 35 --review-calibration
```

Space saves a detected pair; Q ends capture. New images are saved as demosaiced BGR PNGs. After capture ends and cameras close, a worker thread runs the existing pipeline's checkerboard detection, intrinsic calibration and fixed-intrinsic stereo calibration. The command waits for it to finish. Results go under `session_new/calibration_review/`, including `review_summary.json`, calibration files and per-pair vertical-error rankings. The review uses the same views for fitting and evaluation, so it is not independent validation. A failed fit is reported in the summary and returns an error; captures are retained. The option does not perform asynchronous checking while capture is active.
