> Project layout note: commands run from the repository root. The active pipeline is `stereo_pipeline.py`; the copy alongside this document is the original workflow snapshot. Existing experiment output is under `test/`.

# Repeatable stereo calibration: Basler a2A3840-45ucPRO + checkerboard + AprilTags

**Workflow:** calibrate intrinsics once → estimate current stereo rig → undistort/rectify all pairs → check epipolar alignment and metric scale → optionally generate SGBM depth/PLY or use any disparity model.

Input folders as provided:

```text
data_acq_stereo_vision/
├── data/calibration_images/stereo_calibration_images/
│   ├── pair_001_left_41958448.png
│   ├── pair_001_right_41958460.png
│   └── ... through pair_030; pair_020 is in excluded/
└── data/captures/stereo_capture_pair_close/
    ├── pair_20261023T161855_248110Z_left_41958448.png
    ├── pair_20261023T161855_248110Z_right_41958460.png
    └── ... 4 pairs total
```

Photos are already demosaiced BGR8 PNG according to the provided capture JSON. The code expects this format (not raw Bayer planes). Left/right images must have the **same resolution** as the calibrated images: 3840×2160 in your data.

## Prerequisites

- Python 3.10+ recommended.
- `pip install -r docs/stereo_calibration_workflow/requirements.txt` installs NumPy and a **GUI-enabled** OpenCV-contrib wheel.
- A physical checkerboard with known **inner-corner count** (columns × rows) and **square edge length in meters**. *The number of inner intersections is NOT the number of printed squares.*
- For AprilTag reprojection, know the **actual edge length of the detected tag square** (not the sheet or white margin), and its family, such as `DICT_APRILTAG_36h11`.
- The cameras must remain **fixed relative to each other for all stereo checkerboard poses used in one rig calibration**. Move the checkerboard, not the cameras. Subsequent camera movement requires a NEW rig estimate.

The placeholders below use a **9×6 inner-corner checkerboard with a 25 mm square edge** and an **80 mm AprilTag edge**. These are ONLY example measurements; replace them with the dimensions of YOUR targets. Otherwise, metric depths and camera baseline will be wrong. The checkerboard must be observed at diverse tilts, distances and positions across the whole frame. Avoid a fully symmetric square grid for robust correspondence ordering.

Run the commands from `data_acq_stereo_vision/` with `stereo_pipeline.py` accessible, e.g., `python /path/to/stereo_pipeline.py ...`.

## A. First setup: calibrate each camera once

```bash
python stereo_pipeline.py intrinsics \
  --images data/calibration_images/stereo_calibration_images \
  --cols 6 --rows 9 --square-size 0.080 \
  --out output/calibration/intrinsics.npz \
  --debug
```

Output: `intrinsics.npz` (K1, D1, K2, D2, size); `.json` diagnostic summary. For each of the 30 pairs, the terminal reports corner detection success in each image. Debug windows overlay corner ordering and write previews to `output/debug_previews/checkerboard/`.

Reusing the intrinsics is valid **only if** lens, focus, zoom, resolution, lens mounting, and relevant optics have not changed.

## B. Initial stereo geometry from 30 checkerboard pairs

```bash
python stereo_pipeline.py rig-board \
  --images data/calibration_images/stereo_calibration_images \
  --cols 6 --rows 9 --square-size 0.080 \
  --intrinsics calibration/intrinsics.npz \
  --out output/calibration/rig_v1 \
  --alpha 0 \
  --debug
```

Outputs: `calibration/rig_v1/rig.npz` (K, D, R, T, rectification R1/R2/P1/P2/Q); `rig_report.json` (RMS and epipolar error); `camera_geometry.json` (model-agnostic matrices); `foundationstereo_intrinsics.txt` (FoundationStereo's expected 3×3 K + baseline in meters).

`--alpha 0` crops/remaps to reduce invalid image borders. `--alpha 1` retains the whole original frame and may produce black borders. The stored `P1`, `P2`, `Q` always correspond to the selected rectification.

### After moving either camera: re-estimate ONLY the rig

Capture **new, static**, matching left/right views of the **same detected AprilTags** after repositioning the cameras. Multiple view pairs improve robustness; if tags are mounted to a rigid object, they need not share a known board-coordinate system for this implementation because each shared tag is used as an independent pose estimate.

```bash
python stereo_pipeline.py rig-tags \
  --images data/captures/stereo_capture_pair_close \
  --intrinsics calibration/intrinsics.npz \
  --tag-size 0.080 --tag-family DICT_APRILTAG_36h11 \
  --out output/calibration/rig_v2 \
  --debug
```

`rig-tags` computes the relative transform of the right camera with respect to the left camera using a known-size tag in both views; it robustly combines estimates from multiple shared tag observations. It is less accurate than a well-captured checkerboard rig fit in some cases; compare baseline and epipolar errors and reject obviously inconsistent tag poses.

You may instead **recapture a few synchronized checkerboard pairs** after repositioning and run `rig-board` again with the saved intrinsics. This is useful if tag poses are too noisy.

## C. Apply the new calibration to ANY matched stereo PNG pairs

```bash
python stereo_pipeline.py rectify \
  --images data/captures/stereo_capture_pair_close \
  --rig test/calibration/rig_v1/rig.npz \
  --out output/results/rig_v1_objects \
  --save-undistorted \
  --sgbm --compare-raw --sgbm-scale 0.5 \
  --num-disparities 256 --min-depth 0.2 --max-depth 8 \
  --debug
```

```bash
python scripts/stereo_calibration_workflow/stereo_pipeline.py rectify \
  --images data/calibration_images/stereo_calibration_images \
  --rig test/right_intrinsics_experiment/calibration/rig/rig.npz \
  --out output/results/test \
  --save-undistorted \
  --sgbm --compare-raw --sgbm-scale 1 \
  --num-disparities 1024 --min-depth 0.2 --max-depth 5 \
  --debug
```

If you have moved the camera, substitute `rig_v2/rig.npz` instead. `rig_v1` and `rig_v2` are NOT interchangeable for the same physical setup.

- `results/rig_v1_objects/undistorted/` contains lens-undistorted-only images (for monocular use).
- `results/rig_v1_objects/rectified/` contains **undistorted + epipolar-rectified** lossless stereo PNGs for stereo models.
- `results/rig_v1_objects/sgbm/<pair>/disparity_px.npy` is SGBM pixel disparity measured at `--sgbm-scale` resolution.
- `results/rig_v1_objects/sgbm/<pair>/depth_m.npy` is metric depth, `NaN` where invalid.
- `results/rig_v1_objects/sgbm/<pair>/pointcloud_m.ply` is a colored metric 3D point cloud in the rectified-left camera coordinate system.
- `results/rig_v1_objects/raw_sgbm/` is an **illustrative invalid baseline**, not usable as metric depth. *StereoSGBM on unrectified images violates its epipolar assumption.*
- `results/rig_v1_objects/geometry.npz`, `camera_geometry.json` and `foundationstereo_intrinsics.txt` are exported alongside results.

Use `--max-pairs 2` for a quick first smoke test and `--no-debug` for unattended/headless processing. `--sgbm` is **optional**; skip it to rectify all 4K stereo pairs faster.

## D. Validate against the AprilTags independently

Best is to calibrate the current rig using **checkerboard** photos then validate on different **AprilTag** object photos:

```bash
python stereo_pipeline.py validate-tags \
  --images data/captures/stereo_capture_pair_close \
  --rig calibration/rig_v1/rig.npz \
  --tag-size 0.100 --tag-family DICT_APRILTAG_36h11 \
  --out output/results/tag_validation_v1 \
  --debug
```

Outputs:
- `tag_validation.csv`: per tag / image pair, vertical epipolar error, triangulated square edge vs physical tag edge, and tag depth from four triangulated corners.
- `tag_validation_summary.json`: median vertical epipolar error, median *absolute* tag edge-size error percentage.

If tag views **also produced the rig** (e.g. `rig-tags` on the exact same inputs), the reported errors are **in-sample** and not independent; retain separate validation pairs if you want a fair generalization test. Test the actual reconstructed **depth error** against tape-measured distances when possible.

### Interpretation

- Aim for low held-out median and P95 `|y_left-y_right|` for correctly matched checkerboard/tag corners after rectification. Subpixel median is a useful sign, but not a guarantee of correct depth.
- RMS reprojection errors should be low relative to image resolution and detection accuracy; large errors may indicate bad corners, wrong board pattern, motion, or board orientation ambiguity.
- Tag edge-size relative error checks metric **scale**. If the tag is 80 mm and the reconstructed edges are 83 mm, the relative error is about +3.75%.
- The raw left/right **vertical-coordinate difference is not itself a valid rectification error**; cameras may have different optical axes. It is reported only for a visible before/after sanity check.
- `hardware_synchronized: false` is fine for **static** scenes; moving scenes need hardware synchronization or careful timestamp matching.

## E. FoundationStereo / other stereo networks

[Official FoundationStereo repository](https://github.com/NVlabs/FoundationStereo) requires rectified inputs and an intrinsic text file with flattened 3×3 **rectified** K and baseline in meters. The script generates it automatically.

```bash
python scripts/run_demo.py \
  --left_file /absolute/path/results/rig_v1_objects/rectified/PAIR_NAME_left.png \
  --right_file /absolute/path/results/rig_v1_objects/rectified/PAIR_NAME_right.png \
  --intrinsic_file /absolute/path/results/rig_v1_objects/foundationstereo_intrinsics.txt \
  --ckpt_dir ./weights/23-51-11/model_best_bp2.pth \
  --out_dir ./test_outputs/your_pair \
  --hiera 1 --low_memory 1 --valid_iters 16
```

Your FoundationStereo fork/build may not accept every optional flag; check `python scripts/run_demo.py --help`. The checkpoint path above follows your prior command and must exist in your environment.

If another model saves a **real horizontal pixel-disparity map** as a `.npy` or `.npz`, convert it to metric depth/PLY:

```bash
python stereo_pipeline.py depth-from-disparity \
  --disparity path/to/model_disparity.npy \
  --rectified-left results/rig_v1_objects/rectified/PAIR_NAME_left.png \
  --rig calibration/rig_v1/rig.npz \
  --out results/your_model \
  --min-depth 0.2 --max-depth 8
```

If the model outputs disparity in pixel units at its **own reduced resolution** (usual case), keep `--disparity-units array-resolution` (default); if it outputs full-resolution disparity values in a smaller map, set `--disparity-units full-resolution`. Adjust disparity sign convention if your model differs from the standard left-minus-right convention. Disparity must be in **pixels**, not disparity normalized to [0,1] or a unitless inverse-depth prediction.

For monocular models, use `undistorted/` images or `rectified/` images depending on model input requirements. **Calibration doesn't make scale-ambiguous monocular depth metric by itself.** If the model predicts inverse depth or scale-invariant depth, it needs a separate metric calibration stage.

## F. DEBUG = True

The code defaults to `DEBUG = True` near the top of `stereo_pipeline.py`; each command supports `--debug` / `--no-debug`.

OpenCV GUI windows show:
- Checkerboard detected corners and their indexing.
- AprilTag borders/IDs.
- Raw, undistorted, rectified image pairs with horizontal reference lines.
- SGBM disparity/depth previews, if enabled.

Press `p` or Space to pause any preview; press `q` to stop opening windows and continue the batch. Debug images are saved in `output/debug_previews/`. On a headless machine with no display, previews are written to disk automatically instead.

## G. Smoke test and limitations

```bash
python docs/stereo_calibration_workflow/self_test.py
```

The synthetic test exercises the archived pipeline copy beside the test, not the active root pipeline. It checks AprilTag pose composition, rectified epipolar error, 3D tag scale, disparity-resolution scaling, `Q`, and rig output. **No user images are bundled**. The data shown in the chat are file listings/metadata, not access to the original PNGs, so the reported errors must be determined by running on your actual files.

Planar AprilTags may give ambiguous/noisy pose estimates (especially small/far/oblique tags); reject outliers and prefer multiple images. For large camera baselines and/or almost non-overlapping views, there may be too little overlapping image content for dense stereo matching, even when geometry is correct. If you physically swap the camera order or arrange a near-vertical baseline, conventional horizontal stereo disparity models may fail; the CLI warns/errors accordingly.

## Docs

- [OpenCV stereoCalibrate and stereoRectify](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenCV ArUco/AprilTag marker detection and pose](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [NVlabs FoundationStereo](https://github.com/NVlabs/FoundationStereo)
