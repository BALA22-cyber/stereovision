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

## Capture stereo scene pairs

Connect both Basler cameras and run from the project root in a Python environment with `pypylon`, NumPy, and GUI-enabled OpenCV installed, with the Basler pylon runtime available. The defaults identify the left camera as serial `41958448` and the right as `41958460`; use `--left-serial` and `--right-serial` to change them. A graphical desktop is needed for the preview.

```bash
# Preview and save one pair with Space; the program then exits.
python scripts/capture/capture_stereo_pair.py \
  --output data/captures/session_new

# Keep the preview open and save multiple pairs with Space.
python scripts/capture/capture_stereo_pair.py \
  --continuous --output data/captures/session_new
```

Press Q or Esc to stop. The default output is `data/captures/stereo_capture_pair`; the script can add timestamp-named pairs to an existing directory. Each pair has two full-resolution, demosaiced BGR PNGs and a JSON file containing camera settings and timing information. The cameras are **not hardware synchronized**: keep the scene still while each pair is acquired.

## Capture stereo checkerboard calibration images

Keep both cameras fixed relative to each other throughout the session. Move the checkerboard through varied positions, tilts, and distances, making sure it is visible in both views; hold it still for each pair. The cameras are not hardware synchronized.

```bash
# Manual: press Space to save each detected pair; Q or Esc ends capture.
python scripts/capture/capture_stereo_calibration.py \
  --output data/calibration_images/session_new \
  --cols 9 --rows 6 --square-mm 80 --count 35

# Automatic: save detected pairs at 2-second intervals.
python scripts/capture/capture_stereo_calibration.py \
  --output data/calibration_images/session_auto \
  --cols 9 --rows 6 --square-mm 80 --count 35 \
  --auto --interval-sec 2
```

The program saves a pair only when the checkerboard is detected in both views, and stops after `--count` pairs or when you quit. Defaults are 9 columns × 6 rows of **inner corners**, 80 mm squares, 25 pairs, and a 2-second automatic interval. Measure your actual board and adjust these values as needed. The camera serial options are the same as for scene capture.

**Use a new, empty output folder for each calibration session.** The default folder is `data/calibration_images/stereo_calibration_images`; the script refuses a nonempty folder to protect existing captures. It writes numbered demosaiced BGR PNG pairs and `capture_metadata.json` with board dimensions and camera settings.

Add `--review-calibration` to either command to fit intrinsics and a stereo rig after capture and report per-pair alignment errors. This runs after the cameras close; the command waits for it to finish. Outputs are written to `calibration_review/` within the session folder. These are in-sample diagnostics, not independent validation. A failed review returns an error and retains the captures.

For all options, run either script with `--help`. If the cameras are not detected, check their connections, the selected serials, and the pylon runtime from the terminal environment used to run the command.

## ChArUco calibration boards

The capture and calibration tools also support OpenCV ChArUco boards with partially visible corners. `--cols` and `--rows` still mean **inner corners**: the 11 × 8 specification in `docs/data_capture_notes.md` corresponds to a 12 × 9 square OpenCV board. Both cameras must see at least six common, non-collinear ChArUco corners by default. Use `--min-charuco-corners` to change the threshold.

Both boards were reported to use `DICT_4X4_50`, but their dimensions need confirmation before capture: a standard 12 × 9 square board has 54 markers, while this dictionary contains 50. Check whether 11 × 8 describes squares rather than inner corners, or whether the board uses a different dictionary or a custom layout. Do not use the example dimensions below until confirmed. The example uses `DICT_4X4_1000` solely to illustrate command syntax; it is not the reported board dictionary. The implementation assumes the standard OpenCV ChArUco board and its default marker IDs. If the board was printed using the legacy OpenCV layout, add `--charuco-legacy` consistently to capture, intrinsics, and rig commands. Custom marker IDs or non-OpenCV layouts are not supported.

```bash
# Substitute the actual board dictionary; this example is for board 1.
python scripts/capture/capture_stereo_calibration.py \
  --output data/calibration_images/charuco_board1_session \
  --board-type charuco --cols 11 --rows 8 --square-mm 100 \
  --marker-mm 72 --dictionary DICT_4X4_1000 --count 35 \
  --review-calibration
```

For board 2, use `--square-mm 70 --marker-mm 48` and a separate output folder. `--auto --interval-sec 2` is also available. The optional review runs after capture, using the same board parameters; its errors are in-sample.

To calibrate later without the review option, use the same dictionary and board layout as capture; sizes here are in **meters**:

```bash
python scripts/stereo_calibration_workflow/stereo_pipeline.py intrinsics \
  --images data/calibration_images/charuco_board1_session \
  --board-type charuco --cols 11 --rows 8 --square-size 0.100 \
  --marker-size 0.072 --dictionary DICT_4X4_1000 \
  --out test/charuco_board1/intrinsics.npz --no-debug

python scripts/stereo_calibration_workflow/stereo_pipeline.py rig-board \
  --images data/calibration_images/charuco_board1_session \
  --board-type charuco --cols 11 --rows 8 --square-size 0.100 \
  --marker-size 0.072 --dictionary DICT_4X4_1000 \
  --intrinsics test/charuco_board1/intrinsics.npz \
  --out test/charuco_board1/rig --no-debug
```

For board 2 use `--square-size 0.070 --marker-size 0.048`. Calibrate each board session separately; check physical square and marker measurements and validate the resulting rig on independent images before using metric depth. ChArUco requires `opencv-contrib-python` with the ArUco module; the board must be generated with a compatible dictionary and marker layout.
