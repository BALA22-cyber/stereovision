# 10_30_2026:

# Camera Calibration Setup

## Cameras

- Left: Basler a2A3840-45ucPRO, serial `41958448`
- Right: Basler a2A3840-45ucPRO, serial `41958460`
- Both cameras are fixed in their final positions during calibration capture.

## Checkerboard

- Pattern: `9 x 6` **inner corner intersections** (columns x rows), not squares.
- Printed square side length: `80 mm`.
- OpenCV pattern size: `(9, 6)`.

## AprilTags visible in the scene

- IDs: `0` and `1`
- Stated physical tag size: `100 mm` (confirm this is the printed tag side length).
- Tag family/dictionary: not yet specified; record it before using the tags for pose estimation.

## Capture plan

- Collect paired left/right images for each checkerboard pose, with the board visible in both cameras.
- Keep the board still while each pair is captured.
- Target 20-30 pairs with varied distance, tilt, and image position, covering the centers and edges of both views.
- Keep camera resolution and focus unchanged throughout the dataset.

## Access status

- macOS USB inspection sees both cameras.
- Most recent pypylon enumeration returned zero cameras; resolve that before capture.



## Checkerboard - Charuco 1 specifications.

- Pattern: `11 x 8` **inner corner intersections** (columns x rows), not squares.
- Printed square side length: `100 mm`.
- Marker size : `72 mm`
- OpenCV pattern size: `(11, 8)`.

## Checkerboard - Charuco 2 specifications.

- Pattern: `11 x 8` **inner corner intersections** (columns x rows), not squares.
- Printed square side length: `70 mm`.
- Marker size : `48 mm`
- OpenCV pattern size: `(11, 8)`.


new april tag measurement - 200 mm

## ChArUco dictionary clarification

- Both boards were reported to use `DICT_4X4_50`.
- Confirm whether 11 × 8 counts squares or inner corners. A standard OpenCV board with 11 × 8 inner corners has 12 × 9 squares and needs 54 distinct marker IDs; `DICT_4X4_50` contains 50.

## Capture 1 — requested, not captured

- Target: ChArUco board 2; 70 mm squares, 48 mm markers; reported dictionary `DICT_4X4_50`.
- Requested: 40 stereo pairs in `data/calibration_images/capture_1`.
- Status: capture did not start because the recorded 11 × 8 inner-corner dimensions require 54 markers for a standard OpenCV board, exceeding the dictionary size. Confirm the printed board layout before capturing.

## Capture 1 — alternate layout trial

- Live capture launched with 11 × 8 squares (10 × 7 inner corners), `DICT_4X4_50`, square size 70 mm, marker size 48 mm, target 40 pairs.
- Output: `data/calibration_images/capture_1`. This board interpretation is experimental and must be verified against the physical board. See capture metadata for the pairs actually saved.

- Capture 1 was restarted in automatic mode at a 2-second interval, with a target of 40 pairs. The initial manual session saved zero pairs. The 11 × 8-square layout remains an interpretation to verify.

## 2026-10-30 calibration sessions

- The completed 40-pair `capture_1` ChArUco session was renamed to `data/calibration_images/charuco_70_48`. The 11 × 8-square layout interpretation remains to be independently verified.
- Checkerboard capture started in `data/calibration_images/checkerboard_80mm`, with 9 × 6 inner corners, 80 mm squares, 40-pair target, and automatic 2-second interval.
- ChArUco in-sample review is in `charuco_70_48/calibration_review/`: stereo RMS 4.32 px; median vertical error 0.91 px; P95 6.71 px; estimated baseline 0.0965 m. This is preliminary and not independent validation.

- Checkerboard session completed with 40 saved pairs; 36 jointly detected in the calibration review. Fit: stereo RMS 2.28 px, median vertical error 0.64 px, P95 3.43 px, baseline 0.1014 m.
- Cross-board epipolar diagnostics: ChArUco rig on checkerboard median 0.74 px / P95 3.43 px; checkerboard rig on ChArUco median 0.80 px / P95 9.51 px. See `data/calibration_images/cross_board_validation.json`. These do not verify metric scale.

## 2026-10-30 scene capture

- Output: `data/captures/2026-10-30_6_4`; target: 10 manually triggered stereo pairs.
- Scene contains AprilTags of family `tag36h11` (`DICT_APRILTAG_36h11`), with different IDs and reported constant edge size 200 mm; confirm the edge measurement refers to the detected tag square.
- Captures are not hardware synchronized; keep the scene still during each pair.



- stereo camera distance: 

inner distance 72.6 mm
outer distance 130.75 mm
thickness of camera module: 29.1 mm
