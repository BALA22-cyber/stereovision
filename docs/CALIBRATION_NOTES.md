# Camera Calibration Setup

## Cameras

- Left: Basler a2A3840-45ucPRO, serial `41958448`
- Right: Basler a2A3840-45ucPRO, serial `41958460`
- Keep both cameras fixed in their final positions during calibration capture.

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
