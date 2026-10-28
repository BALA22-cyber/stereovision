"""Apply saved stereo rectification maps to a left/right image pair."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_color(path, pixel_format):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if pixel_format == "BayerRG8":
        return cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path, help="input image from left camera")
    parser.add_argument("--right", required=True, type=Path, help="input image from right camera")
    parser.add_argument("--calibration", type=Path, default=Path("output/legacy/stereo_calibration_output"))
    parser.add_argument("--output", type=Path, default=Path("rectified_pair"))
    args = parser.parse_args()

    maps_path = args.calibration / "stereo_calibration.npz"
    left_info_path = args.calibration / "left_intrinsics.json"
    right_info_path = args.calibration / "right_intrinsics.json"
    maps = np.load(maps_path)
    left_info = json.loads(left_info_path.read_text())
    right_info = json.loads(right_info_path.read_text())

    left = load_color(args.left, left_info["camera"].get("pixel_format"))
    right = load_color(args.right, right_info["camera"].get("pixel_format"))
    if left.shape[:2] != tuple(maps["image_size"])[::-1] or right.shape[:2] != tuple(maps["image_size"])[::-1]:
        raise SystemExit("Input image resolution does not match the calibration map resolution.")

    left_rectified = cv2.remap(left, maps["map_left_x"], maps["map_left_y"], cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right, maps["map_right_x"], maps["map_right_y"], cv2.INTER_LINEAR)
    args.output.mkdir(parents=True, exist_ok=True)
    left_path = args.output / "left_rectified.png"
    right_path = args.output / "right_rectified.png"
    cv2.imwrite(str(left_path), left_rectified)
    cv2.imwrite(str(right_path), right_rectified)
    print(f"Saved {left_path} and {right_path}")


if __name__ == "__main__":
    main()
