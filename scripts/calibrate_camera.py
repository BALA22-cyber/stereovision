"""Estimate one camera's intrinsics from capture_calibration.py images."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="folder containing checkerboard PNGs")
    parser.add_argument("--cols", type=int, required=True, help="inner corners across")
    parser.add_argument("--rows", type=int, required=True, help="inner corners down")
    parser.add_argument("--square-mm", type=float, required=True, help="square side length in mm")
    parser.add_argument("--output", type=Path, default=Path("camera_intrinsics"))
    parser.add_argument("--undistort", action="store_true", help="also write undistorted copies of input images")
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    object_template = np.zeros((args.rows * args.cols, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    object_template *= args.square_mm
    object_points, image_points = [], []
    image_size = None
    accepted, rejected = [], []

    paths = sorted(p for p in args.image_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    if not paths:
        raise SystemExit(f"No supported images found in {args.image_dir}")

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected.append((path.name, "could not read image"))
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            rejected.append((path.name, f"size {current_size} differs from {image_size}"))
            continue

        found, corners = cv2.findChessboardCornersSB(image, pattern)
        if not found:
            rejected.append((path.name, "checkerboard not detected"))
            continue
        refined = cv2.cornerSubPix(
            image, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
        )
        object_points.append(object_template.copy())
        image_points.append(refined)
        accepted.append(path)

    if len(accepted) < 8:
        raise SystemExit(f"Only {len(accepted)} usable views; capture at least 8, preferably 15-25 varied views.")

    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view_errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
        per_view_errors.append(float(cv2.norm(img, projected, cv2.NORM_L2) / len(projected) ** 0.5))

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "intrinsics.npz", camera_matrix=camera_matrix, distortion_coefficients=distortion,
             image_size=np.array(image_size), rms=np.array(rms))
    report = {
        "image_size_px": list(image_size),
        "board_inner_corners": [args.cols, args.rows],
        "square_size_mm": args.square_mm,
        "views_used": len(accepted),
        "views_rejected": [{"file": n, "reason": reason} for n, reason in rejected],
        "rms_reprojection_error_px": float(rms),
        "mean_per_view_error_px": float(np.mean(per_view_errors)),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "distortion_order": "OpenCV: k1, k2, p1, p2, k3 (additional terms may follow)",
    }
    (args.output / "intrinsics.json").write_text(json.dumps(report, indent=2) + "\n")

    if args.undistort:
        undistorted_dir = args.output / "undistorted"
        undistorted_dir.mkdir(exist_ok=True)
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, image_size, 0, image_size)
        for path in accepted:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            corrected = cv2.undistort(image, camera_matrix, distortion, None, new_matrix)
            cv2.imwrite(str(undistorted_dir / path.name), corrected)
        report["undistorted_output"] = str(undistorted_dir)
        report["undistorted_valid_roi_xywh"] = [int(v) for v in roi]
        (args.output / "intrinsics.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"Used {len(accepted)} views; rejected {len(rejected)}.")
    print(f"RMS reprojection error: {rms:.4f} px")
    print("Camera matrix K:\n", camera_matrix)
    print("Distortion coefficients:\n", distortion.reshape(-1))
    print(f"Wrote calibration to {args.output.resolve()}")
    if rejected:
        print("Rejected:", ", ".join(f"{name} ({why})" for name, why in rejected))


if __name__ == "__main__":
    main()
