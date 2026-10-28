"""Calibrate a Basler stereo pair and save intrinsics/rectification maps."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def gray_for_calibration(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def color_for_preview(image, pixel_format):
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if pixel_format == "BayerRG8":
        return cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/calibration_images/stereo_calibration_images"))
    parser.add_argument("--output", type=Path, default=Path("output/legacy/stereo_calibration_output"))
    parser.add_argument("--cols", type=int, help="override inner-corner count across")
    parser.add_argument("--rows", type=int, help="override inner-corner count down")
    parser.add_argument("--square-mm", type=float, help="override checker square side length in mm")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="rectification crop parameter: 0 crops invalid borders; 1 keeps all pixels")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("alpha must be between 0 and 1")

    metadata_path = args.images / "capture_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing capture metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    cols, rows = metadata["board_inner_corners_cols_rows"]
    square_mm = metadata["square_size_mm"]
    cols = args.cols if args.cols is not None else cols
    rows = args.rows if args.rows is not None else rows
    square_mm = args.square_mm if args.square_mm is not None else square_mm
    pattern = (int(cols), int(rows))
    pairs = metadata.get("pairs", [])
    if not pairs:
        raise SystemExit("No stereo pairs are listed in capture_metadata.json")

    object_template = np.zeros((rows * cols, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_template *= float(square_mm)

    object_points = []
    left_points = []
    right_points = []
    usable_pairs = []
    rejected_pairs = []
    image_size = None

    for pair in pairs:
        left_path = args.images / pair["left_image"]
        right_path = args.images / pair["right_image"]
        left_image = read_image(left_path)
        right_image = read_image(right_path)
        left_size = (left_image.shape[1], left_image.shape[0])
        right_size = (right_image.shape[1], right_image.shape[0])
        if left_size != right_size:
            rejected_pairs.append({"pair": pair.get("pair"), "reason": "left/right image sizes differ"})
            continue
        if image_size is None:
            image_size = left_size
        if left_size != image_size:
            rejected_pairs.append({"pair": pair.get("pair"), "reason": "image size differs from other pairs"})
            continue

        left_found, lcorners = cv2.findChessboardCornersSB(gray_for_calibration(left_image), pattern)
        right_found, rcorners = cv2.findChessboardCornersSB(gray_for_calibration(right_image), pattern)
        if not (left_found and right_found):
            rejected_pairs.append({
                "pair": pair.get("pair"),
                "reason": f"board detection left={bool(left_found)} right={bool(right_found)}",
            })
            continue

        object_points.append(object_template.copy())
        left_points.append(lcorners.astype(np.float32))
        right_points.append(rcorners.astype(np.float32))
        usable_pairs.append(pair)

    if len(usable_pairs) < 8:
        raise SystemExit(f"Only {len(usable_pairs)} usable pairs; need at least 8 varied board poses.")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    left_rms, K_left, D_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        object_points, left_points, image_size, None, None, criteria=criteria
    )
    right_rms, K_right, D_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        object_points, right_points, image_size, None, None, criteria=criteria
    )

    stereo_rms, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        K_left,
        D_left,
        K_right,
        D_right,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    R_left, R_right, P_left, P_right, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=args.alpha,
        newImageSize=image_size,
    )
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, D_left, R_left, P_left, image_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, D_right, R_right, P_right, image_size, cv2.CV_32FC1
    )

    # Check how closely matched checkerboard corners land on the same rows.
    vertical_errors = []
    for lcorners, rcorners in zip(left_points, right_points):
        lrect = cv2.undistortPoints(lcorners, K_left, D_left, R=R_left, P=P_left)
        rrect = cv2.undistortPoints(rcorners, K_right, D_right, R=R_right, P=P_right)
        vertical_errors.extend(np.abs(lrect[:, 0, 1] - rrect[:, 0, 1]).tolist())

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "stereo_calibration.npz",
        K_left=K_left, D_left=D_left, K_right=K_right, D_right=D_right,
        R=R, T=T, E=E, F=F,
        R_left=R_left, R_right=R_right, P_left=P_left, P_right=P_right, Q=Q,
        map_left_x=map_left_x, map_left_y=map_left_y,
        map_right_x=map_right_x, map_right_y=map_right_y,
        image_size=np.array(image_size),
        roi_left=np.array(roi_left), roi_right=np.array(roi_right),
    )

    report = {
        "left_camera": metadata["camera_settings"]["left"],
        "right_camera": metadata["camera_settings"]["right"],
        "board_inner_corners_cols_rows": [cols, rows],
        "square_size_mm": float(square_mm),
        "image_size_px": list(image_size),
        "pairs_in_metadata": len(pairs),
        "pairs_used": len(usable_pairs),
        "pairs_rejected": rejected_pairs,
        "left_mono_rms_reprojection_error_px": float(left_rms),
        "right_mono_rms_reprojection_error_px": float(right_rms),
        "stereo_rms_reprojection_error_px": float(stereo_rms),
        "mean_rectified_vertical_disparity_px": float(np.mean(vertical_errors)),
        "baseline_mm": float(np.linalg.norm(T)),
        "K_left": K_left.tolist(),
        "D_left": D_left.reshape(-1).tolist(),
        "K_right": K_right.tolist(),
        "D_right": D_right.reshape(-1).tolist(),
        "R_left_to_right": R.tolist(),
        "T_left_to_right_mm": T.reshape(-1).tolist(),
        "R_rect_left": R_left.tolist(),
        "R_rect_right": R_right.tolist(),
        "P_rect_left": P_left.tolist(),
        "P_rect_right": P_right.tolist(),
        "Q_disparity_to_depth": Q.tolist(),
        "valid_roi_left_xywh": [int(v) for v in roi_left],
        "valid_roi_right_xywh": [int(v) for v in roi_right],
        "rectification_alpha": float(args.alpha),
    }
    (args.output / "stereo_calibration.json").write_text(json.dumps(report, indent=2) + "\n")

    for side, camera, K, D, rms in (
        ("left", metadata["camera_settings"]["left"], K_left, D_left, left_rms),
        ("right", metadata["camera_settings"]["right"], K_right, D_right, right_rms),
    ):
        intrinsic_report = {
            "camera": camera,
            "image_size_px": list(image_size),
            "board_inner_corners_cols_rows": [cols, rows],
            "square_size_mm": float(square_mm),
            "mono_rms_reprojection_error_px": float(rms),
            "camera_matrix_K": K.tolist(),
            "distortion_coefficients": D.reshape(-1).tolist(),
            "distortion_order": "OpenCV: k1, k2, p1, p2, k3",
        }
        (args.output / f"{side}_intrinsics.json").write_text(json.dumps(intrinsic_report, indent=2) + "\n")
        np.savez_compressed(
            args.output / f"{side}_intrinsics.npz",
            camera_matrix=K,
            distortion_coefficients=D,
            image_size=np.array(image_size),
            rms=np.array(rms),
        )

    # Write one full-resolution rectified pair and an easy-to-inspect side-by-side preview.
    first = usable_pairs[0]
    left_raw = read_image(args.images / first["left_image"])
    right_raw = read_image(args.images / first["right_image"])
    pixel_left = metadata["camera_settings"]["left"].get("pixel_format")
    pixel_right = metadata["camera_settings"]["right"].get("pixel_format")
    left_color = color_for_preview(left_raw, pixel_left)
    right_color = color_for_preview(right_raw, pixel_right)
    left_rect = cv2.remap(left_color, map_left_x, map_left_y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_color, map_right_x, map_right_y, cv2.INTER_LINEAR)
    cv2.imwrite(str(args.output / "pair_001_left_rectified.png"), left_rect)
    cv2.imwrite(str(args.output / "pair_001_right_rectified.png"), right_rect)

    target_width = 960
    target_height = int(image_size[1] * target_width / image_size[0])
    left_small = cv2.resize(left_rect, (target_width, target_height))
    right_small = cv2.resize(right_rect, (target_width, target_height))
    preview = np.hstack((left_small, right_small))
    for y in range(0, target_height, 40):
        cv2.line(preview, (0, y), (preview.shape[1] - 1, y), (0, 0, 255), 1)
    cv2.imwrite(str(args.output / "rectification_preview.png"), preview)

    print(f"Pairs used: {len(usable_pairs)}/{len(pairs)}")
    print(f"Left RMS: {left_rms:.4f} px; right RMS: {right_rms:.4f} px; stereo RMS: {stereo_rms:.4f} px")
    print(f"Mean rectified vertical disparity: {np.mean(vertical_errors):.4f} px")
    print(f"Estimated baseline: {np.linalg.norm(T):.3f} mm")
    print(f"Wrote calibration and rectified example to {args.output.resolve()}")
    if rejected_pairs:
        print(f"Rejected {len(rejected_pairs)} pairs; see stereo_calibration.json")


if __name__ == "__main__":
    main()
