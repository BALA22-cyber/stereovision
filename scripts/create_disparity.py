"""Rectify a captured stereo pair and compute a StereoSGBM disparity map.

Reference: https://docs.opencv.org/4.8.0/d2/d85/classcv_1_1StereoSGBM.html
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from rectify_stereo import load_color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=Path("data/calibration_images/stereo_calibration_images/pair_001_left_41958448.png"))
    parser.add_argument("--right", type=Path, default=Path("data/calibration_images/stereo_calibration_images/pair_001_right_41958460.png"))
    parser.add_argument("--calibration", type=Path, default=Path("output/legacy/stereo_calibration_output"))
    parser.add_argument("--output", type=Path, default=Path("output/legacy/disparity_pair_001"))
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.65,
                        help="rectification field of view; wider than the calibration preview")
    parser.add_argument("--min-disparity", type=int, default=256,
                        help="search start in processing pixels; default tuned for calibration pair 1")
    parser.add_argument("--num-disparities", type=int, default=512)
    args = parser.parse_args()
    if not 0 < args.scale <= 1 or not 0 <= args.alpha <= 1:
        parser.error("scale must be in (0,1] and alpha in [0,1]")
    if args.num_disparities <= 0 or args.num_disparities % 16:
        parser.error("num-disparities must be a positive multiple of 16")

    c = np.load(args.calibration / "stereo_calibration.npz")
    report = json.loads((args.calibration / "stereo_calibration.json").read_text())
    left = load_color(args.left, report["left_camera"].get("pixel_format"))
    right = load_color(args.right, report["right_camera"].get("pixel_format"))
    original_size = tuple(int(v) for v in c["image_size"])
    if left.shape[:2][::-1] != original_size or right.shape[:2][::-1] != original_size:
        raise SystemExit("Input images must match the calibrated resolution.")
    size = tuple(round(v * args.scale) for v in original_size)
    if args.num_disparities + max(args.min_disparity, 0) >= size[0]:
        parser.error("disparity search must fit within the processing image width")
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        c["K_left"], c["D_left"], c["K_right"], c["D_right"], original_size,
        c["R"], c["T"], flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=args.alpha, newImageSize=size,
    )
    mx1, my1 = cv2.initUndistortRectifyMap(c["K_left"], c["D_left"], r1, p1, size, cv2.CV_32FC1)
    mx2, my2 = cv2.initUndistortRectifyMap(c["K_right"], c["D_right"], r2, p2, size, cv2.CV_32FC1)
    lrect = cv2.remap(left, mx1, my1, cv2.INTER_LINEAR)
    rrect = cv2.remap(right, mx2, my2, cv2.INTER_LINEAR)
    lg = cv2.cvtColor(lrect, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(rrect, cv2.COLOR_BGR2GRAY)
    block = 5

    def matcher(minimum):
        return cv2.StereoSGBM_create(
            minDisparity=minimum, numDisparities=args.num_disparities, blockSize=block,
            P1=8 * block * block, P2=32 * block * block,
            disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
            speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    print(f"Computing disparity at {size[0]}x{size[1]}...", flush=True)
    disparity = matcher(args.min_disparity).compute(lg, rg).astype(np.float32) / 16.0
    reverse_min = -args.min_disparity - args.num_disparities + 1
    reverse = matcher(reverse_min).compute(rg, lg).astype(np.float32) / 16.0
    yy, xx = np.indices(disparity.shape)
    xr = np.rint(xx - disparity).astype(np.int32)
    in_bounds = (xr >= 0) & (xr < size[0])
    xr_safe = np.clip(xr, 0, size[0] - 1)
    consistent = np.abs(disparity + reverse[yy, xr_safe]) <= 1.5
    lmask = (mx1 >= 1) & (mx1 < original_size[0] - 1) & (my1 >= 1) & (my1 < original_size[1] - 1)
    rmask = (mx2 >= 1) & (mx2 < original_size[0] - 1) & (my2 >= 1) & (my2 < original_size[1] - 1)
    valid = (disparity > args.min_disparity) & in_bounds & consistent & lmask & rmask[yy, xr_safe]
    disparity[~valid] = np.nan
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "disparity_px.npy", disparity)
    cv2.imwrite(str(args.output / "valid_mask.png"), valid.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output / "left_rectified.png"), lrect)
    cv2.imwrite(str(args.output / "right_rectified.png"), rrect)
    np.savez_compressed(args.output / "rectification.npz", R_left=r1, R_right=r2, P_left=p1, P_right=p2,
                        Q=q, image_size=np.array(size), map_left_x=mx1, map_left_y=my1,
                        map_right_x=mx2, map_right_y=my2)
    if not valid.any():
        raise SystemExit("No consistent disparities found; raw arrays and rectified images were saved.")
    lo, hi = np.percentile(disparity[valid], [2, 98])
    normalized = np.clip((np.nan_to_num(disparity, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    cv2.imwrite(str(args.output / "disparity_color.png"), colored)
    # Standalone figure with units and a shared legend; black denotes invalid pixels.
    figure = np.full((size[1] + 115, size[0], 3), 245, np.uint8)
    figure[:size[1]] = colored
    cv2.putText(figure, "Pair 1 disparity | black: invalid / occluded / inconsistent", (25, size[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    x0, x1 = 30, min(size[0] - 30, 1000)
    gradient = np.tile(np.linspace(0, 255, x1 - x0, dtype=np.uint8), (20, 1))
    figure[size[1] + 40:size[1] + 60, x0:x1] = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    cv2.putText(figure, f"{lo:.1f} px", (x0, size[1] + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
    cv2.putText(figure, f"{hi:.1f} px (larger disparity = nearer)", (max(x0 + 140, x1 - 480), size[1] + 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
    cv2.imwrite(str(args.output / "disparity_map.png"), figure)
    summary = {
        "left_image": str(args.left), "right_image": str(args.right),
        "processing_size_px": list(size), "rectification_alpha": args.alpha,
        "disparity_units": "pixels in these rectified processing images (not original sensor pixels)",
        "min_disparity": args.min_disparity, "num_disparities": args.num_disparities,
        "valid_fraction_of_image": float(valid.mean()), "color_range_px": [float(lo), float(hi)],
        "invalid_value_in_npy": "NaN", "left_right_consistency_tolerance_px": 1.5,
        "calibration_mean_vertical_disparity_px_original_rectification": report.get("mean_rectified_vertical_disparity_px"),
    }
    (args.output / "disparity_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {args.output / 'disparity_map.png'}; valid pixels: {valid.mean():.1%}")


if __name__ == "__main__":
    main()
