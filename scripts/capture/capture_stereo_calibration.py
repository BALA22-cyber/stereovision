"""Capture paired checkerboard images from the left and right Basler cameras.

Press Space to save a pair manually, or pass --auto to save at a fixed interval
when OpenCV detects the configured checkerboard in both views. Keep the board
still while a pair is acquired. Press q or Esc to finish early.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from pypylon import pylon


DEFAULT_LEFT_SERIAL = "41958448"
DEFAULT_RIGHT_SERIAL = "41958460"


def feature_value(camera, name):
    try:
        node = getattr(camera, name)
        if node.IsReadable():
            return node.GetValue()
    except (AttributeError, RuntimeError):
        pass
    return None


def detect_board(frame, pattern):
    if frame.ndim == 2:
        gray = frame
    elif frame.shape[2] == 1:
        gray = frame[:, :, 0]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.findChessboardCornersSB(gray, pattern)


def display_frame(frame, pattern, found, corners, label, board_type="checkerboard"):
    if frame.ndim == 2:
        view = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:
        view = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        view = frame.copy()
    if found:
        if board_type == "charuco":
            cv2.aruco.drawDetectedCornersCharuco(view, corners[0], corners[1].reshape(-1, 1))
        else:
            cv2.drawChessboardCorners(view, pattern, corners, found)
    status = "BOARD FOUND" if found else "BOARD NOT FOUND"
    color = (0, 220, 0) if found else (0, 0, 255)
    cv2.putText(view, f"{label}: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    height, width = view.shape[:2]
    scale = min(960 / width, 600 / height, 1.0)
    return cv2.resize(view, (int(width * scale), int(height * scale)))


def review_calibration(images, cols, rows, square_mm, board_type="checkerboard", marker_mm=None, dictionary=None, charuco_legacy=False, min_charuco_corners=6):
    """Run the existing pipeline after capture, in a worker thread."""
    started = time.monotonic()
    out = images / "calibration_review"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "review_summary.json"
    report = {"status": "running", "measurement_scope": "in-sample; not independent validation",
              "square_size_mm": square_mm, "input_images": str(images)}
    try:
        pipeline_path = Path(__file__).resolve().parent.parent / "stereo_calibration_workflow" / "stereo_pipeline.py"
        spec = importlib.util.spec_from_file_location("capture_review_stereo_pipeline", pipeline_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load pipeline: {pipeline_path}")
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        dbg = pipeline.Debugger(False, out / "debug")
        base = dict(images=images, cols=cols, rows=rows, square_size=square_mm / 1000.0,
                    board_type=board_type, marker_size=marker_mm / 1000.0 if marker_mm else None,
                    dictionary=dictionary, charuco_legacy=charuco_legacy, min_charuco_corners=min_charuco_corners)
        target = pipeline.make_charuco_target(argparse.Namespace(**base)) if board_type == "charuco" else None
        intrinsics_path = out / "intrinsics.npz"
        try:
            pipeline.calibrate_intrinsics(argparse.Namespace(**base, out=intrinsics_path), dbg)
            pipeline.calibrate_rig_board(argparse.Namespace(**base, intrinsics=intrinsics_path,
                                                            out=out / "rig", alpha=0.0), dbg)
            rig = pipeline.as_dict(out / "rig" / "rig.npz")
            pair_metrics = []
            for name, left_path, right_path in pipeline.camera_pairs(images):
                left, right = pipeline.read_pair(left_path, right_path, rig["size"])
                if target:
                    dl, dr = target.detect(left), target.detect(right)
                    shared = target.shared(dl, dr)
                    cl, cr = (shared[1], shared[2]) if shared is not None else (None, None)
                else:
                    cl = pipeline.detect_checkerboard(left, cols, rows)
                    cr = pipeline.detect_checkerboard(right, cols, rows)
                row = {"pair": name, "left_detected": cl is not None, "right_detected": cr is not None}
                if cl is not None and cr is not None:
                    row.update(pipeline.epi_metrics([cl], [cr], rig))
                pair_metrics.append(row)
            ranked = sorted((row for row in pair_metrics if "median_vertical_epipolar_error_px" in row),
                            key=lambda row: row["median_vertical_epipolar_error_px"], reverse=True)
            report.update(status="complete", pairs=pair_metrics,
                          pairs_ranked_by_median_vertical_error=[row["pair"] for row in ranked],
                          rig_diagnostics=json.loads((out / "rig" / "rig_report.json").read_text())["diagnostics"])
        finally:
            dbg.close()
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"[review] {report['status']}; {report['elapsed_seconds']:.2f}s; {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/calibration_images/stereo_calibration_images"))
    parser.add_argument("--cols", type=int, default=9, help="inner checkerboard corners across (default: 9)")
    parser.add_argument("--rows", type=int, default=6, help="inner checkerboard corners down (default: 6)")
    parser.add_argument("--square-mm", type=float, default=80.0, help="checker square side length in mm (default: 80)")
    parser.add_argument("--count", type=int, default=25, help="number of stereo pairs to collect (default: 25)")
    parser.add_argument("--left-serial", default=DEFAULT_LEFT_SERIAL)
    parser.add_argument("--right-serial", default=DEFAULT_RIGHT_SERIAL)
    parser.add_argument("--auto", action="store_true",
                        help="automatically save valid pairs at the configured interval")
    parser.add_argument("--interval-sec", type=float, default=2.0,
                        help="seconds between automatic pair captures (default: 2)")
    parser.add_argument("--board-type", choices=["checkerboard", "charuco"], default="checkerboard")
    parser.add_argument("--marker-mm", type=float, help="ChArUco marker edge length in mm")
    parser.add_argument("--dictionary", help="OpenCV ArUco dictionary matching the printed board")
    parser.add_argument("--charuco-legacy", action="store_true", help="use the legacy OpenCV ChArUco layout")
    parser.add_argument("--min-charuco-corners", type=int, default=6, help="minimum non-collinear shared corners")
    parser.add_argument("--review-calibration", action="store_true",
                        help="after capture, run pipeline calibration and per-pair review in a worker thread")
    args = parser.parse_args()

    if (args.cols < 2 or args.rows < 2 or args.square_mm <= 0 or args.count < 1
            or args.interval_sec <= 0):
        parser.error("cols/rows must be >= 2; square-mm, count, and interval-sec must be positive")

    target = None
    if args.board_type == "charuco":
        if args.marker_mm is None or not args.dictionary:
            parser.error("ChArUco requires --marker-mm and --dictionary matching the printed board")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stereo_calibration_workflow"))
        from charuco_support import CharucoTarget
        try:
            target = CharucoTarget(args.cols, args.rows, args.square_mm, args.marker_mm,
                                   args.dictionary, args.charuco_legacy, args.min_charuco_corners)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))

    if args.output.exists():
        if not args.output.is_dir():
            parser.error(f"Output path is not a directory: {args.output}")
        if any(args.output.iterdir()):
            parser.error(
                f"Output folder is not empty: {args.output}. "
                "Choose a new --output folder to preserve previous captures."
            )

    factory = pylon.TlFactory.GetInstance()
    devices = list(factory.EnumerateDevices())
    by_serial = {device.GetSerialNumber(): device for device in devices}
    required = [args.left_serial, args.right_serial]
    missing = [serial for serial in required if serial not in by_serial]
    if missing:
        seen = ", ".join(
            f"{d.GetModelName()} ({d.GetSerialNumber()})" for d in devices
        ) or "none"
        raise SystemExit(
            f"Could not find required camera serial(s): {', '.join(missing)}. "
            f"pypylon currently sees: {seen}. Check that the .venv and pylon runtime are active."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    pattern = (args.cols, args.rows)
    left_info, right_info = by_serial[args.left_serial], by_serial[args.right_serial]
    left = pylon.InstantCamera(factory.CreateDevice(left_info))
    right = pylon.InstantCamera(factory.CreateDevice(right_info))
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment.Value = pylon.OutputBitAlignment_MsbAligned
    captures = []

    left.Open()
    try:
        right.Open()
        try:
            left.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            right.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            next_auto_capture = time.monotonic() + args.interval_sec
            camera_settings = {
                "left": {
                    "model": left_info.GetModelName(),
                    "serial": args.left_serial,
                    "width_px": feature_value(left, "Width"),
                    "height_px": feature_value(left, "Height"),
                    "pixel_format": feature_value(left, "PixelFormat"),
                    "exposure_time_us": feature_value(left, "ExposureTime"),
                    "gain": feature_value(left, "Gain"),
                },
                "right": {
                    "model": right_info.GetModelName(),
                    "serial": args.right_serial,
                    "width_px": feature_value(right, "Width"),
                    "height_px": feature_value(right, "Height"),
                    "pixel_format": feature_value(right, "PixelFormat"),
                    "exposure_time_us": feature_value(right, "ExposureTime"),
                    "gain": feature_value(right, "Gain"),
                },
            }
            print(f"Left: {left_info.GetModelName()} serial {args.left_serial}")
            print(f"Right: {right_info.GetModelName()} serial {args.right_serial}")
            print(f"Board: {args.cols} x {args.rows} inner corners; square size {args.square_mm:g} mm")
            if args.auto:
                print(f"Automatic capture every {args.interval_sec:g} seconds when both views detect the board; q/Esc quits.")
                print("Move the board to a new position between captures to collect varied calibration views.")
            else:
                print("Hold the board still in both views. Space saves a valid pair; q/Esc quits.")

            while left.IsGrabbing() and right.IsGrabbing() and len(captures) < args.count:
                with left.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as lres:
                    if not lres.GrabSucceeded():
                        print(f"Left grab failed: {lres.ErrorDescription}")
                        continue
                    left_frame = converter.Convert(lres).Array.copy()
                with right.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as rres:
                    if not rres.GrabSucceeded():
                        print(f"Right grab failed: {rres.ErrorDescription}")
                        continue
                    right_frame = converter.Convert(rres).Array.copy()

                if target:
                    left_corners, right_corners = target.detect(left_frame), target.detect(right_frame)
                    left_found, right_found = left_corners is not None, right_corners is not None
                    shared = target.shared(left_corners, right_corners)
                    usable = shared is not None
                else:
                    left_found, left_corners = detect_board(left_frame, pattern)
                    right_found, right_corners = detect_board(right_frame, pattern)
                    usable = left_found and right_found
                left_view = display_frame(left_frame, pattern, left_found, left_corners, "LEFT", args.board_type)
                right_view = display_frame(right_frame, pattern, right_found, right_corners, "RIGHT", args.board_type)
                if left_view.shape[0] != right_view.shape[0]:
                    target_height = min(left_view.shape[0], right_view.shape[0])
                    left_view = cv2.resize(left_view, (int(left_view.shape[1] * target_height / left_view.shape[0]), target_height))
                    right_view = cv2.resize(right_view, (int(right_view.shape[1] * target_height / right_view.shape[0]), target_height))
                preview = np.hstack((left_view, right_view))
                if args.auto:
                    remaining = max(0.0, next_auto_capture - time.monotonic())
                    capture_status = f"AUTO in {remaining:.1f}s" if remaining > 0 else "AUTO: READY"
                else:
                    capture_status = "SPACE: SAVE PAIR"
                cv2.putText(preview, f"Saved {len(captures)}/{args.count} pairs  |  {capture_status}",
                            (20, preview.shape[0] - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)
                cv2.imshow("Stereo checkerboard capture", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if args.auto:
                    if time.monotonic() < next_auto_capture:
                        continue
                elif key != ord(" "):
                    continue
                if not usable:
                    print("Pair skipped: target must have enough corresponding corners in both views.")
                    if args.auto:
                        next_auto_capture = time.monotonic() + args.interval_sec
                    continue

                pair_index = len(captures) + 1
                left_name = f"pair_{pair_index:03d}_left_{args.left_serial}.png"
                right_name = f"pair_{pair_index:03d}_right_{args.right_serial}.png"
                if not cv2.imwrite(str(args.output / left_name), left_frame):
                    raise RuntimeError(f"Could not write {args.output / left_name}")
                if not cv2.imwrite(str(args.output / right_name), right_frame):
                    raise RuntimeError(f"Could not write {args.output / right_name}")
                captures.append({
                    "pair": pair_index,
                    "captured_utc": datetime.now(timezone.utc).isoformat(),
                    "left_image": left_name,
                    "right_image": right_name,
                    "left_size_px": [int(left_frame.shape[1]), int(left_frame.shape[0])],
                    "right_size_px": [int(right_frame.shape[1]), int(right_frame.shape[0])],
                })
                metadata = {
                    "camera_settings": camera_settings,
                    "stored_pixel_format": "BGR8 PNG (demosaiced by pylon)",
                    "hardware_synchronized": False,
                    "board_type": args.board_type,
                    "charuco_dictionary": args.dictionary if target else None,
                    "charuco_marker_size_mm": args.marker_mm if target else None,
                    "charuco_legacy_pattern": args.charuco_legacy if target else None,
                    "min_charuco_corners": args.min_charuco_corners if target else None,
                    "board_inner_corners_cols_rows": [args.cols, args.rows],
                    "square_size_mm": args.square_mm,
                    "pairs": captures,
                }
                (args.output / "capture_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"Saved pair {pair_index}: {left_name} + {right_name}")
                if args.auto:
                    next_auto_capture = time.monotonic() + args.interval_sec
        finally:
            right.Close()
    finally:
        left.Close()
        cv2.destroyAllWindows()

    print(f"Finished with {len(captures)} paired views in {args.output.resolve()}")
    if args.review_calibration:
        print("[review] Running detection and calibration on a background worker; waiting for results.")
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="calibration-review") as executor:
            executor.submit(review_calibration, args.output, args.cols, args.rows, args.square_mm,
                            args.board_type, args.marker_mm, args.dictionary, args.charuco_legacy,
                            args.min_charuco_corners).result()


if __name__ == "__main__":
    main()
