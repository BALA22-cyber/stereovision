"""Capture checkerboard views from a Basler camera for camera calibration.

With OpenCV installed, a live preview appears and Space saves a frame only
when the configured checkerboard is detected. Without OpenCV, press Enter in
the terminal to save the current frame, or type q to stop.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pypylon import pylon


def readable_value(camera, feature_name):
    camera_metadata = {}
    try:
        feature = getattr(camera, feature_name)
        if feature.IsReadable():
            return feature.GetValue()
    except (AttributeError, RuntimeError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("calibration_images"))
    parser.add_argument("--cols", type=int, required=True,
                        help="number of inner checkerboard corners across the board")
    parser.add_argument("--rows", type=int, required=True,
                        help="number of inner checkerboard corners down the board")
    parser.add_argument("--square-mm", type=float, required=True,
                        help="measured side length of one printed square, in mm")
    parser.add_argument("--count", type=int, default=25,
                        help="stop after this many accepted images (default: 25)")
    parser.add_argument("--serial", help="select a camera by serial number")
    args = parser.parse_args()

    if args.cols < 2 or args.rows < 2 or args.square_mm <= 0 or args.count < 1:
        parser.error("cols/rows must be >= 2, square-mm and count must be positive")

    try:
        import cv2
    except ImportError:
        cv2 = None

    factory = pylon.TlFactory.GetInstance()
    devices = list(factory.EnumerateDevices())
    if args.serial:
        devices = [d for d in devices if d.GetSerialNumber() == args.serial]
    if not devices:
        raise SystemExit("No matching Basler camera found. Check USB/pylon Viewer and serial selection.")
    if len(devices) > 1 and not args.serial:
        print("Multiple cameras found; using the first. Select with --serial.")

    device = devices[0]
    args.output.mkdir(parents=True, exist_ok=True)
    board = (args.cols, args.rows)
    saved = []
    camera = pylon.InstantCamera(factory.CreateDevice(device))
    camera.Open()
    try:
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(f"Camera: {device.GetModelName()} serial {device.GetSerialNumber()}")
        print(f"Board: {args.cols} x {args.rows} inner corners; square {args.square_mm:g} mm")
        if cv2 is not None:
            print("Live preview: move the board through varied positions/tilts; Space saves a detected view, q quits.")
        else:
            print("OpenCV is unavailable, so there is no preview or automatic board check.")
            print("Press Enter to save a frame after checking the board position in pylon Viewer; type q to quit.")

        while camera.IsGrabbing() and len(saved) < args.count:
            with camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as result:
                if not result.GrabSucceeded():
                    print(f"Frame grab failed: {result.ErrorDescription}", file=sys.stderr)
                    continue

                frame = result.Array.copy()
                detected = False
                corners = None
                if cv2 is not None:
                    gray = frame
                    if frame.ndim == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected, corners = cv2.findChessboardCornersSB(gray, board)
                    preview = frame.copy()
                    if detected:
                        cv2.drawChessboardCorners(preview, board, corners, detected)
                    cv2.putText(preview, f"Saved {len(saved)}/{args.count}  " +
                                ("BOARD FOUND" if detected else "board not found"),
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 220, 0) if detected else (0, 0, 255), 2)
                    cv2.imshow("Basler calibration capture", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    if key != ord(" "):
                        continue
                    if not detected:
                        print("Checkerboard not detected; frame skipped.")
                        continue
                else:
                    answer = input(f"Save frame {len(saved) + 1}/{args.count}? [Enter/q] ").strip().lower()
                    if answer == "q":
                        break

                index = len(saved) + 1
                filename = f"image_{index:03d}.png"
                image = pylon.PylonImage()
                image.AttachGrabResultBuffer(result)
                try:
                    image.Save(pylon.ImageFileFormat_Png, str(args.output / filename))
                finally:
                    image.Release()
                saved.append({
                    "file": filename,
                    "captured_utc": datetime.now(timezone.utc).isoformat(),
                    "width_px": int(result.GetWidth()),
                    "height_px": int(result.GetHeight()),
                    "board_inner_corners": [args.cols, args.rows],
                    "square_size_mm": args.square_mm,
                })
                print(f"Saved {args.output / filename} ({len(saved)}/{args.count})")
        camera_metadata = {
            "camera_model": device.GetModelName(),
            "camera_serial": device.GetSerialNumber(),
            "width_setting_px": readable_value(camera, "Width"),
            "height_setting_px": readable_value(camera, "Height"),
            "pixel_format": readable_value(camera, "PixelFormat"),
            "exposure_time_us": readable_value(camera, "ExposureTime"),
            "gain": readable_value(camera, "Gain"),
        }
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
        camera.Close()

    metadata = {
        **camera_metadata,
        "board_inner_corners": [args.cols, args.rows],
        "square_size_mm": args.square_mm,
        "images": saved,
    }
    (args.output / "capture_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(f"Finished: {len(saved)} images and metadata in {args.output.resolve()}")


if __name__ == "__main__":
    main()
