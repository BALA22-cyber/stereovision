"""Preview the stereo cameras; Space saves a pair and Q quits.

Use --continuous to keep capturing additional pairs with each Space press.
Images are full-resolution, lossless BGR PNGs, converted from the sensor format.
The cameras stream independently; this script does not hardware synchronize them.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
from pypylon import pylon


def camera_settings(camera, info):
    settings = {"model": info.GetModelName(), "serial": info.GetSerialNumber()}
    for feature in ("Width", "Height", "PixelFormat", "ExposureTime", "Gain"):
        try:
            settings[feature] = getattr(camera, feature).GetValue()
        except (AttributeError, RuntimeError):
            settings[feature] = None
    return settings


def grab(camera, converter):
    with camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as result:
        if not result.GrabSucceeded():
            raise RuntimeError(result.ErrorDescription)
        image = converter.Convert(result).Array.copy()
        return image, {
            "host_received_utc": datetime.now(timezone.utc).isoformat(),
            "camera_timestamp_ticks": int(result.GetTimeStamp()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", action="store_true",
                        help="keep running: each Space saves another pair; Q quits")
    parser.add_argument("--count", type=int, help="stop after this many saved pairs (requires --continuous for more than one)")
    parser.add_argument("--output", type=Path, default=Path("data/captures/stereo_capture_pair"))
    parser.add_argument("--left-serial", default="41958448")
    parser.add_argument("--right-serial", default="41958460")
    args = parser.parse_args()
    if args.count is not None and (args.count < 1 or (args.count > 1 and not args.continuous)):
        parser.error("--count must be positive; counts above 1 require --continuous")
    if args.left_serial == args.right_serial:
        parser.error("left and right serials must differ")

    factory = pylon.TlFactory.GetInstance()
    devices = {d.GetSerialNumber(): d for d in factory.EnumerateDevices()}
    missing = [s for s in (args.left_serial, args.right_serial) if s not in devices]
    if missing:
        raise SystemExit(f"Camera(s) not found: {', '.join(missing)}. Detected serials: {', '.join(devices) or 'none'}")

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment.Value = pylon.OutputBitAlignment_MsbAligned
    window = "Stereo capture: SPACE saves | Q quits"
    count = 0
    with ExitStack() as cleanup:
        cameras = []
        settings = {}
        for side, serial in (("left", args.left_serial), ("right", args.right_serial)):
            camera = pylon.InstantCamera(factory.CreateDevice(devices[serial]))
            camera.Open()
            cleanup.callback(camera.Close)
            settings[side] = camera_settings(camera, devices[serial])
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            cameras.append(camera)
        cleanup.callback(cv2.destroyAllWindows)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1600, 500)
        print(f"Left: {args.left_serial}; right: {args.right_serial}")
        print("SPACE saves a pair; Q or Esc quits. Keep the scene still for paired captures.")
        print("Continuous capture enabled." if args.continuous else "The program exits after the first saved pair.")

        while True:
            left, left_timing = grab(cameras[0], converter)
            right, right_timing = grab(cameras[1], converter)
            views = []
            for side, image in (("LEFT", left), ("RIGHT", right)):
                height = 450
                view = cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))
                cv2.putText(view, side, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                views.append(view)
            preview = np.hstack(views)
            cv2.putText(preview, f"Saved {count} pairs | SPACE: save | Q: quit", (20, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key != ord(" "):
                continue

            args.output.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            prefix = f"pair_{stamp}"
            left_name = f"{prefix}_left_{args.left_serial}.png"
            right_name = f"{prefix}_right_{args.right_serial}.png"
            written = []
            try:
                for name, image in ((left_name, left), (right_name, right)):
                    path = args.output / name
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError(f"Failed to save {path}")
                    written.append(path)
                metadata = {
                    "camera_settings": settings,
                    "left_image": left_name,
                    "right_image": right_name,
                    "stored_pixel_format": "BGR8 PNG (demosaiced)",
                    "hardware_synchronized": False,
                    "left_timing": left_timing,
                    "right_timing": right_timing,
                }
                metadata_path = args.output / f"{prefix}.json"
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            except Exception:
                for path in written:
                    path.unlink(missing_ok=True)
                raise
            count += 1
            print(f"Saved pair {count}: {left_name} + {right_name}")
            if not args.continuous or (args.count is not None and count >= args.count):
                break
    print(f"Finished: {count} pairs saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
