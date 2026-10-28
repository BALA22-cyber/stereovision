"""Enumerate Basler cameras and save one frame from the first camera."""

from pathlib import Path

from pypylon import pylon


def main() -> None:
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        raise SystemExit("No Basler cameras found.")

    for device in devices:
        print(f"Found {device.GetModelName()} (serial {device.GetSerialNumber()})")

    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    camera.Open()
    try:
        camera.StartGrabbingMax(1)
        result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f"Capture failed: {result.ErrorDescription}")
            image = pylon.PylonImage()
            image.AttachGrabResultBuffer(result)
            try:
                output = Path("basler_capture.png")
                image.Save(pylon.ImageFileFormat_Png, str(output))
                print(f"Saved {result.GetWidth()}x{result.GetHeight()} image to {output.resolve()}")
            finally:
                image.Release()
        finally:
            result.Release()
    finally:
        camera.Close()


if __name__ == "__main__":
    main()
