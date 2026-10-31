"""Standalone camera diagnostic, independent of the rest of this project.

Tries every plausible device index against every plausible OpenCV backend on
this platform, reads a few frames from each combination that opens, and
reports resolution + mean brightness. Run this directly:

    python scripts/diagnose_camera.py

If every combination reports near-zero brightness, the camera access is
genuinely blocked at the OS level (permission, or another process holding
the device exclusively) rather than anything in this project's code. If one
combination reports real brightness, that tells us the working
device_index/backend to put in `src/config.py`.
"""
import platform
import sys
import time

import cv2


def candidate_backends() -> list[tuple[str, int]]:
    system = platform.system()
    candidates = [("CAP_ANY (default)", cv2.CAP_ANY)]
    if system == "Darwin":
        candidates.append(("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION))
    elif system == "Windows":
        candidates.append(("CAP_DSHOW", cv2.CAP_DSHOW))
        candidates.append(("CAP_MSMF", cv2.CAP_MSMF))
    else:
        candidates.append(("CAP_V4L2", cv2.CAP_V4L2))
    return candidates


def probe(index: int, backend_name: str, backend: int, n_frames: int = 40,
           timeout_s: float = 4.0) -> None:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"  index={index} backend={backend_name:<20} -> could not open")
        cap.release()
        return

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    start = time.time()
    best_brightness = 0.0
    frames_read = 0
    while time.time() - start < timeout_s:
        ok, frame = cap.read()
        if ok and frame is not None:
            frames_read += 1
            b = float(frame.mean())
            best_brightness = max(best_brightness, b)
            if b >= 5.0:
                break
        time.sleep(0.03)

    status = "LOOKS REAL" if best_brightness >= 5.0 else "black/near-black"
    print(f"  index={index} backend={backend_name:<20} opened={w:.0f}x{h:.0f} "
          f"frames_read={frames_read} max_brightness={best_brightness:.2f}/255 "
          f"[{status}]")
    cap.release()


def main() -> int:
    print(f"Platform: {platform.system()} {platform.release()}  "
          f"OpenCV: {cv2.__version__}")
    print("Probing camera indices 0-3 against each available backend "
          "(this takes a few seconds per combination)...\n")

    for index in range(4):
        for backend_name, backend in candidate_backends():
            probe(index, backend_name, backend)
    print("\nIf every line above says 'black/near-black', the camera feed "
          "itself is blocked (OS permission for this exact process, or "
          "another app holding the camera exclusively) -- this is outside "
          "what this project's code can work around.\n"
          "If one line says 'LOOKS REAL', note its index/backend and we'll "
          "wire that into src/config.py / src/ui/app.py:open_camera().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
