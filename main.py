from __future__ import annotations

import argparse
import sys

from src.config import CALIBRATION_FILE


def _load_or_run_calibration(force: bool = False):
    from src.calibration.calibrator import GazeCalibrator
    if not force and CALIBRATION_FILE.exists():
        print(f"[main] loading existing calibration from {CALIBRATION_FILE}")
        return GazeCalibrator.load(CALIBRATION_FILE)

    import cv2
    from src.sensing.face_tracker import FaceTracker
    from src.ui.app import open_camera, cfg_model_path
    from src.ui.calibration_screen import run_calibration

    cap = open_camera()
    face_tracker = FaceTracker(str(cfg_model_path("face")))
    try:
        calibrator = run_calibration(face_tracker, cap)
    finally:
        face_tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    return calibrator


def cmd_calibrate(_args: argparse.Namespace) -> None:
    _load_or_run_calibration(force=True)


def cmd_run(args: argparse.Namespace) -> None:
    from src.ui.app import run_assistive_demo
    calibrator = _load_or_run_calibration(force=args.recalibrate)
    run_assistive_demo(calibrator)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.ui.app import run_evaluation
    calibrator = _load_or_run_calibration(force=args.recalibrate)
    modes = args.modes.split(",") if args.modes else None
    run_evaluation(calibrator, args.participant, modes=modes, with_tlx=args.tlx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="Run the <30s calibration sequence.")
    p_cal.set_defaults(func=cmd_calibrate)

    p_run = sub.add_parser("run", help="Run the interactive assistive demo UI.")
    p_run.add_argument("--recalibrate", action="store_true",
                        help="Force a new calibration even if one is saved.")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("evaluate", help="Run the 3-mode comparison study.")
    p_eval.add_argument("--participant", required=True, help="Participant ID for logging.")
    p_eval.add_argument("--modes", default=None,
                         help="Comma-separated subset of fixed_dwell,adaptive_dwell,"
                              "smooth_pursuit (default: all three).")
    p_eval.add_argument("--tlx", action="store_true",
                         help="Collect a NASA-TLX (RTLX) questionnaire after each mode.")
    p_eval.add_argument("--recalibrate", action="store_true",
                         help="Force a new calibration even if one is saved.")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt as e:
        print(f"\n[main] aborted: {e}")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
