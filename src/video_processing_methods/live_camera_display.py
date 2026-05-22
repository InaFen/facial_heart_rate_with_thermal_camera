import argparse
import time

try:
    import cv2
except ImportError as exc:
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None
import numpy as np

try:
    from pylepton import Lepton
except ImportError:
    Lepton = None


if cv2 is not None:
    COLORMAPS = {
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
    }
else:
    COLORMAPS = {}


def ensure_cv2_available():
    if cv2 is None:
        raise ImportError("OpenCV is required for live display.") from _cv2_import_error


def prepare_display_frame(frame, colormap, min_val=None, max_val=None):
    ensure_cv2_available()
    frame = np.asarray(frame)
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = frame.astype(np.float32)
    vmin = np.nanmin(frame) if min_val is None else float(min_val)
    vmax = np.nanmax(frame) if max_val is None else float(max_val)

    if vmax <= vmin:
        scaled = np.zeros_like(frame, dtype=np.uint8)
    else:
        scaled = np.clip((frame - vmin) / (vmax - vmin), 0.0, 1.0)
        scaled = (scaled * 255.0).astype(np.uint8)

    if colormap is None:
        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    return cv2.applyColorMap(scaled, COLORMAPS[colormap])


def stream_lepton_pylepton(spi_device, colormap, min_val, max_val, fps_limit, flip="none"):
    ensure_cv2_available()
    if Lepton is None:
        raise ImportError("pylepton is not available. Install it or use --backend v4l2.")

    window_name = "Lepton 3.5 (pylepton)"
    last_frame_time = 0.0

    with Lepton(spi_device) as lepton:
        while True:
            frame, _ = lepton.capture()
            display_frame = prepare_display_frame(frame, colormap, min_val, max_val)
            # apply requested flip
            if flip == "h":
                display_frame = cv2.flip(display_frame, 1)
            elif flip == "v":
                display_frame = cv2.flip(display_frame, 0)
            elif flip == "hv":
                display_frame = cv2.flip(display_frame, -1)
            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if fps_limit:
                now = time.time()
                wait = (1.0 / fps_limit) - (now - last_frame_time)
                if wait > 0:
                    time.sleep(wait)
                last_frame_time = time.time()

    cv2.destroyAllWindows()


def stream_lepton_v4l2(video_device, colormap, min_val, max_val, fps_limit, flip="none"):
    ensure_cv2_available()
    window_name = "Lepton 3.5 (v4l2)"
    cap = cv2.VideoCapture(video_device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video device {video_device}")

    last_frame_time = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = prepare_display_frame(frame, colormap, min_val, max_val)
        # apply requested flip
        if flip == "h":
            display_frame = cv2.flip(display_frame, 1)
        elif flip == "v":
            display_frame = cv2.flip(display_frame, 0)
        elif flip == "hv":
            display_frame = cv2.flip(display_frame, -1)
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if fps_limit:
            now = time.time()
            wait = (1.0 / fps_limit) - (now - last_frame_time)
            if wait > 0:
                time.sleep(wait)
            last_frame_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


def stream_lepton_live(backend, spi_device, video_device, colormap, min_val, max_val, fps_limit, flip="none"):
    if colormap == "none":
        colormap = None

    if backend in {"auto", "pylepton"} and Lepton is not None:
        stream_lepton_pylepton(spi_device, colormap, min_val, max_val, fps_limit, flip=flip)
        return

    if backend in {"auto", "v4l2"}:
        stream_lepton_v4l2(video_device, colormap, min_val, max_val, fps_limit, flip=flip)
        return

    raise ValueError(f"Unsupported backend: {backend}")


def main():
    parser = argparse.ArgumentParser(description="Live display for Lepton 3.5 thermal camera")
    parser.add_argument("--backend", choices=["auto", "pylepton", "v4l2"], default="auto")
    parser.add_argument("--spi-device", default="/dev/spidev0.0")
    parser.add_argument("--video-device", default="/dev/v4l/by-id/usb-GroupGets_PureThermal__fw:v1.3.0__80030034-5101-3133-3332-373300000000-video-index0")
    parser.add_argument("--colormap", choices=["inferno", "jet", "turbo", "none"], default="inferno")
    parser.add_argument("--min", type=float, default=None, dest="min_val")
    parser.add_argument("--max", type=float, default=None, dest="max_val")
    parser.add_argument("--fps-limit", type=float, default=None)
    parser.add_argument("--flip", choices=["none", "h", "v", "hv"], default="v", help="Flip display: h=horizontal, v=vertical, hv=both")
    args = parser.parse_args()

    stream_lepton_live(
        backend=args.backend,
        spi_device=args.spi_device,
        video_device=args.video_device,
        colormap=args.colormap,
        min_val=args.min_val,
        max_val=args.max_val,
        fps_limit=args.fps_limit,
        flip=args.flip,
    )


if __name__ == "__main__":
    main()
