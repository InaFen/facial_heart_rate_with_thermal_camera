from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

CAMERA_TYPE = "mwir"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DATA_SEPARATOR = "\t"
FPS = 25

videos_and_npy_vario = Path("/data3/ina/may_27/LWIR/Daten")
videos_and_npy_mwir_julian = Path("/data3/ina/31_03_2026_MWIR/Julian")
videos_and_npy_mwir_ina = Path("/data3/ina/20_04_2026_MWIR/Ina") 
videos_and_npy_mwir_paulo = Path("/data3/ina/20_04_2026_MWIR/Paulo") 

DATA_SECTION_HEADER = "[Data]"
INPUT_DIR = videos_and_npy_mwir_paulo
SAVE_NPY = False
NPY_SUFFIX = "_thermal_data_all_frames"

# Video source: "asc" parses raw .asc folders, "npy" renders existing .npy stacks.
SOURCE_MODE = "npy"
# Folder searched recursively for *<NPY_SUFFIX>.npy files when SOURCE_MODE == "npy".
NPY_INPUT_DIR = INPUT_DIR

if CAMERA_TYPE == "mwir":
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 512
    DATA_SEPARATOR = ";"
    FPS = 100
elif CAMERA_TYPE != "vario":
    raise ValueError(f"Unsupported CAMERA_TYPE: {CAMERA_TYPE}")


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def parse_asc_frame(asc_path: Path) -> np.ndarray:
    text = asc_path.read_text(encoding="latin-1", errors="replace")
    lines = text.splitlines()

    data_start = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == DATA_SECTION_HEADER:
            data_start = index + 1
            break

    if data_start is None:
        raise ValueError(f"Missing [Data] section in {asc_path}")

    data_lines = [line for line in lines[data_start:] if line.strip()]
    if len(data_lines) < FRAME_HEIGHT:
        raise ValueError(
            f"Not enough data rows in {asc_path}: expected {FRAME_HEIGHT}, got {len(data_lines)}"
        )

    frame = np.empty((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
    for row in range(FRAME_HEIGHT):
        line = data_lines[row].replace(",", ".")
        values = np.fromstring(line, sep=DATA_SEPARATOR, dtype=np.float32)

        if values.size != FRAME_WIDTH and DATA_SEPARATOR != "\t":
            values = np.fromstring(line, sep="\t", dtype=np.float32)

        if values.size != FRAME_WIDTH:
            values = np.fromstring(line, sep=" ", dtype=np.float32)

        if values.size != FRAME_WIDTH:
            raise ValueError(
                f"Row {row} in {asc_path} has {values.size} columns, expected {FRAME_WIDTH}"
            )

        frame[row] = values

    return frame


def to_grayscale_uint8(frame_celsius: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(frame_celsius, None, 0, 255, cv2.NORM_MINMAX)
    frame_gray = normalized.astype(np.uint8)
    # Replicate the single grayscale channel to BGR (R == G == B) so the video
    # stays grayscale, matching thermal_video_generator.py for the lepton camera.
    return cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)


def create_thermal_video(input_dir: Path, fps: int = 25) -> Path:
    asc_files = sorted(input_dir.glob("*.asc"), key=natural_sort_key)
    if not asc_files:
        raise ValueError(f"No .asc files found in {input_dir}")

    total_frames = len(asc_files)
    print(f"Found {total_frames} .asc files in {input_dir}")

    output_video = input_dir.parent / f"{input_dir.name}_video_visualization.mp4"
    output_base = input_dir.parent / f"{input_dir.name}{NPY_SUFFIX}"

    first_frame = parse_asc_frame(asc_files[0])
    height, width = first_frame.shape

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_video}")

    writer.write(to_grayscale_uint8(first_frame))

    memmap_array = None
    if SAVE_NPY:
        memmap_path = output_base.with_suffix(".npy")
        memmap_array = np.lib.format.open_memmap(
            memmap_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, height, width),
        )
        memmap_array[0] = first_frame

    last_reported = 0
    report_every = max(1, total_frames // 20)
    for index, asc_file in enumerate(asc_files[1:], start=2):
        frame = parse_asc_frame(asc_file)
        if frame.shape != (height, width):
            raise ValueError(
                f"Frame shape mismatch in {asc_file}: got {frame.shape}, expected {(height, width)}"
            )
        if memmap_array is not None:
            memmap_array[index - 1] = frame
        writer.write(to_grayscale_uint8(frame))

        if index - last_reported >= report_every or index == total_frames:
            elapsed = (index - 1) / fps
            remaining = (total_frames - index) / fps
            percent = (index / total_frames) * 100
            print(
                f"Processed {index}/{total_frames} frames ({percent:.1f}%). "
                f"Elapsed ~{elapsed:.1f}s, remaining ~{remaining:.1f}s"
            )
            last_reported = index

    writer.release()
    print(f"Wrote {len(asc_files)} frames to {output_video} at {fps} FPS")

    if memmap_array is not None:
        memmap_array.flush()
        print(f"Saved numpy array to {output_base.with_suffix('.npy')}")
    return output_video


def create_thermal_video_from_npy(npy_path: Path, fps: int = 25) -> Path:
    """Render a grayscale visualization video from an existing .npy thermal stack.

    The .npy file is expected to hold raw values with shape
    (num_frames, height, width), as written by create_thermal_video when
    SAVE_NPY is enabled.
    """
    frames = np.load(npy_path, mmap_mode="r")
    if frames.ndim != 3:
        raise ValueError(
            f"Expected a 3D array (frames, height, width) in {npy_path}, got shape {frames.shape}"
        )

    total_frames, height, width = frames.shape
    print(f"Loaded {total_frames} frames from {npy_path} (shape {frames.shape})")

    stem = npy_path.stem
    base_name = stem[: -len(NPY_SUFFIX)] if stem.endswith(NPY_SUFFIX) else stem
    output_video = npy_path.parent / f"{base_name}_video_visualization.mp4"

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_video}")

    last_reported = 0
    report_every = max(1, total_frames // 20)
    for index in range(1, total_frames + 1):
        frame = np.array(frames[index - 1])
        writer.write(to_grayscale_uint8(frame))

        if index - last_reported >= report_every or index == total_frames:
            elapsed = (index - 1) / fps
            remaining = (total_frames - index) / fps
            percent = (index / total_frames) * 100
            print(
                f"Processed {index}/{total_frames} frames ({percent:.1f}%). "
                f"Elapsed ~{elapsed:.1f}s, remaining ~{remaining:.1f}s"
            )
            last_reported = index

    writer.release()
    print(f"Wrote {total_frames} frames to {output_video} at {fps} FPS")
    return output_video


def main() -> None:
    if SOURCE_MODE == "npy":
        npy_files = sorted(NPY_INPUT_DIR.rglob(f"*{NPY_SUFFIX}.npy"))
        if not npy_files:
            npy_files = sorted(NPY_INPUT_DIR.rglob("*.npy"))
        if not npy_files:
            raise ValueError(f"No .npy files found under {NPY_INPUT_DIR}")

        total_files = len(npy_files)
        print(f"Found {total_files} .npy files under {NPY_INPUT_DIR}")

        for index, npy_file in enumerate(npy_files, start=1):
            print(f"[{index}/{total_files}] Processing {npy_file}")
            create_thermal_video_from_npy(npy_file, fps=FPS)
        return

    if SOURCE_MODE != "asc":
        raise ValueError(f"Unsupported SOURCE_MODE: {SOURCE_MODE}")

    sub_sub_folders = [
        path
        for path in INPUT_DIR.glob("*/*")
        if path.is_dir()
    ]
    if not sub_sub_folders:
        raise ValueError(f"No sub-sub-folders found under {INPUT_DIR}")

    total_folders = len(sub_sub_folders)
    print(f"Found {total_folders} sub-sub-folders under {INPUT_DIR}")

    for index, folder in enumerate(sorted(sub_sub_folders), start=1):
        print(f"[{index}/{total_folders}] Processing {folder}")
        create_thermal_video(folder, fps=FPS)


if __name__ == "__main__":
    main()
