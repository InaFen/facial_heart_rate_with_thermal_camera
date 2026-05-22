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

DATA_SECTION_HEADER = "[Data]"
INPUT_DIR = Path("/mnt/wwn-0x50000398325009f6-part1/Masterarbeit/MWIR")
SAVE_NPY = True
NPY_SUFFIX = "_thermal_data_all_frames"

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


def to_colored_uint8(frame_celsius: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(frame_celsius, None, 0, 255, cv2.NORM_MINMAX)
    frame_gray = normalized.astype(np.uint8)
    return cv2.applyColorMap(frame_gray, cv2.COLORMAP_INFERNO)


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

    writer.write(to_colored_uint8(first_frame))

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
        writer.write(to_colored_uint8(frame))

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


def main() -> None:
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
