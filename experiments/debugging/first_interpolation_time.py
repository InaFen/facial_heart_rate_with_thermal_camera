""" Get the time of the first interpolated frame in a directory of .gray files. """
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations


def load_fps_from_config(config_path: Path) -> float | None:
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("processing", {}).get("fps")


def find_first_interpolation_seconds(gray_dir: Path, fps: float | None) -> tuple[int | None, float | None]:
    _, sorted_names = sort_grayfiles_with_interpolations(str(gray_dir))
    first_index = None
    for idx, name in enumerate(sorted_names):
        if "_interpolated" in name:
            first_index = idx
            break
    if first_index is None:
        return None, None
    seconds = None if fps is None else first_index / fps
    return first_index, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Report when the first interpolated .gray frame appears.")
    parser.add_argument(
        "--gray-dir",
        default="data/lepton/processed_lepton_data/05032026_Wasserhahn_interpolated",
        help="Directory containing .gray frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames per second (overrides config).",
    )
    parser.add_argument(
        "--config",
        default="src/approaches/new_approach/configs.json",
        help="Config file used to read fps if not provided.",
    )
    args = parser.parse_args()

    gray_dir = Path(args.gray_dir)
    if not gray_dir.exists():
        raise FileNotFoundError(f"Gray directory not found: {gray_dir}")

    fps = args.fps
    if fps is None:
        fps = load_fps_from_config(Path(args.config))

    first_index, seconds = find_first_interpolation_seconds(gray_dir, fps)

    if first_index is None:
        print("No interpolated frames found.")
        return

    if seconds is None:
        print(f"First interpolated frame at index {first_index} (fps unknown).")
    else:
        print(f"First interpolated frame at index {first_index}, time {seconds:.3f} s (fps={fps}).")


if __name__ == "__main__":
    main()
