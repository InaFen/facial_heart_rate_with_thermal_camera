from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class FramePixelStats:
	frame_number: int
	pixel_count: int
	mean_temperature: float
	file_path: Path


FRAME_NUMBER_PATTERN = re.compile(r"_(\d+)\s+-\s+Stats\.txt$", re.IGNORECASE)
PIXEL_COUNT_PATTERN = re.compile(r"Number of Pixels\s+(\d+)")
MEAN_TEMPERATURE_PATTERN = re.compile(r"Mean \[C\]\s+([0-9]+\.?[0-9]*)")


def extract_frame_number(file_path: Path) -> int:
	match = FRAME_NUMBER_PATTERN.search(file_path.name)
	if not match:
		raise ValueError(f"Could not extract frame number from filename: {file_path.name}")
	return int(match.group(1))


def extract_pixel_count(lines: Iterable[str], file_path: Path) -> int:
	for line in lines:
		match = PIXEL_COUNT_PATTERN.search(line)
		if match:
			return int(match.group(1))
	raise ValueError(f"Could not find pixel count in file: {file_path}")


def extract_mean_temperature_from_csv(frame_number: int, folder: Path, threshold: float | None = None) -> float:
	"""
	Find the mean temperature for a frame, keeping only pixels above a given threshold (if provided).
	If threshold is None, use all pixels.
	"""
	csv_file = folder / f"Rec-temp-000005_{frame_number}.csv"
	if not csv_file.exists():
		raise FileNotFoundError(f"CSV file not found for frame {frame_number}: {csv_file}")
	pixel_values = []
	with csv_file.open("r", encoding="utf-8") as f:
		for line in f:
			if line.strip() == "" or "=" in line:
				continue
			pixel_values.extend([float(val) for val in line.strip().split(",") if val])
	if not pixel_values:
		raise ValueError(f"No pixel values found in {csv_file}")
	if threshold is not None:
		pixel_values = [val for val in pixel_values if val > threshold]
		print(frame_number, len(pixel_values))

	if not pixel_values:
		raise ValueError(f"No pixel values above threshold {threshold} in {csv_file}")
    
	return sum(pixel_values) / len(pixel_values)


def collect_frame_pixel_stats(folder_path: str | Path, threshold: float | None = None) -> List[FramePixelStats]:
	folder = Path(folder_path)
	if not folder.exists() or not folder.is_dir():
		raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

	stats_files = sorted(folder.glob("* - Stats.txt"))
	results: List[FramePixelStats] = []

	for stats_file in stats_files:
		frame_number = extract_frame_number(stats_file)
		with stats_file.open("r", encoding="utf-8", errors="replace") as handle:
			lines = list(handle)
			pixel_count = extract_pixel_count(lines, stats_file)
		mean_temperature = extract_mean_temperature_from_csv(frame_number, folder, threshold=threshold)
		results.append(
			FramePixelStats(
				frame_number=frame_number,
				pixel_count=pixel_count,
				mean_temperature=mean_temperature,
				file_path=stats_file,
			)
		)

	return results


def write_csv(stats: List[FramePixelStats], output_path: str | Path) -> None:
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)

	with output.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(["frame_number", "pixel_count", "mean_temperature", "file_path"])
		for item in stats:
			writer.writerow(
				[item.frame_number, item.pixel_count, item.mean_temperature, str(item.file_path)]
			)


def plot_pixel_counts_and_mean(
	stats: List[FramePixelStats], frames_per_second: float
) -> None:
	if frames_per_second <= 0:
		raise ValueError("frames_per_second must be greater than zero.")

	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise ImportError(
			"matplotlib is required for plotting. Install it with 'pip install matplotlib'."
		) from exc

	try:
		import numpy as np
	except ImportError as exc:
		raise ImportError(
			"numpy is required for FFT plotting. Install it with 'pip install numpy'."
		) from exc

	times = [(item.frame_number - 1) / frames_per_second for item in stats]
	counts = [item.pixel_count for item in stats]
	means = [item.mean_temperature for item in stats]

	fig = plt.figure(figsize=(10, 10))
	grid = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
	axis_top = fig.add_subplot(grid[0, 0])
	axis_bottom = fig.add_subplot(grid[1, 0], sharex=axis_top)
	axis_fft_counts = fig.add_subplot(grid[2, 0])
	axis_fft_means = fig.add_subplot(grid[3, 0], sharex=axis_fft_counts)
	axis_top.plot(times, counts, marker="o", linewidth=1)
	axis_top.set_ylabel("Number of Pixels")
	axis_top.set_title("Pixel Amount per Frame")
	axis_top.grid(True, alpha=0.3)

	axis_bottom.plot(times, means, color="tab:red", marker="o", linewidth=1)
	axis_bottom.set_xlabel("Time (s)")
	axis_bottom.set_ylabel("Mean Temperature (C)")
	axis_bottom.set_title("Mean Temperature per Frame")
	axis_bottom.grid(True, alpha=0.3)

	mean_centered = np.array(means, dtype=float) - float(np.mean(means))
	counts_centered = np.array(counts, dtype=float) - float(np.mean(counts))
	frequencies = np.fft.rfftfreq(len(times), d=1.0 / frames_per_second)
	fft_counts = np.abs(np.fft.rfft(counts_centered))
	fft_means = np.abs(np.fft.rfft(mean_centered))

	axis_fft_counts.plot(frequencies, fft_counts, label="Pixels FFT")
	axis_fft_counts.set_ylabel("Amplitude")
	axis_fft_counts.set_title("FFT of Pixel Count")
	axis_fft_counts.grid(True, alpha=0.3)
	axis_fft_counts.legend(loc="upper right")

	axis_fft_means.plot(frequencies, fft_means, label="Mean Temp FFT", color="tab:red")
	axis_fft_means.set_xlabel("Frequency (Hz)")
	axis_fft_means.set_ylabel("Amplitude")
	axis_fft_means.set_title("FFT of Mean Temperature")
	axis_fft_means.grid(True, alpha=0.3)
	axis_fft_means.legend(loc="upper right")

	fig.tight_layout()
	plt.show()

def create_mean_temperature_colormap_video(
	folder: Path,
	output_video_path: Path,
	threshold: float,
	width: int = 320,
	height: int = 240,
	fps: float = 8.58811,
	codec: str = "mp4v",
	colormap: int = 2  # cv2.COLORMAP_JET
) -> None:
	"""
	For each frame in the folder, create a color-mapped image based on temperature values.
	Pixels below threshold can be dimmed or set to a background color.
	"""
	import cv2
	import numpy as np
	stats_files = sorted(folder.glob("* - Stats.txt"))
	# Pair each frame path with its frame number for sorting
	frame_paths_with_numbers = []
	for f in stats_files:
		frame_number = extract_frame_number(f)
		csv_path = folder / f"Rec-temp-000005_{frame_number}.csv"
		frame_paths_with_numbers.append((frame_number, csv_path))
	# Sort by frame number
	frame_paths_with_numbers.sort(key=lambda x: x[0])
	frame_paths = [csv_path for _, csv_path in frame_paths_with_numbers]
	fourcc = cv2.VideoWriter_fourcc(*codec)
	out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height), isColor=True)
	vmin, vmax = None, None
	# First pass: find min/max temperature for normalization
	for csv_file in frame_paths:
		if not csv_file.exists():
			continue
		pixel_values = []
		with csv_file.open("r", encoding="utf-8") as f:
			for line in f:
				if line.strip() == "" or "=" in line:
					continue
				pixel_values.extend([float(val) for val in line.strip().split(",") if val])
		if len(pixel_values) != width * height:
			continue
		arr = np.array(pixel_values, dtype=float)
		arr_masked = arr[arr > threshold]
		if arr_masked.size > 0:
			arr_min, arr_max = arr_masked.min(), arr_masked.max()
			vmin = arr_min if vmin is None else min(vmin, arr_min)
			vmax = arr_max if vmax is None else max(vmax, arr_max)
	if vmin is None or vmax is None or vmin == vmax:
		vmin, vmax = 20.0, 40.0  # fallback
	# Second pass: write video
	for csv_file in frame_paths:
		if not csv_file.exists():
			continue
		pixel_values = []
		with csv_file.open("r", encoding="utf-8") as f:
			for line in f:
				if line.strip() == "" or "=" in line:
					continue
				pixel_values.extend([float(val) for val in line.strip().split(",") if val])
		if len(pixel_values) != width * height:
			continue
		arr = np.array(pixel_values, dtype=float).reshape((height, width))
		norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
		norm_uint8 = (norm * 255).astype(np.uint8)
		color_img = cv2.applyColorMap(norm_uint8, colormap)
		# Optionally dim or gray out pixels below threshold
		below = arr <= threshold
		color_img[below] = [40, 40, 40]  # dark gray for not-used pixels
		out.write(color_img)
	out.release()
	print(f"Video saved to {output_video_path}")

def main() -> None:
	input_folder = Path(
		"/home/arbeit/Dokumente/master_thesis/data/flir_a300/Rec-temp-0000005"
	)
	output_csv: Path | None = "data/flir_a300/experiments/pixel_amount_per_frame_Rec-temp-0000005.csv"
	frames_per_second = 8.58811

	# Set threshold to a float value to keep only pixels above this value for mean calculation, or None to use all pixels
	threshold = 0.0  # e.g., 30.0
	stats = collect_frame_pixel_stats(input_folder, threshold=threshold)
	stats.sort(key=lambda item: item.frame_number)

	if output_csv:
		write_csv(stats, output_csv)
	else:
		try:
			for item in stats:
				print(
					f"{item.frame_number},{item.pixel_count},{item.mean_temperature},{item.file_path}"
				)
		except BrokenPipeError:
			return

	plot_pixel_counts_and_mean(stats, frames_per_second)

	# Create a color video showing which pixels are used for mean temperature calculation
	output_video_path = Path("data/flir_a300/experiments/mean_temperature_colormap_video.mp4")
	create_mean_temperature_colormap_video(
		input_folder,
		output_video_path,
		threshold=threshold,
		width=320,
		height=240,
		fps=frames_per_second,
		codec="mp4v",
		colormap=2  # cv2.COLORMAP_JET
	)


if __name__ == "__main__":
	main()
