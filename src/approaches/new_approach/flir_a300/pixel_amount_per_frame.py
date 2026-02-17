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



def extract_mean_temperature_from_csv(frame_number: int, folder: Path) -> float:
	# Find the corresponding CSV file for the frame
	# Example: Rec-temp-000007_515.csv for frame 515
	csv_file = folder / f"Rec-temp-000007_{frame_number}.csv"
	if not csv_file.exists():
		raise FileNotFoundError(f"CSV file not found for frame {frame_number}: {csv_file}")
	pixel_values = []
	with csv_file.open("r", encoding="utf-8") as f:
		for line in f:
			# Skip header/meta lines
			if line.strip() == "" or "=" in line:
				continue
			# Split by comma and convert to float
			pixel_values.extend([float(val) for val in line.strip().split(",") if val])
	if not pixel_values:
		raise ValueError(f"No pixel values found in {csv_file}")
	return sum(pixel_values) / len(pixel_values)


def collect_frame_pixel_stats(folder_path: str | Path) -> List[FramePixelStats]:
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
		mean_temperature = extract_mean_temperature_from_csv(frame_number, folder)
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


def main() -> None:
	input_folder = Path(
		"/home/arbeit/Dokumente/master_thesis/data/flir_a300/Rec-temp-000007"
	)
	output_csv: Path | None = "data/flir_a300/experiments/pixel_amount_per_frame_Rec-temp-000007.csv"
	frames_per_second = 8.5882

	stats = collect_frame_pixel_stats(input_folder)
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


if __name__ == "__main__":
	main()
