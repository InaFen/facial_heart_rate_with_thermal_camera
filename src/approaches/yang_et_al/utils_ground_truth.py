import numpy as np
from scipy.signal import find_peaks


def load_ground_truth_signal(path: str, flip: bool = False) -> tuple[np.ndarray, float]:
	ground_truth_data = np.load(path)
	signal = ground_truth_data["signal"]
	if flip:
		signal = -signal
	fs = float(ground_truth_data["fs"])
	return signal, fs


def get_ground_truth_segment(
	signal: np.ndarray,
	fs: float,
	start_time: float,
	end_time: float,
	signal_offset: float = 0.0,
	ecg_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
	gt_start_time = max(0.0, (start_time + signal_offset) - ecg_offset)
	gt_end_time = max(gt_start_time, (end_time + signal_offset) - ecg_offset)
	gt_start_idx = int(gt_start_time * fs)
	gt_end_idx = int(gt_end_time * fs)
	segment = signal[gt_start_idx:gt_end_idx]
	time_axis = np.arange(len(segment)) / fs + start_time
	return segment, time_axis


def detect_peaks_in_signal(
	signal: np.ndarray,
	fs: float,
	min_distance_s: float = 0.5,
	prominence: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	min_distance = int(min_distance_s * fs)
	peaks, _ = find_peaks(signal, distance=max(1, min_distance), prominence=prominence)
	peak_times = peaks / fs
	return peaks, peak_times


def mean_hr_from_peak_times(peak_times: np.ndarray) -> float | None:
	if len(peak_times) < 2:
		return None
	ibi = np.diff(peak_times)
	hr_bpm = 60.0 / ibi
	return float(np.mean(hr_bpm))
