from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _print_np_content(np_path: Path, loaded_obj) -> tuple[np.ndarray, float | None]:
	"""Print what is contained in a .npy/.npz and return signal + fs if available."""
	signal = None
	sampling_rate = None

	print(f"[visualize_gt_npy] Loaded file: {np_path}")

	if isinstance(loaded_obj, np.lib.npyio.NpzFile):
		keys = list(loaded_obj.files)
		print(f"[visualize_gt_npy] NPZ keys: {keys}")

		for key in keys:
			value = loaded_obj[key]
			if isinstance(value, np.ndarray):
				print(
					f"[visualize_gt_npy] key='{key}': ndarray, "
					f"shape={value.shape}, dtype={value.dtype}"
				)
			else:
				print(f"[visualize_gt_npy] key='{key}': type={type(value)}")

		if "signal" in loaded_obj.files:
			signal = np.asarray(loaded_obj["signal"], dtype=float)

		if "fs" in loaded_obj.files:
			fs_value = loaded_obj["fs"]
			if isinstance(fs_value, np.ndarray) and fs_value.ndim == 0:
				fs_value = fs_value.item()
			try:
				sampling_rate = float(fs_value)
			except (TypeError, ValueError):
				sampling_rate = None

		if "units" in loaded_obj.files:
			units_value = loaded_obj["units"]
			if isinstance(units_value, np.ndarray) and units_value.ndim == 0:
				units_value = units_value.item()
			print(f"[visualize_gt_npy] units: {units_value}")
	else:
		arr = np.asarray(loaded_obj)
		print(
			f"[visualize_gt_npy] NPY content: ndarray, "
			f"shape={arr.shape}, dtype={arr.dtype}"
		)
		signal = arr.astype(float)

	return signal, sampling_rate


def save_signal_plot(np_path: str | Path, output_path: str | Path | None = None) -> Path:
	"""Load a saved ground-truth .npy/.npz file, print its content, and save a signal plot."""
	np_path = Path(np_path)
	if not np_path.exists():
		raise FileNotFoundError(f"File not found: {np_path}")

	loaded = np.load(np_path, allow_pickle=True)
	try:
		signal, sampling_rate = _print_np_content(np_path, loaded)
	finally:
		if isinstance(loaded, np.lib.npyio.NpzFile):
			loaded.close()

	if signal is None or signal.size == 0:
		raise ValueError("No signal data found to plot.")

	if sampling_rate is not None and sampling_rate > 0:
		time_axis = np.arange(signal.size) / sampling_rate
		x_label = "Time (s)"
	else:
		time_axis = np.arange(signal.size)
		x_label = "Sample index"

	if output_path is None:
		output_path = np_path.parent / f"{np_path.stem}_signal_plot.png"
	else:
		output_path = Path(output_path)

	output_path.parent.mkdir(parents=True, exist_ok=True)

	plt.figure(figsize=(14, 4))
	plt.plot(time_axis, signal, color="C0", linewidth=1.0)
	plt.xlabel(x_label)
	plt.ylabel("Amplitude")
	plt.title(f"Signal plot: {np_path.name}")
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close()

	print(f"[visualize_gt_npy] Saved signal plot to: {output_path}")
	return output_path


if __name__ == "__main__":
	# Update this path to the .npz/.npy created by offset_detection_ground_truth.py
	INPUT_NP_PATH = Path(
		"data/ground_truth/opensignals/extracted_npy_files/20042026/Ina/Ina_3/opensignals_0007800F80FA_2026-04-20_15-14-22.npz"
	)

	save_signal_plot(INPUT_NP_PATH)
