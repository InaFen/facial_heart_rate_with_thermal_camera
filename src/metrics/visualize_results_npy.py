from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _as_python_scalar(value: Any) -> Any:
	"""Convert numpy scalar-like values to native Python scalars."""
	if isinstance(value, np.generic):
		return value.item()
	return value


def _try_parse_json_string(value: Any) -> Any:
	"""Parse JSON if value looks like a JSON-encoded string; otherwise return as-is."""
	if isinstance(value, bytes):
		try:
			value = value.decode("utf-8")
		except Exception:
			return value
	if isinstance(value, str):
		text = value.strip()
		if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
			try:
				return json.loads(text)
			except json.JSONDecodeError:
				return value
	return value


def _to_named_dict(loaded: Any, source_path: Path) -> dict[str, Any]:
	"""Normalize loaded npy/npz content into a dictionary."""
	if isinstance(loaded, np.lib.npyio.NpzFile):
		return {key: loaded[key] for key in loaded.files}

	if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
		obj = loaded.item()
		if isinstance(obj, dict):
			return obj
		return {source_path.stem: obj}

	if isinstance(loaded, np.ndarray):
		return {source_path.stem: loaded}

	if isinstance(loaded, dict):
		return loaded

	return {source_path.stem: loaded}


def _print_content_summary(content: dict[str, Any]) -> None:
	"""Print human-readable summary of all entries in content."""
	print("\n=== Stored Entries ===")
	for key, value in content.items():
		parsed = _try_parse_json_string(_as_python_scalar(value))

		if isinstance(parsed, np.ndarray):
			print(f"- {key}: ndarray, shape={parsed.shape}, dtype={parsed.dtype}")
		elif isinstance(parsed, dict):
			print(f"- {key}: dict with keys={list(parsed.keys())}")
		elif isinstance(parsed, (list, tuple)):
			print(f"- {key}: {type(parsed).__name__} (len={len(parsed)})")
		else:
			print(f"- {key}: {parsed}")


def _plot_array(key: str, arr: np.ndarray) -> None:
	"""Plot array with reasonable defaults based on dimensionality."""
	array = np.asarray(arr)

	if array.ndim == 0:
		print(f"{key}: scalar value = {array.item()}")
		return

	if array.ndim == 1:
		plt.figure(figsize=(10, 4))
		plt.plot(array)
		plt.title(f"{key} (1D signal)")
		plt.xlabel("Index")
		plt.ylabel("Value")
		plt.grid(alpha=0.3)
		plt.tight_layout()
		return

	if array.ndim == 2:
		if 1 in array.shape:
			plt.figure(figsize=(10, 4))
			plt.plot(array.reshape(-1))
			plt.title(f"{key} (flattened 2D signal)")
			plt.xlabel("Index")
			plt.ylabel("Value")
			plt.grid(alpha=0.3)
			plt.tight_layout()
			return

		if array.shape[1] <= 12:
			plt.figure(figsize=(10, 5))
			for column_idx in range(array.shape[1]):
				plt.plot(array[:, column_idx], label=f"col {column_idx}")
			plt.title(f"{key} (2D: columns as signals)")
			plt.xlabel("Index")
			plt.ylabel("Value")
			plt.grid(alpha=0.3)
			plt.legend(loc="best", fontsize=8)
			plt.tight_layout()
			return

		plt.figure(figsize=(7, 6))
		plt.imshow(array, aspect="auto", cmap="viridis")
		plt.title(f"{key} (2D heatmap)")
		plt.xlabel("Column")
		plt.ylabel("Row")
		plt.colorbar(label="Value")
		plt.tight_layout()
		return

	# ndim >= 3
	flat_for_plot = array.reshape(array.shape[0], -1)
	mean_signal = flat_for_plot.mean(axis=1)
	plt.figure(figsize=(10, 4))
	plt.plot(mean_signal)
	plt.title(f"{key} (mean over trailing dims)")
	plt.xlabel("Index (first dimension)")
	plt.ylabel("Mean value")
	plt.grid(alpha=0.3)
	plt.tight_layout()


def visualize_npy_contents(path: str | Path, show_plots: bool = True) -> dict[str, Any]:
	"""Load `.npy`/`.npz`, print all stored information, and plot available signals.

	This function tries to visualize everything sensible:
	- prints scalars/metadata (e.g. heart rate)
	- prints array shapes/dtypes
	- plots 1D/2D/ND arrays with adaptive defaults

	Returns:
		Dictionary of loaded content entries.
	"""
	source_path = Path(path)
	if not source_path.exists():
		raise FileNotFoundError(f"File not found: {source_path}")

	loaded = np.load(source_path, allow_pickle=True)
	content = _to_named_dict(loaded, source_path)

	# Parse JSON-like strings (common for metadata saved in npz)
	parsed_content: dict[str, Any] = {}
	for key, value in content.items():
		if isinstance(value, np.ndarray) and value.ndim == 0:
			parsed_content[key] = _try_parse_json_string(_as_python_scalar(value))
		else:
			parsed_content[key] = _try_parse_json_string(value)

	print(f"\nLoaded: {source_path}")
	_print_content_summary(parsed_content)

	# Explicitly print heart-rate-like entries, if present
	for candidate in ["heart_rate_estimate", "heart_rate_estimate_bpm", "heart_rate"]:
		if candidate in parsed_content:
			print(f"\n{candidate}: {parsed_content[candidate]}")

	if show_plots:
		for key, value in parsed_content.items():
			if isinstance(value, np.ndarray):
				_plot_array(key, value)
			elif isinstance(value, (list, tuple)) and len(value) > 0:
				arr = np.asarray(value)
				if np.issubdtype(arr.dtype, np.number):
					_plot_array(key, arr)

		plt.show()

	if isinstance(loaded, np.lib.npyio.NpzFile):
		loaded.close()

	return parsed_content


def main() -> None:
	PATH = "saved_results/yang_et_al/05032026_before_after_sport_interpolated_temp_0_00-10_00s.npz"
	SHOW_PLOTS = True
	visualize_npy_contents(PATH, show_plots=SHOW_PLOTS)


if __name__ == "__main__":
	main()
