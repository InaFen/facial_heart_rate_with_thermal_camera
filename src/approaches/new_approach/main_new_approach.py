from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, iirnotch, welch

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.approaches.new_approach.classes.user_input import Config
from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline
from src.approaches.yang_et_al.utils_post_processing import apply_butterworth_bandpass_filter


DEFAULT_CONFIGS_PATH = "src/approaches/new_approach/configs_new_approach.yaml"
CONFIG_PATH_ENV_VAR = "NEW_APPROACH_CONFIG_PATH"
PLOTS_SAVE_DIR_ENV_VAR = "NEW_APPROACH_PLOTS_SAVE_DIR"
DEFAULT_OUTPUT_ROOT = Path("saved_results/new_approach")

FILTER_TYPE = "bandpass"  # "none", "highpass", "bandpass", "highpass and notch"
GAUSSIAN_KERNEL_SIZE = 7
GAUSSIAN_SIGMA = None
HIGHPASS_CUTOFF_HZ = 0.4
BANDPASS_LOW_HZ = 0.4
BANDPASS_HIGH_HZ = 4.0
FILTER_ORDER = 5
NOTCH_FREQUENCY_HZ = 1.0
NOTCH_QUALITY_FACTOR = 30.0


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_")


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_output_root() -> Path:
    plots_save_dir = os.environ.get(PLOTS_SAVE_DIR_ENV_VAR)
    if plots_save_dir:
        return Path(plots_save_dir)
    return DEFAULT_OUTPUT_ROOT


def interpolate_nan_columns(signal_matrix: np.ndarray) -> np.ndarray:
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    if signal_matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array (time, pixels), got shape {signal_matrix.shape}")

    result = signal_matrix.copy()
    time_len, pixel_count = result.shape
    time_axis = np.arange(time_len)
    finite_global = np.isfinite(result)
    global_mean = float(np.nanmean(result[finite_global])) if finite_global.any() else 0.0
    if not np.isfinite(global_mean):
        global_mean = 0.0

    for pixel_idx in range(pixel_count):
        series = result[:, pixel_idx]
        finite = np.isfinite(series)
        if finite.sum() == 0:
            result[:, pixel_idx] = global_mean
        elif finite.sum() == 1:
            result[:, pixel_idx] = series[finite][0]
        else:
            result[~finite, pixel_idx] = np.interp(time_axis[~finite], time_axis[finite], series[finite])

    return result


def extract_roi_signals(
    roi_pixel_data: list[dict[str, dict[str, object]]],
    roi_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    coords_per_frame: list[np.ndarray] = []
    values_per_frame: list[np.ndarray] = []

    for frame_data in roi_pixel_data:
        roi_data = frame_data.get(roi_name)
        if roi_data is None:
            continue

        coords_per_frame.append(np.asarray(roi_data["coords"], dtype=int))
        values_per_frame.append(np.asarray(roi_data["adc_values"], dtype=float))

    if not coords_per_frame or not values_per_frame:
        raise ValueError(f"No pixel data found for ROI '{roi_name}'.")

    return coords_per_frame, values_per_frame


def pad_frame_signals(frame_signals: list[np.ndarray]) -> np.ndarray:
    if not frame_signals:
        raise ValueError("frame_signals must not be empty")

    max_length = max(signal.size for signal in frame_signals)
    padded = np.full((len(frame_signals), max_length), np.nan, dtype=float)
    for index, signal_values in enumerate(frame_signals):
        padded[index, : signal_values.size] = signal_values
    return padded


def plot_signals_with_mean(
    signals: np.ndarray,
    fps: float,
    title: str,
    save_path: Path,
) -> None:
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError(f"Expected a 2D array (time, pixels), got shape {signals.shape}")

    time_axis = np.arange(signals.shape[0]) / float(fps)
    mean_signal = np.nanmean(signals, axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for pixel_idx in range(signals.shape[1]):
        ax.plot(time_axis, signals[:, pixel_idx], color="tab:blue", alpha=0.15, linewidth=0.8)
    ax.plot(time_axis, mean_signal, color="black", linewidth=2.5, label="Mean signal")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    save_figure(fig, save_path)


def plot_pixelwise_psd(
    signals: np.ndarray,
    fps: float,
    title: str,
    save_path: Path,
) -> None:
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError(f"Expected a 2D array (time, pixels), got shape {signals.shape}")
    if signals.shape[0] < 2:
        raise ValueError("At least 2 frames are required for PSD plotting.")

    centered = signals - np.nanmean(signals, axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0)
    nperseg = min(256, centered.shape[0])
    freqs, psd = welch(centered, fs=fps, nperseg=nperseg, axis=0)
    mean_psd = psd.mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for pixel_idx in range(psd.shape[1]):
        ax.plot(freqs, psd[:, pixel_idx], color="tab:blue", alpha=0.12, linewidth=0.8)
    ax.plot(freqs, mean_psd, color="black", linewidth=2.5, label="Mean PSD")
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    save_figure(fig, save_path)


def normalize_roi_coords(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    coords = np.asarray(coords, dtype=int)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinates with shape (n_pixels, 2), got {coords.shape}")

    xs = coords[:, 0]
    ys = coords[:, 1]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return xs - x_min, ys - y_min, x_max - x_min + 1, y_max - y_min + 1


def reduce_spatial_noise_gaussian_roi(
    frame_signals: list[np.ndarray],
    coords_per_frame: list[np.ndarray],
    kernel_size: int,
    sigma: float | None = None,
) -> np.ndarray:
    if len(frame_signals) != len(coords_per_frame):
        raise ValueError("frame_signals and coords_per_frame must have the same length")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

    sigma_used = float(sigma) if sigma is not None else kernel_size / 2.0
    smoothed_frames: list[np.ndarray] = []

    for values, coords in zip(frame_signals, coords_per_frame):
        values = np.asarray(values, dtype=float)
        coords = np.asarray(coords, dtype=int)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Expected coordinates with shape (n_pixels, 2), got {coords.shape}")
        if values.size != coords.shape[0]:
            raise ValueError(
                f"Pixel count mismatch for ROI smoothing: coords has {coords.shape[0]} pixels but values has {values.size}."
            )

        rel_xs, rel_ys, width, height = normalize_roi_coords(coords)
        image = np.zeros((height, width), dtype=float)
        mask = np.zeros((height, width), dtype=float)
        finite = np.isfinite(values)
        image[rel_ys[finite], rel_xs[finite]] = np.nan_to_num(values[finite], nan=0.0)
        mask[rel_ys[finite], rel_xs[finite]] = 1.0

        filtered_image = gaussian_filter(image, sigma=sigma_used)
        filtered_mask = gaussian_filter(mask, sigma=sigma_used)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                filtered_image,
                filtered_mask,
                out=np.zeros_like(filtered_image),
                where=filtered_mask > 0,
            )
        smoothed_frames.append(normalized[rel_ys, rel_xs])

    return pad_frame_signals(smoothed_frames)


def apply_highpass_filter(signals: np.ndarray, cutoff_hz: float, fs: float, order: int) -> np.ndarray:
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive. Got fs={fs}.")

    nyquist = 0.5 * fs
    if not (0 < cutoff_hz < nyquist):
        raise ValueError(f"Highpass cutoff must satisfy 0 < cutoff < Nyquist. Got cutoff={cutoff_hz}, Nyquist={nyquist}.")

    b, a = butter(order, cutoff_hz / nyquist, btype="high")
    return filtfilt(b, a, np.asarray(signals, dtype=float), axis=0)


def apply_notch_filter(signals: np.ndarray, notch_frequency_hz: float, fs: float, q_factor: float) -> np.ndarray:
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive. Got fs={fs}.")

    nyquist = 0.5 * fs
    if not (0 < notch_frequency_hz < nyquist):
        raise ValueError(
            f"Notch frequency must satisfy 0 < frequency < Nyquist. Got frequency={notch_frequency_hz}, Nyquist={nyquist}."
        )

    b, a = iirnotch(notch_frequency_hz / nyquist, q_factor)
    return filtfilt(b, a, np.asarray(signals, dtype=float), axis=0)


def apply_selected_filter(signals: np.ndarray, configs: Config) -> np.ndarray:
    filter_type = FILTER_TYPE.lower().strip()

    if filter_type == "none":
        return np.asarray(signals, dtype=float)

    if filter_type == "bandpass":
        return apply_butterworth_bandpass_filter(
            np.asarray(signals, dtype=float),
            configs.bandpass_lowcut,
            configs.bandpass_highcut,
            configs.fps,
            configs.bandpass_order,
        )

    filtered = apply_highpass_filter(np.asarray(signals, dtype=float), HIGHPASS_CUTOFF_HZ, configs.fps, FILTER_ORDER)

    if filter_type == "highpass and notch":
        return apply_notch_filter(filtered, NOTCH_FREQUENCY_HZ, configs.fps, NOTCH_QUALITY_FACTOR)

    if filter_type == "highpass":
        return filtered

    raise ValueError(f"Unsupported FILTER_TYPE: {FILTER_TYPE}")


def process_roi(
    roi_name: str,
    roi_pixel_data: list[dict[str, dict[str, object]]],
    configs: Config,
    output_root: Path,
) -> None:
    coords_per_frame, raw_frame_signals = extract_roi_signals(roi_pixel_data, roi_name)
    raw_signals = interpolate_nan_columns(pad_frame_signals(raw_frame_signals))

    roi_dir = output_root / safe_name(roi_name)
    plot_signals_with_mean(
        raw_signals,
        configs.fps,
        f"{roi_name} - Raw mean signal",
        roi_dir / "01_raw_mean_signal.png",
    )
    plot_pixelwise_psd(
        raw_signals,
        configs.fps,
        f"{roi_name} - Raw PSD",
        roi_dir / "01_raw_psd.png",
    )

    smoothed_signals = reduce_spatial_noise_gaussian_roi(
        raw_frame_signals,
        coords_per_frame=coords_per_frame,
        kernel_size=GAUSSIAN_KERNEL_SIZE,
        sigma=GAUSSIAN_SIGMA,
    )
    smoothed_signals = interpolate_nan_columns(smoothed_signals)
    plot_signals_with_mean(
        smoothed_signals,
        configs.fps,
        f"{roi_name} - Gaussian smoothed mean signal (kernel={GAUSSIAN_KERNEL_SIZE})",
        roi_dir / "02_gaussian_smoothed_mean_signal.png",
    )
    plot_pixelwise_psd(
        smoothed_signals,
        configs.fps,
        f"{roi_name} - Gaussian smoothed PSD (kernel={GAUSSIAN_KERNEL_SIZE})",
        roi_dir / "02_gaussian_smoothed_psd.png",
    )

    filtered_signals = apply_selected_filter(smoothed_signals, configs)
    plot_signals_with_mean(
        filtered_signals,
        configs.fps,
        f"{roi_name} - Filtered mean signal ({FILTER_TYPE})",
        roi_dir / f"03_{safe_name(FILTER_TYPE)}_mean_signal.png",
    )
    plot_pixelwise_psd(
        filtered_signals,
        configs.fps,
        f"{roi_name} - Filtered PSD ({FILTER_TYPE})",
        roi_dir / f"03_{safe_name(FILTER_TYPE)}_psd.png",
    )


def main() -> None:
    config_path = os.environ.get(CONFIG_PATH_ENV_VAR) or os.environ.get("CONFIG_PATH") or DEFAULT_CONFIGS_PATH
    configs = Config.from_yaml(config_path)
    configs.plot_steps = False
    output_root = get_output_root()

    print("Expects one big ROI")
    roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)
    if not roi_pixel_data:
        raise ValueError("No ROI pixel data was returned by the ROI pipeline.")

    roi_names = list(roi_pixel_data[0].keys())
    if not roi_names:
        raise ValueError("No ROI names were found in the first frame.")

    selected_rois = [roi_name for roi_name in roi_names if not configs.used_rois or roi_name in configs.used_rois]
    if not selected_rois:
        raise ValueError(
            f"None of the configured ROIs {configs.used_rois!r} are available. Available ROIs: {roi_names!r}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for roi_name in selected_rois:
        print(f"Processing ROI: {roi_name}")
        process_roi(roi_name, roi_pixel_data, configs, output_root)


if __name__ == "__main__":
    main()
