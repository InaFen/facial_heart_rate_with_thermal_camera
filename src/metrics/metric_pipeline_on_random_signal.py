import numpy as np
import os
import numpy as np
import tempfile
import zlib
from pathlib import Path
import matplotlib.pyplot as plt
import json


from metric_pipeline import evaluate_prediction_against_ground_truth, append_metrics_summary_row

def time_shuffle_signal(
    signal: np.ndarray,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Randomly permute the time samples of a signal.

    The signal's value distribution is preserved, but its temporal
    structure is destroyed.
    """
    signal = np.asarray(signal)

    rng = np.random.default_rng(random_seed)
    shuffled_signal = signal.copy()

    # Shuffle along the time axis (axis 0).
    permutation = rng.permutation(len(signal))
    shuffled_signal = shuffled_signal[permutation]

    return shuffled_signal


def plot_original_vs_shuffled_signal(
    original_signal: np.ndarray,
    shuffled_signal: np.ndarray,
    output_path: str,
    fs: float = 8.76,
):
    """
    Plot the original and time-shuffled signals against time.
    """

    original_signal = np.asarray(original_signal).squeeze()
    shuffled_signal = np.asarray(shuffled_signal).squeeze()

    if original_signal.ndim != 1 or shuffled_signal.ndim != 1:
        raise ValueError("Expected one-dimensional signals.")

    if len(original_signal) != len(shuffled_signal):
        raise ValueError("Original and shuffled signals must have equal length.")

    time = np.arange(len(original_signal)) / fs

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 7),
        sharex=True,
    )

    axes[0].plot(time, original_signal, linewidth=0.8)
    axes[0].set_title("Original predicted signal")
    axes[0].set_ylabel("Signal amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, shuffled_signal, linewidth=0.8)
    axes[1].set_title("Time-shuffled predicted signal")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Signal amplitude")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Original vs. time-shuffled signal")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_time_shuffled_signal(
    results: str,
    output_csv: str,
    random_seed: int = 42,
    **evaluation_kwargs,
):
    """
    Evaluate a time-shuffled version of the predicted signal.

    Parameters
    ----------
    results : str
        Path to the original NPZ file.
    output_csv : str
        CSV file where the shuffled evaluation results are stored.
    random_seed : int
        Seed for reproducible shuffling.
    **evaluation_kwargs
        Additional arguments passed to
        evaluate_prediction_against_ground_truth().
    """

    # Load original data
    data = np.load(results, allow_pickle=True)

    # Copy all arrays and metadata
    shuffled_data = {
        key: data[key].copy()
        for key in data.files
    }

    # Extract and shuffle predicted signal
    predicted_signal = shuffled_data["predicted_signal"]

    shuffled_data["predicted_signal"] = time_shuffle_signal(
        predicted_signal,
        random_seed=random_seed,
    )

    # Save shuffled NPZ file persistently
    shuffled_folder = Path("saved_results/new_approach/random_experiment")
    shuffled_folder.mkdir(parents=True, exist_ok=True)

    shuffled_npz_path = shuffled_folder / (
        Path(results).stem + "_time_shuffled.npz"
    )

    np.savez_compressed(
        shuffled_npz_path,
        **shuffled_data,
    )

    # Run the original evaluation pipeline
    output_data = evaluate_prediction_against_ground_truth(
        results=str(shuffled_npz_path),
        output_root=shuffled_folder/ "metric_results",
        tolerance_ms=0,
    )

    plot_path = shuffled_folder/ "metric_results" / Path(str(shuffled_npz_path)).stem / ( "original_vs_shuffled.png"
                )
            
    plot_original_vs_shuffled_signal(original_signal=predicted_signal,
                                            shuffled_signal=shuffled_data["predicted_signal"],
                                            output_path=plot_path,
                                            fs=output_data["predicted_fs"])

    return output_data


def batch_evaluate_time_shuffled(
    input_folder: str,
    output_csv: str,
    random_seed: int = 42,
    individuals=None,
    **evaluation_kwargs,
):
    """
    Run the time-shuffled evaluation for all NPZ files in a folder.

    Each NPZ recording receives its own reproducible random shuffle.
    """

    input_folder = Path(input_folder)

    npz_files = sorted(input_folder.rglob("*.npz"))

    if individuals is not None:
        individuals = set(individuals)

        npz_files = [
            f for f in npz_files
            if any(individual in str(f) for individual in individuals)
        ]

    print(f"Found {len(npz_files)} NPZ files.")

    all_results = []

    for i, npz_file in enumerate(npz_files, start=1):

        print(
            f"[{i}/{len(npz_files)}] "
            f"Evaluating time-shuffled signal: {npz_file.name}"
        )

        # Derive a reproducible but file-specific seed
        file_seed = (
            random_seed
            + zlib.crc32(str(npz_file).encode("utf-8"))
        ) % (2**32)

        try:
            output_data = evaluate_time_shuffled_signal(
                results=str(npz_file),
                output_csv=output_csv,
                random_seed=file_seed,
                **evaluation_kwargs,
            )

            all_results.append(output_data)

        except Exception as e:
            print(f"Error processing {npz_file}: {e}")

    print(
        f"Finished time-shuffled evaluation. "
        f"Successfully evaluated {len(all_results)} "
        f"out of {len(npz_files)} recordings."
    )

    return all_results


if __name__ == "__main__":
    input_folder = "saved_results/new_approach/for_random_experiment"

    output_csv = "saved_results/time_shuffled_baseline.csv"

    results = batch_evaluate_time_shuffled(
        input_folder=input_folder,
        output_csv=output_csv,
        random_seed=42,
    )