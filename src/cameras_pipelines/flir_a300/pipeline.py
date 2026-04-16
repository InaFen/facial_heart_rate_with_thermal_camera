from utils_data_extraction import (
    define_roi_pixel_from_first_frame,
    extract_frame_number_from_csv,
    read_all_thermal_frames,
    create_roi_video,
)
from pathlib import Path
import sys 
import numpy as np

from utils_roi_data import compute_roi_mean_temperature_all_frames, plot_roi_mean_temperature
from utils_subrois import (
    generate_subroi_bounds,
    generate_subroi_bounds_multi,
    compute_mean_temperatures_for_roi_bounds,
    select_subroi_by_frequency_power,
    roi_pixels_from_bounds,
    save_subroi_candidate_previews,
    expand_sizes_with_half_double,
)

PROJECT_ROOT = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, PROJECT_ROOT)

from src.approaches.new_approach.utils.utils_add_ground_truth import load_ground_truth_signal_and_fs


def select_roi_frame_csv(file_for_roi: Path, start_frame_index: int) -> Path:
    csv_files = sorted(file_for_roi.parent.glob("*.csv"), key=extract_frame_number_from_csv)
    if not csv_files:
        raise FileNotFoundError(f"No CSV frames found in {file_for_roi.parent}")
    clamped_index = min(max(start_frame_index, 0), len(csv_files) - 1)
    return csv_files[clamped_index]

def pipeline(
    file_for_roi,
    fps,
    path_ground_truth,
    start_seconds: float,
    duration_seconds: float,
    offset_gt: float,
    use_subroi_selection,
    use_exclusion_roi,
    subroi_width: int | None = None,
    subroi_height: int | None = None,
    subroi_sizes: list[tuple[int, int]] | None = None,
    subroi_size_fractions: list[float] | None = None,
    subroi_step_x: int | None = None,
    subroi_step_y: int | None = None,
    target_frequency: float = 1.95,
):
    start_frame = int(round(start_seconds * fps))
    end_frame = start_frame + int(round(duration_seconds * fps))

    roi_csv = select_roi_frame_csv(file_for_roi, start_frame)
    roi_info = define_roi_pixel_from_first_frame(
        csv_file_path=roi_csv,
        fps=fps,
        output_video_dir="data/flir_a300/experiments",
        start_frame=start_frame,
        end_frame=end_frame,
    )
    roi_pixels = roi_info["roi_pixels"]
    roi_bounds = roi_info["roi_bounds"]

    if use_subroi_selection:
        if subroi_sizes is not None or subroi_size_fractions is not None:
            subroi_bounds_list = generate_subroi_bounds_multi(
                roi_bounds,
                sizes=subroi_sizes,
                size_fractions=subroi_size_fractions,
                step_x=subroi_step_x,
                step_y=subroi_step_y,
            )
        elif subroi_width is not None and subroi_height is not None:
            expanded_sizes = expand_sizes_with_half_double(
                [(subroi_width, subroi_height)],
                roi_bounds=roi_bounds,
            )
            subroi_bounds_list = generate_subroi_bounds_multi(
                roi_bounds,
                sizes=expanded_sizes,
                step_x=subroi_step_x,
                step_y=subroi_step_y,
            )
        else:
            subroi_bounds_list = generate_subroi_bounds(
                roi_bounds,
                subroi_width=subroi_width,
                subroi_height=subroi_height,
                step_x=subroi_step_x,
                step_y=subroi_step_y,
            )
        save_subroi_candidate_previews(
            frame_dir=file_for_roi.parent,
            roi_bounds_list=subroi_bounds_list,
            output_dir=Path("data/flir_a300/experiments") / f"{roi_csv.stem}_subroi_candidates",
            base_name=roi_csv.stem,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        _, mean_temperatures_list = compute_mean_temperatures_for_roi_bounds(
            frame_dir=file_for_roi.parent,
            roi_bounds_list=subroi_bounds_list,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        best_index, _ = select_subroi_by_frequency_power(
            mean_temperatures_list,
            fps=fps,
            target_frequency=target_frequency,
        )
        selected_subroi_bounds = subroi_bounds_list[best_index]
        roi_pixels = roi_pixels_from_bounds(selected_subroi_bounds)

        frames_for_video = read_all_thermal_frames(
            file_for_roi.parent,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        create_roi_video(
            frames_for_video,
            roi_bounds,
            output_dir="data/flir_a300/experiments",
            base_name=roi_csv.stem,
            fps=fps,
            subroi_bounds=selected_subroi_bounds,
            suffix="subroi_visualization",
        )

    ground_truth_signal, fps_gt, gt_units = load_ground_truth_signal_and_fs(file_path=path_ground_truth)
    start_sample_gt = int(round(start_seconds * fps_gt))
    end_sample_gt = start_sample_gt + int(round(duration_seconds * fps_gt)) + int(round(offset_gt * fps_gt))
    end_sample_gt = min(end_sample_gt, len(ground_truth_signal))
    ground_truth_signal = ground_truth_signal[start_sample_gt:end_sample_gt]

    frame_numbers, mean_temperatures = compute_roi_mean_temperature_all_frames(
        frame_dir=file_for_roi.parent,
        roi_pixels=roi_pixels,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    if use_exclusion_roi:
        exclusion_info = define_roi_pixel_from_first_frame(
            csv_file_path=roi_csv,
            fps=fps,
            output_video_dir=Path("data/flir_a300/experiments") / "exclusion_roi",
            start_frame=start_frame,
            end_frame=end_frame,
        )
        exclusion_pixels = exclusion_info["roi_pixels"]
        _, exclusion_mean = compute_roi_mean_temperature_all_frames(
            frame_dir=file_for_roi.parent,
            roi_pixels=exclusion_pixels,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        length = min(len(mean_temperatures), len(exclusion_mean))
        mean_temperatures = list(
            (np.array(mean_temperatures[:length]) - np.array(exclusion_mean[:length]))
        )
        frame_numbers = frame_numbers[:length]
    plot_roi_mean_temperature(
        frame_numbers,
        mean_temperatures,
        fps,
        ecg_signal=ground_truth_signal,
        ecg_fs=fps_gt,
        ecg_units=gt_units,
        offset=offset_gt,
    )

if __name__ == "__main__":
    input_folder = Path("data/flir_a300/18022026/Rec-temp-000009")
    #input_folder = Path("data/flir_a300/Rec-temp-000007")
    output_csv: Path | None = "data/flir_a300/experiments/pixel_amount_per_frame_Rec-temp-000009.csv"
    fps = 8.582353
    file_for_roi = input_folder / "Rec-temp-000009_1.csv"
    start_seconds = 20.0
    duration_seconds= 10.0

    use_subroi_selection = False
    use_exclusion_roi = False

    groud_truth = "data/ground_truth/opensignals/extracted_npy_files/18022026/flir_a300_rec_000009/opensignals_0007800F80FA_2026-02-18_13-32-40.npz"
    #groud_truth = "data/ground_truth/opensignals/extracted_npy_files/12022026/flir_a300_rec_000007/opensignals_0007800F80FA_2026-02-12_15-37-52.npz"
    offset_gt_seconds = 6.6
    offset_gt_seconds = 17


    pipeline(
        file_for_roi,
        fps,
        groud_truth,
        start_seconds,
        duration_seconds,
        offset_gt=offset_gt_seconds,
        use_exclusion_roi=use_exclusion_roi,
        use_subroi_selection=use_subroi_selection,
    )