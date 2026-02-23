from utils_data_extraction import define_roi_pixel_from_first_frame
from pathlib import Path
import sys 

from utils_roi_data import compute_roi_mean_temperature_all_frames, plot_roi_mean_temperature

PROJECT_ROOT = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, PROJECT_ROOT)

from src.approaches.new_approach.utils.utils_add_ground_truth import load_ground_truth_signal_and_fs

def pipeline(file_for_roi, fps, path_ground_truth):
    roi_info = define_roi_pixel_from_first_frame(csv_file_path=file_for_roi, fps=fps, output_video_dir="data/flir_a300/experiments")
    roi_pixels = roi_info["roi_pixels"]

    ground_truth_signal, fps_gt, _ = load_ground_truth_signal_and_fs(file_path=path_ground_truth)

    frame_numbers, mean_temperatures = compute_roi_mean_temperature_all_frames(frame_dir=file_for_roi.parent, roi_pixels=roi_pixels)
    plot_roi_mean_temperature(frame_numbers, mean_temperatures, fps, ecg_signal=ground_truth_signal, ecg_fs=fps_gt)

if __name__ == "__main__":
    input_folder = Path("data/flir_a300/18022026/Rec-temp-000009")
    output_csv: Path | None = "data/flir_a300/experiments/pixel_amount_per_frame_Rec-temp-000009.csv"
    fps = 8.582353
    file_for_roi = input_folder / "Rec-temp-000009_1.csv"

    groud_truth = "data/ground_truth/opensignals/extracted_npy_files/18022026/flir_a300_rec_000009/opensignals_0007800F80FA_2026-02-18_13-32-40.npz"



    pipeline(file_for_roi, fps, groud_truth)