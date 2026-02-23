from utils_data_extraction import define_roi_pixel_from_first_frame
from pathlib import Path

def pipeline(file_for_roi, fps):
    roi_pixels = define_roi_pixel_from_first_frame(csv_file_path=file_for_roi, fps=fps, output_video_dir="data/flir_a300/experiments")
    print(roi_pixels)

if __name__ == "__main__":
    input_folder = Path("data/flir_a300/18022026/Rec-temp-000009")
    output_csv: Path | None = "data/flir_a300/experiments/pixel_amount_per_frame_Rec-temp-000009.csv"
    fps = 8.582353
    file_for_roi = input_folder / "Rec-temp-000009_1.csv"


    pipeline(file_for_roi, fps)