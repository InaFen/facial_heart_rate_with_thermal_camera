import sys
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from src.approaches.new_approach.classes.user_input import Config

from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline

CONFIGS_PATH = "src/approaches/new_approach/configs_new_approach.yaml"

configs = Config.from_yaml(CONFIGS_PATH)
output_dir = os.path.dirname(configs.path_video_thermal_with_roi_masks)


roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)

# Spatial averaging of each ROI (should be very small ROIs)
# to average random noise out (by sqrt(n))
# Create matrix with rows: subregions, columns: time frames, values: mean pixel intensity of the subregion in the frame
result = defaultdict(list)
subregions_names = []

for frame in roi_pixel_data:
    for roi, values in frame.items():
        if roi not in result:
            subregions_names.append(roi)  # remember first-seen order
        result[roi].append(values["mean_kelvin_multiplied_100"])

mean_values_subregions = [result[roi] for roi in subregions_names]
raw_before_averaging = np.array(mean_values_subregions)

# Temporal averaging over ROI as much as FPS allows by applying moving average
minimum_effective_fps = 4
input_fps = float(configs.fps)
averaging_window_size_frames = max(1, int(np.floor(input_fps / minimum_effective_fps)))

if averaging_window_size_frames > 1:
    temporally_averaged_values_subregions = []
    for roi_signal_values in mean_values_subregions:
        roi_signal_array = np.asarray(roi_signal_values)
        if roi_signal_array.size == 0:
            temporally_averaged_values_subregions.append(roi_signal_array.tolist())
            continue
        averaging_kernel = np.ones(averaging_window_size_frames) / averaging_window_size_frames
        smoothed_roi_signal = np.convolve(roi_signal_array, averaging_kernel, mode="same")
        temporally_averaged_values_subregions.append(smoothed_roi_signal.tolist())
    mean_values_subregions = temporally_averaged_values_subregions

if configs.plot_steps:
    raw_after_averaging = np.array(mean_values_subregions)

    t = np.arange(raw_before_averaging.shape[1])

    fig, ax = plt.subplots(figsize=(10, 4))

    for i in range(raw_before_averaging.shape[0]):
        ax.plot(
            t,
            raw_before_averaging[i],
            label=f"{subregions_names[i]} (before)",
            alpha=0.4,
        )
        ax.plot(
            t,
            raw_after_averaging[i],
            label=f"{subregions_names[i]} (after)",
            linewidth=2,
        )

    ax.set_title("Mean values subregions (before vs after averaging)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()

# Bandpass filter in area of NETD -> needs to be determined first

# Wiener filter, Wavelet denoising

# Peak detection, Peakt ot peak