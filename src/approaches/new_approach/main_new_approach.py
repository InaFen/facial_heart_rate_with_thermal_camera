"""
Based on Yang et al. """

import sys

PROJECT_ROOT = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, PROJECT_ROOT)

from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline
from src.approaches.new_approach.pipelines.denoising_graph_application import apply_denoising_graph_pipeline
from src.approaches.yang_et_al.utils_post_processing import transform_flatten_signal_to_original_shape, apply_butterworth_bandpass_filter
from src.approaches.yang_et_al.utils_plotting import plot_denoised_signal_unflat
from src.approaches.new_approach.pipelines.comparison_to_ground_truth import compare_signal_to_ground_truth_pipeline
from src.approaches.new_approach.classes.user_input import Config
from src.approaches.new_approach.utils.utils_denoised_baseline import add_random_baseline_signal
from src.approaches.new_approach.utils.utils_stft import get_frequencies_over_time_spectrogram, get_frequencies_over_time_cwt
from src.approaches.new_approach.utils.utils_detrend import detrend_signal
from src.approaches.new_approach.utils.utils_fft import plot_fft_each_roi, get_dominant_frequency_per_roi, print_dominant_frequencies

def main():
    configs = Config.from_json(CONFIGS_PATH)

    # Get ROI data
    roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)

    # Apply timespan filter if configured
    if configs.timespan_signal_to_look_at is not None:
        start_time, end_time = configs.timespan_signal_to_look_at
        start_frame = int(start_time * configs.fps)
        end_frame = int(end_time * configs.fps)
        roi_pixel_data = roi_pixel_data[start_frame:end_frame]
        print(f"Applied timespan filter: {start_time}s to {end_time}s (frames {start_frame} to {end_frame})")

    # Apply graph denoising
    denoised_signal_all_rois, signal_each_roi_seperate, roi_names = apply_denoising_graph_pipeline(roi_pixel_data=roi_pixel_data, configs=configs)

    denoised_signal_each_roi_seperate = transform_flatten_signal_to_original_shape(block_features=signal_each_roi_seperate, flattened_signal=denoised_signal_all_rois)
    
    # Add random baseline ROI to denoised signal (control signal for validation)
    denoised_signal_each_roi_seperate, baseline_names = add_random_baseline_signal(
        denoised_signal_each_roi_seperate, seed=42, amplitude_scale=0.2
    )
    roi_names = list(roi_names) + baseline_names  # Append baseline to ROI names list

    # Remove mean (remove DC component - constant offset)
    denoised_signal_each_roi_seperate = detrend_signal(denoised_signal_each_roi_seperate, roi_names, plot =configs.plot_steps)
    
    #if configs.plot_steps:
    #    plot_denoised_signal_unflat(denoised_signal_unflat=denoised_signal_each_roi_seperate, roi_names=roi_names)

    # Get frequencies over time using STFT and CWT
    #get_frequencies_over_time_spectrogram(denoised_signal_each_roi_seperate, fps=configs.fps, window_seconds=5.0, roi_names=roi_names, save_path=None, plot=True)
    
    #get_frequencies_over_time_cwt(denoised_signal_each_roi_seperate, fps=configs.fps, roi_names=roi_names, save_path=None, plot=True)


    # Apply bandpass filter
    #bandpassed_signal_each_roi_seperate = apply_butterworth_bandpass_filter(signal=denoised_signal_each_roi_seperate, lowcut=configs.bandpass_lowcut, highcut=configs.bandpass_highcut, fs=configs.fps, order=configs.bandpass_order)
    bandpassed_signal_each_roi_seperate = denoised_signal_each_roi_seperate  # No additional filtering for now

    if configs.plot_steps:
        plot_denoised_signal_unflat(denoised_signal_unflat=bandpassed_signal_each_roi_seperate, roi_names=roi_names, title="Bandpassed Denoised Signal per ROI", fps=configs.fps, show_seconds=True)

    # Plot FFT for each ROI
    plot_fft_each_roi(bandpassed_signal_each_roi_seperate, roi_names, fps=configs.fps, show_peaks=True)
    
    # Get and print dominant frequencies and heart rate estimates
    dominant_frequencies, heart_rates = get_dominant_frequency_per_roi(bandpassed_signal_each_roi_seperate, roi_names, fps=configs.fps)
    print_dominant_frequencies(dominant_frequencies, heart_rates)

    # Compare to ground truth signal
    compare_signal_to_ground_truth_pipeline(configs=configs, bandpassed_signal_each_roi_seperate=bandpassed_signal_each_roi_seperate, roi_names=roi_names, timespan_filter=configs.timespan_signal_to_look_at)


"""

# PCA 
prinicpal_components = pca(data=bandpassed_signal_each_roi_seperate, num_components=4)
#plot_principal_components_in_seperate_plots(principal_components=prinicpal_components)

# Find most periodic component
selected_pca_component = find_most_periodic_pca_component(principal_components=prinicpal_components)
#plot_chosen_component(chosen_component=selected_pca_component, fps = fps)

# Apply FFT to the selected PCA component
fft_selected = np.abs(np.fft.fft(selected_pca_component, n=502))

# Calculate the frequencies corresponding to the FFT bins
frequencies = np.fft.fftfreq(502, d=1/fps)

# Find the dominant frequency
dominant_frequency_index = np.argmax(fft_selected)
dominant_frequency = frequencies[dominant_frequency_index]

# Estimate heart rate
heart_rate_estimate = dominant_frequency * 60
print(f"Estimated heart rate: {heart_rate_estimate:.2f} beats per minute")
"""

if __name__ == "__main__":
    CONFIGS_PATH = "src/approaches/new_approach/configs.json"
    main()
