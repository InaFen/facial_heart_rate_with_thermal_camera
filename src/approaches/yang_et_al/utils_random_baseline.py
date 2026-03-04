import numpy as np

from src.approaches.yang_et_al.utils_plotting import (
	plot_flattened_signal_each_roi,
	plot_original_vs_denoised,
	plot_denoised_signal_unflat,
	plot_principal_components_in_seperate_plots,
	plot_chosen_component,
)
from src.approaches.yang_et_al.utils_post_processing import (
	pca,
	apply_butterworth_bandpass_filter,
	transform_flatten_signal_to_original_shape,
	find_most_periodic_pca_component,
)
from src.approaches.yang_et_al.utils_spatial_temporal_graph import denoise_via_laplacian


def run_random_baseline(
	signal_all_rois_without_nans: np.ndarray,
	signal_each_roi_seperate: list,
	roi_names: list,
	laplacian: np.ndarray,
	fps: float,
	plot_steps: bool,
	random_seed: int,
) -> np.ndarray:
	"""Run the random baseline pipeline and return the chosen PCA component."""
	rng = np.random.default_rng(random_seed)
	random_signal_all_rois = rng.standard_normal(signal_all_rois_without_nans.shape)
	# Match random signal scale to the original signal magnitude.
	original_scale = float(np.std(signal_all_rois_without_nans))
	random_scale = float(np.std(random_signal_all_rois))
	if random_scale > 0:
		random_signal_all_rois = random_signal_all_rois * (original_scale / random_scale)

	# Laplacian denoising on the random baseline.
	random_denoised_signal_all_rois = denoise_via_laplacian(
		Y=random_signal_all_rois,
		L=laplacian,
	)

	if plot_steps:
		plot_flattened_signal_each_roi(
			signal_flattened=random_signal_all_rois,
			block_features=signal_each_roi_seperate,
			roi_names=roi_names,
			title_info="Random: Before denoising",
		)
		plot_flattened_signal_each_roi(
			signal_flattened=random_denoised_signal_all_rois,
			block_features=signal_each_roi_seperate,
			roi_names=roi_names,
			title_info="Random: After denoising",
		)
		plot_original_vs_denoised(
			signal_flattened_orig=random_signal_all_rois,
			signal_flattened_denoised=random_denoised_signal_all_rois,
			block_features=signal_each_roi_seperate,
			roi_names=roi_names,
			title_info="Random: Original vs Denoised Signals per ROI",
		)

	random_denoised_each_roi = transform_flatten_signal_to_original_shape(
		block_features=signal_each_roi_seperate,
		flattened_signal=random_denoised_signal_all_rois,
	)
	# Bandpass filter in the physiological band.
	random_bandpassed_each_roi = apply_butterworth_bandpass_filter(
		signal=random_denoised_each_roi,
		lowcut=0.667,
		highcut=1.667,
		fs=fps,
		order=2,
	)

	if plot_steps:
		plot_denoised_signal_unflat(
			denoised_signal_unflat=random_bandpassed_each_roi,
			roi_names=roi_names,
			fps=fps,
			title="Random: Denoised and Bandpassed Signals per ROI",
		)

	# PCA and component selection.
	random_principal_components = pca(data=random_bandpassed_each_roi, num_components=4)
	if plot_steps:
		plot_principal_components_in_seperate_plots(
			principal_components=random_principal_components
		)

	random_selected_pca_component = find_most_periodic_pca_component(
		principal_components=random_principal_components
	)
	if plot_steps:
		plot_chosen_component(
			chosen_component=random_selected_pca_component,
			fps=fps,
		)

	return random_selected_pca_component
