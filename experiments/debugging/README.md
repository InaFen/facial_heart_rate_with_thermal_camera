# Debugging Experiments

Noise-related:
- `analyze_random_noise.py`: Simulates heartbeat recovery from noisy thermal pixels (single vs multi-pixel averaging + bandpass filtering) and plots comparison signals.

File structure related:
- `first_interpolation_time.py`: Finds the first `_interpolated` `.gray` frame in a directory and reports its frame index/time (using FPS from CLI or config).

Camera tests:
- `save_thermal_frames.py`: Records thermal frames to timestamped image files and analyzes frame timing/FPS from saved filenames.
- `test_camera.py`: Minimal OpenCV camera sanity check that displays a live feed and exits on `q`.
