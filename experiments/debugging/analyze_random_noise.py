import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ----------------------------
# Simulation parameters
# ----------------------------
# Sampling settings: define time grid for a 10-second signal
fs = 8  # sampling frequency (frames per second)
t = np.arange(0, 10, 1/fs)  # 10 seconds
# Heart rate parameters (varying over time)
hr_bpm_mean = 60  # average heart rate in bpm
hr_bpm_std = 6    # short-term variability in bpm
hr_bpm = hr_bpm_mean + np.cumsum(np.random.normal(0, hr_bpm_std, size=len(t))) / fs
hr_bpm = np.clip(hr_bpm, 45, 90)  # keep within a plausible range

# Convert heart rate to instantaneous frequency (Hz) and integrate phase
f_hr_inst = hr_bpm / 60.0
phase = 2 * np.pi * np.cumsum(f_hr_inst) / fs

# Heartbeat signal (ideal underlying signal, same for all pixels)
hb_amplitude = 0.01  # heartbeat amplitude in °C
heartbeat = hb_amplitude * np.sin(phase)

# Random noise parameters (uniform noise added per pixel)
noise_amplitude = 0.05
num_pixels = 16  # simulate multiple pixels in ROI
single_pixel_only = False  # set True to analyze only a single pixel

# Pixel behavior options
# Some pixels follow the heartbeat, some only drift, some are noise-only.
# Fractions are relative to the ROI size and should sum to <= 1.0.
hr_pixel_fraction = 0.3    # fraction of pixels influenced by heart rate
drift_pixel_fraction = 0.25 # fraction with slow temperature drift only

drift_amplitude = 0.05      # amplitude of slow drift in °C
drift_period_s = 20         # drift period in seconds

# ----------------------------
# Simulate multiple pixels with noise
# ----------------------------
# Each pixel observes a different underlying signal, plus independent noise
# We partition the ROI into heartbeat pixels, drift-only pixels, and noise-only pixels.
pixel_count = 1 if single_pixel_only else num_pixels
num_hr_pixels = int(round(pixel_count * hr_pixel_fraction))
num_drift_pixels = int(round(pixel_count * drift_pixel_fraction))
num_noise_only_pixels = max(0, pixel_count - num_hr_pixels - num_drift_pixels)

# Slow drift to mimic baseline temperature changes unrelated to HR
drift_signal = drift_amplitude * np.sin(2 * np.pi * t / drift_period_s)

measured_pixels = []
for idx in range(pixel_count):
    if idx < num_hr_pixels:
        base_signal = heartbeat
    elif idx < num_hr_pixels + num_drift_pixels:
        base_signal = drift_signal
    else:
        base_signal = np.zeros_like(t)

    # Add independent sensor noise per pixel
    noise = np.random.uniform(-noise_amplitude, noise_amplitude, size=len(t))
    measured_pixels.append(base_signal + noise)

measured_pixels = np.array(measured_pixels)

# ----------------------------
# Spatial averaging across pixels
# ----------------------------
# Averaging across ROI pixels reduces uncorrelated noise
measured_avg = measured_pixels[0] if single_pixel_only else np.mean(measured_pixels, axis=0)

# ----------------------------
# Band-pass filter to extract heartbeat
# ----------------------------
# Keep only plausible heart-rate frequencies
lowcut = 0.7  # Hz
highcut = 3.0  # Hz
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = butter(2, [low, high], btype='band')
filtered_single = filtfilt(b, a, measured_pixels[0])  # single noisy pixel
filtered_avg = filtfilt(b, a, measured_avg)            # averaged pixels or single pixel

# ----------------------------
# Plot results
# ----------------------------
# Compare raw vs filtered signals for single-pixel and averaged ROI
peaks, _ = find_peaks(heartbeat, distance=max(1, int(0.3 * fs)))

plt.figure(figsize=(14, 7))
plt.plot(t, heartbeat, label='True Heartbeat', linestyle='--', color='blue')
if single_pixel_only:
    plt.plot(t, measured_pixels[0], label='Raw Single Pixel', color='gray', alpha=0.5)
    plt.plot(t, filtered_single, label='Single Pixel (Bandpass-filtered)', color='red')
else:
    plt.plot(t, measured_avg, label=f'Averaged {num_pixels} pixels', color='green', alpha=0.7)
    plt.plot(t, filtered_avg, label=f'Averaged + Bandpass-filtered {num_pixels} pixels', color='orange', linewidth=2)

# Annotate instantaneous HR above heartbeat peaks
for peak_idx in peaks:
    plt.text(
        t[peak_idx],
        heartbeat[peak_idx] + 0.01,
        f"{hr_bpm[peak_idx]:.0f}",
        ha='center',
        va='bottom',
        fontsize=8,
        color='blue'
    )
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Heartbeat Signal Recovery from Thermal Pixel Noise')
plt.legend()
plt.grid(True)
plt.show()