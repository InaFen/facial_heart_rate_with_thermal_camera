import numpy as np
from scipy.signal import find_peaks, detrend
import matplotlib.pyplot as plt


def get_signal_and_heart_rate_from_opensignals_npz(
    file_name: str, plot_fft: bool = False, start_sec: float = None, end_sec: float = None
):
    """
    Load OpenSignals npz file and extract signal with heart rate.
    
    Args:
        file_name: Path to .npz file containing 'signal', 'fs', and 'units'
    
    Returns:
        dict with keys:
            - 'signal': raw signal array
            - 'fs': sampling frequency (Hz)
            - 'units': signal units (e.g., 'mV')
            - 'peak_times': array of peak times in seconds
            - 'heart_rates': array of heart rates (BPM) at each peak
            - 'mean_hr': mean heart rate across all peaks
    """
    # Load data
    with np.load(file_name) as data:
        signal = data['signal'].astype(np.float64)
        fs = int(data['fs'].item())
        units = str(data['units'].item())
    

    # Restrict to time window if specified
    if start_sec is not None or end_sec is not None:
        n = len(signal)
        t = np.arange(n) / fs
        mask = np.ones(n, dtype=bool)
        if start_sec is not None:
            mask &= t >= start_sec
        if end_sec is not None:
            mask &= t <= end_sec
        signal = signal[mask]
        # Recompute time axis for restricted signal
        t = t[mask]
    else:
        t = np.arange(len(signal)) / fs

    # Detrend and flip signal for peak detection
    signal_detrended = detrend(signal, type='linear')
    signal_detrended = -signal_detrended  # Flip if peaks point down
    
    # Find peaks

    min_peak_distance = int(0.5 * fs)  # Minimum 0.5s between peaks
    signal_std = float(np.std(signal_detrended))
    prominence = max(0.1 * signal_std, 0.001)

    peaks, _ = find_peaks(signal_detrended, distance=min_peak_distance, prominence=prominence)

    # Calculate heart rates from inter-peak intervals
    peak_times = peaks / fs
    if start_sec is not None or end_sec is not None:
        # Adjust peak_times to match restricted time axis
        peak_times = t[peaks]

    heart_rates = []
    for i in range(1, len(peaks)):
        interval_sec = (peaks[i] - peaks[i-1]) / fs
        if interval_sec > 1e-6:
            hr = 60.0 / interval_sec
            heart_rates.append(hr)

    # Add first HR (duplicate of second if available)
    if len(heart_rates) > 0:
        heart_rates.insert(0, heart_rates[0])
        heart_rates = np.array(heart_rates)
        mean_hr = np.mean(heart_rates)
    else:
        heart_rates = np.array([])
        mean_hr = 0.0
    
    # Plot signal and heart rate
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Top plot: Signal with detected peaks
    axes[0].plot(t, signal, 'b-', linewidth=0.5, alpha=0.7, label='Raw Signal')
    axes[0].plot(t, signal_detrended, 'g-', linewidth=0.8, label='Detrended (flipped)')
    axes[0].plot(peak_times, signal_detrended[peaks], 'ro', markersize=4, label=f'Peaks (n={len(peaks)})')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel(f'Amplitude ({units})')
    axes[0].set_title('ECG Signal with Detected Peaks')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bottom plot: Heart rate over time
    if len(heart_rates) > 0:
        axes[1].plot(peak_times, heart_rates, 'r.-', linewidth=1.5, markersize=6, label='Heart Rate')
        axes[1].axhline(mean_hr, color='k', linestyle='--', linewidth=1, label=f'Mean HR: {mean_hr:.1f} BPM')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Heart Rate (BPM)')
        axes[1].set_title('Heart Rate Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    if plot_fft:
        signal_centered = signal - np.mean(signal)
        frequencies = np.fft.rfftfreq(len(signal_centered), d=1.0 / fs)
        fft_values = np.abs(np.fft.rfft(signal_centered))

        plt.figure(figsize=(10, 4))
        plt.plot(frequencies, fft_values, color='tab:purple', linewidth=1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('FFT of ECG Signal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return {
        'signal': signal,
        'fs': fs,
        'units': units,
        'peak_times': peak_times,
        'heart_rates': heart_rates,
        'mean_hr': mean_hr
    }
    
if __name__ == "__main__":
    file_path = "data/ground_truth/opensignals/extracted_npy_files/12022026/flir_a300_rec_000007/opensignals_0007800F80FA_2026-02-12_15-37-52.npz"
    enable_fft_plot = True
    start_sec = 5.0  # e.g., 10.0
    end_sec = 10.0   # e.g., 60.0
    result = get_signal_and_heart_rate_from_opensignals_npz(
        file_path, plot_fft=enable_fft_plot, start_sec=start_sec, end_sec=end_sec
    )
    print(f"Sampling Frequency: {result['fs']} Hz")
    print(f"Signal Units: {result['units']}")
    print(f"Number of Peaks Detected: {len(result['peak_times'])}")
    print(f"Mean Heart Rate: {result['mean_hr']:.2f} BPM")