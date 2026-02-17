import numpy as np
import matplotlib.pyplot as plt

def generate_qrs_wave(fs, duration=0.1, amplitude=1.0):
    # QRS complex: sharp peak, short duration
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Use a Gaussian for QRS
    qrs = amplitude * np.exp(-((t - duration/2)**2) / (2 * (duration/10)**2))
    return qrs

def generate_artificial_ecg_with_qrs(duration_sec=10, fs=250, heart_rate_bpm=70, noise_level=0.05):
    t = np.arange(0, duration_sec, 1/fs)
    period = 60.0 / heart_rate_bpm
    ecg = np.zeros_like(t)
    qrs_duration = 0.08  # typical QRS duration in seconds
    qrs_amplitude = 1.0
    qrs_wave = generate_qrs_wave(fs, duration=qrs_duration, amplitude=qrs_amplitude)
    # Insert QRS complexes at each beat
    beat_times = np.arange(0, duration_sec, period)
    for bt in beat_times:
        idx = int(bt * fs)
        if idx + len(qrs_wave) < len(ecg):
            ecg[idx:idx+len(qrs_wave)] += qrs_wave
    # Add P and T waves (smaller, slower)
    ecg += 0.15 * np.sin(2 * np.pi * t / period)  # P wave
    ecg += 0.1 * np.sin(2 * np.pi * t / (period/2))  # T wave
    # Add noise
    ecg += noise_level * np.random.randn(len(t))
    return t, ecg, fs

def save_artificial_ecg_with_qrs_npz(filename, duration_sec=10, fs=250, heart_rate_bpm=70, noise_level=0.05):
    t, ecg, fs_val = generate_artificial_ecg_with_qrs(duration_sec, fs, heart_rate_bpm, noise_level)
    np.savez(filename, signal=ecg, fs=fs_val, units='mV')
    print(f"Artificial ECG with QRS saved to {filename}")
    plt.figure(figsize=(10, 4))
    plt.plot(t, ecg, label='Artificial ECG with QRS')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.title('Artificial ECG Signal with QRS Complex')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    save_artificial_ecg_with_qrs_npz("/home/arbeit/Dokumente/master_thesis/data/ground_truth/opensignals/extracted_npy_files/artificial_ecg_qrs.npz", duration_sec=60, fs=250, heart_rate_bpm=72, noise_level=0.03)
