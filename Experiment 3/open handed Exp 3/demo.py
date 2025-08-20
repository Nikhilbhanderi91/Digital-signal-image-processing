import librosa
import numpy as np
import matplotlib.pyplot as plt

def cross_correlation(signal1, signal2):
    return np.correlate(signal1, signal2, mode='full')

def autocorrelation(signal):
    return np.correlate(signal, signal, mode='full')

# Load only first 60 seconds of each audio (mono)
audio_path1 = "/Users/nikhilbhanderi/Desktop/Experiment 3/Audio/Vande Mataram (Maa Tujhe Salaam)-(Mr-Jat.in).wav"
audio_path2 = "/Users/nikhilbhanderi/Desktop/Experiment 3/Audio/Vande Mataram Karaoke -HQ.wav"

signal1, sr1 = librosa.load(audio_path1, sr=None, mono=True, duration=60)
signal2, sr2 = librosa.load(audio_path2, sr=None, mono=True, duration=60)

# Ensure both signals have the same length
min_len = min(len(signal1), len(signal2))
signal1 = signal1[:min_len]
signal2 = signal2[:min_len]

# Compute correlations
cross_corr = cross_correlation(signal1, signal2)
auto_corr = autocorrelation(signal1)

# Create lags
lags_cross = np.arange(-len(signal1) + 1, len(signal2))
lags_auto = np.arange(-len(signal1) + 1, len(signal1))

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(lags_cross, cross_corr)
plt.title("Cross-Correlation (First 1 min)")
plt.xlabel("Lag")
plt.ylabel("Magnitude")

plt.subplot(2, 1, 2)
plt.plot(lags_auto, auto_corr)
plt.title("Autocorrelation (Signal 1, First 1 min)")
plt.xlabel("Lag")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.savefig("./Audio_Correlation_1min.png")
print("File saved at Audio_Correlation_1min.png")
plt.show()
