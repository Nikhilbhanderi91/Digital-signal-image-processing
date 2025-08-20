import matplotlib.pyplot as plt
import numpy as np
import librosa

audio_path1 = "/Users/nikhilbhanderi/Desktop/Experiment 3/Audio/Vande Mataram (Maa Tujhe Salaam)-(Mr-Jat.in).wav"
audio_path2 = "/Users/nikhilbhanderi/Desktop/Experiment 3/Audio/Vande Mataram Karaoke -HQ.wav"

signal1, sr1 = librosa.load(audio_path1, sr=None, duration=60)
signal2, sr2 = librosa.load(audio_path2, sr=None, duration=60)

min_len = min(len(signal1), len(signal2))
signal1 = signal1[:min_len]
signal2 = signal2[:min_len]

def cross_correlation(sig1, sig2):
    return np.correlate(sig1, sig2, mode='full')

def autocorrelation(sig):
    return np.correlate(sig, sig, mode='full')

cross_corr = cross_correlation(signal1, signal2)
auto_corr = autocorrelation(signal1)


lags_cross = np.arange(-len(signal1) + 1, len(signal2))
lags_auto = np.arange(-len(signal1) + 1, len(signal1))


plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(lags_cross, cross_corr, color='b')
plt.title("Cross-correlation (Audio1 vs Audio2)")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(lags_auto, auto_corr, color='g')
plt.title("Autocorrelation (Audio1)")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid(True)

plt.tight_layout()
plt.savefig("./Audio_Correlation.png")
print("File Saved at Audio_Correlation.png")
plt.show()
