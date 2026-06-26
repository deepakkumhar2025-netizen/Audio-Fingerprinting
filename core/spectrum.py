import numpy as np
from scipy.signal import spectrogram

class Spectrogram:
    """Generates a log-magnitude Short-Time Fourier Transform (STFT) matrix."""
    def __init__(self, nfft: int = 4096, ovr: int = 2048):
        self.nfft = nfft
        self.ovr = ovr

    def compute(self, sig: np.ndarray, sr: int) -> np.ndarray:
        """Computes the spectrogram and returns the log-amplitude layout."""
        # Use standard periodic Hann window via scipy's spectrogram implementation
        _, _, spts = spectrogram(
            sig, 
            fs=sr, 
            window="hann", 
            nperseg=self.nfft, 
            noverlap=self.ovr, 
            mode="psd"
        )
        
        # Avoid log(0) with a structural floor before converting to decibels
        spts = np.where(spts == 0, 1e-10, spts)
        return 10 * np.log10(spts)
