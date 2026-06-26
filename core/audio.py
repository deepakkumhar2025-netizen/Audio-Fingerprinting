import librosa
import numpy as np
from config import Config

class AudioLoader:
    """Handles reading and converting diverse audio formats using librosa."""
    def __init__(self, sr: int = Config.SR):
        self.sr = sr

    def load(self, path: str) -> tuple[np.ndarray, float]:
        """Loads any audio file (mp3, wav, etc.), forces mono, and returns duration."""
        # librosa automatically converts to mono and resamples to self.sr on the fly
        sig, fs = librosa.load(path, sr=self.sr, mono=True)
        
        dur = float(librosa.get_duration(y=sig, sr=fs))
        return sig, dur
