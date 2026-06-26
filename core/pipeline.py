from typing import List, Tuple
import numpy as np
from core.audio import AudioLoader
from core.spectrum import Spectrogram
from core.peaks import PeakDetector
from core.fingerprint import FingerprintGenerator

class FingerprintPipeline:
    """Coordinates downstream discrete transformations from audio files to hash keys."""
    def __init__(self):
        self.loader = AudioLoader()
        self.spectrogram = Spectrogram()
        self.detector = PeakDetector()
        self.generator = FingerprintGenerator()

    def process(self, path: str) -> Tuple[List[Tuple[int, int]], float]:
        """Runs loading, spectral generation, peak filtering, and hashing."""
        sig, dur = self.loader.load(path)
        spts = self.spectrogram.compute(sig, self.loader.sr)
        pks = self.detector.detect(spts)
        fps = self.generator.generate(pks)
        return fps, dur
