import numpy as np
from scipy.ndimage import maximum_filter

class PeakDetector:
    """Extracts prominent local constellation points using a max filter footprint."""
    def __init__(self, size: int = 20, min_db: float = -45.0):
        self.size = size      # Size of neighborhood region box
        self.min_db = min_db  # Structural amplitude floor cutoff

    def detect(self, spts: np.ndarray) -> list[tuple[int, int]]:
        """Finds regional peaks inside the 2D spectrogram surface."""
        # Define a boolean mask where local value equals regional max filter response
        local_max = (spts == maximum_filter(spts, size=(self.size, self.size)))
        
        # Filter background floor noise
        amp_mask = (spts > self.min_db)
        peaks = local_max & amp_mask
        
        # Find structural coordinate positions matching conditions
        f_idx, t_idx = np.where(peaks)
        
        # Zip coordinates into structural structural landmarks list [(freq_bin, time_bin)]
        return list(zip(f_idx, t_idx))
