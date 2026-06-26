import hashlib
from typing import List, Tuple

class FingerprintGenerator:
    """Generates combinatorial anchor-target hashes from structural peaks."""
    def __init__(self, dt_min: int = 0, dt_max: int = 200, fan: int = 15):
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.fan = fan

    def generate(self, pks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Pairs constellation points to construct specific hashes and timestamps."""
        fps = []
        n = len(pks)
        
        # Sort landmarks primarily by time index
        pks = sorted(pks, key=lambda x: x[1])

        for i in range(n):
            for j in range(i + 1, n):
                f1, t1 = pks[i]
                f2, t2 = pks[j]
                dt = t2 - t1

                if dt < self.dt_min:
                    continue
                if dt > self.dt_max:
                    continue # Out of range, subsequent points will be further out

                # Construct reproducible integer hash using 32 bits of an MD5 checksum
                hk = f"{f1}|{f2}|{dt}".encode("utf-8")
                hsh = int(hashlib.md5(hk).hexdigest()[:8], 16)
                
                fps.append((hsh, int(t1)))
                
                # Enforce structural fan-out constraint boundary
                if len(fps) % self.fan == 0:
                    break
                    
        return fps
