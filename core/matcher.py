from typing import Dict, List, Tuple, Optional

class Matcher:
    """Matches a query fingerprint footprint list using the memory lookup cache."""
    def __init__(self, idx: Dict[int, List[Tuple[int, int]]]):
        self.idx = idx

    def match(self, fp: List[Tuple[int, int]]) -> Optional[Tuple[Tuple[int, int], int]]:
        """Calculates counter offsets and determines the highest-voted match match."""
        vt: Dict[Tuple[int, int], int] = {}

        for h, qt in fp:
            if h not in self.idx:
                continue

            # Tabulate delta alignment time calculations across matches
            for sid, at in self.idx[h]:
                k = (sid, at - qt)
                vt[k] = vt.get(k, 0) + 1

        if not vt:
            return None

        # Extract entry containing the largest absolute consensus counter value
        win_k = max(vt, key=lambda x: vt[x])
        score = vt[win_k]
        
        return win_k, score
