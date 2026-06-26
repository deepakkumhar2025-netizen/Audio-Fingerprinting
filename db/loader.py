from utils.timer import Timer

class MemoryIndex:
    def __init__(self, repo):
        self.repo = repo
        self.idx = {}

    def load(self):
        with Timer("Loading SQLite fingerprints into RAM cache"):
            self.idx.clear()
            recs = self.repo.fetch_all_fps()
            
            for r in recs:
                # Force both hash and anchor values to explicit native integers
                hsh_key = int(r.hash)
                sid = int(r.sid)
                anchor_val = int(r.anchor)
                
                if hsh_key not in self.idx:
                    self.idx[hsh_key] = []
                self.idx[hsh_key].append((sid, anchor_val))

