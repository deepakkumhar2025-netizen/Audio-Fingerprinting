class SongModel:
    """Data carrier representing a row in the songs table."""
    def __init__(self, sid: int, name: str, dur: float):
        self.id = sid
        self.name = name
        self.dur = dur

class FingerprintModel:
    """Data carrier representing a row in the fingerprints table."""
    def __init__(self, hsh: int, sid: int, anc: int):
        self.hash = hsh
        self.sid = sid
        self.anchor = anc
