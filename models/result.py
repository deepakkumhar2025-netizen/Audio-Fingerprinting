class RecognitionResult:
    """Carries calculated identity and verification metadata back to the interface."""
    def __init__(self, sid: int, name: str, score: int, off: int):
        self.sid = sid
        self.name = name
        self.score = score
        self.offset = off
