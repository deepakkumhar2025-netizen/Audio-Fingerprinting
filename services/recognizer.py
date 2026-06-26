from typing import Optional
from core.pipeline import FingerprintPipeline
from core.matcher import Matcher
from db.repository import FingerprintRepository
from models.result import RecognitionResult

class RecognizerService:
    """Exposes high-level query verification workflows using current cache states."""
    def __init__(self, repo: FingerprintRepository, mat: Matcher):
        self.repo = repo
        self.mat = mat
        self.pipe = FingerprintPipeline()

    def recognize(self, path: str) -> Optional[RecognitionResult]:
        """Matches input snippet points directly against tracking dictionary indexes."""
        fps, _ = self.pipe.process(path)
        res = self.mat.match(fps)
        
        if not res:
            return None
            
        (sid, off), score = res
        
        # Resolve identity parameters from repository records
        songs = self.repo.fetch_all_songs()
        match_song = next((s for s in songs if s.id == sid), None)
        name = match_song.name if match_song else "Unknown"
        
        return RecognitionResult(sid, name, score, off)
