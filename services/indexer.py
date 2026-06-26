from core.pipeline import FingerprintPipeline
from db.repository import FingerprintRepository
from db.loader import MemoryIndex

class IndexerService:
    """Handles parsing and ingestion of files into Postgres and active memory."""
    def __init__(self, repo: FingerprintRepository, idx: MemoryIndex):
        self.repo = repo
        self.idx = idx
        self.pipe = FingerprintPipeline()

    def index_file(self, path: str, name: str):
        """Processes track, registers relational bounds, and syncs memory state."""
        fps, dur = self.pipe.process(path)
        
        # Save song metadata row
        sid = self.repo.insert_song(name, dur)
        
        # Map values into bulk payload format: (hash, song_id, anchor)
        db_records = [(h, sid, at) for h, at in fps]
        self.repo.insert_fps(db_records)
        
        # Refresh operational memory index boundaries
        self.idx.load()
