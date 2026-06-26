import os
import glob
from db.client import DBClient
from db.repository import FingerprintRepository
from db.loader import MemoryIndex
from services.indexer import IndexerService
from utils.logger import setup_logger

def bulk_populate(audio_dir: str):
    """Scans a directory for audio files and indexes them into the SQLite database."""
    log = setup_logger()
    
    # Initialize database architecture
    cli = DBClient()
    repo = FingerprintRepository(cli)
    repo.init_db()
    
    midx = MemoryIndex(repo)
    idx_svc = IndexerService(repo, midx)
    
    # Supported media types
    exts = ["*.mp3", "*.wav", "*.m4a", "*.flac"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(audio_dir, ext)))
        
    if not files:
        log.warning(f"No audio files found in directory: {audio_dir}")
        return

    log.info(f"Found {len(files)} tracks to process.")
    
    for idx, path in enumerate(files, start=1):
        name = os.path.basename(path)
        log.info(f"[{idx}/{len(files)}] Processing: {name}")
        try:
            idx_svc.index_file(path, name)
        except Exception as e:
            log.error(f"Failed to process {name}: {str(e)}")
            
    log.info("Bulk population complete! Verification snapshot:")
    songs = repo.fetch_all_songs()
    log.info(f"Total songs registered in database file: {len(songs)}")

if __name__ == "__main__":
    # Replace this string path with the directory holding your reference songs
    TARGET_DIR = "./songs"
    bulk_populate(TARGET_DIR)
