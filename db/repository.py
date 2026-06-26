from typing import List, Tuple
from db.client import DBClient
from db.models import SongModel, FingerprintModel

class FingerprintRepository:
    """Handles all SQLite file operations for songs and generated hashes."""
    def __init__(self, cli: DBClient):
        self.cli = cli

    def init_db(self, schema_path: str = "db/schema.sql"):
        """Initializes local database tables and indices using the schema file."""
        with open(schema_path, "r") as f:
            schema = f.read()
        
        with self.cli.get_conn() as conn:
            conn.executescript(schema)
            conn.commit()

    def insert_song(self, name: str, dur: float) -> int:
        """Inserts a new song record and returns its autoincremented ID."""
        q = "INSERT INTO songs (name, duration) VALUES (?, ?);"
        with self.cli.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(q, (name, dur))
            sid = cur.lastrowid
            conn.commit()
        return sid

    def insert_fps(self, fps: List[Tuple[int, int, int]]):
        """Performs a highly optimized batch insert of fingerprints into the file."""
        q = "INSERT INTO fingerprints (hash, song_id, anchor) VALUES (?, ?, ?);"
        with self.cli.get_conn() as conn:
            conn.executemany(q, fps)
            conn.commit()

    def fetch_all_songs(self) -> List[SongModel]:
        """Retrieves all songs registered in the catalog."""
        q = "SELECT id, name, duration FROM songs;"
        res = []
        with self.cli.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(q)
            for r in cur.fetchall():
                res.append(SongModel(r[0], r[1], r[2]))
        return res

    def fetch_all_fps(self) -> List[FingerprintModel]:
        """Streams all fingerprints efficiently for the initial in-memory warm-up."""
        q = "SELECT hash, song_id, anchor FROM fingerprints;"
        res = []
        with self.cli.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(q)
            for r in cur.fetchall():
                res.append(FingerprintModel(r[0], r[1], r[2]))
        return res
