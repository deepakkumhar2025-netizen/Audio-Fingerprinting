CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    duration REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS fingerprints (
    hash INTEGER NOT NULL,
    song_id INTEGER NOT NULL,
    anchor INTEGER NOT NULL,
    FOREIGN KEY(song_id) REFERENCES songs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fp_hash ON fingerprints(hash);
CREATE INDEX IF NOT EXISTS idx_fp_song_id ON fingerprints(song_id);
CREATE INDEX IF NOT EXISTS idx_fp_hash_song ON fingerprints(hash, song_id);
