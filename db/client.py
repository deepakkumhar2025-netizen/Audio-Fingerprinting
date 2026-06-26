import sqlite3

class DBClient:
    """Manages connections to a local file-based SQLite database."""
    def __init__(self, db_path: str = "audiofp.db"):
        self.db_path = db_path

    def get_conn(self):
        """Returns a connection to the local SQLite file."""
        return sqlite3.connect(self.db_path)
