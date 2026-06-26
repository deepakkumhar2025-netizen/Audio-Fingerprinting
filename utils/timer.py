import time
import logging

class Timer:
    """A context manager to measure and log execution times cleanly."""
    def __init__(self, desc: str):
        self.desc = desc
        self.logger = logging.getLogger("audiofp")

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dur = time.perf_counter() - self.start
        self.logger.info(f"{self.desc} completed in {dur:.4f} seconds")
