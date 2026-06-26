import logging
import sys

def setup_logger():
    """Initializes a standardized application logger printing to stdout."""
    log = logging.getLogger("audiofp")
    
    # Avoid adding handlers multiple times if re-imported
    if not log.handlers:
        log.setLevel(logging.INFO)
        hnd = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s", "%Y-%m-%d %H:%M:%S")
        hnd.setFormatter(fmt)
        log.addHandler(hnd)
        
    return log
