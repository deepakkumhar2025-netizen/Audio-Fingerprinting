import os
import streamlit as st
from utils.logger import setup_logger
from db.client import DBClient
from db.repository import FingerprintRepository
from db.loader import MemoryIndex
from core.matcher import Matcher
from services.recognizer import RecognizerService

# Initialize system log framework immediately on startup
log = setup_logger()

@st.cache_resource
def init_backend():
    """Bootstraps and caches structural core database architecture links in read-only mode."""
    cli = DBClient()
    repo = FingerprintRepository(cli)
    
    # NOTICE: repo.init_db() is completely removed.
    # The application now strictly assumes audiofp.db exists and is fully populated.
    
    midx = MemoryIndex(repo)
    midx.load()  # Warm cache once out of the static SQLite asset file
    
    mat = Matcher(midx.idx)
    rec_svc = RecognizerService(repo, mat)
    
    return rec_svc

# Spin up internal service dependency injection layers
rec_svc = init_backend()

st.title("Audio FP Engine")
st.caption("Production Ready — Read-Only Memory Lookup Index")

# Minimalist main presentation workspace for running identification queries
st.subheader("Query Identification")
qry_file = st.file_uploader("Upload Query Fragment (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"], key="qry")

if qry_file:
    # Explicitly catch and isolate uploaded query bytes to disk
    t_path = f"temp_qry_{qry_file.name}"
    with open(t_path, "wb") as f:
        f.write(qry_file.read())
        
    if st.button("Identify"):
        with st.spinner("Searching memory map..."):
            res = rec_svc.recognize(t_path)
            
        if res:
            st.metric(label="Match Found", value=res.name)
            col1, col2 = st.columns(2)
            col1.metric(label="Confidence (Votes)", value=res.score)
            col2.metric(label="Time Offset (Bins)", value=res.offset)
        else:
            st.error("No matches found in structural memory cache index.")
            
    # Always clean up temporary storage artifacts immediately
    if os.path.exists(t_path):
        os.remove(t_path)
