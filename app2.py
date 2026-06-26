import os
import time
import tempfile
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import Counter

from db.client import DBClient
from db.repository import FingerprintRepository
from db.loader import MemoryIndex
from core.matcher import Matcher
from services.recognizer import RecognizerService
from utils.logger import setup_logger

# =====================================================
# SYSTEM PARAMETERS & CONFIGURATION
# =====================================================
SR = 44100  
MIN_SCORE = 5  

PANEL_BG = "#FFFFFF"
BORDER = "#E2E5EC"
TEXT = "#111827"
TEXT_DIM = "#6B7280"
AXIS_TEXT = "#374151"
CYAN = "#0891B2"
PURPLE = "#7C3AED"

CHART_CONFIG = {"displayModeBar": False, "staticPlot": True} # staticPlot removes browser hover-event overhead

log = setup_logger()

@st.cache_resource
def init_backend():
    cli = DBClient()
    repo = FingerprintRepository(cli)
    midx = MemoryIndex(repo)
    midx.load()  
    mat = Matcher(midx.idx)
    rec_svc = RecognizerService(repo, mat)
    return rec_svc, repo

rec_svc, repo = init_backend()

# =====================================================
# ADVANCED IDENTIFICATION (With Timings & Structural Vectors)
# =====================================================
def identify_song_detailed(path: str, rec_service: RecognizerService):
    timings = {}
    
    t0 = time.perf_counter()
    sig, dur = rec_service.pipe.loader.load(path)
    t1 = time.perf_counter()
    timings["Audio Load"] = (t1 - t0) * 1000
    
    spts = rec_service.pipe.spectrogram.compute(sig, rec_service.pipe.loader.sr)
    t2 = time.perf_counter()
    timings["Spectrogram"] = (t2 - t1) * 1000
    
    pks = rec_service.pipe.detector.detect(spts)
    t3 = time.perf_counter()
    timings["Constellation"] = (t3 - t2) * 1000
    
    fps = rec_service.pipe.generator.generate(pks)
    t4 = time.perf_counter()
    timings["Hashing"] = (t4 - t3) * 1000
    
    t5 = time.perf_counter()
    vt = {}
    for h, qt in fps:
        if h not in rec_service.mat.idx:
            continue
        for sid, at in rec_service.mat.idx[h]:
            k = (sid, at - qt)
            vt[k] = vt.get(k, 0) + 1
    t6 = time.perf_counter()
    timings["Lookup"] = (t6 - t5) * 1000
    
    candidates = []
    song_votes = {}
    for (sid, off), votes in vt.items():
        if sid not in song_votes:
            song_votes[sid] = []
        song_votes[sid].append(votes)
        
    songs = rec_service.repo.fetch_all_songs()
    song_map = {s.id: s.name for s in songs}
    
    for sid, votes_list in song_votes.items():
        candidates.append((song_map.get(sid, f"ID {sid}"), max(votes_list)))
        
    candidates.sort(key=lambda c: c[1], reverse=True)
    raw_offsets = [at - qt for h, qt in fps if h in rec_service.mat.idx for sid, at in rec_service.mat.idx[h]]
    
    best_song, best_score = None, 0
    if candidates and candidates[0][1] >= MIN_SCORE:
        best_song, best_score = candidates[0]
        
    return {
        "song": best_song,
        "score": best_score,
        "total_hashes": len(fps),
        "candidates": candidates[:5],
        "timings": timings,
        "spts": spts,
        "peaks": pks,
        "offsets": raw_offsets,
        "duration": dur,
        "hashes": fps
    }

# =====================================================
# HARDENED PLOTLY VISUALIZATION BUILDERS
# =====================================================
def plotly_spectrogram(spts, height=300):
    if spts.shape[1] > 1000:
        spts = spts[:, ::4]
        
    zmin = float(np.percentile(spts, 5))
    zmax = float(np.percentile(spts, 99.5))
    fig = go.Figure(data=go.Heatmap(z