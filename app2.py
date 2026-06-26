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
    fig = go.Figure(data=go.Heatmap(z=spts, colorscale="Viridis", zmin=zmin, zmax=zmax, showscale=False))
    fig.update_layout(
        height=height, margin=dict(l=45, r=10, t=10, b=40),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=AXIS_TEXT, size=12),
        xaxis=dict(title="Time Windows", color=AXIS_TEXT),
        yaxis=dict(title="Frequency Channels", color=AXIS_TEXT)
    )
    return fig

def plotly_constellation(peaks, height=300):
    pks_arr = np.array(peaks)
    
    if len(pks_arr) > 500:
        step = len(pks_arr) // 500
        pks_arr = pks_arr[::step]
        
    x = pks_arr[:, 1] if len(pks_arr) > 0 else []
    y = pks_arr[:, 0] if len(pks_arr) > 0 else []
    
    fig = go.Figure(data=go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=4, color=CYAN, opacity=0.85)))
    fig.update_layout(
        height=height, margin=dict(l=45, r=10, t=10, b=40),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=AXIS_TEXT, size=12),
        xaxis=dict(title="Time Bins", color=AXIS_TEXT),
        yaxis=dict(title="Frequency Bins", color=AXIS_TEXT)
    )
    return fig

def plotly_histogram(offsets, height=300):
    fig = go.Figure()
    if offsets:
        counts = Counter(offsets)
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]
        
        if len(xs) > 1000:
            xs = xs[::2]
            ys = ys[::2]
            
        best_idx = max(range(len(ys)), key=lambda i: ys[i]) if ys else 0
        colors = [PURPLE] * len(xs)
        if colors:
            colors[best_idx] = CYAN
        fig.add_bar(x=xs, y=ys, marker_color=colors)
    fig.update_layout(
        height=height, margin=dict(l=45, r=10, t=30, b=40),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=AXIS_TEXT, size=12),
        xaxis=dict(title="Calculated Alignment Time Offsets", color=AXIS_TEXT),
        yaxis=dict(title="Coincidence Votes", color=AXIS_TEXT)
    )
    return fig

def plotly_candidates(candidates, height=260):
    fig = go.Figure()
    if candidates:
        ordered = list(reversed(candidates))
        names = [c[0] for c in ordered]
        scores = [c[1] for c in ordered]
        colors = [PURPLE] * (len(ordered) - 1) + [CYAN]
        fig.add_bar(x=scores, y=names, orientation="h", marker_color=colors, text=scores, textposition="outside")
    fig.update_layout(
        height=height, margin=dict(l=10, r=40, t=10, b=40),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT, size=12),
        yaxis=dict(automargin=True)
    )
    return fig

def metric_card_html(label, value):
    return f'<div style="background:#FFF; border-radius:16px; padding:16px; border:1px solid #E2E5EC; text-align:center;"><div style="color:#6B7280; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;">{label}</div><div style="font-family:monospace; color:#111827; font-size:1.35rem; font-weight:600; margin-top:4px;">{value}</div></div>'

# =====================================================
# FRONTEND ENGINE LAYOUT
# =====================================================
st.set_page_config(page_title="Acoustic Fingerprinting Dashboard", page_icon="🎵", layout="wide")

st.markdown(
    f'<div style="padding: 2rem; border-radius: 24px; background: linear-gradient(135deg, rgba(8,145,178,0.08), rgba(124,58,237,0.08)); border: 1px solid #E2E5EC; margin-bottom: 1.5rem;">'
    f'<div style="font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, {CYAN}, {PURPLE}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🎵 EE200 Audio Fingerprinting</div>'
    f'<div style="color: #6B7280; margin-top: 4px; font-size: 0.95rem;">Production Environment — Real-Time Diagnostic Dashboard</div>'
    f'</div>',
    unsafe_allow_html=True
)

all_songs_meta = repo.fetch_all_songs()
left_col, right_col = st.columns([1, 4])

with left_col:
    st.markdown("##### System Index Status")
    st.markdown(metric_card_html("Indexed Tracks", f"{len(all_songs_meta)}"), unsafe_allow_html=True)
    st.divider()
    st.caption("Active Configuration:")
    st.caption(f"Sample Rate: {SR} Hz \n\nMatching Floor Cutoff: {MIN_SCORE} Hits")

with right_col:
    tab_identify, tab_batch = st.tabs(["🎯 Single Query Identification", "📦 Batch Processing Workspace"])
    
    # -------------------------------------------------
    # TAB 1: SINGLE QUERY IDENTIFICATION
    # -------------------------------------------------
    with tab_identify:
        uploaded = st.file_uploader("Upload Audio Fragment Query", type=["mp3", "wav", "m4a"], key="single_uploader")
        
        if uploaded:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                query_path = tmp.name
                
            with st.spinner("Analyzing audio fingerprint structures..."):
                result = identify_song_detailed(query_path, rec_svc)
            
            confidence = (result["score"] / result["total_hashes"] * 100) if result["total_hashes"] else 0
            
            st.divider()
            st.markdown(f"**Playing File Source Fragment:** `{uploaded.name}`")
            st.audio(uploaded.getvalue())
            
            with st.container():
                st.markdown("##### DSP Pipeline Processing Velocities")
                t_cols = st.columns(len(result["timings"]))
                for col, (stage, ms) in zip(t_cols, result["timings"].items()):
                    col.markdown(metric_card_html(stage, f"{ms:.1f} ms"), unsafe_allow_html=True)
                
            st.markdown("##### Analysis Output")
            if result["song"]:
                st.markdown(
                    f'<div style="background: linear-gradient(135deg, {CYAN}, {PURPLE}); border-radius: 24px; padding: 28px 32px; margin-bottom:1.5rem;">'
                    f'<div style="color: rgba(255,255,255,0.85); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;">MATCH IDENTIFIED</div>'
                    f'<div style="font-size: 1.9rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{result["song"]}</div>'
                    f'<div style="color: rgba(255,255,255,0.9); font-size: 0.95rem; font-family:monospace;">Consensus Score Matrix Cluster: {result["score"]} · {confidence:.1f}% Confidence Match</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background: #F1F2F6; border: 1px solid {BORDER}; border-radius: 24px; padding: 28px 32px; margin-bottom:1.5rem;">'
                    f'<div style="color: {TEXT_DIM}; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;">NO TARGET MATCH MATCHED</div>'
                    f'<div style="font-size: 1.9rem; font-weight: 700; color: {TEXT}; margin: 4px 0;">Acoustic Footprint Missing</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            if result["candidates"]:
                st.markdown("##### Candidate Ranking Distributions")
                st.plotly_chart(plotly_candidates(result["candidates"]), use_container_width=True, config=CHART_CONFIG, key="single_cand")
                
            st.markdown("##### Pipeline Signal Deconstruction Visualization")
            e1, e2 = st.columns(2)
            with e1:
                st.caption("Step 1: Computed Audio Log-dB Spectrogram")
                st.plotly_chart(plotly_spectrogram(result["spts"]), use_container_width=True, config=CHART_CONFIG, key="single_spec")
            with e2:
                st.caption("Step 2: Extracted Structural Local Maxima Constellation Field Map")
                st.plotly_chart(plotly_constellation(result["peaks"]), use_container_width=True, config=CHART_CONFIG, key="single_const")
            
            st.caption("Step 3: Scatter Offset Delta Consensual Cross-Correlation Alignment Density Profile")
            st.plotly_chart(plotly_histogram(result["offsets"]), use_container_width=True, config=CHART_CONFIG, key="single_hist")
            
            try:
                os.remove(query_path)
            except Exception:
                pass

    # -------------------------------------------------
    # TAB 2: BATCH PROCESSING WORKSPACE (ADJUSTED FOR STRICT EVALUATION)
    # -------------------------------------------------
    with tab_batch:
        st.markdown("##### 📦 Processing Run Engine Operations")
        batch_files = st.file_uploader("Drop Multiple Audio Targets for Processing Ingestion", type=["mp3", "wav", "m4a"], accept_multiple_files=True, key="batch_uploader")
        
        if batch_files:
            ui_rows = []
            evaluation_rows = []
            progress = st.progress(0, text="Iterating batch entries...")
            
            for i, file in enumerate(batch_files):
                suffix = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.getbuffer())
                    path = tmp.name
                    
                res = identify_song_detailed(path, rec_svc)
                confidence = (res["score"] / res["total_hashes"] * 100) if res["total_hashes"] else 0
                
                # Strip extension for evaluation requirement
                prediction_clean = os.path.splitext(res["song"])[0] if res["song"] else "None"
                
                # Main diagnostic UI view data
                ui_rows.append([
                    file.name,
                    res["song"] if res["song"] else "No match identified",
                    f"{res['score']} votes",
                    f"{confidence:.1f}%",
                    f"{res['duration']:.2f}s"
                ])
                
                # Evaluation format: exactly 'filename' and 'prediction'
                evaluation_rows.append([file.name, prediction_clean])
                    
                progress.progress((i + 1) / len(batch_files), text=f"Processed and matched: {file.name}")
                try:
                    os.remove(path)
                except Exception:
                    pass
                
            progress.empty()
            
            df_ui = pd.DataFrame(ui_rows, columns=["Filename", "Acoustic Prediction", "Raw Cluster Score", "Confidence", "Length"])
            df_eval = pd.DataFrame(evaluation_rows, columns=["filename", "prediction"])
            
            st.markdown("##### Batch Processing Predictions Consolidated View")
            st.dataframe(df_ui, use_container_width=True, hide_index=True)
            
            # Download button generates the exact 2-column layout format
            st.download_button(
                label="📥 Download results.csv (Strict Evaluation Format)",
                data=df_eval.to_csv(index=False),
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_btn_res"
            )
