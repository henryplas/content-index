import os, sys, json, glob, time, io
from pathlib import Path
import streamlit as st
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import faiss

# make local src/ importable
sys.path.append(os.getcwd())
from src.embed.clip_embed import ClipEmbedder

ART_DIR = Path("artifacts")

@st.cache_resource
def load_all():
    # load segment index + meta (required)
    seg_index = faiss.read_index(str(ART_DIR / "faiss_segments.index"))
    seg_meta  = [json.loads(l) for l in open(ART_DIR / "meta_segments.jsonl")]
    # also load asset index for the other tab (optional)
    asset_index = faiss.read_index(str(ART_DIR / "faiss_hnsw.index"))
    asset_meta  = [json.loads(l) for l in open(ART_DIR / "meta.jsonl")]
    embedder = ClipEmbedder()
    return seg_index, seg_meta, asset_index, asset_meta, embedder

def pick_frames_for_window(frames_dir: str, start_sec: int, end_sec: int, max_frames: int = 8):
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frames: return []
    start = max(0, int(start_sec)); end = min(len(frames), int(end_sec))
    if end <= start: return []
    # subsample evenly up to max_frames
    count = end - start
    step = max(1, count // max_frames)
    picks = frames[start:end:step][:max_frames]
    return picks

def make_gif(frame_paths, fps=4):
    # returns bytes of an animated GIF built from frames
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    # ensure a nice size
    imgs = [im.resize((min(320, im.width), int(min(320, im.width)*im.height/im.width))) for im in imgs]
    buf = io.BytesIO()
    imageio.mimsave(buf, imgs, format="GIF", duration=1.0/max(fps,1))
    buf.seek(0)
    return buf

st.set_page_config(page_title="Segment Search Viewer", layout="wide")
st.title("🎬 Segment-level Media Search")

# load once
try:
    seg_index, seg_meta, asset_index, asset_meta, embedder = load_all()
except Exception as e:
    st.error(f"Failed to load indices or embeddings: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Segment search (timestamps)", "Whole-asset search"])

with tab1:
    col_l, col_r = st.columns([2,1])
    with col_l:
        q = st.text_input("Query", value="bunny in the forest", placeholder="Describe a moment…")
    with col_r:
        k = st.slider("Top-K", 1, min(20, len(seg_meta)), min(10, len(seg_meta)))

    if q:
        t0 = time.time()
        v = embedder.text_embed(q)
        D, I = seg_index.search(v, k)
        ms = (time.time() - t0) * 1000
        st.caption(f"Search time: {ms:.1f} ms  •  segments: {len(seg_meta)}")

        for rank, (sim, idx) in enumerate(zip(D[0], I[0]), 1):
            i = int(idx)
            if i < 0: continue
            m = seg_meta[i]
            st.markdown(f"**{rank}. {m['asset_id']}**  —  `{m['start_sec']}s → {m['end_sec']}s`  ·  sim `{float(sim):.3f}`")
            c1, c2 = st.columns([3,2])
            with c1:
                # play the video at the segment start (Streamlit supports start_time)
                st.video(m["video_path"], start_time=int(m["start_sec"]))
            with c2:
                picks = pick_frames_for_window(m["frames_dir"], m["start_sec"], m["end_sec"], max_frames=8)
                if picks:
                    st.caption("Thumbnails")
                    st.image(picks, use_container_width=True)
                    if st.button(f"Make GIF for {m['segment_id']}", key=f"gif_{rank}"):
                        gif = make_gif(picks, fps=4)
                        st.image(gif)
                        st.download_button("Download GIF", data=gif, file_name=f"{m['segment_id']}.gif", mime="image/gif")

with tab2:
    q2 = st.text_input("Query (asset-level)", value="spaceship with robots")
    k2 = st.slider("Top-K (assets)", 1, min(10, len(asset_meta)), min(3, len(asset_meta)))
    if q2:
        v2 = embedder.text_embed(q2)
        D2, I2 = asset_index.search(v2, k2)
        for rank, (sim, idx) in enumerate(zip(D2[0], I2[0]), 1):
            m = asset_meta[int(idx)]
            st.markdown(f"**{rank}. {m['asset_id']}** · sim `{float(sim):.3f}`")
            st.video(m["video_path"])
