import os, sys, json, glob
import streamlit as st
from PIL import Image
import numpy as np
import faiss

sys.path.append(os.getcwd())
from src.embed.clip_embed import ClipEmbedder

@st.cache_resource
def load_resources():
    idx = faiss.read_index("artifacts/faiss_hnsw.index")
    meta = [json.loads(l) for l in open("artifacts/meta.jsonl")]
    emb = ClipEmbedder()
    return idx, meta, emb

def show_frames(frames_dir, n=6):
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))[:n]
    if not frames: 
        st.write("(no frames)")
        return
    cols = st.columns(min(n, 6))
    for i, p in enumerate(frames):
        with cols[i % len(cols)]:
            st.image(Image.open(p), use_container_width=True)

st.title("Content Index Viewer")
idx, meta, emb = load_resources()

q = st.text_input("Query", value="bunny in the forest")
k = st.slider("Top-K", 1, min(10, len(meta)), min(5, len(meta)))

if st.button("Search") or q:
    v = emb.text_embed(q)
    D, I = idx.search(v, k)
    st.subheader("Results")
    for rank, (d, i) in enumerate(zip(D[0], I[0]), 1):
        i = int(i)
        if i < 0: 
            continue
        m = meta[i]
        st.markdown(f"**{rank}. {m['asset_id']}**  • score: `{float(d):.3f}`")
        show_frames(m["frames_dir"])
        st.video(m["video_path"])
