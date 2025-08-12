# Content Index — Multimodal Segment Search

A tiny media search engine. It ingests videos → extracts 1 fps frames + audio (ffmpeg) → transcribes to text (Whisper) → embeds visuals + text (CLIP) → builds a Faiss vector index → serves semantic search with FastAPI. Bonus: segment-level retrieval (30s windows) + a Streamlit viewer.

---

## 🚀 Live Demo (replace with your links)
- API docs: https://\<PUBLIC_API_URL\>/docs
- Segment viewer: https://\<PUBLIC_VIEWER_URL\>:8501

On Lightning Studio, set ports 8000 (API) and 8501 (Streamlit) to Public.

---

## 🧭 How it works
1. Preprocess: ffmpeg → 1 fps frames (artifacts/frames/<asset>/*.jpg) + mono 16 kHz audio.
2. ASR: Whisper → artifacts/transcripts/<asset>.json.
3. Embeddings: CLIP image (frames) + CLIP text (transcript); fuse with weight alpha; L2-normalize.
4. Index: Faiss (inner product ≡ cosine on normalized vectors).
   - Asset-level: faiss_hnsw.index, meta.jsonl (1 vector/video).
   - Segment-level (30s): faiss_segments.index, meta_segments.jsonl.
5. Serve: FastAPI /query (assets) and /query_segments (segments).
6. View: Streamlit app to preview hits, jump to timestamps, export GIFs.

---

## ⏱ Quickstart

API:
    export PYTHONPATH=.
    uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
    # open http://127.0.0.1:8000/docs

Queries (examples):
    curl -s http://127.0.0.1:8000/healthz | python -m json.tool
    curl -sG --data-urlencode 'q=bunny in the forest' http://127.0.0.1:8000/query | python -m json.tool
    curl -sG --data-urlencode 'q=spaceship with robots' http://127.0.0.1:8000/query_segments | python -m json.tool

Viewer:
    streamlit run apps/segments_viewer.py --server.address 0.0.0.0 --server.port 8501

---

## 📷 Screens & Artifacts (files expected in this repo)
- API screenshot → docs/api.png
- Viewer screenshot → docs/viewer.png
- Contact sheets → docs/contact_<asset>.jpg
- Segment GIFs → docs/seg_bunny.gif, docs/seg_robots.gif, docs/seg_snowy.gif
- Latency histogram (optional) → docs/latency.png

---

## 🧾 API
- GET /healthz  → { ok, assets, asset_index_ntotal, segments, segment_index_ntotal }
- GET /query?q=...&k=...  → top-K assets with similarity
- GET /query_segments?q=...&k=...  → top-K segments with start_sec/end_sec

---

## 🧱 Repo layout (key bits)
    src/util/        ffmpeg, preprocessing, whisper
    src/embed/       CLIP embedder
    src/index/       asset + segment index builders
    src/serve/app.py FastAPI
    apps/            Streamlit viewer
    artifacts/       frames/, audio/, transcripts/, meta*.jsonl, faiss*.index

---

## 🔧 Tuning
- alpha (vision vs text): higher = visuals, lower = transcript
- seg_seconds (window size): 15–60s
- frame_step (subsample): 2–3 for speed
- CLIP backbone: ViT-B/32 (fast) ↔ ViT-L/14 (better)
