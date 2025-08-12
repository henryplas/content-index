# Content Index (MVP)

This project builds a text-to-video retrieval index using:
- CLIP (image/text) embeddings
- 1 fps keyframes averaged per asset
- Whisper transcripts (text) fused into the vector
- Faiss HNSW index
- FastAPI serving

Quickstart:

    bash scripts/run_all.sh
    # then open http://127.0.0.1:8000/docs
