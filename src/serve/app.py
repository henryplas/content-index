from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, Query
import faiss
from src.embed.clip_embed import ClipEmbedder

app = FastAPI(title="Content Index API", version="0.3.0")

# whole-asset index (cosine/IP)
A_INDEX_PATH = Path("artifacts/faiss_hnsw.index")
A_META_PATH  = Path("artifacts/meta.jsonl")

# segment index (cosine/IP)
S_INDEX_PATH = Path("artifacts/faiss_segments.index")
S_META_PATH  = Path("artifacts/meta_segments.jsonl")

asset_index = faiss.read_index(str(A_INDEX_PATH))
asset_meta  = [json.loads(l) for l in open(A_META_PATH)]
segment_index = faiss.read_index(str(S_INDEX_PATH)) if S_INDEX_PATH.exists() else None
segment_meta  = [json.loads(l) for l in open(S_META_PATH)] if S_META_PATH.exists() else []

embedder = ClipEmbedder()

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "assets": len(asset_meta),
        "asset_index_ntotal": asset_index.ntotal,
        "segments": len(segment_meta),
        "segment_index_ntotal": (segment_index.ntotal if segment_index is not None else 0),
    }

@app.get("/query")
def query(q: str = Query(...), k: int = 5):
    n = len(asset_meta)
    k = max(1, min(k, n))
    v = embedder.text_embed(q)
    D, I = asset_index.search(v, k)
    results = []
    for sim, i in zip(D[0], I[0]):
        i = int(i)
        if i < 0: continue
        m = asset_meta[i]
        results.append({
            "asset_id": m["asset_id"],
            "score": float(sim),
            "video_path": m["video_path"],
            "frames_dir": m["frames_dir"],
            "transcript_json": m["transcript_json"],
        })
    return {"mode": "asset", "q": q, "k": k, "results": results}

@app.get("/query_segments")
def query_segments(q: str = Query(...), k: int = 10):
    if segment_index is None:
        return {"error": "segment index not built; run build_segments.py"}
    n = len(segment_meta)
    k = max(1, min(k, n))
    v = embedder.text_embed(q)
    D, I = segment_index.search(v, k)
    results = []
    for sim, i in zip(D[0], I[0]):
        i = int(i)
        if i < 0: continue
        m = segment_meta[i]
        results.append({
            "asset_id": m["asset_id"],
            "segment_id": m["segment_id"],
            "start_sec": m["start_sec"],
            "end_sec": m["end_sec"],
            "score": float(sim),
            "video_path": m["video_path"],
        })
    return {"mode": "segment", "q": q, "k": k, "results": results}
