from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
import numpy as np, faiss
from tqdm import tqdm
from src.embed.clip_embed import ClipEmbedder
from src.util.transcribe import transcript_text

def fuse(image_vec: np.ndarray, text_vec: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    v = alpha * image_vec + (1 - alpha) * text_vec
    # L2-normalize so dot == cosine
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype("float32")

def build(preprocessed_jsonl: str | Path, transcripts_dir: str | Path, artifacts_dir: str | Path, alpha: float = 0.7):
    artifacts = Path(artifacts_dir); artifacts.mkdir(parents=True, exist_ok=True)
    embedder = ClipEmbedder()
    meta: List[Dict] = []
    vecs: List[np.ndarray] = []

    with open(preprocessed_jsonl) as f:
        for line in tqdm(f, desc="Embedding assets"):
            rec = json.loads(line)
            asset_id = rec["asset_id"]
            frames_dir = rec["frames_dir"]
            tjson = Path(transcripts_dir) / f"{asset_id}.json"
            text = transcript_text(tjson) if tjson.exists() else asset_id

            v_img = embedder.image_dir_embed(frames_dir)
            v_txt = embedder.text_embed(text)
            v = fuse(v_img, v_txt, alpha=alpha)
            vecs.append(v)
            meta.append({
                "asset_id": asset_id,
                "frames_dir": frames_dir,
                "transcript_json": str(tjson),
                "video_path": rec.get("video_path",""),
            })

    X = np.vstack(vecs).astype("float32")      # already normalized
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)               # inner product (cosine for normalized)
    index.add(X)

    faiss.write_index(index, str(artifacts / "faiss_hnsw.index"))  # reuse same filename
    with open(artifacts / "meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    print(f"Indexed {len(meta)} assets -> {artifacts/'faiss_hnsw.index'}")
