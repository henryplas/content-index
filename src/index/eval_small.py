from __future__ import annotations
import json, time
import faiss
from src.embed.clip_embed import ClipEmbedder

def load_meta(meta_path: str):
    rows = []
    with open(meta_path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    idx = faiss.read_index("artifacts/faiss_hnsw.index")
    meta = load_meta("artifacts/meta.jsonl")
    emb = ClipEmbedder()
    queries = [
        "bunny in the forest",
        "spaceship and robots",
        "snowy mountains and a girl with a sword",
        "dialogue inside a house",
    ]
    for q in queries:
        v = emb.text_embed(q)
        t0 = time.time()
        D, I = idx.search(v, 3)
        lat = (time.time() - t0) * 1000
        print(f"\nQ: {q}  (latency: {lat:.2f} ms)")
        for rank, (d, i) in enumerate(zip(D[0], I[0]), 1):
            print(f"  {rank}. {meta[i]['asset_id']}  score={1-d:.3f}")

if __name__ == "__main__":
    main()
