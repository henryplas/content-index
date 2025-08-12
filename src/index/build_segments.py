from __future__ import annotations
import json, glob, math
from pathlib import Path
from typing import List, Dict
import numpy as np, faiss
from tqdm import tqdm
from src.embed.clip_embed import ClipEmbedder
import json as _json

def _load_transcript_segments(json_path: Path):
    if not json_path.exists(): return []
    with open(json_path) as f:
        j = _json.load(f)
    return j.get("segments", [])

def _text_for_window(segments, t0: float, t1: float) -> str:
    if not segments: return ""
    keep = [s["text"].strip() for s in segments if not (s["end"] <= t0 or s["start"] >= t1)]
    return " ".join(keep) if keep else ""

def build_segments(preprocessed_jsonl: str | Path,
                   transcripts_dir: str | Path,
                   artifacts_dir: str | Path,
                   seg_seconds: int = 30,
                   alpha: float = 0.8,
                   frame_step: int = 1):
    """
    seg_seconds: window length (seconds) since we extracted at 1 fps
    alpha: weight for image vs text in fusion
    frame_step: sample every Nth frame inside a window for speed (1 = use all)
    """
    artifacts = Path(artifacts_dir); artifacts.mkdir(parents=True, exist_ok=True)
    embedder = ClipEmbedder()
    meta: List[Dict] = []
    vecs: List[np.ndarray] = []

    with open(preprocessed_jsonl) as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Assets"):
        rec = json.loads(line)
        asset = rec["asset_id"]
        frames_dir = Path(rec["frames_dir"])
        video_path = rec.get("video_path","")
        tjson = Path(transcripts_dir) / f"{asset}.json"
        tsegments = _load_transcript_segments(tjson)

        frames = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))
        if not frames: 
            continue

        n = len(frames)                   # 1 fps -> n seconds
        nsegs = math.ceil(n / seg_seconds)
        for s in range(nsegs):
            s0 = s * seg_seconds
            s1 = min((s + 1) * seg_seconds, n)
            # select only frames in this window (optionally subsample)
            win_paths = frames[s0:s1:frame_step]
            if not win_paths:
                continue

            # embed only the window's frames
            v_img = embedder.image_paths_embed(win_paths)

            # transcript slice for [s0, s1)
            txt = _text_for_window(tsegments, float(s0), float(s1)) or asset
            v_txt = embedder.text_embed(txt)

            # fuse and normalize (cosine-ready)
            v = alpha * v_img + (1 - alpha) * v_txt
            v = v / np.linalg.norm(v, axis=1, keepdims=True)

            vecs.append(v.astype("float32"))
            meta.append({
                "asset_id": asset,
                "segment_id": f"{asset}_{s0:06d}_{s1:06d}",
                "start_sec": s0,
                "end_sec": int(s1),
                "frames_dir": str(frames_dir),
                "video_path": video_path,
                "transcript_json": str(tjson),
            })

    if not vecs:
        raise RuntimeError("No segments produced")

    X = np.vstack(vecs).astype("float32")  # already normalized
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)           # inner product = cosine for normalized
    index.add(X)

    faiss.write_index(index, str(artifacts / "faiss_segments.index"))
    with open(artifacts / "meta_segments.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    print(f"Built {len(meta)} segments -> {artifacts/'faiss_segments.index'}")
