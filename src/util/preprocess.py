from __future__ import annotations
import json
from pathlib import Path
from .ffmpeg_tools import extract_audio_wav, extract_frames_1fps, ensure_dir

def find_videos(raw_dir: str | Path) -> list[Path]:
    raw = Path(raw_dir)
    return sorted([p for p in raw.rglob("*") if p.suffix.lower() in {".mp4", ".mkv", ".m4v", ".mov"}])

def slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

def build_manifest(raw_dir: str | Path, out_manifest: str | Path) -> list[dict]:
    vids = find_videos(raw_dir)
    items = []
    for v in vids:
        asset_id = slugify(v.stem)
        items.append({"asset_id": asset_id, "path": str(v)})
    with open(out_manifest, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return items

def preprocess_assets(manifest_path: str | Path, work_root: str | Path) -> list[dict]:
    work = Path(work_root)
    ensure_dir(work)
    out_items = []
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            asset_id = rec["asset_id"]
            video = Path(rec["path"])
            frames_dir = work / "frames" / asset_id
            audio_wav  = work / "audio" / f"{asset_id}.wav"
            extract_frames_1fps(video, frames_dir)
            extract_audio_wav(video, audio_wav)
            out_items.append({
                "asset_id": asset_id,
                "video_path": str(video),
                "frames_dir": str(frames_dir),
                "audio_wav": str(audio_wav),
            })
    out_jsonl = Path(work_root) / "preprocessed.jsonl"
    with open(out_jsonl, "w") as f:
        for it in out_items:
            f.write(json.dumps(it) + "\n")
    return out_items
