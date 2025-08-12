from __future__ import annotations
import json
from pathlib import Path
import whisper
from .ffmpeg_tools import ensure_dir

def transcribe_all(preprocessed_jsonl: str | Path, out_dir: str | Path, model_name: str = "small"):
    ensure_dir(out_dir)
    model = whisper.load_model(model_name)
    outs = []
    with open(preprocessed_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            asset_id = rec["asset_id"]
            audio = rec["audio_wav"]
            out_json = Path(out_dir) / f"{asset_id}.json"
            result = model.transcribe(audio, language="en")
            with open(out_json, "w") as fo:
                json.dump(result, fo)
            outs.append({"asset_id": asset_id, "transcript_json": str(out_json)})
    return outs

def transcript_text(json_path: str | Path) -> str:
    with open(json_path) as f:
        j = json.load(f)
    segs = j.get("segments", [])
    return " ".join(s.get("text","").strip() for s in segs) if segs else j.get("text","")
