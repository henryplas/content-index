import subprocess
from pathlib import Path

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def extract_audio_wav(video_path: str | Path, out_wav: str | Path, sr: int = 16000):
    ensure_dir(Path(out_wav).parent)
    run(["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", str(sr), str(out_wav)])

def extract_frames_1fps(video_path: str | Path, out_dir: str | Path):
    ensure_dir(out_dir)
    run(["ffmpeg", "-y", "-i", str(video_path), "-vf", "fps=1", f"{out_dir}/frame_%06d.jpg"])
