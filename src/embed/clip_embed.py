from __future__ import annotations
from pathlib import Path
import numpy as np
import torch, open_clip
from PIL import Image

class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "laion2b_s32b_b82k", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.inference_mode()
    def image_paths_embed(self, frame_paths: list[str], batch_size: int = 32) -> np.ndarray:
        if not frame_paths:
            raise RuntimeError("No frame paths provided")
        feats = []
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i+batch_size]
            imgs = [self.preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in batch_paths]
            batch = torch.cat(imgs).to(self.device)
            f = self.model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
        all_feats = torch.cat(feats, dim=0)
        mean_feat = all_feats.mean(dim=0, keepdim=True)
        mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
        return mean_feat.cpu().numpy().astype("float32")

    @torch.inference_mode()
    def image_dir_embed(self, frames_dir: str | Path, max_frames: int | None = None) -> np.ndarray:
        frames = sorted(Path(frames_dir).glob("*.jpg"))
        if max_frames is not None:
            frames = frames[:max_frames]
        return self.image_paths_embed([str(p) for p in frames])

    @torch.inference_mode()
    def text_embed(self, text: str) -> np.ndarray:
        toks = self.tokenizer([text]).to(self.device)
        f = self.model.encode_text(toks)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype("float32")
