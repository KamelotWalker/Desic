"""The Laya-style decision network (requires PyTorch).

    encoder backbone (bidirectional)            ModernBERT / mmBERT / any HF encoder,
          │                                     or a tiny built-in encoder
    2-layer decision transformer
          │
    ├── option scorer on every [MASK] marker  → logits → softmax over *this* question's answers
    └── act head on [CLS]                      → P(the top answer is right)  (act / escalate)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .text import MARK, ScratchTokenizer

class ScratchBackbone(nn.Module):
    kind = "scratch"

    def __init__(self, vocab_size: int = 16384, hidden: int = 128, layers: int = 2, heads: int = 4, max_len: int = 256) -> None:
        super().__init__()
        self.tokenizer = ScratchTokenizer(vocab_size)
        self.hidden_size = hidden
        self.max_len = max_len
        self.mask_token_id = ScratchTokenizer.MASK
        self.cfg = {"kind": "scratch", "vocab_size": vocab_size, "hidden": hidden, "layers": layers, "heads": heads, "max_len": max_len}
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def encode(self, first: str, second: str) -> list[int] | None:
        return self.tokenizer.encode(first, second, self.max_len)

    def forward(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        x = self.emb(ids) + self.pos(pos)
        return self.norm(self.enc(x, src_key_padding_mask=attn == 0))


class HFBackbone(nn.Module):
    """Any Hugging Face encoder with a mask token (ModernBERT, mmBERT, BERT, DeBERTa …)."""

    kind = "hf"

    def __init__(self, name_or_path: str, max_len: int = 512) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError("install the neural extra: pip install 'desic[neural]'") from e
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if self.tokenizer.mask_token is None:
            raise ValueError(f"{name_or_path} has no mask token; use an encoder such as ModernBERT or mmBERT")
        self.model = AutoModel.from_pretrained(name_or_path)
        self.hidden_size = self.model.config.hidden_size
        self.max_len = min(max_len, getattr(self.tokenizer, "model_max_length", max_len) or max_len)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cfg = {"kind": "hf", "name": name_or_path, "max_len": self.max_len}

    def encode(self, first: str, second: str) -> list[int] | None:
        first = first.replace(MARK, self.tokenizer.mask_token)
        n_first = len(self.tokenizer(first)["input_ids"])
        if n_first > self.max_len - 8:
            return None
        return self.tokenizer(first, second or " ", truncation="only_second", max_length=self.max_len)["input_ids"]

    def forward(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids, attention_mask=attn).last_hidden_state


def build_backbone(spec: dict) -> nn.Module:
    if spec.get("kind") == "scratch":
        return ScratchBackbone(**{k: v for k, v in spec.items() if k in ("vocab_size", "hidden", "layers", "heads", "max_len")})
    return HFBackbone(spec["name"], int(spec.get("max_len", 512)))


def backbone_spec(name: str, max_len: int | None = None) -> dict:
    if name == "scratch":
        return {"kind": "scratch", "max_len": max_len or 256}
    return {"kind": "hf", "name": name, "max_len": max_len or 512}


class DecisionNet(nn.Module):
    def __init__(self, backbone: nn.Module, head_layers: int = 2) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = backbone.hidden_size
        heads = next(h for h in (hidden // 64, 8, 4, 2, 1) if h >= 1 and hidden % h == 0)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.act = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.head_layers = head_layers

    def forward(self, ids: torch.Tensor, attn: torch.Tensor, markers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns option logits [B, K] (padded with -inf) and act logits [B]."""
        h = self.backbone(ids, attn)
        h = self.head(h, src_key_padding_mask=attn == 0)
        scores = self.scorer(h).squeeze(-1)  # [B, T]
        logits = torch.full(markers.shape, float("-inf"), device=scores.device, dtype=scores.dtype)
        valid = markers >= 0
        gathered = torch.gather(scores, 1, markers.clamp(min=0))
        logits = torch.where(valid, gathered, logits)
        return logits, self.act(h[:, 0]).squeeze(-1)


def collate(batch_ids: list[list[int]], mask_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = max(len(x) for x in batch_ids)
    ids = torch.zeros((len(batch_ids), t), dtype=torch.long)
    attn = torch.zeros((len(batch_ids), t), dtype=torch.long)
    positions = []
    for i, x in enumerate(batch_ids):
        ids[i, : len(x)] = torch.tensor(x)
        attn[i, : len(x)] = 1
        positions.append([j for j, tok in enumerate(x) if tok == mask_id])
    k = max(len(p) for p in positions)
    markers = torch.full((len(batch_ids), k), -1, dtype=torch.long)
    for i, p in enumerate(positions):
        markers[i, : len(p)] = torch.tensor(p)
    return ids.to(device), attn.to(device), markers.to(device)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_net(net: DecisionNet, directory: Path, extra: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    bb = net.backbone
    config = {"backbone": bb.cfg, "head_layers": net.head_layers, **extra}
    if bb.kind == "hf":
        bb.model.save_pretrained(directory / "backbone")
        bb.tokenizer.save_pretrained(directory / "backbone")
        config["backbone"] = {**bb.cfg, "name": str(directory / "backbone"), "source": bb.cfg["name"]}
        torch.save({k: v for k, v in net.state_dict().items() if not k.startswith("backbone.")}, directory / "head.pt")
    else:
        torch.save(net.state_dict(), directory / "model.pt")
    (directory / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))


def load_net(directory: Path, device: torch.device) -> tuple[DecisionNet, dict]:
    config = json.loads((directory / "config.json").read_text())
    net = DecisionNet(build_backbone(config["backbone"]), config.get("head_layers", 2))
    if (directory / "model.pt").exists():
        net.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    else:
        net.load_state_dict(torch.load(directory / "head.pt", map_location="cpu", weights_only=True), strict=False)
    net.to(device).eval()
    return net, config
