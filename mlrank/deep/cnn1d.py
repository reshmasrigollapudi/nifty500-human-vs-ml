"""1D-CNN on raw price sequences (needs torch).

Input per (T, symbol): the last 252 daily log
returns up to the T-1 close, scaled by that
stock's own volatility. Target: label_rank.
Purpose: test whether a model that learns its
own features beats hand-built features + GBM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data_loader import MarketData

SEQ_LEN = 252


def build_sequences(md: MarketData, panel: pd.DataFrame, seq_len: int = SEQ_LEN):
    """Returns float32 array (n_rows, 1, seq_len)
    aligned with panel.index. Missing history is
    zero-padded on the left."""
    c = md.close.ffill(limit=5)
    r = np.log(c / c.shift(1)).to_numpy(np.float32)
    cal = md.dates
    col = {s: i for i, s in enumerate(md.symbols)}
    dates = panel.index.get_level_values("date")
    syms = panel.index.get_level_values("symbol")
    dec_pos = cal.get_indexer(dates) - 1
    out = np.zeros((len(panel), 1, seq_len), dtype=np.float32)
    for i, (p, s) in enumerate(zip(dec_pos, syms)):
        j = col[s]
        lo = max(0, p - seq_len + 1)
        seg = r[lo:p + 1, j]
        seg = np.nan_to_num(seg)
        sd = seg.std() or 1.0
        out[i, 0, seq_len - len(seg):] = seg / sd
    return out


def _model(torch):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Dropout(0.2), nn.Linear(32, 1),
    )


class CNN1DRanker:
    name = "cnn1d"

    def __init__(self, epochs: int = 8, batch: int = 512, lr: float = 1e-3,
                 weight_decay: float = 1e-4, seed: int = 0, device: str | None = None):
        import torch
        self.torch = torch
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.wd, self.seed = weight_decay, seed
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_seq: np.ndarray, y: np.ndarray):
        torch = self.torch
        torch.manual_seed(self.seed)
        self.net_ = _model(torch).to(self.device)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        Xt = torch.from_numpy(X_seq)
        yt = torch.from_numpy(np.asarray(y, np.float32)).view(-1, 1)
        n = len(Xt)
        for ep in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n)
            total = 0.0
            for k in range(0, n, self.batch):
                b = perm[k:k + self.batch]
                xb, yb = Xt[b].to(self.device), yt[b].to(self.device)
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net_(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            print(f"      epoch {ep + 1}/{self.epochs} mse={total / n:.5f}")
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.net_.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(X_seq), 4096):
                xb = torch.from_numpy(X_seq[k:k + 4096]).to(self.device)
                out.append(self.net_(xb).cpu().numpy().ravel())
        return np.concatenate(out)
