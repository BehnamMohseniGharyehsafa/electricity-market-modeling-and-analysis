"""Compact, explicitly documented implementations of modern neural forecasters.

These are benchmark adaptations of published ideas, not official paper reproductions.
Every model maps a seven-day context to all 24 prices in one forward pass.
"""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import TARGET


class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.calendar = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 24))


class DLinear(Forecaster):
    """Moving-average decomposition with separate seasonal and trend linear heads."""

    def __init__(self, context=168, channels=12):
        super().__init__()
        self.seasonal = nn.Linear(context, 24)
        self.trend = nn.Linear(context, 24)
        self.pool = nn.AvgPool1d(25, stride=1)

    def forward(self, history, calendar):
        x = history[:, :, TARGET]
        padded = nn.functional.pad(x[:, None], (12, 12), mode="replicate")
        trend = self.pool(padded).squeeze(1)
        return self.seasonal(x - trend) + self.trend(trend) + self.calendar(calendar)


class LSTM(Forecaster):
    def __init__(self, context=168, channels=12):
        super().__init__()
        self.encoder = nn.LSTM(channels, 48, batch_first=True)
        self.head = nn.Linear(48, 24)

    def forward(self, history, calendar):
        _, (state, _) = self.encoder(history)
        return self.head(state[-1]) + self.calendar(calendar)


class PatchTST(Forecaster):
    """Channel-independent patch attention, NYC target, reversible instance scaling.

    Only the NYC channel is used here: this is intentionally a univariate adaptation.
    Patches: length 24, stride 12; learned positions; two pre-norm encoder blocks.
    """

    def __init__(self, context=168, channels=12):
        super().__init__()
        patches = (context - 24) // 12 + 1
        self.embed = nn.Linear(24, 48)
        self.position = nn.Parameter(torch.zeros(1, patches, 48))
        layer = nn.TransformerEncoderLayer(
            48, 4, 96, dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(patches * 48, 24)

    def forward(self, history, calendar):
        x = history[:, :, TARGET]
        mean = x.mean(dim=1, keepdim=True)
        scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.1)
        patches = ((x - mean) / scale).unfold(1, 24, 12)
        encoded = self.encoder(self.embed(patches) + self.position)
        return self.head(encoded.flatten(1)) * scale + mean + self.calendar(calendar)


class ITransformer(Forecaster):
    """Invert axes: each zone's full history is a token; attention mixes zones."""

    def __init__(self, context=168, channels=12):
        super().__init__()
        self.embed = nn.Linear(context, 48)
        layer = nn.TransformerEncoderLayer(
            48, 4, 96, dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(48, 24)

    def forward(self, history, calendar):
        # Price-only tokens; the historical DST flag is used by the tabular/LSTM models.
        x = history[:, :, :-1]
        mean = x.mean(dim=1, keepdim=True)
        scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.1)
        encoded = self.encoder(self.embed(((x - mean) / scale).transpose(1, 2)))
        target = self.head(encoded[:, TARGET])
        return (
            target * scale[:, 0, TARGET, None] + mean[:, 0, TARGET, None] + self.calendar(calendar)
        )


ARCHITECTURES = {
    "DLinear": DLinear,
    "LSTM": LSTM,
    "PatchTST": PatchTST,
    "iTransformer": ITransformer,
}


def train_neural(name, samples, train_mask, val_mask, seed=42, epochs=100):
    """Fit transformations on training only; restore lowest validation-MAE checkpoint."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    center = samples.history[train_mask, :, :-1].mean()
    scale = max(float(samples.history[train_mask, :, :-1].std()), 1.0)
    history = samples.history.copy()
    history[:, :, :-1] = (history[:, :, :-1] - center) / scale
    x = torch.tensor(history)
    c = torch.tensor(samples.calendar)
    y = torch.tensor((samples.target - center) / scale)
    net = ARCHITECTURES[name](context=history.shape[1], channels=history.shape[2])
    loader = DataLoader(
        TensorDataset(x[train_mask], c[train_mask], y[train_mask]),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.L1Loss()
    best, best_state, stale, best_epoch = float("inf"), None, 0, 0
    trace = []
    for epoch in range(epochs):
        net.train()
        for xb, cb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(xb, cb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(x[val_mask], c[val_mask]), y[val_mask]))
        trace.append({"epoch": epoch + 1, "validation_mae": val * scale})
        if val < best - 1e-5:
            best, best_state, stale, best_epoch = val, copy.deepcopy(net.state_dict()), 0, epoch + 1
        else:
            stale += 1
        if stale >= 15:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        predictions = torch.cat(
            [net(xb, cb) for xb, cb in DataLoader(TensorDataset(x, c), batch_size=128)]
        ).numpy()
    metadata = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(trace),
        "parameters": sum(p.numel() for p in net.parameters()),
        "history_center": float(center),
        "history_scale": scale,
        "trace": trace,
    }
    checkpoint = {
        "state_dict": best_state,
        "center": float(center),
        "scale": scale,
        "architecture": name,
        "context": history.shape[1],
    }
    return predictions * scale + center, metadata, checkpoint
