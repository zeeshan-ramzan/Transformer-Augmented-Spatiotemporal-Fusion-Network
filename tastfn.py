"""
tastfn.py - Transformer-Augmented Spatiotemporal Fusion Network (TASTFN), the proposed method.

Architecture (Methods section of the paper)
-------------------------------------------
1. Dual-stream input encoder: the 10 features of each month are split into the
   agrometeorological stream (6 weather variables) and the NDVI stream (4 NDVI
   statistics); each stream is projected to d_model = 64 and layer-normalised.
2. Phenological positional encoding (PPE), added to the agrometeorological stream:
       PPE(t, 2i)   = sin(t / 10000^(2i/d_model)) * cos(2*pi*m_t/12)
       PPE(t, 2i+1) = cos(t / 10000^(2i/d_model)) * cos(2*pi*m_t/12)
   (t = position in the window, m_t = calendar month).
3. Multi-head self-attention encoder: L = 2 pre-norm transformer layers,
   N_h = 8 heads (d_k = 8), feed-forward width d_ff = 256, padding mask.
4. NDVI branch: two-layer feed-forward network per time step, then a masked mean
   over the window -> z_ndvi.
5. Cross-modal attention gating (CMAG): z_ndvi is the query, the T encoded
   agrometeorological steps are keys and values,
       A = softmax((z_ndvi Wq)(Z_agro Wk)^T / sqrt(d_model) + M),  c = A (Z_agro Wv)
       alpha = sigmoid(Wg [c; z_ndvi] + bg),  f = alpha * c + (1 - alpha) * z_ndvi
6. Residual regression decoder: widths [128, 64, 32], ReLU, dropout 0.30,
       f(l+1) = ReLU(W f(l) + b) + P f(l),  then a linear output unit.

Training: AdamW (lr 1e-3, weight decay 1e-4), cosine annealing with warm restarts
(T_0 = 50 epochs, eta_min = 1e-5), MSE loss, batch size 32, at most 500 epochs,
early stopping with patience 30 on validation RMSE; stratified 10-fold CV.

Each sample is a window of T = 6 consecutive monthly observations of one plot
ending at the target month t; earlier missing positions are zero-padded and masked.
Only the features of the window are used as inputs (never the yields), so no
target information leaks from neighbouring months.

Outputs (results/tastfn/): metrics.json, oof_predictions.csv, learning_curves.csv,
attention_layer2.npy (mean T x T matrix), gates_by_month.csv, complexity.json.

Usage:
  python tastfn.py --data data/nthri_tea_yield.csv
  python tastfn.py --data data/nthri_tea_yield.csv --epochs 50          # quick run
  python tastfn.py --early-stop-on-test   # early stopping on the held-out fold (paper protocol)
"""
from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import (AGRO_FEATURES, NDVI_FEATURES, TARGET, base_parser, format_metrics, load_data,
                    mean_metrics, measure_ops, regression_metrics, save_json, save_predictions, set_seed,
                    stratified_folds)


# ============================================================================ configuration
@dataclass
class TASTFNConfig:
    model: str = "tastfn"            # "tastfn" | "mlp_flat" | "mlp_seq"  (the last two are ablation baselines)
    window: int = 6                  # T
    d_model: int = 64
    n_heads: int = 8
    d_ff: int = 256
    n_layers: int = 2                # L
    attn_dropout: float = 0.1
    dropout: float = 0.30            # decoder dropout
    pe: str = "ppe"                  # "ppe" | "sinusoidal" | "none"
    fusion: str = "cmag"             # "cmag" | "concat"
    residual_decoder: bool = True
    decoder_widths: List[int] = field(default_factory=lambda: [128, 64, 32])
    agro_features: List[str] = field(default_factory=lambda: list(AGRO_FEATURES))
    ndvi_features: List[str] = field(default_factory=lambda: list(NDVI_FEATURES))
    lr: float = 1e-3
    weight_decay: float = 1e-4
    t0: int = 50
    eta_min: float = 1e-5
    batch_size: int = 32
    epochs: int = 500
    patience: int = 30


# ============================================================================ model
def phenological_encoding(months: torch.Tensor, d_model: int, pe: str = "ppe") -> torch.Tensor:
    """months: (B, T) calendar months 1..12 -> (B, T, d_model)."""
    B, T = months.shape
    pos = torch.arange(1, T + 1, device=months.device, dtype=torch.float32).view(1, T, 1)
    i = torch.arange(0, d_model, 2, device=months.device, dtype=torch.float32)
    div = torch.pow(10000.0, i / d_model).view(1, 1, -1)
    enc = torch.zeros(B, T, d_model, device=months.device)
    enc[..., 0::2] = torch.sin(pos / div)
    enc[..., 1::2] = torch.cos(pos / div)
    if pe == "ppe":
        enc = enc * torch.cos(2 * math.pi * months.float() / 12.0).unsqueeze(-1)
    return enc


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer that also returns the attention weights."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, pad_mask):
        x = self.ln1(h)
        a, w = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=True, average_attn_weights=False)
        h = h + self.drop(a)                                   # H(l) = H(l-1) + MHSA(LN(H(l-1)))
        h = h + self.drop(self.ffn(self.ln2(h)))               # position-wise FFN
        return h, w                                            # w: (B, heads, T, T)


class CMAG(nn.Module):
    """Cross-modal attention gating."""

    def __init__(self, d_model):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wg = nn.Linear(2 * d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, z_ndvi, Z_agro, pad_mask):
        q = self.wq(z_ndvi).unsqueeze(1)                       # (B, 1, d)
        k, v = self.wk(Z_agro), self.wv(Z_agro)                # (B, T, d)
        scores = (q @ k.transpose(1, 2)) / self.scale          # (B, 1, T)
        scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        A = torch.softmax(scores, dim=-1)
        c = (A @ v).squeeze(1)                                 # (B, d)
        alpha = torch.sigmoid(self.wg(torch.cat([c, z_ndvi], dim=-1)))
        f = alpha * c + (1 - alpha) * z_ndvi
        return f, alpha, A.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, d_in, widths, dropout, residual=True):
        super().__init__()
        self.layers, self.proj = nn.ModuleList(), nn.ModuleList()
        self.residual = residual
        for w in widths:
            self.layers.append(nn.Linear(d_in, w))
            self.proj.append(nn.Linear(d_in, w, bias=False) if residual else nn.Identity())
            d_in = w
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_in, 1)

    def forward(self, f):
        for lin, P in zip(self.layers, self.proj):
            h = self.drop(F.relu(lin(f)))
            f = h + P(f) if self.residual else h            # f(l+1) = ReLU(W f(l) + b) + P f(l)
        return self.out(f).squeeze(-1)


class TASTFN(nn.Module):
    def __init__(self, cfg: TASTFNConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.n_agro, self.n_ndvi = len(cfg.agro_features), len(cfg.ndvi_features)
        self.proj_agro = nn.Linear(self.n_agro, d)
        self.proj_ndvi = nn.Linear(self.n_ndvi, d)
        self.ln_agro, self.ln_ndvi = nn.LayerNorm(d), nn.LayerNorm(d)
        self.encoder = nn.ModuleList([EncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.attn_dropout)
                                      for _ in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(d)
        self.ndvi_ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        if cfg.fusion == "cmag":
            self.cmag = CMAG(d)
        else:
            self.concat = nn.Linear(2 * d, d)
        self.decoder = Decoder(d, cfg.decoder_widths, cfg.dropout, cfg.residual_decoder)

    @staticmethod
    def masked_mean(x, valid):
        w = valid.unsqueeze(-1).float()
        return (x * w).sum(1) / w.sum(1).clamp(min=1.0)

    def forward(self, x, months, valid, return_extras=False):
        """x: (B, T, n_agro + n_ndvi), months: (B, T), valid: (B, T) bool."""
        pad = ~valid
        xa, xn = x[..., : self.n_agro], x[..., self.n_agro:]
        h = self.ln_agro(self.proj_agro(xa))
        if self.cfg.pe != "none":
            h = h + phenological_encoding(months, self.cfg.d_model, self.cfg.pe)
        attn = []
        for layer in self.encoder:
            h, w = layer(h, pad)
            attn.append(w)
        Z = self.ln_out(h)
        z_ndvi = self.masked_mean(self.ndvi_ffn(self.ln_ndvi(self.proj_ndvi(xn))), valid)
        if self.cfg.fusion == "cmag":
            f, alpha, A = self.cmag(z_ndvi, Z, pad)
        else:
            f = self.concat(torch.cat([self.masked_mean(Z, valid), z_ndvi], -1))
            alpha, A = None, None
        y = self.decoder(f)
        if return_extras:
            return y, {"attn": attn, "alpha": alpha, "cross_attn": A}
        return y


class FlatMLP(nn.Module):
    """Ablation baselines: 3-layer MLP on the current month only (mlp_flat) or on the
    flattened window (mlp_seq, 'temporal sequencing')."""

    def __init__(self, cfg: TASTFNConfig, n_features: int):
        super().__init__()
        self.seq = cfg.model == "mlp_seq"
        d_in = n_features * (cfg.window if self.seq else 1)
        layers = []
        for w in cfg.decoder_widths:
            layers += [nn.Linear(d_in, w), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d_in = w
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, months, valid, return_extras=False):
        z = (x * valid.unsqueeze(-1).float()).flatten(1) if self.seq else x[:, -1, :]
        y = self.net(z).squeeze(-1)
        return (y, {"attn": [], "alpha": None, "cross_attn": None}) if return_extras else y


def build_model(cfg: TASTFNConfig) -> nn.Module:
    if cfg.model in ("mlp_flat", "mlp_seq"):
        return FlatMLP(cfg, len(cfg.agro_features) + len(cfg.ndvi_features))
    return TASTFN(cfg)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_flops(cfg: TASTFNConfig) -> int:
    """Analytic FLOPs per observation (1 multiply-accumulate = 2 FLOPs)."""
    T, d, na, nn_ = cfg.window, cfg.d_model, len(cfg.agro_features), len(cfg.ndvi_features)
    macs = T * na * d + T * nn_ * d
    macs += cfg.n_layers * (T * d * 3 * d + 2 * T * T * d + T * d * d + 2 * T * d * cfg.d_ff)
    macs += 2 * T * d * d                                          # NDVI feed-forward branch
    if cfg.fusion == "cmag":
        macs += d * d + 2 * T * d * d + 2 * T * d + 2 * d * d       # q, K, V, scores, context, gate
    else:
        macs += 2 * d * d
    d_in = d
    for w in cfg.decoder_widths:
        macs += d_in * w * (2 if cfg.residual_decoder else 1)
        d_in = w
    macs += d_in
    return int(2 * macs)


# ============================================================================ data
def build_windows(df: pd.DataFrame, T: int) -> np.ndarray:
    """Row indices of the T-month window ending at each row (-1 = padding).
    df must be sorted by farm_id and date (load_data does this)."""
    win = np.full((len(df), T), -1, dtype=np.int64)
    for _, idx in df.groupby("farm_id", sort=False).indices.items():
        idx = np.sort(idx)
        for j, row in enumerate(idx):
            take = idx[max(0, j - T + 1): j + 1]
            win[row, T - len(take):] = take
    return win


def window_tensors(Xs: np.ndarray, months: np.ndarray, win: np.ndarray, rows: np.ndarray, device):
    w = win[rows]
    valid = w >= 0
    safe = np.where(valid, w, 0)
    x = Xs[safe] * valid[..., None]
    m = np.where(valid, months[safe], 1)
    return (torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(m, dtype=torch.long, device=device),
            torch.tensor(valid, dtype=torch.bool, device=device))


# ============================================================================ training
@torch.no_grad()
def predict(model, x, m, v, batch=1024, extras=False):
    model.eval()
    ys, alphas, attn2, cross = [], [], [], []
    for i in range(0, len(x), batch):
        y, ex = model(x[i:i + batch], m[i:i + batch], v[i:i + batch], return_extras=True)
        ys.append(y)
        if extras:
            if ex["alpha"] is not None:
                alphas.append(ex["alpha"].mean(-1))
                cross.append(ex["cross_attn"])
            if ex["attn"]:
                attn2.append(ex["attn"][-1].mean(1))              # last layer, mean over heads
    out = {"y": torch.cat(ys).cpu().numpy()}
    if extras:
        out["alpha"] = torch.cat(alphas).cpu().numpy() if alphas else None
        out["attn_last"] = torch.cat(attn2).cpu().numpy() if attn2 else None
        out["cross_attn"] = torch.cat(cross).cpu().numpy() if cross else None
    return out


def train_fold(cfg, Xs, months, win, y_s, tr, val, te, device, seed, log_prefix=""):
    """Train on rows `tr`, early-stop on rows `val`, monitor RMSE on `te` each epoch."""
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.t0, eta_min=cfg.eta_min)
    xtr, mtr, vtr = window_tensors(Xs, months, win, tr, device)
    ytr = torch.tensor(y_s["scaled"][tr], dtype=torch.float32, device=device)
    xva, mva, vva = window_tensors(Xs, months, win, val, device)
    xte, mte, vte = window_tensors(Xs, months, win, te, device)
    inv = y_s["inverse"]
    y_raw = y_s["raw"]
    best, best_rmse, wait, curve = None, float("inf"), 0, []
    n = len(tr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            loss = F.mse_loss(model(xtr[b], mtr[b], vtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step(epoch + i / n)
        val_rmse = float(np.sqrt(np.mean((inv(predict(model, xva, mva, vva)["y"]) - y_raw[val]) ** 2)))
        te_rmse = float(np.sqrt(np.mean((inv(predict(model, xte, mte, vte)["y"]) - y_raw[te]) ** 2)))
        curve.append({"epoch": epoch + 1, "val_rmse": val_rmse, "test_fold_rmse": te_rmse})
        if val_rmse < best_rmse - 1e-6:
            best_rmse, best, wait = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best)
    print(f"{log_prefix} stopped at epoch {len(curve)}  best val RMSE={best_rmse:.2f}")
    return model, curve, (xte, mte, vte)


def prepare_data(df: pd.DataFrame, cfg: TASTFNConfig):
    feats = cfg.agro_features + cfg.ndvi_features
    X = df[feats].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    months = df["month"].to_numpy(dtype=int)
    win = build_windows(df, cfg.window if cfg.model != "mlp_flat" else 1)
    if cfg.model == "mlp_flat":
        win = np.repeat(win, cfg.window, axis=1)
        win[:, :-1] = -1
    return X, y, months, win


def run_cv(cfg: TASTFNConfig, args, name: str = "tastfn", df: Optional[pd.DataFrame] = None,
           save_extras: bool = True, verbose: bool = True) -> Dict:
    set_seed(args.seed)
    device = torch.device(args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu"))
    df = load_data(args.data) if df is None else df
    X, y, months, win = prepare_data(df, cfg)
    folds = stratified_folds(y, args.folds, args.seed)
    oof = np.zeros(len(y))
    fold_id = np.zeros(len(y), dtype=int)
    per_fold, curves, ops = [], [], []
    alpha_all = np.full(len(y), np.nan)
    attn_sum, attn_n = None, 0
    t_start = time.perf_counter()
    for k, (tr_all, te) in enumerate(folds):
        sx = StandardScaler().fit(X[tr_all])                     # statistics of the training rows only
        Xs = sx.transform(X)
        sy = StandardScaler().fit(y[tr_all].reshape(-1, 1))
        y_s = {"raw": y, "scaled": sy.transform(y.reshape(-1, 1)).ravel(),
               "inverse": lambda p, sy=sy: sy.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()}
        if getattr(args, "early_stop_on_test", False):
            tr, val = tr_all, te
        else:
            tr, val = train_test_split(tr_all, test_size=0.1, random_state=args.seed + k)
        model, curve, (xte, mte, vte) = train_fold(cfg, Xs, months, win, y_s, tr, val, te, device,
                                                    args.seed + k, f"[{name}] fold {k + 1:2d}")
        res = predict(model, xte, mte, vte, extras=save_extras)
        pred = y_s["inverse"](res["y"])
        oof[te], fold_id[te] = pred, k
        m = regression_metrics(y[te], pred)
        per_fold.append(m)
        for c in curve:
            curves.append({"fold": k + 1, **c})
        if save_extras and res.get("alpha") is not None:
            alpha_all[te] = res["alpha"]
        if save_extras and res.get("attn_last") is not None:
            full = vte.all(dim=1).cpu().numpy()                 # windows without padding
            if full.any():
                s = res["attn_last"][full].sum(0)
                attn_sum = s if attn_sum is None else attn_sum + s
                attn_n += int(full.sum())
        ops.append(measure_ops(lambda _x, model=model: predict(model, xte, mte, vte)["y"], np.zeros(len(te))))
        if verbose:
            print(f"[{name}] fold {k + 1:2d}/{len(folds)}  {format_metrics(m)}")
    t_total = time.perf_counter() - t_start
    summary = mean_metrics(per_fold)
    model_cpu = build_model(cfg)
    summary.update({"OPS": float(np.median(ops)), "train_time_s": t_total,
                    "parameters": count_parameters(model_cpu),
                    "flops_per_observation": count_flops(cfg) if cfg.model == "tastfn" else None})
    print(f"[{name}] MEAN  {format_metrics(summary)}  params={summary['parameters']}")
    out = os.path.join(args.out_dir, name)
    save_json({"model": name, "config": asdict(cfg), "mean": summary, "folds": per_fold}, os.path.join(out, "metrics.json"))
    save_predictions(df, oof, fold_id, os.path.join(out, "oof_predictions.csv"))
    pd.DataFrame(curves).to_csv(os.path.join(out, "learning_curves.csv"), index=False)
    if save_extras and attn_sum is not None:
        np.save(os.path.join(out, "attention_layer2.npy"), attn_sum / max(attn_n, 1))
    if save_extras and not np.all(np.isnan(alpha_all)):
        g = pd.DataFrame({"month": df["month"].to_numpy(), "alpha": alpha_all}).dropna()
        g.groupby("month")["alpha"].agg(["mean", "std", "count"]).reset_index().to_csv(
            os.path.join(out, "gates_by_month.csv"), index=False)
    save_json({"parameters": summary["parameters"], "flops_per_observation": summary["flops_per_observation"],
               "train_time_s_10fold": t_total, "OPS": summary["OPS"]}, os.path.join(out, "complexity.json"))
    return {"summary": summary, "oof": oof, "fold_id": fold_id, "folds": folds}


def add_common_args(p):
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--early-stop-on-test", action="store_true",
                   help="use the held-out fold for early stopping (protocol of the paper); "
                        "default uses 10%% of the training rows as an inner validation set")
    return p


if __name__ == "__main__":
    p = add_common_args(base_parser(__doc__))
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=256)
    a = p.parse_args()
    config = TASTFNConfig(window=a.window, n_layers=a.layers, n_heads=a.heads, d_model=a.d_model, d_ff=a.d_ff,
                          epochs=a.epochs, patience=a.patience, batch_size=a.batch_size)
    print(f"TASTFN parameters: {count_parameters(build_model(config)):,}   FLOPs/observation: {count_flops(config):,}")
    run_cv(config, a, "tastfn")
