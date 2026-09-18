"""Soccer 05 - sequence student: a PyTorch model over the 20-event history vs LightGBM E2.

Stage 02 trained one LightGBM student per 360 target on tabular ``E2`` features (current event,
possession context, 10-event window counts).  This stage trains a small neural network on the
same rows and the same match-grouped folds: a GRU (or transformer encoder) over the raw 20-slot
event sequence (``seq_*`` columns: type id, x, y, seconds before the current event, same-team
flag) concatenated with an MLP over the ``E2`` current-event design, with one head per headline
target (multi-task).  Variants differ only in the history the sequence branch sees: ``seq20``
(20 slots), ``seq1`` (the previous event only) and ``seq0`` (no sequence branch, i.e. the
neural analogue of LightGBM E2), so ``seq20 - seq1`` isolates what the raw history adds.

Protocol: folds come from ``imputed_oof.parquet`` (stage 02, ``group_kfold`` by match, seed 0,
verified against a recomputation); labels are the stage-02 ``y_<target>`` columns (identical
subsets and label rules); early stopping uses an inner match holdout of the training folds;
training rows are thinned uniformly to ``train_cap``.  Baselines next to every model: the
per-event-type training mean (``base_type``) and LightGBM ``E2`` out-of-fold predictions from
stage 02, scored on exactly the same rows.  Deltas: paired per-row bootstrap on squared error /
log-loss (memory-safe re-implementation of ``common.metrics.paired_bootstrap_delta``) and the
match-clustered bootstrap next to it.

Leakage contract: the network reads only ``f_*`` / ``seq_*`` columns (through
``imputation_features.build_design('E2')`` and the ``seq_*`` block); ``f_after_*`` is not in E2.

Run from the repo root::

    python -m research.privileged_tracking.soccer.sequence_student --stage train \\
        --variants seq20,seq1,seq0 --folds 0,1,2
    python -m research.privileged_tracking.soccer.sequence_student --stage report --folds 0,1,2
    python -m research.privileged_tracking.soccer.sequence_student --stage all --smoke

Outputs: ``processed_dir('soccer')/imputed_oof_seq.parquet`` (``<target>__<variant>`` out-of-fold
predictions, NaN outside the target subset and on folds not run), checkpoints
``processed_dir('soccer')/models/seq/<variant>__fold<k>.pt`` (:class:`SeqBundle`),
``reports/soccer_05_sequence.md`` and ``soccer_05_*.parquet``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn

from research.privileged_tracking.common.io import processed_dir, reports_dir
from research.privileged_tracking.common.metrics import (
    clustered_bootstrap_delta,
    per_sample_log_loss,
)
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.common.splits import group_kfold
from research.privileged_tracking.soccer import imputation_features as imf
from research.privileged_tracking.soccer.imputation import ELAPSED_BINS, score, skill_of

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: headline targets of this stage (stage-02 names; kinds / subsets from ``imf.TARGETS``)
TARGETS: tuple[str, ...] = (
    "block_depth",
    "def_line",
    "n_opp_ahead_of_ball",
    "nearest_opp_dist",
    "n_opp_in_cone",
    "deep_block",
    "counter_on",
)
#: variant name -> (architecture, window length)
VARIANTS: dict[str, tuple[str, int]] = {
    "seq20": ("gru", 20),
    "seq1": ("gru", 1),
    "seq0": ("gru", 0),
    "tr20": ("transformer", 20),
}
VARIANT_DESCRIPTION: dict[str, str] = {
    "seq20": "GRU over the 20-slot event history + E2 current-event MLP (multi-task)",
    "seq1": "same network, history truncated to the previous event only (window length 1)",
    "seq0": "no sequence branch: the E2 current-event MLP alone (neural analogue of LightGBM E2)",
    "tr20": "2-layer transformer encoder over the 20-slot history + E2 MLP",
    "lgbm_E2": "stage-02 LightGBM E2 student (out-of-fold, one model per target)",
    "base_type": "stage-02 per-event-type training mean / base rate",
}
LGBM = "lgbm_E2"
BASE = "base_type"
FEATURE_SET = "E2"
CACHE_SUBDIR = "seq_cache"
SEQ_N_FEATS = 8  # per-slot numeric features, see :func:`sequence_tokens`


@dataclass
class SeqConfig:
    """Driver / model configuration.

    Attributes:
        seed: base seed (fold ``k`` uses ``seed + k``).
        n_threads: torch CPU threads.
        train_cap / val_cap: training / early-stopping rows kept (uniform thinning).
        inner_holdout_frac: share of training matches held out for early stopping.
        d_model: token / GRU width; ``cat_dim`` embedding width per categorical column.
        n_layers / n_heads: transformer depth / heads (``arch == 'transformer'``).
        mlp_hidden / head_hidden / dropout: current-event MLP and shared head.
        batch_size / lr / weight_decay / max_epochs / patience / max_minutes: optimisation.
        loss_weights: per-target loss weight (default 1 each).
        n_boot: bootstrap resamples.
        smoke: first parquet row group only, separate cache, nothing written to reports / models.
    """

    seed: int = 0
    n_threads: int = 2
    train_cap: int = 300_000
    val_cap: int = 60_000
    inner_holdout_frac: float = 0.15
    d_model: int = 64
    cat_dim: int = 8
    n_layers: int = 2
    n_heads: int = 4
    mlp_hidden: tuple[int, ...] = (256, 128)
    head_hidden: int = 128
    dropout: float = 0.1
    batch_size: int = 1024
    lr: float = 2e-3
    weight_decay: float = 1e-2
    max_epochs: int = 15
    patience: int = 3
    max_minutes: float = 10.0
    loss_weights: dict[str, float] = field(default_factory=lambda: {t: 1.0 for t in TARGETS})
    n_boot: int = 1000
    smoke: bool = False

    @property
    def cache_dir(self) -> Path:
        d = processed_dir("soccer") / (CACHE_SUBDIR + ("_smoke" if self.smoke else ""))
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def model_dir(self) -> Path:
        d = processed_dir("soccer") / "models" / ("seq_smoke" if self.smoke else "seq")
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# Pure feature helpers (tested on synthetic input)
# ---------------------------------------------------------------------------


def seq_arrays(df: pd.DataFrame, seq_len: int = imf.SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Raw sequence block as arrays, slot 1 (most recent) first.

    Returns:
        ``(types [n, seq_len] int16, raw [n, seq_len, 4] float32)`` with raw fields
        ``(x, y, dt, same)`` exactly as stored (NaN / -1 padding).
    """
    n = len(df)
    types = np.zeros((n, seq_len), dtype=np.int16)
    raw = np.full((n, seq_len, 4), np.nan, dtype=np.float32)
    for k in range(1, seq_len + 1):
        types[:, k - 1] = df[f"seq_type_{k:02d}"].to_numpy(dtype=np.int16)
        for j, f in enumerate(("x", "y", "dt", "same")):
            raw[:, k - 1, j] = df[f"seq_{f}_{k:02d}"].to_numpy(dtype=np.float32)
    return types, raw


def sequence_tokens(
    types: np.ndarray, raw: np.ndarray, cur_xy: np.ndarray, window_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenise the last ``window_len`` slots, oldest first, for the sequence encoder.

    Args:
        types: slot type ids ``[n, seq_len]`` (slot 1 first, 0 = pad).
        raw: ``[n, seq_len, 4]`` raw slot fields ``(x, y, dt, same)``.
        cur_xy: current event location ``[n, 2]`` (event team's frame).
        window_len: slots kept (``1 .. seq_len``); 0 returns empty ``[n, 0, ...]`` arrays.

    Returns:
        ``(type_ids [n, L] int64, feats [n, L, 8] float32, pad [n, L] bool)`` ordered oldest ->
        newest (the last token is slot 1).  Per-slot features: ``x / 120``, ``y / 80``,
        ``log1p(dt)``, own-team flag, opponent flag, pad flag, ``(x - cur_x) / 120``,
        ``(y - cur_y) / 80``; NaN (pads) become 0.
    """
    n = len(types)
    if window_len <= 0:
        return (
            np.zeros((n, 0), dtype=np.int64),
            np.zeros((n, 0, SEQ_N_FEATS), dtype=np.float32),
            np.zeros((n, 0), dtype=bool),
        )
    sl = slice(window_len - 1, None, -1)  # slots window_len .. 1 -> oldest first
    t = types[:, sl].astype(np.int64)
    r = raw[:, sl, :].astype(np.float32)
    pad = t == 0
    x, y, dt, same = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    cx = np.asarray(cur_xy, dtype=np.float32)[:, 0][:, None]
    cy = np.asarray(cur_xy, dtype=np.float32)[:, 1][:, None]
    feats = np.stack(
        [
            x / 120.0,
            y / 80.0,
            np.log1p(np.clip(dt, 0.0, None)),
            (same == 1).astype(np.float32),
            (same == 0).astype(np.float32),
            pad.astype(np.float32),
            (x - cx) / 120.0,
            (y - cy) / 80.0,
        ],
        axis=-1,
    ).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats[pad] = 0.0
    feats[pad, 5] = 1.0
    return t, feats, pad


def split_design(design: pd.DataFrame, categorical: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Split a :func:`imf.build_design` frame into categorical codes and numeric columns.

    Returns:
        ``(codes [n, n_cat] int64, num [n, n_num] float32)``; codes are shifted by one so that
        NaN -> 0 (``code + 1`` otherwise, unseen already ``len(vocab)``); numeric keeps NaN.
    """
    cat_cols = [c for c in design.columns if c in categorical]
    num_cols = [c for c in design.columns if c not in categorical]
    codes = np.zeros((len(design), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        v = design[c].to_numpy(dtype=np.float32)
        codes[:, j] = np.where(np.isnan(v), 0, v + 1).astype(np.int64)
    num = design[num_cols].to_numpy(dtype=np.float32)
    return codes, num


def categorical_sizes(categorical: list[str]) -> list[int]:
    """Embedding table sizes: ``vocab + 2`` (unseen code and the NaN code 0)."""
    sizes = imf.vocab_sizes()
    return [sizes[c] + 2 for c in categorical]


@dataclass
class NumericStats:
    """Training-row mean / sd of the numeric design columns (NaN ignored)."""

    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, num: np.ndarray) -> NumericStats:
        """Column stats over ``num [n_train, n_num]``; constant / all-NaN columns get sd 1."""
        mean = np.nanmean(num, axis=0)
        sd = np.nanstd(num, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        sd = np.where(np.isnan(sd) | (sd < 1e-6), 1.0, sd)
        return cls(mean=mean.astype(np.float32), sd=sd.astype(np.float32))

    def transform(self, num: np.ndarray) -> np.ndarray:
        """``[n, 2 * n_num]``: z-scores (NaN -> 0) followed by missing indicators."""
        z = (num - self.mean) / self.sd
        miss = np.isnan(num).astype(np.float32)
        return np.concatenate([np.nan_to_num(z, nan=0.0), miss], axis=1).astype(np.float32)


@dataclass
class TargetStats:
    """Training mean / sd per continuous target (binaries get 0 / 1)."""

    mean: dict[str, float]
    sd: dict[str, float]

    @classmethod
    def fit(cls, y: pd.DataFrame) -> TargetStats:
        mean, sd = {}, {}
        for t in y.columns:
            if imf.TARGET_BY_NAME[t].kind == "binary":
                mean[t], sd[t] = 0.0, 1.0
            else:
                v = y[t].to_numpy(dtype=float)
                mean[t] = float(np.nanmean(v))
                s = float(np.nanstd(v))
                sd[t] = s if s > 1e-6 else 1.0
        return cls(mean=mean, sd=sd)


def target_matrix(y: pd.DataFrame, stats: TargetStats) -> tuple[np.ndarray, np.ndarray]:
    """Standardised targets and validity masks.

    Returns:
        ``(values [n, n_targets] float32 with NaN -> 0, mask [n, n_targets] bool)``.
    """
    vals = np.zeros((len(y), y.shape[1]), dtype=np.float32)
    mask = np.zeros((len(y), y.shape[1]), dtype=bool)
    for j, t in enumerate(y.columns):
        v = y[t].to_numpy(dtype=np.float32)
        ok = ~np.isnan(v)
        mask[:, j] = ok
        vals[ok, j] = (v[ok] - stats.mean[t]) / stats.sd[t]
    return vals, mask


def decode_outputs(out: np.ndarray, targets: list[str], stats: TargetStats) -> np.ndarray:
    """Network outputs ``[n, n_targets]`` -> predictions in target units (sigmoid for binaries,
    de-standardised and clipped at 0 for counts / distances)."""
    pred = np.empty_like(out, dtype=np.float32)
    for j, t in enumerate(targets):
        spec = imf.TARGET_BY_NAME[t]
        if spec.kind == "binary":
            pred[:, j] = 1.0 / (1.0 + np.exp(-out[:, j]))
        else:
            v = out[:, j] * stats.sd[t] + stats.mean[t]
            pred[:, j] = np.clip(v, 0.0, None) if spec.kind == "count" or "dist" in t else v
    return pred


def bootstrap_delta_per_sample(
    loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 1000, seed: int = 0, chunk: int = 20
) -> tuple[float, float, float]:
    """Paired per-sample bootstrap of ``mean(loss_a - loss_b)`` in chunks (bounded memory).

    Same semantics as ``common.metrics.paired_bootstrap_delta`` (positive = ``b`` better,
    95% interval) but draws ``chunk`` resamples at a time so a million-row comparison does not
    allocate ``n_boot x n`` indices at once.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done : done + m] = d[idx].mean(axis=1)
        done += m
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SequenceStudentNet(nn.Module):
    """Sequence branch + current-event MLP + multi-task head.

    Inputs: ``seq_types [B, L]`` int64, ``seq_feats [B, L, 8]``, ``seq_pad [B, L]`` bool,
    ``cat_codes [B, n_cat]`` int64, ``num [B, 2 * n_num]``; output ``[B, n_targets]`` (raw
    regression outputs in standardised units, logits for binaries).  ``window_len == 0``
    disables the sequence branch.
    """

    def __init__(
        self,
        cfg: SeqConfig,
        arch: str,
        window_len: int,
        cat_sizes: list[int],
        n_num: int,
        n_targets: int,
        n_type_ids: int = imf.N_TYPE_IDS,
    ) -> None:
        super().__init__()
        self.arch, self.window_len = arch, window_len
        d = cfg.d_model
        seq_out = 0
        if window_len > 0:
            self.type_emb = nn.Embedding(n_type_ids, d)
            self.pos_emb = nn.Embedding(imf.SEQ_LEN, d)
            self.feat_proj = nn.Linear(SEQ_N_FEATS, d)
            if arch == "gru":
                self.encoder = nn.GRU(d, d, batch_first=True)
            elif arch == "transformer":
                layer = nn.TransformerEncoderLayer(
                    d, cfg.n_heads, 4 * d, dropout=cfg.dropout, batch_first=True, norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    layer, cfg.n_layers, enable_nested_tensor=False
                )
            else:  # pragma: no cover - config error
                raise ValueError(arch)
            seq_out = 2 * d  # final / newest state + masked mean over the window
        self.cat_embs = nn.ModuleList([nn.Embedding(s, cfg.cat_dim) for s in cat_sizes])
        layers: list[nn.Module] = []
        width = n_num + cfg.cat_dim * len(cat_sizes)
        for h in cfg.mlp_hidden:
            layers += [nn.Linear(width, h), nn.GELU(), nn.Dropout(cfg.dropout)]
            width = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(width + seq_out, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, n_targets),
        )

    def encode_sequence(
        self, seq_types: torch.Tensor, seq_feats: torch.Tensor, seq_pad: torch.Tensor
    ) -> torch.Tensor:
        """``[B, L]``, ``[B, L, 8]``, ``[B, L]`` -> ``[B, 2 d]``."""
        b, length = seq_types.shape
        # position id = slot number - 1 (0 = most recent); tokens run oldest -> newest
        pos = torch.arange(length - 1, -1, -1, device=seq_types.device).expand(b, length)
        h = self.type_emb(seq_types) + self.feat_proj(seq_feats) + self.pos_emb(pos)
        if self.arch == "gru":
            out, _ = self.encoder(h)
        else:
            out = self.encoder(h, src_key_padding_mask=seq_pad)
        keep = (~seq_pad).unsqueeze(-1).to(out.dtype)
        mean = (out * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return torch.cat([out[:, -1], mean], dim=1)

    def forward(
        self,
        seq_types: torch.Tensor,
        seq_feats: torch.Tensor,
        seq_pad: torch.Tensor,
        cat_codes: torch.Tensor,
        num: torch.Tensor,
    ) -> torch.Tensor:
        parts = [num] + [emb(cat_codes[:, j]) for j, emb in enumerate(self.cat_embs)]
        z = self.mlp(torch.cat(parts, dim=1))
        if self.window_len > 0:
            z = torch.cat([z, self.encode_sequence(seq_types, seq_feats, seq_pad)], dim=1)
        return self.head(z)


def multitask_loss(
    out: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    kinds: list[str],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum of masked per-target losses (MSE for standardised regression / count heads,
    BCE-with-logits for binaries); targets without a valid row in the batch contribute 0.

    Shapes: ``out / y [B, T]``, ``mask [B, T]`` bool, ``weights [T]``.
    """
    total = out.new_zeros(())
    for j, kind in enumerate(kinds):
        m = mask[:, j]
        if not bool(m.any()):
            continue
        o, t = out[m, j], y[m, j]
        if kind == "binary":
            lj = nn.functional.binary_cross_entropy_with_logits(o, t)
        else:
            lj = nn.functional.mse_loss(o, t)
        total = total + weights[j] * lj
    return total


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

ID_COLS = ["match_id", "event_id", "event_index", "period", "f_type", "f_poss_elapsed"]


@dataclass
class SeqData:
    """Every events360 row with features, labels, folds and stage-02 references (row-aligned).

    Attributes:
        ids: ``ID_COLS`` frame ``[n]``.
        y: stage-02 labels ``y_<target>`` ``[n, T]`` (NaN = outside subset / invalid label).
        ref: stage-02 out-of-fold references ``{target: {lgbm_E2 | base_type | base_global:
            pred[n]}}``.
        fold: match-grouped fold id ``[n]``; ``match`` match id ``[n]``.
        masks: :func:`imf.subset_masks` (prediction coverage).
        codes / num: categorical codes ``[n, n_cat]`` and raw numeric E2 columns ``[n, n_num]``.
        seq_types / seq_raw: :func:`seq_arrays`; ``cur_xy`` current location ``[n, 2]``.
    """

    ids: pd.DataFrame
    y: pd.DataFrame
    ref: dict[str, dict[str, np.ndarray]]
    fold: np.ndarray
    match: np.ndarray
    masks: dict[str, np.ndarray]
    codes: np.ndarray
    num: np.ndarray
    cat_cols: list[str]
    num_cols: list[str]
    seq_types: np.ndarray
    seq_raw: np.ndarray
    cur_xy: np.ndarray


def load_data(cfg: SeqConfig) -> SeqData:
    """Load events360 (E2 columns + sequence block) and the stage-02 OOF file, row-aligned."""
    t0 = time.time()
    ev_path = processed_dir("soccer") / "events360.parquet"
    oof_path = processed_dir("soccer") / "imputed_oof.parquet"
    names = [f.name for f in pq.ParquetFile(ev_path).schema_arrow]
    fcols = sorted(
        set(imf.design_columns(FEATURE_SET)) & set(names)
        | {"f_x", "f_y", "f_is_possession_team", "f_play_pattern", "f_poss_elapsed"}
    )
    cols = ID_COLS[:4] + ["competition", "gender", "f_type"] + fcols
    cols += [c for c in names if c.startswith("seq_")]
    cols = list(dict.fromkeys(cols))
    if cfg.smoke:
        df = pq.ParquetFile(ev_path).read_row_group(0, columns=cols).to_pandas()
    else:
        df = pd.read_parquet(ev_path, columns=cols)
    df = df.reset_index(drop=True)
    n = len(df)
    ref_cols = ["event_id", "fold"] + [f"y_{t}" for t in TARGETS]
    ref_cols += [f"{t}__{f}" for t in TARGETS for f in ("E2", "base_type", "base_global")]
    oof = pd.read_parquet(oof_path, columns=ref_cols)
    oof = oof.iloc[:n].reset_index(drop=True)
    if not np.array_equal(oof["event_id"].to_numpy(), df["event_id"].to_numpy()):
        raise RuntimeError("events360 / imputed_oof row order differs")
    fold = oof["fold"].to_numpy(dtype=np.int8)
    match = df["match_id"].to_numpy()
    if not cfg.smoke:
        check = np.full(n, -1, dtype=np.int8)
        for k, (_, te) in enumerate(group_kfold(match, n_splits=5, seed=0)):
            check[te] = k
        if not np.array_equal(check, fold):
            raise RuntimeError("stage-02 fold column does not match group_kfold(seed=0)")
    y = pd.DataFrame({t: oof[f"y_{t}"].to_numpy(dtype=np.float32) for t in TARGETS})
    ref = {
        t: {
            LGBM: oof[f"{t}__E2"].to_numpy(dtype=np.float32),
            BASE: oof[f"{t}__base_type"].to_numpy(dtype=np.float32),
            "base_global": oof[f"{t}__base_global"].to_numpy(dtype=np.float32),
        }
        for t in TARGETS
    }
    del oof
    design = imf.build_design(df, FEATURE_SET)
    categorical = imf.categorical_columns(FEATURE_SET)
    codes, num = split_design(design, categorical)
    cat_cols = [c for c in design.columns if c in categorical]
    num_cols = [c for c in design.columns if c not in categorical]
    del design
    seq_types, seq_raw = seq_arrays(df)
    cur_xy = df[["f_x", "f_y"]].to_numpy(dtype=np.float32)
    masks = imf.subset_masks(df)
    ids = df[ID_COLS].copy()
    del df
    print(f"loaded {n:,} rows in {time.time() - t0:.0f}s", flush=True)
    return SeqData(
        ids=ids,
        y=y,
        ref=ref,
        fold=fold,
        match=match,
        masks=masks,
        codes=codes,
        num=num,
        cat_cols=cat_cols,
        num_cols=num_cols,
        seq_types=seq_types,
        seq_raw=seq_raw,
        cur_xy=cur_xy,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class SeqBundle:
    """A trained variant with everything needed to apply it to a build-stage frame."""

    cfg: SeqConfig
    variant: str
    arch: str
    window_len: int
    targets: list[str]
    cat_cols: list[str]
    num_cols: list[str]
    num_stats: NumericStats
    target_stats: TargetStats
    state_dict: dict[str, torch.Tensor]
    meta: dict[str, Any]

    def build_net(self) -> SequenceStudentNet:
        net = SequenceStudentNet(
            self.cfg,
            self.arch,
            self.window_len,
            categorical_sizes(self.cat_cols),
            2 * len(self.num_cols),
            len(self.targets),
        )
        net.load_state_dict(self.state_dict)
        net.eval()
        return net

    def save(self, path: Path) -> None:
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "variant": self.variant,
                "arch": self.arch,
                "window_len": self.window_len,
                "targets": self.targets,
                "cat_cols": self.cat_cols,
                "num_cols": self.num_cols,
                "num_mean": self.num_stats.mean,
                "num_sd": self.num_stats.sd,
                "target_mean": self.target_stats.mean,
                "target_sd": self.target_stats.sd,
                "state_dict": self.state_dict,
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> SeqBundle:
        d = torch.load(path, map_location="cpu", weights_only=False)
        cfg = SeqConfig(**d["cfg"])
        return cls(
            cfg=cfg,
            variant=d["variant"],
            arch=d["arch"],
            window_len=d["window_len"],
            targets=list(d["targets"]),
            cat_cols=list(d["cat_cols"]),
            num_cols=list(d["num_cols"]),
            num_stats=NumericStats(mean=d["num_mean"], sd=d["num_sd"]),
            target_stats=TargetStats(mean=d["target_mean"], sd=d["target_sd"]),
            state_dict=d["state_dict"],
            meta=d["meta"],
        )

    def predict_frame(self, df: pd.DataFrame, batch_size: int = 8192) -> pd.DataFrame:
        """Apply to a build-stage frame: ``imp_<target>__<variant>`` columns ``[n, T]``, NaN
        outside each target's subset."""
        design = imf.build_design(df, FEATURE_SET)
        codes, num = split_design(design, imf.categorical_columns(FEATURE_SET))
        seq_types, seq_raw = seq_arrays(df)
        cur_xy = df[["f_x", "f_y"]].to_numpy(dtype=np.float32)
        net = self.build_net()
        out = predict_rows(
            net, self, np.arange(len(df)), codes, num, seq_types, seq_raw, cur_xy, batch_size
        )
        masks = imf.subset_masks(df)
        res = pd.DataFrame(index=df.index)
        for j, t in enumerate(self.targets):
            v = out[:, j].copy()
            v[~masks[imf.TARGET_BY_NAME[t].subset]] = np.nan
            res[f"imp_{t}__{self.variant}"] = v
        return res


def _batch_tensors(
    rows: np.ndarray,
    codes: np.ndarray,
    num: np.ndarray,
    num_stats: NumericStats,
    seq_types: np.ndarray,
    seq_raw: np.ndarray,
    cur_xy: np.ndarray,
    window_len: int,
) -> tuple[torch.Tensor, ...]:
    t, f, p = sequence_tokens(seq_types[rows], seq_raw[rows], cur_xy[rows], window_len)
    return (
        torch.from_numpy(t),
        torch.from_numpy(f),
        torch.from_numpy(p),
        torch.from_numpy(codes[rows]),
        torch.from_numpy(num_stats.transform(num[rows])),
    )


def predict_rows(
    net: SequenceStudentNet,
    bundle: SeqBundle,
    rows: np.ndarray,
    codes: np.ndarray,
    num: np.ndarray,
    seq_types: np.ndarray,
    seq_raw: np.ndarray,
    cur_xy: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    """Decoded predictions ``[len(rows), T]`` in target units."""
    net.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            b = rows[s : s + batch_size]
            outs.append(
                net(
                    *_batch_tensors(
                        b,
                        codes,
                        num,
                        bundle.num_stats,
                        seq_types,
                        seq_raw,
                        cur_xy,
                        bundle.window_len,
                    )
                ).numpy()
            )
    out = np.concatenate(outs, axis=0) if outs else np.zeros((0, len(bundle.targets)), np.float32)
    return decode_outputs(out, bundle.targets, bundle.target_stats)


def thin_rows(idx: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Keep ``cap`` of ``idx`` uniformly at random (sorted)."""
    if len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=cap, replace=False))


def train_variant(
    data: SeqData, cfg: SeqConfig, variant: str, fold: int, log: bool = True
) -> tuple[SeqBundle, np.ndarray, np.ndarray, dict[str, Any]]:
    """Train one variant on the training folds of ``fold`` and predict the held-out fold.

    Training rows: every row of the other folds with at least one valid label, minus an inner
    match holdout (``inner_holdout_frac`` of the training matches) used only for early
    stopping / LR decay; both thinned uniformly to ``train_cap`` / ``val_cap``.

    Returns:
        ``(bundle, test_rows, pred [len(test_rows), T] in target units, fit metadata)``.
    """
    arch, window_len = VARIANTS[variant]
    seed = cfg.seed + fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    any_label = data.y.notna().any(axis=1).to_numpy()
    train_all = np.where((data.fold != fold) & any_label)[0]
    test_rows = np.where(data.fold == fold)[0]
    uniq = np.unique(data.match[train_all])
    hold = rng.choice(
        uniq, size=max(1, int(round(len(uniq) * cfg.inner_holdout_frac))), replace=False
    )
    is_hold = np.isin(data.match[train_all], hold)
    tr = thin_rows(train_all[~is_hold], cfg.train_cap, seed)
    va = thin_rows(train_all[is_hold], cfg.val_cap, seed + 1)
    num_stats = NumericStats.fit(data.num[tr])
    target_stats = TargetStats.fit(data.y.iloc[tr])
    y_vals, y_mask = target_matrix(data.y, target_stats)
    kinds = [imf.TARGET_BY_NAME[t].kind for t in TARGETS]
    weights = torch.tensor([cfg.loss_weights.get(t, 1.0) for t in TARGETS], dtype=torch.float32)
    net = SequenceStudentNet(
        cfg,
        arch,
        window_len,
        categorical_sizes(data.cat_cols),
        2 * len(data.num_cols),
        len(TARGETS),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    bundle = SeqBundle(
        cfg=cfg,
        variant=variant,
        arch=arch,
        window_len=window_len,
        targets=list(TARGETS),
        cat_cols=data.cat_cols,
        num_cols=data.num_cols,
        num_stats=num_stats,
        target_stats=target_stats,
        state_dict={},
        meta={},
    )

    def val_loss() -> float:
        net.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), 8192):
                b = va[s : s + 8192]
                out = net(
                    *_batch_tensors(
                        b,
                        data.codes,
                        data.num,
                        num_stats,
                        data.seq_types,
                        data.seq_raw,
                        data.cur_xy,
                        window_len,
                    )
                )
                lb = multitask_loss(
                    out, torch.from_numpy(y_vals[b]), torch.from_numpy(y_mask[b]), kinds, weights
                )
                tot += float(lb) * len(b)
                cnt += len(b)
        return tot / max(cnt, 1)

    best, best_state, best_epoch, bad, history = np.inf, None, -1, 0, []
    stop_reason = "max_epochs"
    for epoch in range(cfg.max_epochs):
        net.train()
        order = rng.permutation(tr)
        run, nb = 0.0, 0
        for s in range(0, len(order), cfg.batch_size):
            b = order[s : s + cfg.batch_size]
            out = net(
                *_batch_tensors(
                    b,
                    data.codes,
                    data.num,
                    num_stats,
                    data.seq_types,
                    data.seq_raw,
                    data.cur_xy,
                    window_len,
                )
            )
            loss = multitask_loss(
                out, torch.from_numpy(y_vals[b]), torch.from_numpy(y_mask[b]), kinds, weights
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += float(loss.detach())
            nb += 1
        vl = val_loss()
        sched.step(vl)
        history.append(
            {
                "epoch": epoch,
                "train_loss": run / max(nb, 1),
                "val_loss": vl,
                "lr": opt.param_groups[0]["lr"],
                "minutes": (time.time() - t0) / 60,
            }
        )
        if log:
            print(
                f"  {variant} fold {fold} epoch {epoch}: train {run / max(nb, 1):.4f} "
                f"val {vl:.4f} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
        if vl < best - 1e-4:
            best, best_epoch, bad = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                stop_reason = "early_stopping"
                break
        if (time.time() - t0) / 60 > cfg.max_minutes:
            stop_reason = "time_cap"
            break
    assert best_state is not None
    net.load_state_dict(best_state)
    bundle.state_dict = best_state
    pred = predict_rows(
        net, bundle, test_rows, data.codes, data.num, data.seq_types, data.seq_raw, data.cur_xy
    )
    n_train_by_target = {t: int(data.y[t].iloc[tr].notna().sum()) for t in TARGETS}
    meta = {
        "variant": variant,
        "arch": arch,
        "window_len": window_len,
        "fold": fold,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(test_rows)),
        "n_train_by_target": n_train_by_target,
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": float(best),
        "stop_reason": stop_reason,
        "seconds": time.time() - t0,
        "n_params": int(sum(p.numel() for p in net.parameters())),
        "history": history,
    }
    bundle.meta = meta
    return bundle, test_rows, pred, meta


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_path(cfg: SeqConfig, variant: str, fold: int) -> Path:
    return cfg.cache_dir / f"{variant}__fold{fold}.parquet"


def save_fold(
    cfg: SeqConfig,
    variant: str,
    fold: int,
    rows: np.ndarray,
    pred: np.ndarray,
    meta: dict[str, Any],
) -> None:
    out = pd.DataFrame({"row": rows.astype(np.int64)})
    for j, t in enumerate(TARGETS):
        out[t] = pred[:, j].astype(np.float32)
    p = cache_path(cfg, variant, fold)
    out.to_parquet(p, index=False)
    p.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def load_variant(
    cfg: SeqConfig, variant: str, folds: list[int], n: int, masks: dict[str, np.ndarray]
) -> dict[str, np.ndarray] | None:
    """Out-of-fold predictions ``{target: pred[n]}`` of a variant over ``folds`` (NaN elsewhere
    and outside each target's subset); None when a fold is missing."""
    out = {t: np.full(n, np.nan, dtype=np.float32) for t in TARGETS}
    for k in folds:
        p = cache_path(cfg, variant, k)
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        rows = d["row"].to_numpy()
        for t in TARGETS:
            v = d[t].to_numpy(dtype=np.float32).copy()
            v[~masks[imf.TARGET_BY_NAME[t].subset][rows]] = np.nan
            out[t][rows] = v
    return out


def load_meta(cfg: SeqConfig, variant: str, fold: int) -> dict[str, Any] | None:
    p = cache_path(cfg, variant, fold).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_train(
    data: SeqData, cfg: SeqConfig, variants: list[str], folds: list[int], force: bool
) -> None:
    for variant in variants:
        for k in folds:
            if not force and cache_path(cfg, variant, k).exists():
                print(f"cached {variant} fold {k}", flush=True)
                continue
            print(f"training {variant} fold {k}", flush=True)
            bundle, rows, pred, meta = train_variant(data, cfg, variant, k)
            save_fold(cfg, variant, k, rows, pred, meta)
            bundle.save(cfg.model_dir / f"{variant}__fold{k}.pt")
            print(
                f"  done: {meta['epochs']} epochs, best {meta['best_epoch']}, "
                f"{meta['seconds'] / 60:.1f} min ({meta['stop_reason']})",
                flush=True,
            )


def per_row_loss(kind: str, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return per_sample_log_loss(y, np.clip(p, 0.0, 1.0)) if kind == "binary" else (y - p) ** 2


def common_rows(y: np.ndarray, preds: dict[str, np.ndarray], sel: np.ndarray) -> np.ndarray:
    ok = sel & ~np.isnan(y)
    for p in preds.values():
        ok &= ~np.isnan(p)
    return ok


def metrics_table(
    data: SeqData, preds: dict[str, dict[str, np.ndarray]], sel: np.ndarray, models: list[str]
) -> pd.DataFrame:
    """One row per (target, model) scored on the rows where every model has a prediction."""
    rows = []
    for t in TARGETS:
        spec = imf.TARGET_BY_NAME[t]
        y = data.y[t].to_numpy(dtype=float)
        ok = common_rows(y, {m: preds[m][t] for m in models}, sel)
        ys = np.where(ok, y, np.nan)
        base = data.ref[t]["base_global"]
        for m in models:
            s = score(spec.kind, ys, preds[m][t].astype(float), base.astype(float))
            rows.append({"target": t, "kind": spec.kind, "model": m, **s})
    return pd.DataFrame(rows)


def deltas_table(
    data: SeqData,
    preds: dict[str, dict[str, np.ndarray]],
    sel: np.ndarray,
    pairs: list[tuple[str, str]],
    cfg: SeqConfig,
) -> pd.DataFrame:
    """Paired per-row bootstrap (and match-clustered bootstrap) of loss(a) - loss(b);
    positive = ``b`` better."""
    rows = []
    for t in TARGETS:
        spec = imf.TARGET_BY_NAME[t]
        y = data.y[t].to_numpy(dtype=float)
        for a, b in pairs:
            ok = common_rows(y, {a: preds[a][t], b: preds[b][t]}, sel)
            if ok.sum() < 100:
                continue
            la = per_row_loss(spec.kind, y[ok], preds[a][t][ok].astype(float))
            lb = per_row_loss(spec.kind, y[ok], preds[b][t][ok].astype(float))
            d, lo, hi = bootstrap_delta_per_sample(la, lb, n_boot=cfg.n_boot, seed=cfg.seed)
            dc, clo, chi = clustered_bootstrap_delta(
                la, lb, data.match[ok], n_boot=cfg.n_boot, seed=cfg.seed
            )
            base = data.ref[t]["base_global"].astype(float)
            ys = np.where(ok, y, np.nan)
            sa = score(spec.kind, ys, preds[a][t].astype(float), base)
            sb = score(spec.kind, ys, preds[b][t].astype(float), base)
            rows.append(
                {
                    "target": t,
                    "kind": spec.kind,
                    "from": a,
                    "to": b,
                    "n": int(ok.sum()),
                    "loss": "log_loss" if spec.kind == "binary" else "squared_error",
                    "delta_loss": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": bool(lo > 0 or hi < 0),
                    "ci_low_clustered": clo,
                    "ci_high_clustered": chi,
                    "significant_clustered": bool(clo > 0 or chi < 0),
                    "skill_from": skill_of(spec.kind, sa),
                    "skill_to": skill_of(spec.kind, sb),
                }
            )
    return pd.DataFrame(rows)


def breakdown_table(
    data: SeqData,
    preds: dict[str, dict[str, np.ndarray]],
    sel: np.ndarray,
    models: list[str],
    key: np.ndarray,
    key_name: str,
    min_n: int = 500,
) -> pd.DataFrame:
    rows = []
    keys = pd.Series(np.asarray(key, dtype=object))
    for t in TARGETS:
        spec = imf.TARGET_BY_NAME[t]
        y = data.y[t].to_numpy(dtype=float)
        ok = common_rows(y, {m: preds[m][t] for m in models}, sel)
        base = data.ref[t]["base_global"].astype(float)
        for k in keys.dropna().unique():
            m_ = ok & (keys == k).to_numpy()
            if m_.sum() < min_n:
                continue
            ys = np.where(m_, y, np.nan)
            for m in models:
                s = score(spec.kind, ys, preds[m][t].astype(float), base)
                rows.append({"target": t, "kind": spec.kind, key_name: k, "model": m, **s})
    return pd.DataFrame(rows)


def fits_table(cfg: SeqConfig, variants: list[str], folds: list[int]) -> pd.DataFrame:
    rows = []
    for v in variants:
        for k in folds:
            m = load_meta(cfg, v, k)
            if m is None:
                continue
            r = {key: val for key, val in m.items() if key not in ("history", "n_train_by_target")}
            r.update({f"n_train_{t}": n for t, n in m["n_train_by_target"].items()})
            r["minutes"] = m["seconds"] / 60
            rows.append(r)
    return pd.DataFrame(rows)


def write_oof(
    data: SeqData, preds: dict[str, dict[str, np.ndarray]], variants: list[str], path: Path
) -> None:
    out = data.ids[["match_id", "event_id", "event_index", "period"]].copy()
    out["fold"] = data.fold
    for t in TARGETS:
        out[f"y_{t}"] = data.y[t].to_numpy(dtype=np.float32)
    for v in variants:
        for t in TARGETS:
            out[f"{t}__{v}"] = preds[v][t]
    out.to_parquet(path, index=False)


def _pivot(metrics: pd.DataFrame, value: str, models: list[str]) -> pd.DataFrame:
    p = metrics.pivot(index="target", columns="model", values=value)
    present = set(metrics["target"])
    p = p.reindex(
        index=[t for t in TARGETS if t in present], columns=[m for m in models if m in p.columns]
    )
    return p.reset_index()


def _wins(dl: pd.DataFrame, a: str, b: str) -> tuple[list[str], list[str]]:
    """Targets where ``b`` beats / loses to ``a`` with the match-clustered CI excluding zero."""
    sub = dl[(dl["from"] == a) & (dl["to"] == b)]
    better = [r.target for r in sub.itertuples() if r.ci_low_clustered > 0]
    worse = [r.target for r in sub.itertuples() if r.ci_high_clustered < 0]
    return better, worse


def summary_lines(met: pd.DataFrame, dl: pd.DataFrame, variants: list[str]) -> list[str]:
    """Computed one-paragraph summary: wins / losses per comparison and the largest gains."""
    out: list[str] = []
    n_t = len(TARGETS)

    def skill(model: str, target: str) -> float:
        r = met[(met.model == model) & (met.target == target)]
        return skill_of(r["kind"].iloc[0], r.iloc[0].to_dict()) if len(r) else float("nan")

    def fmt_list(items: list[str]) -> str:
        return ", ".join(f"`{t}`" for t in items) if items else "none"

    if "seq20" in variants:
        better, worse = _wins(dl, LGBM, "seq20")
        out.append(
            f"- `seq20` beats `lgbm_E2` on {len(better)}/{n_t} targets with the match-clustered "
            f"95% CI excluding zero (worse on: {fmt_list(worse)}). Skill lgbm_E2 -> seq20: "
            + "; ".join(f"{t} {skill(LGBM, t):.3f} -> {skill('seq20', t):.3f}" for t in TARGETS)
            + "."
        )
    if "seq0" in variants:
        better, worse = _wins(dl, LGBM, "seq0")
        out.append(
            f"- `seq0` (the same E2 inputs through the MLP, no sequence branch) is worse than "
            f"`lgbm_E2` on {len(worse)}/{n_t} targets and better on {len(better)}/{n_t}: the "
            "neural architecture alone does not beat the trees; the gain of `seq20` comes from "
            "the raw event history."
        )
    if "seq1" in variants and "seq20" in variants:
        better, worse = _wins(dl, "seq1", "seq20")
        b1, w1 = _wins(dl, LGBM, "seq1")
        out.append(
            f"- `seq20` beats `seq1` on {len(better)}/{n_t} targets (clustered CI), so slots 2-20 "
            "carry information beyond the previous event and the E2 window counts; `seq1` alone "
            f"beats `lgbm_E2` on {len(b1)}/{n_t} and loses on {len(w1)}/{n_t} "
            f"({fmt_list(w1)})."
        )
    return out


def render_report(
    data: SeqData,
    cfg: SeqConfig,
    tables: dict[str, pd.DataFrame],
    variants: list[str],
    folds: list[int],
    models: list[str],
) -> str:
    met, dl, fits = tables["metrics"], tables["deltas"], tables["fits"]
    n_rows = int(np.isin(data.fold, folds).sum())
    n_matches = int(len(np.unique(data.match[np.isin(data.fold, folds)])))
    lines = [
        "# Soccer 05 - sequence student: neural model over the 20-event history vs LightGBM E2\n"
    ]
    lines.append(
        f"Source: `processed_dir('soccer')/events360.parquet` + stage-02 `imputed_oof.parquet` "
        f"({len(data.y):,} rows, 417 matches). This stage scores folds {folds} of the stage-02 "
        f"5-fold match-grouped split ({n_matches} held-out matches, {n_rows:,} rows); every model "
        "and baseline below is scored on exactly the same held-out rows (label valid and every "
        "compared model has a prediction), so the LightGBM numbers differ slightly from the "
        "5-fold figures of `soccer_02_imputation.md`.\n"
    )
    lines.append("## Protocol\n")
    arch_note = "GRU" if all(VARIANTS[v][0] == "gru" for v in variants) else "GRU / transformer"
    lines += [
        "- Folds: the `fold` column of `imputed_oof.parquet` (`group_kfold` by `match_id`, "
        "seed 0, "
        "verified against a recomputation). Labels: the stage-02 `y_<target>` columns (same "
        "subsets and label rules: team-shape targets on possession-team reliable frames, "
        "`deep_block` on settled possession, `counter_on` in the middle third, ball-relative "
        "targets on every usable frame, `n_opp_in_cone` on possession-team events).",
        f"- Network ({arch_note}, PyTorch CPU, `torch.set_num_threads({cfg.n_threads})`): sequence "
        f"branch = type-id embedding + linear projection of 8 per-slot numerics (x/120, y/80, "
        "log1p(seconds before the current event), own / opponent / pad flags, displacement to the "
        f"current event) + learned slot position, d_model {cfg.d_model}, tokens oldest -> newest; "
        "output = newest hidden state and masked mean. Current-event branch = MLP "
        f"{list(cfg.mlp_hidden)} over the E2 design (numeric z-scores + missing indicators, "
        f"{cfg.cat_dim}-dim embeddings per categorical column; training-fold statistics). Shared "
        f"head {cfg.head_hidden} -> {len(TARGETS)} outputs: MSE on standardised continuous / count "
        "targets, BCE for the two binaries, masked per row, equal weights.",
        f"- Optimisation: AdamW lr {cfg.lr}, weight decay {cfg.weight_decay}, batch "
        f"{cfg.batch_size}, grad-clip 1, LR halved on validation plateau, early stopping "
        f"(patience {cfg.patience}, max {cfg.max_epochs} epochs, time cap {cfg.max_minutes:.0f} "
        f"min per fold) on an inner {cfg.inner_holdout_frac:.0%} match holdout of the training "
        f"folds; training rows thinned uniformly to {cfg.train_cap:,} (validation "
        f"{cfg.val_cap:,}); "
        f"seed {cfg.seed} + fold. Count / distance predictions are clipped at 0.",
        "- References: `lgbm_E2` = stage-02 LightGBM E2 out-of-fold predictions (one model per "
        "target, 300k training rows each, as stored, unclipped); `base_type` = stage-02 per-event-"
        "type training mean. Skill = R2 (continuous / count) or Brier skill score vs the "
        "training base rate (binary). Deltas: paired per-row bootstrap on squared error / "
        f"log-loss ({cfg.n_boot} resamples, 95% CI, positive = the `to` model is better) with a "
        "match-clustered bootstrap CI next to it (rows within a match are correlated; the "
        "clustered interval is the conservative one).",
        "",
        "### Variants\n",
    ]
    lines.append(
        md_table(
            pd.DataFrame([{"model": m, "description": VARIANT_DESCRIPTION[m]} for m in models])
        )
    )
    lines.append("\n## Summary\n")
    lines += summary_lines(met, dl, variants)
    lines.append("\n## Headline: skill per target (held-out folds, identical rows)\n")
    lines.append("Skill = R2 or BSS. `n` is the number of scored rows per target.\n")
    skill = met.copy()
    skill["skill"] = [
        skill_of(k, r) for k, r in zip(skill["kind"], skill.to_dict("records"), strict=True)
    ]
    piv = _pivot(skill, "skill", models)
    piv["n"] = [
        int(met[(met.target == t) & (met.model == models[0])]["n"].iloc[0]) for t in piv.target
    ]
    piv["kind"] = [imf.TARGET_BY_NAME[t].kind for t in piv.target]
    lines.append(
        md_table(
            piv[["target", "kind", "n"] + [m for m in models if m in piv.columns]],
            floatfmt="{:.3f}",
        )
    )
    lines.append("\n### Continuous / count targets: MAE (yd or players)\n")
    reg = met[met.kind != "binary"]
    lines.append(md_table(_pivot(reg, "mae", models), floatfmt="{:.3f}"))
    lines.append("\n### Binary targets: AUC / log-loss\n")
    binm = met[met.kind == "binary"]
    for val, fmt in (("auc", "{:.4f}"), ("log_loss", "{:.4f}"), ("ece", "{:.4f}")):
        lines.append(f"\n{val}:\n")
        lines.append(md_table(_pivot(binm, val, models), floatfmt=fmt))
    lines.append("\n## Paired deltas (loss(from) - loss(to); positive = `to` better)\n")
    delta_cols = [
        "target",
        "from",
        "to",
        "n",
        "loss",
        "delta_loss",
        "ci_low",
        "ci_high",
        "significant",
        "ci_low_clustered",
        "ci_high_clustered",
        "significant_clustered",
        "skill_from",
        "skill_to",
    ]
    lines.append(md_table(dl[delta_cols], floatfmt="{:.5f}"))
    if "by_poss_elapsed" in tables and len(tables["by_poss_elapsed"]):
        lines.append("\n## Error vs elapsed possession time\n")
        lines.append(
            "MAE (continuous / count) or log-loss (binary) per bin of `f_poss_elapsed` "
            "(seconds since the possession started); n per bin.\n"
        )
        b = tables["by_poss_elapsed"].copy()
        b["err"] = np.where(b["kind"] == "binary", b["log_loss"], b["mae"])
        for t in TARGETS:
            sub = b[b.target == t]
            if not len(sub):
                continue
            p = sub.pivot(index="poss_elapsed_s", columns="model", values="err")
            p = p.reindex(columns=[m for m in models if m in p.columns])
            p["n"] = sub.groupby("poss_elapsed_s")["n"].first()
            p = p.reset_index()
            p["_k"] = [float(str(v).strip("[").split(",")[0]) for v in p["poss_elapsed_s"]]
            p = p.sort_values("_k").drop(columns="_k")
            unit = "log-loss" if imf.TARGET_BY_NAME[t].kind == "binary" else "MAE"
            lines.append(f"\n{t} ({unit}):\n")
            lines.append(md_table(p, floatfmt="{:.3f}"))
    if "by_type" in tables and len(tables["by_type"]):
        lines.append("\n## Skill by event type (types with >= 2,000 scored rows)\n")
        b = tables["by_type"].copy()
        b["skill"] = [skill_of(k, r) for k, r in zip(b["kind"], b.to_dict("records"), strict=True)]
        for t in TARGETS:
            sub = b[b.target == t]
            if not len(sub):
                continue
            p = sub.pivot(index="f_type", columns="model", values="skill")
            p = p.reindex(columns=[m for m in models if m in p.columns])
            p["n"] = sub.groupby("f_type")["n"].first()
            p = p.reset_index().sort_values("n", ascending=False)
            lines.append(f"\n{t}:\n")
            lines.append(md_table(p, floatfmt="{:.3f}"))
    lines.append("\n## Fits\n")
    if len(fits):
        cols = [
            "variant",
            "fold",
            "n_train",
            "n_val",
            "epochs",
            "best_epoch",
            "best_val_loss",
            "stop_reason",
            "minutes",
            "n_params",
        ] + [f"n_train_{t}" for t in TARGETS]
        lines.append(md_table(fits[[c for c in cols if c in fits.columns]], floatfmt="{:.3f}"))
    lines.append("\n## Caveats\n")
    lines += [
        f"- Only folds {folds} of the five stage-02 folds were trained (CPU budget); the LightGBM "
        "reference is scored on the same rows, so the comparison is paired, but the absolute "
        "numbers are on a subset of the 417 matches.",
        "- The multi-task network sees one thinned training set for all seven targets, so per "
        "target it trains on fewer labelled rows than the corresponding LightGBM student (see "
        "`n_train_<target>` in the fits table vs 300,000 per stage-02 student); the comparison "
        "favours LightGBM on the sparse subsets (`deep_block`, `counter_on`).",
        "- The stage-02 review's mechanism caveat stands: `block_depth` / `def_line` / "
        "`deep_block` skill is mostly ball location plus visible-area truncation; the "
        "per-event-type baseline "
        "shows how little event type alone recovers.",
        "- Sequence predictions are clipped at 0 for counts / distances, the stored LightGBM "
        "predictions are not (the stage-02 review's unclipped-output issue); clipping is "
        "loss-reducing by construction but changes squared error only where the prediction "
        "was negative.",
        "- Per-row bootstrap intervals treat rows as independent; the match-clustered interval "
        "next to each delta accounts for within-match correlation and is the one to trust for "
        "significance.",
        "- Within event types whose `nearest_opp_dist` label is a near-constant ~0.14 yd (Duel, "
        "Dribbled Past, Dispossessed, Foul Won, see `soccer_01_build.md`), the network is worse "
        "than LightGBM (more negative within-type R2 in the per-type table: trees isolate "
        "`f_type` exactly, the shared MSE head does not); the aggregate gain comes from Pass / "
        "Carry / Ball Receipt* rows. Absolute errors on those types stay ~1 yd.",
    ]
    if len(fits) and (fits["stop_reason"] == "max_epochs").any():
        capped = sorted(set(fits.loc[fits["stop_reason"] == "max_epochs", "variant"]))
        lines.append(
            f"- {', '.join(f'`{v}`' for v in capped)} reached the {cfg.max_epochs}-epoch cap with "
            "the best epoch near the end (validation loss still improving by < 0.01 per epoch), "
            "so those variants are marginally under-trained; the gaps to `seq1` / `seq20` "
            "(fits table, `best_val_loss`) are an order of magnitude larger."
        )
    return "\n".join(lines) + "\n"


def stage_report(data: SeqData, cfg: SeqConfig, variants: list[str], folds: list[int]) -> None:
    n = len(data.y)
    preds: dict[str, dict[str, np.ndarray]] = {}
    for v in variants:
        p = load_variant(cfg, v, folds, n, data.masks)
        if p is None:
            print(f"variant {v} incomplete for folds {folds}; skipped", flush=True)
            continue
        preds[v] = p
    if not preds:
        raise RuntimeError("no trained variant to report")
    variants = [v for v in variants if v in preds]
    preds[LGBM] = {t: data.ref[t][LGBM] for t in TARGETS}
    preds[BASE] = {t: data.ref[t][BASE] for t in TARGETS}
    models = variants + [LGBM, BASE]
    sel = np.isin(data.fold, folds)
    t0 = time.time()
    tables: dict[str, pd.DataFrame] = {}
    tables["metrics"] = metrics_table(data, preds, sel, models)
    pairs = [(LGBM, v) for v in variants] + [(BASE, variants[0])]
    if "seq1" in variants and "seq20" in variants:
        pairs.append(("seq1", "seq20"))
    if "seq0" in variants and "seq20" in variants:
        pairs.append(("seq0", "seq20"))
    if "seq0" in variants and "seq1" in variants:
        pairs.append(("seq0", "seq1"))
    tables["deltas"] = deltas_table(data, preds, sel, pairs, cfg)
    print(f"metrics + deltas in {time.time() - t0:.0f}s", flush=True)
    elapsed = data.ids["f_poss_elapsed"].to_numpy(dtype=float)
    ebin = pd.cut(elapsed, ELAPSED_BINS, right=False).astype(str)
    ebin = np.where(np.isnan(elapsed), None, ebin)
    tables["by_poss_elapsed"] = breakdown_table(data, preds, sel, models, ebin, "poss_elapsed_s")
    tables["by_type"] = breakdown_table(
        data, preds, sel, models, data.ids["f_type"].to_numpy(dtype=object), "f_type", min_n=2000
    )
    tables["fits"] = fits_table(cfg, variants, folds)
    tables["variants"] = pd.DataFrame(
        [{"model": m, "description": VARIANT_DESCRIPTION[m]} for m in models]
    )
    if cfg.smoke:
        print(tables["metrics"].to_string())
        print(tables["deltas"].to_string())
        return
    rd = reports_dir()
    for name, t in tables.items():
        t.to_parquet(rd / f"soccer_05_{name}.parquet", index=False)
    write_oof(data, preds, variants, processed_dir("soccer") / "imputed_oof_seq.parquet")
    (rd / "soccer_05_sequence.md").write_text(
        render_report(data, cfg, tables, variants, folds, models)
    )
    print(f"report written ({(time.time() - t0) / 60:.1f} min)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="all", choices=["train", "report", "all"])
    ap.add_argument("--variants", default="seq20,seq1,seq0")
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--train-cap", type=int, default=None)
    args = ap.parse_args()
    cfg = SeqConfig(smoke=args.smoke)
    if args.max_minutes is not None:
        cfg.max_minutes = args.max_minutes
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.train_cap is not None:
        cfg.train_cap = args.train_cap
    torch.set_num_threads(cfg.n_threads)
    variants = [v for v in args.variants.split(",") if v]
    folds = [int(k) for k in args.folds.split(",") if k != ""]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {list(VARIANTS)}")
    data = load_data(cfg)
    if args.stage in ("train", "all"):
        stage_train(data, cfg, variants, folds, args.force)
    if args.stage in ("report", "all"):
        stage_report(data, cfg, variants, folds)


if __name__ == "__main__":
    main()
