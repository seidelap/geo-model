"""Tests for the soccer 05 sequence-student helpers on synthetic input (no data access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from research.privileged_tracking.soccer import imputation_features as imf
from research.privileged_tracking.soccer import sequence_student as ss


def _seq_frame() -> pd.DataFrame:
    """Three rows: full history, two real slots then pads, all pads."""
    seq_len = imf.SEQ_LEN
    d: dict[str, object] = {"f_x": [60.0, 30.0, 90.0], "f_y": [40.0, 10.0, 70.0]}
    for k in range(1, seq_len + 1):
        d[f"seq_type_{k:02d}"] = [k % 5 + 1, (1 if k <= 2 else 0), 0]
        d[f"seq_x_{k:02d}"] = [float(k), (12.0 if k <= 2 else np.nan), np.nan]
        d[f"seq_y_{k:02d}"] = [float(2 * k), (8.0 if k <= 2 else np.nan), np.nan]
        d[f"seq_dt_{k:02d}"] = [float(k) - 1.0, (0.5 * k if k <= 2 else np.nan), np.nan]
        d[f"seq_same_{k:02d}"] = [1 if k % 2 else 0, (1 if k <= 2 else -1), -1]
    return pd.DataFrame(d)


def test_seq_arrays_and_tokens_shapes_and_order() -> None:
    df = _seq_frame()
    types, raw = ss.seq_arrays(df)
    assert types.shape == (3, imf.SEQ_LEN) and raw.shape == (3, imf.SEQ_LEN, 4)
    assert types.dtype == np.int16 and raw.dtype == np.float32
    cur = df[["f_x", "f_y"]].to_numpy(dtype=np.float32)
    t, f, p = ss.sequence_tokens(types, raw, cur, 20)
    assert t.shape == (3, 20) and f.shape == (3, 20, ss.SEQ_N_FEATS) and p.shape == (3, 20)
    assert t.dtype == np.int64 and f.dtype == np.float32 and p.dtype == bool
    # oldest first: the last token is slot 1
    assert t[0, -1] == 1 % 5 + 1 and t[0, 0] == 20 % 5 + 1
    assert f[0, -1, 0] == pytest.approx(1.0 / 120.0)  # slot 1 x = 1
    assert f[0, 0, 0] == pytest.approx(20.0 / 120.0)  # slot 20 x = 20
    assert f[0, -1, 2] == pytest.approx(np.log1p(0.0))
    assert f[0, -2, 2] == pytest.approx(np.log1p(1.0))
    # own / opponent flags and displacement to the current event
    assert f[0, -1, 3] == 1.0 and f[0, -1, 4] == 0.0  # slot 1 same-team
    assert f[0, -2, 3] == 0.0 and f[0, -2, 4] == 1.0  # slot 2 opponent
    assert f[0, -1, 6] == pytest.approx((1.0 - 60.0) / 120.0)
    assert f[0, -1, 7] == pytest.approx((2.0 - 40.0) / 80.0)
    assert not p[0].any()
    # row 1: two real slots (the newest two tokens), the rest pads
    assert p[1, -2:].tolist() == [False, False] and p[1, :-2].all()
    assert np.all(f[1, :-2, :5] == 0.0) and np.all(f[1, :-2, 5] == 1.0)
    assert np.all(f[1, :-2, 6:] == 0.0)
    # row 2: all pads, nothing NaN
    assert p[2].all() and not np.isnan(f).any()


def test_sequence_tokens_window_one_and_zero() -> None:
    df = _seq_frame()
    types, raw = ss.seq_arrays(df)
    cur = df[["f_x", "f_y"]].to_numpy(dtype=np.float32)
    t1, f1, p1 = ss.sequence_tokens(types, raw, cur, 1)
    assert t1.shape == (3, 1) and f1.shape == (3, 1, ss.SEQ_N_FEATS)
    assert t1[:, 0].tolist() == types[:, 0].tolist()
    t20, f20, _ = ss.sequence_tokens(types, raw, cur, 20)
    np.testing.assert_allclose(f1[:, 0], f20[:, -1])
    t0, f0, p0 = ss.sequence_tokens(types, raw, cur, 0)
    assert t0.shape == (3, 0) and f0.shape == (3, 0, ss.SEQ_N_FEATS) and p0.shape == (3, 0)


def test_split_design_codes_and_numeric() -> None:
    design = pd.DataFrame(
        {
            "f_type_id": [1.0, np.nan, 36.0],
            "f_x": [10.0, np.nan, 30.0],
            "f_play_pattern": [0.0, 9.0, np.nan],
        }
    )
    codes, num = ss.split_design(design, ["f_type_id", "f_play_pattern"])
    assert codes.shape == (3, 2) and num.shape == (3, 1)
    assert codes[:, 0].tolist() == [2, 0, 37]  # code + 1, NaN -> 0
    assert codes[:, 1].tolist() == [1, 10, 0]
    assert np.isnan(num[1, 0]) and num[2, 0] == 30.0
    sizes = ss.categorical_sizes(["f_type_id", "f_play_pattern"])
    assert sizes == [imf.N_TYPE_IDS + 2, len(imf.VOCABS["f_play_pattern"]) + 2]
    assert codes.max(axis=0).tolist() < sizes


def test_numeric_stats_standardises_and_flags_missing() -> None:
    x = np.array([[1.0, 5.0], [3.0, np.nan], [5.0, 5.0]], dtype=np.float32)
    st = ss.NumericStats.fit(x)
    np.testing.assert_allclose(st.mean, [3.0, 5.0])
    assert st.sd[1] == 1.0  # constant column
    z = st.transform(x)
    assert z.shape == (3, 4)
    np.testing.assert_allclose(z[:, 0], (x[:, 0] - 3.0) / st.sd[0])
    assert z[1, 1] == 0.0 and z[1, 3] == 1.0 and z[0, 3] == 0.0


def test_target_matrix_and_decode_roundtrip() -> None:
    y = pd.DataFrame(
        {
            "block_depth": [40.0, 60.0, np.nan],
            "deep_block": [1.0, np.nan, 0.0],
            "nearest_opp_dist": [2.0, 4.0, 6.0],
        }
    )
    st = ss.TargetStats.fit(y)
    assert st.mean["deep_block"] == 0.0 and st.sd["deep_block"] == 1.0
    vals, mask = ss.target_matrix(y, st)
    assert mask.tolist() == [[True, True, True], [True, False, True], [False, True, True]]
    assert vals[2, 0] == 0.0 and vals[1, 1] == 0.0
    assert vals[0, 0] == pytest.approx((40.0 - 50.0) / st.sd["block_depth"])
    out = vals.copy()
    out[:, 1] = [10.0, 0.0, -10.0]  # logits
    out[2, 2] = -100.0  # far below zero once de-standardised
    pred = ss.decode_outputs(out, list(y.columns), st)
    np.testing.assert_allclose(pred[:2, 0], [40.0, 60.0], atol=1e-4)
    assert pred[0, 1] > 0.99 and pred[1, 1] == pytest.approx(0.5) and pred[2, 1] < 0.01
    assert pred[2, 2] == 0.0  # distance clipped at 0


def test_bootstrap_delta_per_sample_sign_and_coverage() -> None:
    rng = np.random.default_rng(0)
    la = rng.normal(1.0, 0.1, 5000)
    lb = la - 0.2 + rng.normal(0, 0.01, 5000)
    d, lo, hi = ss.bootstrap_delta_per_sample(la, lb, n_boot=200, seed=1, chunk=7)
    assert lo <= d <= hi and lo > 0
    assert d == pytest.approx(np.mean(la - lb))
    d2, lo2, hi2 = ss.bootstrap_delta_per_sample(la, la, n_boot=50, seed=1)
    assert d2 == 0.0 and lo2 == 0.0 and hi2 == 0.0


def _tiny_net(window_len: int, arch: str = "gru") -> ss.SequenceStudentNet:
    cfg = ss.SeqConfig(
        d_model=16, cat_dim=4, mlp_hidden=(16,), head_hidden=8, n_layers=1, n_heads=2
    )
    return ss.SequenceStudentNet(cfg, arch, window_len, [5, 7], 6, len(ss.TARGETS))


@pytest.mark.parametrize(
    "window_len,arch", [(20, "gru"), (1, "gru"), (0, "gru"), (20, "transformer")]
)
def test_net_forward_shapes(window_len: int, arch: str) -> None:
    torch.manual_seed(0)
    net = _tiny_net(window_len, arch)
    b = 4
    seq_types = torch.randint(0, imf.N_TYPE_IDS, (b, window_len))
    seq_feats = torch.randn(b, window_len, ss.SEQ_N_FEATS)
    seq_pad = torch.zeros(b, window_len, dtype=torch.bool)
    if window_len:
        seq_pad[0, 0] = True
    codes = torch.stack([torch.randint(0, 5, (b,)), torch.randint(0, 7, (b,))], dim=1)
    num = torch.randn(b, 6)
    out = net(seq_types, seq_feats, seq_pad, codes, num)
    assert out.shape == (b, len(ss.TARGETS))
    assert torch.isfinite(out).all()


def test_multitask_loss_masks_and_weights() -> None:
    out = torch.zeros(3, 2)
    y = torch.tensor([[1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
    mask = torch.tensor([[True, True], [True, False], [False, True]])
    w = torch.tensor([1.0, 2.0])
    loss = ss.multitask_loss(out, y, mask, ["reg", "binary"], w)
    mse = (1.0 + 4.0) / 2
    bce = float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.zeros(2), torch.tensor([1.0, 1.0])
        )
    )
    assert float(loss) == pytest.approx(mse + 2 * bce)
    none = ss.multitask_loss(out, y, torch.zeros(3, 2, dtype=torch.bool), ["reg", "binary"], w)
    assert float(none) == 0.0


def test_thin_rows_is_seeded_and_sorted() -> None:
    idx = np.arange(100)
    a = ss.thin_rows(idx, 10, 3)
    b = ss.thin_rows(idx, 10, 3)
    assert a.tolist() == b.tolist() and len(a) == 10 and np.all(np.diff(a) > 0)
    assert ss.thin_rows(idx, 200, 0) is idx
