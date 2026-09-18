"""Smoke tests for the soccer 03 payoff driver's fitting helpers on synthetic data (no data
access, tiny LightGBM models)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.privileged_tracking.soccer import payoff as po
from research.privileged_tracking.soccer import payoff_features as pf


def _synthetic(
    n: int = 1200, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Design with one strong numeric feature, one categorical and a state column that carries the
    rest of the signal; labels Bernoulli(sigmoid(...))."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    cat = rng.integers(0, 3, n).astype(np.float32)
    state = rng.normal(0, 1, n)
    logit = 1.2 * x1 + 0.8 * (cat == 2) + 1.5 * state - 1.0
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
    x = pd.DataFrame(
        {"f_x": x1.astype(np.float32), "f_play_pattern": cat, "s_state": state.astype(np.float32)}
    )
    match = rng.integers(0, 30, n)
    fold = match % 5
    return x, y, match, fold


FAST = pf.GbmParams(
    learning_rate=0.1, num_leaves=7, min_data_in_leaf=10, max_rounds=200, early_stopping=20
)


def test_fit_gbm_refit_and_rounds():
    x, y, match, _ = _synthetic()
    booster, rounds = po.fit_gbm(
        FAST, x, y, match, seed=0, categorical=["f_play_pattern"], n_threads=1
    )
    assert 1 <= rounds <= FAST.max_rounds
    p = booster.predict(x, num_iteration=rounds)
    assert p.shape == (len(x),) and (p >= 0).all() and (p <= 1).all()
    assert pf.binary_metrics(y, p)["auc"] > 0.8
    b2, r2_ = po.fit_gbm(
        FAST, x, y, match, seed=0, categorical=["f_play_pattern"], n_threads=1, refit=False
    )
    assert r2_ == rounds  # same seed -> same early-stopped round count


def test_cv_predict_is_out_of_fold_and_state_helps():
    x, y, match, fold = _synthetic()
    oof_full, fits, imp = po.cv_predict(
        x, y, match, fold, FAST, (0,), ["f_play_pattern"], n_threads=1
    )
    oof_event, _, _ = po.cv_predict(
        x[["f_x", "f_play_pattern"]], y, match, fold, FAST, (0,), ["f_play_pattern"], n_threads=1
    )
    assert not np.isnan(oof_full).any() and len(fits) == 5
    assert set(imp["feature"]) == set(x.columns) and imp["gain_share"].sum() == pytest.approx(
        1.0, abs=1e-6
    )
    m_full = pf.binary_metrics(y, oof_full)
    m_event = pf.binary_metrics(y, oof_event)
    assert m_full["log_loss"] < m_event["log_loss"]
    d = pf.paired_delta(y, oof_event, oof_full, match, n_boot=200)
    assert d["delta_log_loss"] > 0
    # two seeds average: still a probability
    oof2, fits2, _ = po.cv_predict(x, y, match, fold, FAST, (0, 1), ["f_play_pattern"], n_threads=1)
    assert len(fits2) == 10 and (oof2 >= 0).all() and (oof2 <= 1).all()
    # a training mask restricts the training rows but every row still gets a prediction
    oof3, fits3, _ = po.cv_predict(
        x,
        y,
        match,
        fold,
        FAST,
        (0,),
        ["f_play_pattern"],
        n_threads=1,
        train_mask=np.arange(len(y)) % 2 == 0,
    )
    assert not np.isnan(oof3).any() and all(f["n_train"] < 0.6 * len(y) for f in fits3)


def test_base_rate_oof_uses_training_folds_only():
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0], dtype=float)
    fold = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    b = po.base_rate_oof(y, fold)
    assert b[0] == pytest.approx(1 / 6) and b[2] == pytest.approx(3 / 6)


def test_distill_oof_nested_student_tracks_teacher():
    x, y, match, fold = _synthetic(n=1500)
    x_teacher = x
    x_student = x[["f_x", "f_play_pattern"]]
    student, seen = po.distill_oof(
        x_teacher, x_student, y, match, fold, FAST, (0,), alpha=1.0, n_threads=1
    )
    assert not np.isnan(student).any() and (student >= 0).all() and (student <= 1).all()
    assert (seen > 0).all() and (seen < 1).all()
    direct, _, _ = po.cv_predict(
        x_student, y, match, fold, FAST, (0,), ["f_play_pattern"], n_threads=1
    )
    # both are event-only; the distilled student should be a sensible probability model
    assert pf.binary_metrics(y, student)["auc"] > 0.65
    assert abs(student.mean() - y.mean()) < 0.1
    blend, _ = po.distill_oof(
        x_teacher, x_student, y, match, fold, FAST, (0,), alpha=0.5, n_threads=1
    )
    assert not np.isnan(blend).any()
    del direct


def test_xg_designs_variants_add_only_state_blocks():
    from research.privileged_tracking.tests.test_soccer_imputation_features import _frame

    df = pd.concat([_frame(seed=s) for s in range(4)], ignore_index=True)
    df = df[df["f_type"] == "Shot"].reset_index(drop=True)
    df = pd.concat([df] * 3, ignore_index=True)
    df["f_shot_body_part"] = (["Right Foot", "Left Foot", "Head"] * 10)[: len(df)]
    df["f_after_duration"] = 1.0
    df["key_pass_id"] = None
    for t in pf.SHOT_STATE:
        for f in ("E0", "E2"):
            df[f"imp_{t}__{f}"] = np.linspace(-0.5, 3.0, len(df))
        df[f"y_{t}"] = np.linspace(0, 3, len(df))
        df[pf.SFF_MAP[t]] = np.linspace(1, 4, len(df))
    for t in pf.ASSIST_STATE:
        for f in ("E0", "E2a"):
            df[f"a_imp_{t}__{f}"] = np.nan
        df[f"a_y_{t}"] = np.nan
    for c in pf.SFF_FULL:
        if c not in df.columns:
            df[c] = np.linspace(0, 1, len(df))
    df["a_has_assist"] = 0.0
    df["a_dt"] = np.nan
    designs = po.xg_designs(df)
    assert set(designs) == {v.name for v in pf.XG_VARIANTS if v.base != "none"}
    ev = set(designs["EVENT"].columns)
    assert all(c.startswith(("f_", "a_")) for c in ev)
    for name in ("EVENT+IMP", "EVENT+IMP(E0)", "EVENT+ORACLE360", "EVENT+ORACLESHOT"):
        extra = set(designs[name].columns) - ev
        assert extra == {f"s_{t}" for t in pf.SHOT_STATE} | {f"as_{t}" for t in pf.ASSIST_STATE}
    assert designs["EVENT+IMP"]["s_n_opp_in_cone"].min() >= 0  # clipped
    assert set(designs["EVENT+ORACLESHOT_full"].columns) - ev >= {f"s_{c}" for c in pf.SFF_FULL}
    assert set(designs["LOC"].columns) <= set(pf.LOC_ONLY)


def test_xg_bundle_round_trip(tmp_path):
    x, y, match, _ = _synthetic(n=600)
    booster, rounds = po.fit_gbm(
        FAST, x, y, match, seed=0, categorical=["f_play_pattern"], n_threads=1
    )
    model = po.XgModel(
        "EVENT",
        list(x.columns),
        ["f_play_pattern"],
        [booster.model_to_string()] * 2,
        rounds,
        len(y),
        float(y.mean()),
    )
    bundle = po.XgBundle(models={"EVENT": model}, params={"n_seeds": 2})
    p = bundle.save(tmp_path)
    assert p.exists()
    loaded = po.XgBundle.load(tmp_path)
    assert loaded.params == {"n_seeds": 2} and set(loaded.models) == {"EVENT"}
    pred = loaded.models["EVENT"].predict(x.assign(extra=1.0))
    np.testing.assert_allclose(pred, booster.predict(x, num_iteration=rounds), rtol=1e-6)
