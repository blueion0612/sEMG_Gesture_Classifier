"""Tests for the feature extraction and the model definitions.

Everything runs on synthetic signals, so the GREAT dataset is not needed. Run
with `pytest`, or directly with `python tests/test_features.py`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semg import features, models  # noqa: E402

FS, WIN = features.FS, features.WIN


def _batch(n_ch=16, n_win=5, length=WIN, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_ch, n_win, length))


def test_window_geometry_matches_the_documented_durations():
    """128 ms windows at 2 kHz with a 50 ms hop."""
    assert FS == 2000
    assert WIN == 256 and WIN / FS == 0.128
    assert features.STRIDE == 100 and features.STRIDE / FS == 0.05


def test_feature_widths_are_four_and_fourteen_per_channel():
    W = _batch()
    thr = np.full(W.shape[0], 0.01)
    hud = features.extract_features(W, thr, rich=False)
    rich = features.extract_features(W, thr, rich=True)
    assert hud.shape == (5, 16 * 4) == (5, 64)
    assert rich.shape == (5, 16 * 14) == (5, 224)
    assert np.isfinite(hud).all() and np.isfinite(rich).all()


def test_rich_contains_the_hudgins_four_but_orders_them_differently():
    """Rich is a superset by content, not a prefix.

    Within one channel the Hudgins block is [mav, zc, ssc, wl] while the rich
    block is [mav, rms, wl, zc, ssc, ...], so the four appear at 0, 3, 4 and 2.
    """
    W = _batch(seed=3)
    thr = np.full(W.shape[0], 0.01)
    hud = features.extract_features(W, thr, rich=False)
    rich = features.extract_features(W, thr, rich=True)

    where_in_rich = {0: 0, 1: 3, 2: 4, 3: 2}   # hudgins index -> rich index
    for c in range(16):
        for h, r in where_in_rich.items():
            assert np.allclose(hud[:, c * 4 + h], rich[:, c * 14 + r]), f"channel {c}, feature {h}"


def test_zero_crossings_count_sign_changes():
    """A square wave crosses zero once per half period."""
    half = 8
    sig = np.tile(np.r_[np.ones(half), -np.ones(half)], WIN // (2 * half))
    W = sig[None, None, :].astype(float)
    zc = features._zero_crossings(W, np.array([0.5]))
    assert zc[0, 0] == WIN // (2 * half) * 2 - 1


def test_threshold_suppresses_noise_in_the_counting_features():
    """Below-threshold ripple must not be counted as crossings."""
    t = np.arange(WIN)
    ripple = 1e-4 * np.sin(2 * np.pi * t / 4.0)
    W = ripple[None, None, :]
    loud = features._zero_crossings(W, np.array([1e-9]))
    quiet = features._zero_crossings(W, np.array([1.0]))
    assert loud[0, 0] > 0
    assert quiet[0, 0] == 0


def test_bandpower_free_features_are_scale_sensitive_as_expected():
    """Doubling the signal doubles MAV, the first feature of each channel block."""
    W = _batch(seed=7)
    thr = np.full(W.shape[0], 0.01)
    a = features.extract_features(W, thr, rich=False)
    b = features.extract_features(2 * W, thr, rich=False)
    assert np.allclose(b[:, 0], 2 * a[:, 0], rtol=1e-6)


def test_every_taught_model_is_a_usable_estimator():
    for name, clf in models.taught_models().items():
        assert hasattr(clf, "fit") and hasattr(clf, "predict"), name


def test_scale_sensitive_models_are_wrapped_with_a_scaler():
    """KNN, logistic regression and the perceptron must be inside a pipeline."""
    m = models.taught_models()
    for name in ("KNN", "LogisticRegression", "Perceptron"):
        assert hasattr(m[name], "steps"), f"{name} is not in a pipeline"
        assert m[name].steps[0][0] == "scaler"
    assert hasattr(models.lda(), "steps")


def test_ensemble_and_baseline_fit_and_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    for clf in (models.soft_voting_ensemble(), models.lda()):
        pred = clf.fit(X, y).predict(X)
        assert pred.shape == y.shape
        assert set(np.unique(pred)) <= {0, 1}


def test_grasp_names_cover_the_six_documented_gestures():
    names = set(features.GRASP_NAMES.values())
    assert names == {"power", "lateral", "tripod", "pointer", "open", "rest"}
    assert sorted(features.GRASP_NAMES) == list(range(1, 7))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
