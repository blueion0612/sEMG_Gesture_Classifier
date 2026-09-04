# -*- coding: utf-8 -*-
"""Window the GREAT sEMG recordings and extract time-domain features.

The dataset is 8 participants, two days of two sessions each, 16 channels at
2 kHz. Two feature sets are produced. Hudgins-4 (MAV, ZC, SSC, WL; 4 per
channel, 64 total) reproduces the baseline of the source paper. Rich-14 adds
ten statistics per channel (224 total) and is what the analysis uses. The
result is cached as an .npz so the extraction runs once."""
import glob
import time
from pathlib import Path

import numpy as np
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
# pandas and h5py are only needed to read the raw dataset, so they are imported
# inside load_features: with the cache present, neither package is required.

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / "data" / "extracted" / "data"   # directory holding the participant folders
CACHE = BASE / "features_cache.npz"

FS = 2000          # sampling rate (Hz)
WIN = 256          # window length, 128 ms
STRIDE = 100       # hop between windows, 50 ms

# Code to name mapping follows the dataset authors' own repository, MoveR_AT_GREAT (3=tripod, 4=pointer)
GRASP_NAMES = {1: "power", 2: "lateral", 3: "tripod",
               4: "pointer", 5: "open", 6: "rest"}


# Time-domain features. W is (channel, window, length); thr is a per-channel amplitude
# threshold that keeps sensor noise out of the ZC, SSC and WAMP counts.
def _zero_crossings(W, thr):
    """Zero crossings: how often the signal changes sign."""
    sign = np.sign(W)
    sign[sign == 0] = -1
    crossed = np.diff(sign, axis=2) != 0
    big_enough = np.abs(np.diff(W, axis=2)) >= thr[:, None, None]
    return np.sum(crossed & big_enough, axis=2)


def _slope_sign_changes(W, thr):
    """Slope sign changes: how often the first difference changes sign."""
    d = np.diff(W, axis=2)
    sign = np.sign(d)
    sign[sign == 0] = -1
    crossed = np.diff(sign, axis=2) != 0
    big_enough = (np.abs(np.diff(d, axis=2)) >= thr[:, None, None]) | \
                 (np.abs(d[:, :, :-1]) >= thr[:, None, None]) | \
                 (np.abs(d[:, :, 1:]) >= thr[:, None, None])
    return np.sum(crossed & big_enough, axis=2)


def _willison_amplitude(W, thr):
    """Willison amplitude: how often adjacent samples differ by more than the threshold."""
    return np.sum(np.abs(np.diff(W, axis=2)) >= thr[:, None, None], axis=2)


def extract_features(W, thr, rich=True):
    """Extract features from a batch of windows as a (window, channel x feature) matrix.

    rich=False gives the Hudgins 4; rich=True gives the extended 14.
    Channels are laid out consecutively: all of channel 0, then all of channel 1.
    """
    C, N, _ = W.shape
    diff = np.diff(W, axis=2)

    mav = np.mean(np.abs(W), axis=2)
    wl = np.sum(np.abs(diff), axis=2)
    zc = _zero_crossings(W, thr).astype(np.float64)
    ssc = _slope_sign_changes(W, thr).astype(np.float64)

    if not rich:
        feats = [mav, zc, ssc, wl]   # the Hudgins 4
    else:
        rms = np.sqrt(np.mean(W ** 2, axis=2))
        wamp = _willison_amplitude(W, thr).astype(np.float64)
        var = np.var(W, axis=2)
        std = np.std(W, axis=2)
        q75, q25 = np.percentile(W, [75, 25], axis=2)
        iqr = q75 - q25
        mean = np.mean(W, axis=2)
        mad = np.mean(np.abs(W - mean[:, :, None]), axis=2)
        skew = np.nan_to_num(stats.skew(W, axis=2))
        kurt = np.nan_to_num(stats.kurtosis(W, axis=2))
        log_var = np.log(var + 1e-12)   # the epsilon keeps log finite when a window is constant
        ssi = np.sum(W ** 2, axis=2)
        feats = [mav, rms, wl, zc, ssc, wamp, var, std,
                 iqr, mad, skew, kurt, log_var, ssi]

    arr = np.stack(feats, axis=2)            # (channel, window, feature)
    arr = np.transpose(arr, (1, 0, 2))       # (window, channel, feature)
    return arr.reshape(N, C * len(feats))


# Loading
def list_blocks():
    """List the recording blocks, one per participant, day and session."""
    blocks = []
    for p in range(1, 9):
        for path in sorted(glob.glob(str(DATA_DIR / f"participant_{p}" / "*"))):
            name = Path(path).name
            day = 1 if "day1" in name else 2
            blocks.append({"participant": p, "day": day, "path": path})
    return blocks


def load_features(use_cache=True):
    """Window the whole dataset, extract both feature sets, and return them.

    Keys:
      Xhud  (window, 64)   Hudgins features
      Xrich (window, 224)  rich features
      y_grasp, g_part, g_pos, g_day, g_trial: labels and grouping keys
    Trial, position and participant are stored alongside so that an evaluation can
    keep every window of one trial on a single side of a split.
    """
    if use_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        print(f"[cache] loaded {CACHE.name}: Xrich {z['Xrich'].shape}")
        return {k: z[k] for k in z.files}

    import pandas as pd   # only needed when reading the raw dataset
    import h5py
    Xhud, Xrich = [], []
    y_grasp, g_part, g_pos, g_day, g_trial = [], [], [], [], []
    trial_id = 0
    t0 = time.time()

    for bi, b in enumerate(list_blocks()):
        labels = pd.read_csv(Path(b["path"]) / "trials.csv")
        with h5py.File(Path(b["path"]) / "emg_data.hdf5", "r") as h:
            for _, row in labels.iterrows():
                key = str(int(row["row_number"]))
                if key not in h:
                    continue
                sig = h[key][:].astype(np.float64)        # (16, time)
                if sig.shape[1] < WIN:
                    continue
                # Per-channel threshold: 1% of that channel's standard deviation
                thr = 0.01 * np.std(sig, axis=1)
                thr = np.where(thr <= 0, 1e-9, thr)
                win = sliding_window_view(sig, WIN, axis=1)[:, ::STRIDE, :]
                N = win.shape[1]
                if N == 0:
                    continue
                Xhud.append(extract_features(win, thr, rich=False))
                Xrich.append(extract_features(win, thr, rich=True))
                y_grasp.append(np.full(N, int(row["grasp"]), np.int16))
                g_part.append(np.full(N, b["participant"], np.int16))
                g_pos.append(np.full(N, int(row["target_position"]), np.int16))
                g_day.append(np.full(N, b["day"], np.int16))
                g_trial.append(np.full(N, trial_id, np.int32))
                trial_id += 1
        print(f"  block {bi+1}/32 done ({time.time()-t0:.0f}s)")

    data = {
        "Xhud": np.concatenate(Xhud).astype(np.float32),
        "Xrich": np.concatenate(Xrich).astype(np.float32),
        "y_grasp": np.concatenate(y_grasp),
        "g_part": np.concatenate(g_part),
        "g_pos": np.concatenate(g_pos),
        "g_day": np.concatenate(g_day),
        "g_trial": np.concatenate(g_trial),
    }
    np.savez_compressed(CACHE, **data)
    print(f"[cache] saved {len(data['y_grasp']):,} windows ({time.time()-t0:.0f}s)")
    return data
