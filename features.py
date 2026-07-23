# -*- coding: utf-8 -*-
"""GREAT sEMG 데이터(참가자 8명, 2일 2세션, 16채널 2kHz)를 윈도우로 자르고 시간영역 특징을 뽑는다.
특징은 두 벌이다. Hudgins 4특징(MAV/ZC/SSC/WL, 채널당 4개=64개)은 원논문 기준선 재현용,
rich 14특징(여기에 통계 10개 추가, 채널당 14개=224개)은 분석용이다. 추출 결과는 .npz로 캐시한다."""
import glob
import time
from pathlib import Path

import numpy as np
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
# pandas, h5py는 원시 데이터 추출 시에만 쓰므로 load_features 안에서 불러온다
# (캐시만 있으면 두 패키지 없이도 실행된다).

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / "data" / "extracted" / "data"   # 참가자 폴더들이 있는 위치
CACHE = BASE / "features_cache.npz"

FS = 2000          # 표본화율 (Hz)
WIN = 256          # 윈도우 길이 = 128ms
STRIDE = 100       # 윈도우 간격 = 50ms

# 코드→이름 매핑은 데이터셋 저자 저장소(MoveR_AT_GREAT) README 기준 (3=tripod, 4=pointer)
GRASP_NAMES = {1: "power", 2: "lateral", 3: "tripod",
               4: "pointer", 5: "open", 6: "rest"}


# 시간영역 특징. W는 (채널, 윈도우수, 길이) 묶음이고, thr는 채널별 진폭 임계값(ZC/SSC/WAMP의 잡음 제거용).
def _zero_crossings(W, thr):
    """영점 교차(ZC): 신호 부호가 바뀌는 횟수."""
    sign = np.sign(W)
    sign[sign == 0] = -1
    crossed = np.diff(sign, axis=2) != 0
    big_enough = np.abs(np.diff(W, axis=2)) >= thr[:, None, None]
    return np.sum(crossed & big_enough, axis=2)


def _slope_sign_changes(W, thr):
    """기울기 부호 변화(SSC): 1차 차분의 부호가 바뀌는 횟수."""
    d = np.diff(W, axis=2)
    sign = np.sign(d)
    sign[sign == 0] = -1
    crossed = np.diff(sign, axis=2) != 0
    big_enough = (np.abs(np.diff(d, axis=2)) >= thr[:, None, None]) | \
                 (np.abs(d[:, :, :-1]) >= thr[:, None, None]) | \
                 (np.abs(d[:, :, 1:]) >= thr[:, None, None])
    return np.sum(crossed & big_enough, axis=2)


def _willison_amplitude(W, thr):
    """WAMP: 인접 표본 차이가 임계값을 넘는 횟수."""
    return np.sum(np.abs(np.diff(W, axis=2)) >= thr[:, None, None], axis=2)


def extract_features(W, thr, rich=True):
    """윈도우 묶음에서 특징을 뽑아 (윈도우수, 채널×특징) 행렬로 반환.

    rich=False면 Hudgins 4특징, rich=True면 확장 14특징.
    채널별 특징을 [채널0의 특징들, 채널1의 특징들, ...] 순으로 펼친다.
    """
    C, N, _ = W.shape
    diff = np.diff(W, axis=2)

    mav = np.mean(np.abs(W), axis=2)
    wl = np.sum(np.abs(diff), axis=2)
    zc = _zero_crossings(W, thr).astype(np.float64)
    ssc = _slope_sign_changes(W, thr).astype(np.float64)

    if not rich:
        feats = [mav, zc, ssc, wl]   # Hudgins 4특징
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
        log_var = np.log(var + 1e-12)   # +1e-12: var=0일 때 log 발산 방지
        ssi = np.sum(W ** 2, axis=2)
        feats = [mav, rms, wl, zc, ssc, wamp, var, std,
                 iqr, mad, skew, kurt, log_var, ssi]

    arr = np.stack(feats, axis=2)            # (채널, 윈도우, 특징)
    arr = np.transpose(arr, (1, 0, 2))       # (윈도우, 채널, 특징)
    return arr.reshape(N, C * len(feats))


# 데이터 적재
def list_blocks():
    """참가자/일자/세션별 측정 블록 목록을 만든다."""
    blocks = []
    for p in range(1, 9):
        for path in sorted(glob.glob(str(DATA_DIR / f"participant_{p}" / "*"))):
            name = Path(path).name
            day = 1 if "day1" in name else 2
            blocks.append({"participant": p, "day": day, "path": path})
    return blocks


def load_features(use_cache=True):
    """전체 데이터를 윈도우로 자르고 특징을 추출해 딕셔너리로 반환.

    반환 키:
      Xhud (윈도우, 64)  Hudgins 특징
      Xrich(윈도우, 224) rich 특징
      y_grasp / g_part / g_pos / g_day / g_trial  : 라벨과 그룹 키
    누수 방지를 위해 시행(g_trial)·위치(g_pos)·피험자(g_part)를 그룹 키로 함께 저장한다.
    """
    if use_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        print(f"[캐시] {CACHE.name} 불러옴: Xrich {z['Xrich'].shape}")
        return {k: z[k] for k in z.files}

    import pandas as pd   # 원시 데이터 추출 시에만 필요
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
                sig = h[key][:].astype(np.float64)        # (16, 시간)
                if sig.shape[1] < WIN:
                    continue
                # 채널별 임계값 = 채널 표준편차의 1% (잡음 무시용)
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
        print(f"  블록 {bi+1}/32 처리 ({time.time()-t0:.0f}s)")

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
    print(f"[캐시] 저장: 총 {len(data['y_grasp']):,}개 윈도우 ({time.time()-t0:.0f}s)")
    return data
