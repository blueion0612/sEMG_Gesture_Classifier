# -*- coding: utf-8 -*-
"""지도학습 전에 데이터 구조를 본다. 클래스 분포와 균형도, 채널 간 상관(다채널이 중복인지),
PCA로 정보가 소수 축에 몰렸는지, KMeans로 손동작이 비지도로 갈라지는지(실루엣과 엘보 SSE,
라벨 일치도 ARI)를 확인한다. 비지도 기법은 표준화한 rich 특징 위에서 돌린다."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from features import GRASP_NAMES

RICH_PER = 14   # 채널당 rich 특징 수 (채널 c의 MAV = 열 c*14)


def _counts(arr):
    u, c = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def run_eda(data, sample_size=40000, seed=42):
    """분포, 채널 상관, PCA, KMeans 지표를 한 번에 계산해 딕셔너리로 돌려준다."""
    Xr = data["Xrich"].astype(np.float64)
    y = data["y_grasp"]
    rng = np.random.RandomState(seed)

    eda = {
        "n_windows": int(len(y)),
        "windows_per_grasp": {GRASP_NAMES[k]: v for k, v in _counts(y).items()},
        "windows_per_subject": _counts(data["g_part"]),
        "windows_per_position": _counts(data["g_pos"]),
    }
    g = list(eda["windows_per_grasp"].values())
    eda["grasp_imbalance_ratio"] = round(max(g) / min(g), 2)

    # 채널 간 상관은 채널별 MAV(16개)로 계산
    mav = Xr[:, [c * RICH_PER for c in range(16)]]
    corr = np.corrcoef(mav.T)
    offdiag = corr[np.triu_indices(16, k=1)]
    eda["channel_corr_mean_abs"] = round(float(np.mean(np.abs(offdiag))), 3)
    eda["channel_corr_max"] = round(float(np.max(offdiag)), 3)

    # 속도를 위해 표본을 뽑아 표준화 후 PCA, KMeans
    samp = rng.choice(len(y), min(sample_size, len(y)), replace=False)
    Xs = StandardScaler().fit_transform(Xr[samp])
    ys = y[samp]

    pca = PCA(random_state=42).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    eda["pca_var_top2"] = round(float(cum[1]), 3)
    eda["pca_n_components_90"] = int(np.searchsorted(cum, 0.90) + 1)
    eda["pca_n_components_95"] = int(np.searchsorted(cum, 0.95) + 1)

    # KMeans: 실루엣과 엘보(SSE) 두 방식으로 K를 보고, 손동작 6종 라벨과의 일치도(ARI)도 본다
    Z = pca.transform(Xs)[:, :10]
    sil, sse = {}, {}
    for k in (2, 3, 4, 5, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        sil[k] = round(float(silhouette_score(Z, km.labels_, sample_size=5000,
                                              random_state=42)), 3)
        sse[k] = round(float(km.inertia_), 1)   # 엘보용 군집 내 제곱합(관성)
    eda["kmeans_silhouette"] = sil
    eda["kmeans_sse_elbow"] = sse
    eda["kmeans_best_k"] = max(sil, key=sil.get)
    km6 = KMeans(n_clusters=6, n_init=10, random_state=42).fit(Z)
    eda["kmeans_ari_vs_grasp"] = round(float(adjusted_rand_score(ys, km6.labels_)), 3)

    return eda


def print_eda(eda):
    """run_eda 결과를 콘솔에 요약 출력."""
    print(f"  윈도우 {eda['n_windows']:,}개, 손동작 균형비 {eda['grasp_imbalance_ratio']}:1")
    print(f"  채널 상관 평균|r| {eda['channel_corr_mean_abs']}, 최대 {eda['channel_corr_max']}")
    print(f"  PCA 상위2성분 {eda['pca_var_top2']*100:.0f}% / 90%에 {eda['pca_n_components_90']}성분 "
          f"/ 95%에 {eda['pca_n_components_95']}성분")
    print(f"  KMeans 실루엣 최적 K={eda['kmeans_best_k']} "
          f"({eda['kmeans_silhouette'][eda['kmeans_best_k']]}), 손동작 일치도 ARI {eda['kmeans_ari_vs_grasp']}")
    print("  비지도로는 손동작이 잘 갈라지지 않아 지도학습이 필요하다")
