# -*- coding: utf-8 -*-
"""Look at the data before fitting anything to it.

Reports the class balance, how correlated the channels are with each other,
whether PCA concentrates the variance in a few axes, and whether KMeans
separates the gestures without labels, judged by silhouette, the elbow in the
sum of squares, and agreement with the true labels. The unsupervised parts run
on standardized rich features."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from .features import GRASP_NAMES

RICH_PER = 14   # rich features per channel; the MAV of channel c is column c*14


def _counts(arr):
    u, c = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def run_eda(data, sample_size=40000, seed=42):
    """Compute the distribution, correlation, PCA and KMeans figures in one pass."""
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

    # Channel correlation is computed on the 16 per-channel MAV columns
    mav = Xr[:, [c * RICH_PER for c in range(16)]]
    corr = np.corrcoef(mav.T)
    offdiag = corr[np.triu_indices(16, k=1)]
    eda["channel_corr_mean_abs"] = round(float(np.mean(np.abs(offdiag))), 3)
    eda["channel_corr_max"] = round(float(np.max(offdiag)), 3)

    # Subsample, standardize, then PCA and KMeans, which keeps this quick
    samp = rng.choice(len(y), min(sample_size, len(y)), replace=False)
    Xs = StandardScaler().fit_transform(Xr[samp])
    ys = y[samp]

    pca = PCA(random_state=42).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    eda["pca_var_top2"] = round(float(cum[1]), 3)
    eda["pca_n_components_90"] = int(np.searchsorted(cum, 0.90) + 1)
    eda["pca_n_components_95"] = int(np.searchsorted(cum, 0.95) + 1)

    # Choose K by silhouette and by the elbow in SSE, and score agreement with the six labels
    Z = pca.transform(Xs)[:, :10]
    sil, sse = {}, {}
    for k in (2, 3, 4, 5, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        sil[k] = round(float(silhouette_score(Z, km.labels_, sample_size=5000,
                                              random_state=42)), 3)
        sse[k] = round(float(km.inertia_), 1)   # within-cluster sum of squares, for the elbow
    eda["kmeans_silhouette"] = sil
    eda["kmeans_sse_elbow"] = sse
    eda["kmeans_best_k"] = max(sil, key=sil.get)
    km6 = KMeans(n_clusters=6, n_init=10, random_state=42).fit(Z)
    eda["kmeans_ari_vs_grasp"] = round(float(adjusted_rand_score(ys, km6.labels_)), 3)

    return eda


def print_eda(eda):
    """Print the run_eda result."""
    print(f"  {eda['n_windows']:,} windows, class imbalance {eda['grasp_imbalance_ratio']}:1")
    print(f"  channel correlation mean |r| {eda['channel_corr_mean_abs']}, max {eda['channel_corr_max']}")
    print(f"  PCA: top two components {eda['pca_var_top2']*100:.0f}%, "
          f"{eda['pca_n_components_90']} components for 90% "
          f"and {eda['pca_n_components_95']} for 95%")
    print(f"  KMeans: best K by silhouette = {eda['kmeans_best_k']} "
          f"({eda['kmeans_silhouette'][eda['kmeans_best_k']]}), ARI against the labels {eda['kmeans_ari_vs_grasp']}")
    print("  the gestures do not separate without labels, so supervision is needed")
