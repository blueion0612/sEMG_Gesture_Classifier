# -*- coding: utf-8 -*-
"""Entry point for the whole analysis. Run with `python main.py`.

In order: load the data and extract features, explore it with PCA and KMeans,
run the full-data bake-off, compare the two feature sets, reproduce the source
paper's protocols, and finish with leave-one-subject-out. Results are printed
and written to results_main.json."""
import sys
import json
from pathlib import Path

# Force UTF-8 on stdout so the output survives a non-UTF-8 Windows console
sys.stdout.reconfigure(encoding="utf-8")

from .features import load_features, GRASP_NAMES
from .eda import run_eda, print_eda
from .evaluation import (holdout_bakeoff, feature_set_comparison,
                        gb_vs_histgb_equivalence, paper_protocol, loso)

OUT = Path(__file__).resolve().parent / "results_main.json"


def main():
    results = {}

    # 1) data and features
    print("\n[1] Loading data and extracting features")
    data = load_features()
    print(f"    {len(data['y_grasp']):,} windows, Hudgins {data['Xhud'].shape[1]} features, rich {data['Xrich'].shape[1]}")

    # 2) EDA
    print("\n[2] Exploratory analysis")
    results["eda"] = run_eda(data)
    print_eda(results["eda"])

    # 3) bake-off
    print("\n[3] Full-data bake-off (same user and position, trial-grouped holdout)")
    bake = holdout_bakeoff(data)
    results["holdout_bakeoff"] = bake
    for name in sorted([k for k in bake if not k.startswith("_")],
                       key=lambda k: -bake[k]["accuracy"]):
        print(f"    {name:14s} accuracy {bake[name]['accuracy']*100:5.2f}  macro-F1 {bake[name]['macro_f1']*100:5.2f}")

    # 4) feature set comparison
    print("\n[4] Feature set comparison (Hudgins-4 to rich-14)")
    feat = feature_set_comparison(data)
    results["feature_comparison"] = feat
    for name, v in feat.items():
        print(f"    {name:14s} {v['hudgins4']*100:5.2f} → {v['rich14']*100:5.2f}  "
              f"({(v['rich14']-v['hudgins4'])*100:+.1f}%p)")

    # 4b) the histogram implementation matches plain gradient boosting
    print("\n[4b] GradientBoosting against HistGradientBoosting, on a subsample")
    eq = gb_vs_histgb_equivalence(data)
    results["gb_equivalence"] = eq
    print(f"    GradientBoosting {eq['GradientBoosting']*100:.1f}  ≈  "
          f"HistGradientBoosting {eq['HistGradientBoosting']*100:.1f}  (gap {eq['abs_gap_pp']} points)")

    # 5) reproduce the source paper's protocols
    print("\n[5] Source paper protocols (LDA fixed, features extended)")
    pp = paper_protocol(data)
    results["paper_protocol"] = pp
    w, c = pp["within_position"], pp["cross_position"]
    print(f"    within position: LDA+Hudgins {w['LDA+Hudgins']*100:.1f}%  ->  LDA+rich {w['LDA+rich']*100:.1f}%")
    print(f"    across positions: LDA+Hudgins {c['LDA+Hudgins']*100:.1f}%  ->  LDA+rich {c['LDA+rich']*100:.1f}%"
          f"  ({c['gain_pp']:+.1f} points, improved for {c['n_subjects_improved']} of 8, Wilcoxon p={c['wilcoxon_p']})")

    # 6) LOSO
    print("\n[6] Leave-one-subject-out")
    lo = loso(data)
    results["loso"] = lo
    for name in sorted([k for k in lo if not k.startswith("_")],
                       key=lambda k: -lo[k]["accuracy"]):
        print(f"    {name:16s} accuracy {lo[name]['accuracy']*100:5.2f}")
    sig = lo["_significance"]
    print(f"    HistGradientBoosting against the paper baseline: {sig['HistGradientBoosting_vs_LDA+Hudgins_gain_pp']:+.1f} points "
          f"(improved for {sig['n_subjects_improved']} of 8, Wilcoxon p={sig['wilcoxon_p']})")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] results written to {OUT.name}")


if __name__ == "__main__":
    main()
