# -*- coding: utf-8 -*-
"""전체 분석 엔트리포인트. python main.py로 실행한다.
순서는 데이터 적재와 특징 추출, 탐색적 분석(PCA, KMeans), 전체데이터 bake-off,
특징셋 비교(Hudgins-4 vs rich-14), 원논문 프로토콜 재현(같은 위치/위치 전이), 미지 사용자(LOSO) 일반화다.
결과는 화면에 출력하고 results_main.json으로 저장한다(보고서를 구동하는 results.json과는 별개의 산출물)."""
import sys
import json
from pathlib import Path

# 한국어 Windows 콘솔(cp949)에서도 출력이 깨지지 않도록 표준출력을 UTF-8로 고정
sys.stdout.reconfigure(encoding="utf-8")

from features import load_features, GRASP_NAMES
from eda import run_eda, print_eda
from evaluation import (holdout_bakeoff, feature_set_comparison,
                        gb_vs_histgb_equivalence, paper_protocol, loso)

OUT = Path(__file__).resolve().parent / "results_main.json"


def main():
    results = {}

    # 1) 데이터 + 특징
    print("\n[1] 데이터 적재 및 특징 추출")
    data = load_features()
    print(f"    윈도우 {len(data['y_grasp']):,}개 / Hudgins {data['Xhud'].shape[1]}특징 / rich {data['Xrich'].shape[1]}특징")

    # 2) EDA
    print("\n[2] 탐색적 데이터 분석")
    results["eda"] = run_eda(data)
    print_eda(results["eda"])

    # 3) 종합 bake-off
    print("\n[3] 전체데이터 종합 bake-off (같은 사용자·위치 내, 시행-그룹 홀드아웃)")
    bake = holdout_bakeoff(data)
    results["holdout_bakeoff"] = bake
    for name in sorted([k for k in bake if not k.startswith("_")],
                       key=lambda k: -bake[k]["accuracy"]):
        print(f"    {name:14s} 정확도 {bake[name]['accuracy']*100:5.2f}  macro-F1 {bake[name]['macro_f1']*100:5.2f}")

    # 4) 특징셋 비교
    print("\n[4] 특징셋 비교 (Hudgins-4 → rich-14)")
    feat = feature_set_comparison(data)
    results["feature_comparison"] = feat
    for name, v in feat.items():
        print(f"    {name:14s} {v['hudgins4']*100:5.2f} → {v['rich14']*100:5.2f}  "
              f"({(v['rich14']-v['hudgins4'])*100:+.1f}%p)")

    # 4b) 후속 모델 동등성 (GradientBoosting ≈ HistGradientBoosting)
    print("\n[4b] 후속 모델 동등성 (GradientBoosting ≈ HistGradientBoosting, 표본)")
    eq = gb_vs_histgb_equivalence(data)
    results["gb_equivalence"] = eq
    print(f"    GradientBoosting {eq['GradientBoosting']*100:.1f}  ≈  "
          f"HistGradientBoosting {eq['HistGradientBoosting']*100:.1f}  (차 {eq['abs_gap_pp']}%p)")

    # 5) 원논문 프로토콜 재현·비교
    print("\n[5] 원논문 프로토콜 재현·비교 (분류기 LDA 고정, 특징만 확장)")
    pp = paper_protocol(data)
    results["paper_protocol"] = pp
    w, c = pp["within_position"], pp["cross_position"]
    print(f"    같은 위치 : LDA+Hudgins {w['LDA+Hudgins']*100:.1f}%  →  LDA+rich {w['LDA+rich']*100:.1f}%")
    print(f"    위치 전이 : LDA+Hudgins {c['LDA+Hudgins']*100:.1f}%  →  LDA+rich {c['LDA+rich']*100:.1f}%"
          f"  ({c['gain_pp']:+.1f}%p, 8명 중 {c['n_subjects_improved']}명 향상, Wilcoxon p={c['wilcoxon_p']})")

    # 6) LOSO
    print("\n[6] 미지 사용자(LOSO) 일반화")
    lo = loso(data)
    results["loso"] = lo
    for name in sorted([k for k in lo if not k.startswith("_")],
                       key=lambda k: -lo[k]["accuracy"]):
        print(f"    {name:16s} 정확도 {lo[name]['accuracy']*100:5.2f}")
    sig = lo["_significance"]
    print(f"    → HistGradientBoosting이 논문 기준선 대비 {sig['HistGradientBoosting_vs_LDA+Hudgins_gain_pp']:+.1f}%p "
          f"(8명 중 {sig['n_subjects_improved']}명, Wilcoxon p={sig['wilcoxon_p']})")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 결과 저장 → {OUT.name}")


if __name__ == "__main__":
    main()
