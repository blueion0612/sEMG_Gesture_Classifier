# -*- coding: utf-8 -*-
"""모델 평가 네 가지를 수행한다.
holdout_bakeoff는 전체데이터 종합 비교(같은 사용자·위치 내, 시행-그룹 홀드아웃),
feature_set_comparison은 Hudgins-4 대 rich-14 특징공학 효과(같은 분할),
paper_protocol은 원논문 프로토콜 재현(같은 위치/위치 전이, 분류기를 고정하고 특징만 확장),
loso는 미지 사용자 일반화와 피험자 단위 유의성을 본다.
누수 방지를 위해 같은 시행의 윈도우는 항상 학습이나 평가 한쪽에만 두고,
위치/사용자 일반화에서는 평가 대상 위치/사용자를 학습에서 완전히 뺀다."""
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from features import GRASP_NAMES
from models import taught_models, soft_voting_ensemble, lda, real_gradient_boosting


# 지표 도우미
def bootstrap_ci(y_true, y_pred, n=1000, seed=0):
    """정확도의 95% 부트스트랩 신뢰구간."""
    rng = np.random.RandomState(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    accs = [np.mean(yt[b] == yp[b]) for b in (rng.randint(0, len(yt), len(yt)) for _ in range(n))]
    return round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)


def _cap_per_trial(idx, trial, cap, rng):
    """시행당 윈도우 수를 cap개로 제한(대규모 평가의 계산량 절감). cap=None이면 전체."""
    if cap is None:
        return idx
    keep = []
    for t in np.unique(trial[idx]):
        ti = idx[trial[idx] == t]
        keep.extend((rng.choice(ti, cap, replace=False) if len(ti) > cap else ti).tolist())
    return np.array(sorted(keep))


# 1) 전체데이터 종합 bake-off
def holdout_bakeoff(data):
    """전체 47만 윈도우를 시행 단위로 80:20 분할해 모든 교안 모델 + 앙상블을 비교."""
    X, y, trial = data["Xrich"], data["y_grasp"], data["g_trial"]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                  .split(X, y, trial))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    models = taught_models()
    models["SoftVote"] = soft_voting_ensemble()
    out, preds = {}, {}
    for name, mdl in models.items():
        m = clone(mdl).fit(Xtr, ytr)
        p = m.predict(Xte)
        out[name] = {"accuracy": round(float(accuracy_score(yte, p)), 4),
                     "macro_f1": round(float(f1_score(yte, p, average="macro")), 4)}
        preds[name] = p

    best = max(out, key=lambda k: out[k]["accuracy"])
    labels = sorted(int(v) for v in np.unique(yte))
    cm = confusion_matrix(yte, preds[best], labels=labels)
    out["_best"] = best
    out["_best_per_class_recall"] = {GRASP_NAMES[labels[i]]: round(float(cm[i, i] / cm[i].sum()), 4)
                                     for i in range(len(labels))}
    return out


# 2) 특징셋 비교 (Hudgins-4 vs rich-14)
def feature_set_comparison(data, cap=12, seed=0):
    """동일 시행-그룹 홀드아웃에서 Hudgins-4(64) 대 rich-14(224)의 정확도를 비교.
    분할·모델을 고정하고 특징만 바꿔, 특징공학의 순수 효과를 본다."""
    y, trial = data["y_grasp"], data["g_trial"]
    rng = np.random.RandomState(seed)
    idx = _cap_per_trial(np.arange(len(y)), trial, cap, rng)
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42)
                  .split(idx, y[idx], trial[idx]))
    tr, te = idx[tr], idx[te]

    out = {}
    for name, mdl in taught_models().items():
        row = {}
        for feat in ("Xhud", "Xrich"):
            m = clone(mdl).fit(data[feat][tr], y[tr])
            row["hudgins4" if feat == "Xhud" else "rich14"] = round(
                float(accuracy_score(y[te], m.predict(data[feat][te]))), 4)
        out[name] = row
    return out


# 2-b) 후속 모델 동등성 (GradientBoosting ≈ HistGradientBoosting)
def gb_vs_histgb_equivalence(data, sample=12000, seed=42):
    """원래 GradientBoosting과 후속 HistGradientBoosting의 정확도가 사실상 같음을 확인.
    원 GB는 전체 데이터 학습이 매우 느려(수 시간) 표본을 뽑아 같은 분할에서 빠르게 비교한다.
    후속 모델의 이점은 정확도가 아니라 속도임을 보이는 통제 실험(보고서 §8)."""
    X, y, trial = data["Xrich"], data["y_grasp"], data["g_trial"]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), min(sample, len(y)), replace=False)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                  .split(idx, y[idx], trial[idx]))
    tr, te = idx[tr], idx[te]
    out = {}
    for name, mdl in (("GradientBoosting", real_gradient_boosting()),
                      ("HistGradientBoosting", taught_models()["HistGradientBoosting"])):
        p = clone(mdl).fit(X[tr], y[tr]).predict(X[te])
        out[name] = round(float(accuracy_score(y[te], p)), 4)
    out["abs_gap_pp"] = round(abs(out["GradientBoosting"] - out["HistGradientBoosting"]) * 100, 2)
    return out


# 3) 원논문 프로토콜 재현·비교
def paper_protocol(data):
    """원논문과 동일한 두 프로토콜에서, 분류기를 LDA로 고정하고 특징만 확장해 비교.
      within : 같은 위치 안에서 시행-그룹 5-fold (논문 ~96% 재현)
      cross  : 한 위치에서 학습 후 같은 날 다른 위치에서 평가 (논문 naive transfer, 84–92%)
    cross에서는 피험자 단위 정확도로 Wilcoxon 유의성도 계산한다."""
    Xh, Xr = data["Xhud"], data["Xrich"]
    y, part, pos, day = data["y_grasp"], data["g_part"], data["g_pos"], data["g_day"]
    trial = data["g_trial"]

    # 비교 모델: LDA(논문 분류기)에 특징만 hud/rich로 바꿔 본다
    def fit_pred(feat, model, tr, te):
        X = Xh if feat == "hud" else Xr
        return clone(model).fit(X[tr], y[tr]).predict(X[te])

    # within-position: (피험자,일,위치) 셀마다 시행-그룹 5-fold
    within = {"LDA+Hudgins": [], "LDA+rich": []}
    for p in np.unique(part):
        for d in (1, 2):
            for ps in np.unique(pos[(part == p) & (day == d)]):
                idx = np.where((part == p) & (day == d) & (pos == ps))[0]
                if len(np.unique(trial[idx])) < 5:
                    continue
                gkf = GroupKFold(5)
                for key, feat in (("LDA+Hudgins", "hud"), ("LDA+rich", "rich")):
                    accs = [accuracy_score(y[idx[te]], fit_pred(feat, lda(), idx[tr], idx[te]))
                            for tr, te in gkf.split(idx, y[idx], trial[idx])]
                    within[key].append(np.mean(accs))
    within = {k: round(float(np.mean(v)), 4) for k, v in within.items()}

    # cross-position: 한 위치 학습, 다른 위치 평가 (피험자별 평균)
    per_subject = {"LDA+Hudgins": [], "LDA+rich": []}
    for p in np.unique(part):
        acc = {"LDA+Hudgins": [], "LDA+rich": []}
        for d in (1, 2):
            cell = (part == p) & (day == d)
            positions = np.unique(pos[cell])
            for src in positions:
                tr = np.where(cell & (pos == src))[0]
                if len(np.unique(y[tr])) < 2:
                    continue
                for tgt in positions:
                    if tgt == src:
                        continue
                    te = np.where(cell & (pos == tgt))[0]
                    acc["LDA+Hudgins"].append(accuracy_score(y[te], fit_pred("hud", lda(), tr, te)))
                    acc["LDA+rich"].append(accuracy_score(y[te], fit_pred("rich", lda(), tr, te)))
        for k in acc:
            per_subject[k].append(np.mean(acc[k]))

    hud = np.array(per_subject["LDA+Hudgins"]); rich = np.array(per_subject["LDA+rich"])
    _, wilcoxon_p = stats.wilcoxon(rich, hud)
    cross = {
        "LDA+Hudgins": round(float(hud.mean()), 4),
        "LDA+rich": round(float(rich.mean()), 4),
        "gain_pp": round(float((rich.mean() - hud.mean()) * 100), 2),
        "n_subjects_improved": int(np.sum(rich > hud)),
        "wilcoxon_p": round(float(wilcoxon_p), 4),
    }
    return {"within_position": within, "cross_position": cross}


# 4) 미지 사용자(LOSO) 일반화
def loso(data, cap=5, seed=13):
    """피험자 한 명을 빼고 학습한 뒤 그 피험자로 평가한다(8겹). 가장 엄격한 일반화 지표다.
    모델별 정확도/CI와, 후속 모델 대 논문 기준선의 피험자 단위 Wilcoxon 유의성을 반환한다."""
    Xh, Xr = data["Xhud"], data["Xrich"]
    y, part, trial = data["y_grasp"], data["g_part"], data["g_trial"]
    rng = np.random.RandomState(seed)

    models = {
        "LDA+Hudgins": ("hud", lda()),
        "LDA+rich": ("rich", lda()),
        "RandomForest": ("rich", taught_models()["RandomForest"]),
        # 교안 GradientBoosting의 빠른 구현(보고서 표7의 LOSO 부스팅 행에 해당)
        "HistGradientBoosting": ("rich", taught_models()["HistGradientBoosting"]),
        "KNN": ("rich", taught_models()["KNN"]),
        "LogReg": ("rich", taught_models()["LogisticRegression"]),
    }
    preds = {k: [] for k in models}
    per_subj = {"LDA+Hudgins": [], "HistGradientBoosting": []}   # 유의성용
    truth = []

    for s in np.unique(part):
        tri = _cap_per_trial(np.where(part != s)[0], trial, cap, rng)
        tei = _cap_per_trial(np.where(part == s)[0], trial, cap, rng)
        truth.append(y[tei])
        for name, (feat, mdl) in models.items():
            X = Xh if feat == "hud" else Xr
            p = clone(mdl).fit(X[tri], y[tri]).predict(X[tei])
            preds[name].append(p)
            if name in per_subj:
                per_subj[name].append(accuracy_score(y[tei], p))

    yt = np.concatenate(truth)
    out = {}
    for name in models:
        yp = np.concatenate(preds[name])
        out[name] = {"accuracy": round(float(accuracy_score(yt, yp)), 4),
                     "macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
                     "ci95": list(bootstrap_ci(yt, yp, n=500, seed=1))}

    hud = np.array(per_subj["LDA+Hudgins"]); gb = np.array(per_subj["HistGradientBoosting"])
    _, wp = stats.wilcoxon(gb, hud)
    out["_significance"] = {
        "HistGradientBoosting_vs_LDA+Hudgins_gain_pp": round(float((gb.mean() - hud.mean()) * 100), 2),
        "n_subjects_improved": int(np.sum(gb > hud)),
        "wilcoxon_p": round(float(wp), 4),
        "note": "유의성 단위는 피험자(n=8). 윈도우 단위 검정은 표본 간 상관으로 과대평가됨",
    }
    return out
