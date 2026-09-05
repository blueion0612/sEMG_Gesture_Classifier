# -*- coding: utf-8 -*-
"""Four evaluations, from the easiest condition to the hardest.

holdout_bakeoff compares every model within one user and one arm position,
splitting by trial. feature_set_comparison isolates the effect of the feature
set by holding the split and the model fixed. paper_protocol reproduces the two
protocols of the source paper, fixing the classifier and extending only the
features. loso holds out a whole participant.

Every window of a trial always lands on one side of a split, and for the
position and user conditions the held-out position or participant is absent
from training entirely."""
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .features import GRASP_NAMES
from .models import taught_models, soft_voting_ensemble, lda, real_gradient_boosting


# Metric helpers
def bootstrap_ci(y_true, y_pred, n=1000, seed=0):
    """95% bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    accs = [np.mean(yt[b] == yp[b]) for b in (rng.randint(0, len(yt), len(yt)) for _ in range(n))]
    return round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)


def _cap_per_trial(idx, trial, cap, rng):
    """Keep at most cap windows per trial, which bounds the cost of the large runs."""
    if cap is None:
        return idx
    keep = []
    for t in np.unique(trial[idx]):
        ti = idx[trial[idx] == t]
        keep.extend((rng.choice(ti, cap, replace=False) if len(ti) > cap else ti).tolist())
    return np.array(sorted(keep))


# 1) Full-data bake-off
def holdout_bakeoff(data):
    """Split all 470k windows 80:20 by trial and compare every model plus the ensemble."""
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


# 2) Feature set comparison, Hudgins-4 against rich-14
def feature_set_comparison(data, cap=12, seed=0):
    """Compare Hudgins-4 (64) with rich-14 (224) on the same trial-grouped holdout.
    The split and the model are fixed, so the difference is the feature set alone."""
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


# 2b) The histogram implementation matches plain GradientBoosting
def gb_vs_histgb_equivalence(data, sample=12000, seed=42):
    """Show that plain and histogram gradient boosting reach the same accuracy.
    Plain GB needs hours on the full set, so the comparison runs on a subsample of
    the same split. The point is that the successor buys speed, not accuracy."""
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


# 3) Reproduction of the source paper's protocols
def paper_protocol(data):
    """Run the paper's two protocols with its classifier, changing only the features.
      within: trial-grouped 5-fold inside one arm position; the paper reports about 96%
      cross:  train at one position, test at another on the same day; the paper's
              naive transfer, 84 to 92%. Significance is Wilcoxon over participants."""
    Xh, Xr = data["Xhud"], data["Xrich"]
    y, part, pos, day = data["y_grasp"], data["g_part"], data["g_pos"], data["g_day"]
    trial = data["g_trial"]

    # The classifier stays LDA, as in the paper; only the feature set changes
    def fit_pred(feat, model, tr, te):
        X = Xh if feat == "hud" else Xr
        return clone(model).fit(X[tr], y[tr]).predict(X[te])

    # within position: trial-grouped 5-fold in each (participant, day, position) cell
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

    # cross position: train at one position, test at another, averaged per participant
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


# 4) Leave-one-subject-out generalization
def loso(data, cap=5, seed=13):
    """Hold out one participant, train on the rest, test on them. Eight folds.
    Returns accuracy and a confidence interval per model, and the Wilcoxon test
    over participants comparing the successor with the paper's baseline."""
    Xh, Xr = data["Xhud"], data["Xrich"]
    y, part, trial = data["y_grasp"], data["g_part"], data["g_trial"]
    rng = np.random.RandomState(seed)

    models = {
        "LDA+Hudgins": ("hud", lda()),
        "LDA+rich": ("rich", lda()),
        "RandomForest": ("rich", taught_models()["RandomForest"]),
        # the fast implementation standing in for the course's GradientBoosting
        "HistGradientBoosting": ("rich", taught_models()["HistGradientBoosting"]),
        "KNN": ("rich", taught_models()["KNN"]),
        "LogReg": ("rich", taught_models()["LogisticRegression"]),
    }
    preds = {k: [] for k in models}
    per_subj = {"LDA+Hudgins": [], "HistGradientBoosting": []}   # kept for the significance test
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
        "note": "significance is over participants (n=8); a window-level test would be inflated by correlation between windows",
    }
    return out
