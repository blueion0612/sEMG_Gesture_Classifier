# -*- coding: utf-8 -*-
"""The classical classifiers from the course, a boosting successor, and an ensemble.

Hyperparameters are left at sensible defaults rather than searched, with
random_state fixed at 42 throughout. Scale-sensitive models (KNN, logistic
regression, perceptron, LDA) are wrapped in a pipeline with StandardScaler so
the scaler is fitted on training data only."""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, AdaBoostClassifier,
                              VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _scaled(clf):
    """Pair a scale-sensitive estimator with a scaler, inside one pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def taught_models():
    """The classifiers covered in the course. Boosting is represented by HistGB."""
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=20, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=80, random_state=42),
        # Plain GradientBoosting takes hours on the full set, so the histogram
        # implementation stands in for it. evaluation.gb_vs_histgb_equivalence shows they agree.
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.1, random_state=42),
        "KNN": _scaled(KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        "LogisticRegression": _scaled(LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42)),
        "Perceptron": _scaled(Perceptron(max_iter=1000, eta0=0.01, random_state=42)),
    }


def real_gradient_boosting():
    """Plain GradientBoosting, used only to show the histogram version matches it."""
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)


def soft_voting_ensemble():
    """Soft voting over three models whose errors are not alike."""
    return VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          n_jobs=-1, random_state=42)),
            ("hgb", HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                                   random_state=42)),
            ("lr", _scaled(LogisticRegression(C=1.0, max_iter=1000,
                                              class_weight="balanced", random_state=42))),
        ],
        voting="soft", n_jobs=-1)


def lda():
    """Linear discriminant analysis, the classifier the source paper used."""
    return _scaled(LinearDiscriminantAnalysis())
