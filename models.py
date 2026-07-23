# -*- coding: utf-8 -*-
"""교안의 고전 분류기와 부스팅 후속 모델, 소프트보팅 앙상블을 정의한다.
하이퍼파라미터는 대규모 탐색 없이 기본값으로 고정했다(random_state=42 통일).
스케일에 민감한 모델(KNN, 로지스틱회귀, 퍼셉트론, LDA)은 StandardScaler를 파이프라인으로 묶어
학습 데이터에만 적합되게 해 누수를 막는다."""
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
    """표준화가 필요한 모델을 StandardScaler와 묶는다."""
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def taught_models():
    """교안에서 다룬 분류기 모음. 부스팅은 후속 HistGB로 대표한다(사유는 아래 주석)."""
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=20, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=80, random_state=42),
        # 교안 GradientBoosting은 전체 데이터 학습이 수 시간 걸려, 같은 방식의 빠른 구현인
        # HistGradientBoosting으로 대표한다. 둘의 동등성은 evaluation.gb_vs_histgb_equivalence에서 확인.
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.1, random_state=42),
        "KNN": _scaled(KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        "LogisticRegression": _scaled(LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42)),
        "Perceptron": _scaled(Perceptron(max_iter=1000, eta0=0.01, random_state=42)),
    }


def real_gradient_boosting():
    """히스토그램이 아닌 원래 GradientBoosting. 후속 모델과의 동등성 확인용."""
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)


def soft_voting_ensemble():
    """서로 다른 오류 양상을 가진 세 모델의 확률을 평균하는 소프트보팅 앙상블."""
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
    """선형판별분석(LDA). 원논문이 쓴 기준선 분류기(교안 외, 재현/비교용)."""
    return _scaled(LinearDiscriminantAnalysis())
