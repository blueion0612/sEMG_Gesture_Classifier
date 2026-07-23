# sEMG Gesture Classifier

> Classical machine-learning bake-off for six-gesture sEMG hand-gesture recognition on the
> public [GREAT dataset](https://doi.org/10.5061/dryad.8sf7m0czv) — course-taught models,
> boosting successors, paper-protocol reproduction, and LOSO evaluation.
> Graduate coursework, Korea University, 2026. Full write-up in Korean below.

---

# sEMG 손동작 인식 분석 코드

표면 근전도(sEMG)로 여섯 손동작(power, lateral, pointer, tripod, open, rest)을 분류하는
프로젝트의 분석 코드입니다. 교안에서 배운 고전 머신러닝 기법을 종합적으로 적용해 비교하고,
부스팅의 후속 모델로 확장하며, 원논문(Kyranou 2025) 기준선을 같은 프로토콜로 재현해 비교합니다.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `features.py`   | GREAT 데이터 적재, 슬라이딩 윈도우, 시간영역 특징 추출(Hudgins-4 / rich-14) |
| `models.py`     | 분류기 정의(교안 모델 + 후속 부스팅 + 소프트보팅) |
| `eda.py`        | 탐색적 분석(클래스 분포, 채널 상관, PCA, KMeans) |
| `evaluation.py` | 평가(종합 bake-off, 특징셋 비교, 논문 프로토콜 재현, LOSO)와 지표 |
| `main.py`       | 전체 파이프라인 실행 엔트리포인트 |

## 실행 방법

```bash
python main.py
```

- 처음 실행하면 원본 데이터에서 특징을 추출해 `features_cache.npz`로 저장하고,
  이후에는 이 캐시를 불러와 바로 실행됩니다.
- 결과는 화면에 출력되고 `results_main.json`으로 저장됩니다.

## 데이터 위치

`features.py` 상단의 `DATA_DIR`이 GREAT 데이터셋(참가자 폴더들이 있는 위치)을 가리킵니다.
기본값은 상위 폴더의 `data/extracted/data` 입니다. 데이터 위치가 다르면 이 값만 바꾸면 됩니다.

데이터셋: Kyranou, Szymaniak & Nazarpour, *EMG Dataset for Gesture Recognition with
Arm Translation* (Dryad: 10.5061/dryad.8sf7m0czv).

데이터 출처·다운로드 방법·파일 구조·실험 설계 등 상세 명세는 **`DATASET.md`** 를 참고하세요.

## 실행 환경

- Python 3.9 이상
- 필요한 패키지: `numpy`, `pandas`, `scipy`, `scikit-learn`, `h5py`

```bash
pip install numpy pandas scipy scikit-learn h5py
```

## 재현 범위

`python main.py`는 핵심 in-scope 파이프라인(탐색적 분석, 전체데이터 bake-off, 특징셋 비교,
GradientBoosting↔HistGradientBoosting 동등성, 논문 프로토콜 재현, LOSO)을 처음부터 다시 계산해
`results_main.json`으로 저장합니다. 표 2의 전체데이터 bake-off 수치(예: KNN 약 98.0%) 같은 핵심 결과가 여기서 재현됩니다.

보고서 본문의 일부 확장 표·그림은 같은 방법론을 적용한 별도 실험에서 나왔습니다. 전체데이터 표준 GradientBoosting의
학습 시간 벤치마크(약 6.7시간), 피험자별 정규화 사다리(enhanced LOSO), XGBoost 후속 계열 비교(표 5의 일부 행),
RandomForest 채널 특징중요도(그림 9)가 그 예입니다. 이 수치들은 보고서와 함께 제출한 `results.json`에 정리돼 있습니다.
즉 `results_main.json`은 이 코드의 산출물이고, `results.json`은 보고서 전체를 구동하는 전체 실험 결과로 서로 구분됩니다.

## AI 활용 고지

본 코드와 보고서 작성에 생성형 AI(Claude, Anthropic)를 코드 초안·디버깅 보조와 문서 정리에
보조 도구로 사용했습니다. 주제·방법 선정, 실험 설계, 모든 수치의 직접 실행·확인은 저자가 수행했습니다.
