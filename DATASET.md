# 데이터셋 명세 — GREAT sEMG (Arm Translation)

본 프로젝트가 사용한 공개 데이터셋의 출처, 다운로드 방법, 구조, 실험 설계를 정리한 문서입니다.
사양은 원논문(Kyranou et al., *Scientific Data*, 2025) 본문, 배포된 데이터의 `recording_parameters.txt`,
그리고 저자 GitHub 저장소(MoveR_AT_GREAT) README에서 확인했습니다.

---

## 1. 한눈에 보기

| 항목 | 내용 |
|------|------|
| 공식 명칭 | **EMG Dataset for Gesture Recognition with Arm Translation** |
| 약칭 | GREAT (저자 GitHub 저장소명 `MoveR_AT_GREAT`에서 유래) |
| 도메인 | 표면 근전도(sEMG) 기반 손동작 인식, 근전 보철·재활·웨어러블 |
| 피험자 | 건강한 성인 남성 8명, 오른손, 연속 2일 측정 |
| 신호 | 16채널 sEMG, 표본화율 2 kHz, 취득 시 20–450 Hz 밴드패스(4차) 적용 |
| 부가 데이터 | 18센서 데이터 글러브(손 운동학), 손가락 위치 데이터 |
| 손동작 | 6종 — power, lateral, tripod, pointer, open, rest |
| 팔 위치 | 3×3 격자 9개 위치 (P5 중립 기준, 45° 오프셋) |
| 시행 수 | 총 4,800 grasp trials (피험자 8 × 일 2 × 블록 2 × 위치 5 × 동작 6 × 반복 5) |
| 파일 포맷 | HDF5(신호) + CSV(라벨), 피험자/일/블록별 폴더 |
| 데이터 라이선스 | **CC0 1.0** (Dryad 공개, 자유 이용) |
| 논문 라이선스 | CC BY 4.0 |

---

## 2. 출처 및 인용

### 데이터 (Dryad 저장소)
- **DOI:** `10.5061/dryad.8sf7m0czv`
- **URL:** https://doi.org/10.5061/dryad.8sf7m0czv
- **공개일:** 2024-11-19
- **라이선스:** CC0 1.0 (저작권 포기, 출처 표기 권장)
- 인용: Kyranou, I., Szymaniak, K. & Nazarpour, K. *EMG dataset for gesture recognition with arm translation.* Dryad (2024). https://doi.org/10.5061/dryad.8sf7m0czv

### 논문 (데이터 디스크립터)
- **제목:** EMG Dataset for Gesture Recognition with Arm Translation
- **저자:** Iris Kyranou, Katarzyna Szymaniak, Kianoush Nazarpour (University of Edinburgh)
- **저널:** *Scientific Data* (2025) **12**:100
- **DOI:** `10.1038/s41597-024-04296-8`
- **URL:** https://doi.org/10.1038/s41597-024-04296-8
- **라이선스:** CC BY 4.0

### 코드 저장소 (저자 제공, 참고용)
- https://github.com/MoveR-Digital-Health-and-Care-Hub/MoveR_AT_GREAT
- 손동작 코드↔이름 매핑의 권위 있는 출처(아래 4절 참고).

---

## 3. 다운로드 방법

1. 웹 브라우저에서 https://doi.org/10.5061/dryad.8sf7m0czv 접속 (Dryad 데이터셋 페이지로 이동).
2. 페이지의 **Download dataset** 버튼으로 전체 압축본을 내려받는다(로그인 불필요, CC0).
3. 압축을 풀면 피험자별 폴더가 나온다(아래 4절 구조 참고).
4. 본 코드는 EMG 신호만 사용하므로 `emg_data.hdf5`와 `trials.csv`만 있으면 동작한다
   (`glove_data.hdf5`, `finger_data.hdf5`는 사용하지 않음).

> 참고: Dryad는 DOI가 영구 식별자다. 위 DOI 링크가 항상 최신 데이터셋 페이지로 연결된다.

---

## 4. 데이터 구조와 파일 포맷

원시 데이터는 **피험자·일·블록별 폴더**로 나뉜다. 폴더명 형식:

```
participant{X}_day{Y}_block{Z}          # X: 피험자, Y∈{1,2}: 일차, Z∈{1,2}: 블록
```

> 명칭 주의: 원논문 본문은 하루의 두 측정을 "session(세션)"으로 부르지만, 실제 배포된 폴더와
> `recording_parameters.txt`는 "block(블록)"이라는 이름을 쓴다. 이 문서는 디스크의 실제 이름(block)을 따른다.

각 폴더에 들어 있는 파일:

| 파일 | 내용 | 포맷 | 본 프로젝트 사용 |
|------|------|------|------------------|
| `emg_data.hdf5`    | 16채널 sEMG, **시행 번호로 인덱싱**(블록당 150 trial) | HDF5 | ✅ 사용 |
| `glove_data.hdf5`  | 18센서 데이터 글러브(손 운동학) | HDF5 | ✖ 미사용 |
| `finger_data.hdf5` | 다섯 손가락 위치 | HDF5 | ✖ 미사용 |
| `trials.csv`       | 라벨(손동작·목표 위치·시행 번호·블록) | CSV | ✅ 사용 |
| `recording_parameters.txt` | 취득 파라미터(필터·창 길이·채널 수 등) | TXT | 참고 |

- `emg_data.hdf5`의 각 키 = 한 시행, 값 = `(16채널, 시간)` 배열.
- `trials.csv`의 열: `row_number`(시행 번호), `target_position`(목표 위치, 1~9), `grasp`(손동작 코드, 1~6), `trial_no`, `block`.
- **손동작 코드↔이름 매핑:** `1=power, 2=lateral, 3=tripod, 4=pointer, 5=open, 6=rest`.
  - 이 매핑은 **저자 저장소(MoveR_AT_GREAT) README**를 따른다. 원논문 본문은 숫자-이름 결속을 명시하지 않고,
    본문 서술 순서와 Table 5의 순서가 서로 달라(예: pointer/tripod 순서) 1차 출처만으로는 확정할 수 없기 때문이다.
    저장소 README와 논문 Table 5는 모두 tripod(3)가 pointer(4)보다 앞선다.

---

## 5. 실험 설계 (수집 방법)

### 장비
- **sEMG:** Delsys **Trigno Quattro** 센서 4개. 각 센서 = 기준 전극 1 + sEMG 채널 4 → 총 **16채널**, **2 kHz** 표본화.
  취득 단계에서 **20–450 Hz 밴드패스(4차) 필터**가 적용되어 제공된다(`recording_parameters.txt`).
  전극은 전완에 두 줄(각 8개)로 부착, 첫 전극은 척측수근신근(extensor carpi ulnaris)에 정렬.
  피부는 70% 이소프로필 알코올로 세척 후 부착.
- **손 운동학:** 18센서 **CyberGlove II** 데이터 글러브(피험자별 1회 캘리브레이션). — 본 프로젝트 미사용.

### 손동작 (6종)
power(강한 주먹 파지), lateral(열쇠 파지), tripod(세 손가락 파지), pointer(검지 지시),
open(손 펼침), rest(휴식). 코드 순서는 4절의 매핑을 따른다.

### 팔 위치 (3×3 격자, 9개)
중립 자세 **P5**를 기준으로 팔을 상하·좌우 45°씩 옮긴 9개 위치. 예: P2 = 위로 45°, P1 = 위 45° + 좌 45°.
하루에 5개 위치만 측정하며, 두 구성으로 나뉜다:

- **Con+** (1일차): 위치 {2, 4, 5, 6, 8}
- **Conx** (2일차): 위치 {1, 3, 5, 7, 9}
- 중앙(P5)만 양일에 공유 → 위치 5의 데이터가 다른 위치의 약 2배.

### 시행(trial) 구성
한 시행 = 지시된 손동작 **5초 유지 + 3초 휴식**. 동일 (손동작, 위치) 조합을 **5회 반복**.
블록당 6동작 × 5위치 × 5반복 = **150 trial**. 하루 2블록, 총 2일 → 피험자당 600 trial,
전체 8명 → **4,800 trial**.

---

## 6. 본 프로젝트에서의 사용

- **신호:** `emg_data.hdf5`의 16채널 sEMG만 사용(운동학 데이터 제외). 데이터는 이미 20–450 Hz로
  밴드패스되어 있어 추가 필터링은 적용하지 않았다.
- **윈도잉:** 연속 신호를 **128 ms(256샘플) 윈도우 · 50 ms(100샘플) 보폭**의 슬라이딩 윈도우로 분할
  → 총 **470,413 윈도우** (4,800 시행).
- **특징:** 각 윈도우에서 채널별 시간영역 특징 추출 — Hudgins 4특징(64차원), 확장 14특징(224차원).
- **누수 차단 그룹 키:** 피험자(`g_part`), 팔 위치(`g_pos`), 일자(`g_day`), 시행(`g_trial`)을 함께 저장해
  시행/위치/사용자 단위 분할에 사용.
- 자세한 특징·전처리는 `features.py`, 분석은 보고서를 참고.

---

## 7. 로컬 배치

`features.py`의 `DATA_DIR`이 가리키는 위치(기본값 `../data/extracted/data`)에 피험자 폴더가 있으면 된다.

```
data/extracted/data/
├── participant_1/
│   ├── participant1_day1_block1/   (emg_data.hdf5, trials.csv, recording_parameters.txt, ...)
│   ├── participant1_day1_block2/
│   ├── participant1_day2_block1/
│   └── participant1_day2_block2/
├── participant_2/
│   └── ...
└── participant_8/
```

데이터 위치가 다르면 `features.py` 상단의 `DATA_DIR` 한 줄만 바꾸면 된다.

---

## 8. 윤리

원논문에 따르면 모든 피험자에게 서면·구두로 실험을 설명하고 동의를 받았으며,
헬싱키 선언 원칙을 따르고 기관 윤리 승인을 받았다(피험자 외형 공개에 대한 별도 동의 포함).
데이터는 CC0로 공개되어 출처 표기하에 자유롭게 사용·재배포할 수 있다.

---

## 참고문헌

- Kyranou, I., Szymaniak, K., & Nazarpour, K. (2025). EMG Dataset for Gesture Recognition with Arm Translation. *Scientific Data*, 12, 100. https://doi.org/10.1038/s41597-024-04296-8
- Kyranou, I., Szymaniak, K., & Nazarpour, K. (2024). *EMG dataset for gesture recognition with arm translation* [Data set]. Dryad. https://doi.org/10.5061/dryad.8sf7m0czv
