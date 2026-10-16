# LPDS / WLACP 리버털 실험 코드 (IEEE Access, Access-2026-46956)

제출본 원고의 프로토콜(데이터셋 9개, 60/15/15 분할과 상한 6000/1000/800, 표 2 하이퍼파라미터, 시드 {42, 7, 13, 99, 2024})을 그대로 재현하고, 두 리뷰어가 요청한 실험을 모두 추가한 독립 실행형 코드입니다. 데이터는 Hugging Face의 `inria-soda/tabular-benchmark`(reg_num)에서 자동으로 내려받아 `data/`에 캐시합니다.

## 1. 설치와 실행

```bash
pip install -r requirements.txt
# 스모크 테스트 (2개 데이터셋, 작은 분할, 수 분)
QUICK=1 N_JOBS=4 bash run_all.sh
# 본 실행 (논문의 9개 데이터셋 x 5시드)
N_JOBS=16 bash run_all.sh
# 벤치마크의 회귀 과제 19개 전체로 확장 (선택 C1)
N_JOBS=16 DATASETS=all OUT=results_all bash run_all.sh
```

- 모든 스크립트는 `(데이터셋, 시드)` 단위 작업으로 나뉘고, 결과는 `results/<실험>/<데이터셋>_s<시드>.json`(+ `.npz`)로 저장됩니다. 이미 있는 결과는 건너뛰므로 중단 후 다시 실행하면 이어서 돕니다. 다시 계산하려면 `--overwrite`를 붙입니다.
- 실험 하나만 돌릴 때는 `python -m experiments.e02_conformal --datasets main --n_jobs 16` 처럼 실행합니다. `--datasets`에는 `main`, `all`, `extra`, `ext6` 또는 쉼표로 구분한 이름을 줄 수 있습니다.
- 작업 하나는 단일 스레드로 돌도록 설정되어 있으므로 `N_JOBS`는 물리 코어 수로 두면 됩니다. 실패한 작업은 `.err` 파일에 traceback이 남습니다.
- 집계는 `python -m analysis.a01_tables` (표), `python -m analysis.a02_stats` (데이터셋 단위 통계)가 `results/tables/`에 CSV와 Markdown으로 씁니다.
- `results/env.json`에 하드웨어와 라이브러리 버전이 기록됩니다(R1-Q3의 재현성 정보).

## 2. 실험과 리뷰 코멘트 대응

| 스크립트 | 내용 | 대응 코멘트 |
|---|---|---|
| `e01_ranking` | 오차 순위: Spearman, PRR(MAE·MSE), rejection curve(oracle·random 기준선 포함). 비교 점수: LPDS, 저장 카운트 LPDS, 트리 기여 가중 LPDS(Var[h_t], \|leaf value\|), IsoForest, AA-Range, Mahalanobis, kNN 거리, LOF, 리프 내 학습 타깃 분산, \|ŷ\|, virtual ensemble, IBUG, 보정 잔차에 적합한 오차 모델(±LPDS 특징). 모델 정확도(R², RMSE), active-feature fraction, LPDS 사분위별 MAE, PC1 3σ 이동 AUROC, LPDS+range check 결합 점수 | R1-Q1, R2-1, R2-2(a)(b)(c), R2-4, R2-7(a)(f)(g), R2-8(a), minor 2 |
| `e02_conformal` | 재적합 없는 구간 비교: Split(보정/비보정 분위수), 제출본 WLACP, test-point mass만 넣은 WLACP, **수준 보정 WLACP(LCP)** 4개 커널(LA, 트리별 정규화 LA, LA², LA⁴)과 검증셋 자동 선택, RBF 커널 LCP, Mondrian(LPDS 사분위), normalized CP(LPDS·오차 모델·IBUG scale), IBUG 가우시안 구간, LoBoost. 지표: marginal coverage, 사분위별 coverage·폭, 제출본 정의의 하위 25%, kNN 거리 기준 희소 부분집단, worst-slab coverage, interval score, ESS, 무한 구간 비율 | R1-Q2, R2-5(a)(b)(c), R2-6(a)(b) |
| `e03_extended` | 재적합 참조 방법: NGBoost(100트리, early stopping), CV+, CQR. 구현 비교: sklearn, XGBoost, LightGBM(leaf-wise, 깊이 제한 없음), LightGBM(max_depth=4). 9개 데이터셋 x 5시드로 메인 표와 시드 통일 | R2-6(b)(e), R2-8(c)(d), minor 5·6·7·8 |
| `e04_sensitivity` | 트리 수, 깊이, 최소 리프 크기, 학습률의 one-at-a-time 변화, LightGBM num_leaves, 검증셋 random search로 튜닝한 LightGBM·XGBoost·sklearn 모델 | R2-7(a) |
| `e05_shift` | 이동 크기 0.5–300σ, 방향 PC1·PC2·PC3·Diag(equal-displacement), 전체 특징 / 연속 특징만 이동, 세 구현의 LPDS AUROC(그림 3 재생성용), LPDS 포화 확인과 결합 점수 | R2-4, R2-7(c), R2-8(c) |
| `e06_calib_size` | n_cal 100–2000, 5시드 평균±SD, 전체·하위 사분위 coverage, 폭 비, max LA, ESS | R2-6(d), 표 10·11 |
| `e07_noise` | Bike_Sharing: 원래 타깃, 잔차 순열(이분산 구조 제거), 가우시안 노이즈 5단계. 합성 데이터: 지지가 조밀한 곳/희소한 곳/균일하게 노이즈를 넣은 통제 실험 | R2-7(e), Section IV-C |
| `e08_ring` | 생성 규칙을 명시한 2D ring 벤치마크(그림 4 재생성용 좌표 저장) | R1-Q3, R2-7(e) |
| `e09_timing` | N에 따른 추론 시간(LPDS, IBUG, kNN, LOF, IsoForest, Mahalanobis), n_cal에 따른 WLACP·LCP·Mondrian·LoBoost 시간 | R1-Q1, R2-1, 표 12–14 |
| `e10_coverage_check` | 수준 보정 WLACP의 유한표본 marginal coverage 몬테카를로 확인 | R1-Q2, R2-5(a) |
| `a02_stats` | 데이터셋 단위 Wilcoxon(사전 지정 단측·양측, Holm 보정), 계층 bootstrap CI, 혼합 모형, 데이터셋별 Spearman bootstrap CI | R2-3 |

## 3. 수준 보정 WLACP (원고의 새 명제)

`core/conformal.py`의 `wlacp_lcp`가 구현입니다. 보정점과 테스트점을 합친 n+1개 점마다

  S_i = Σ_j H(X_i, X_j) 1[V_j < V_i] / Σ_j H(X_i, X_j)   (j = 1..n+1, V는 절대 잔차)

를 계산하고, K = ⌈(1−α)(n+1)⌉일 때 S_{n+1}(y)가 K번째로 작은 S보다 크지 않은 y를 구간에 넣습니다. 이는 Guan(Biometrika, 2023)의 LCP에서 수준 α̃를 "#{i : V_i ≤ Q(α̃; F̂_i)} ≥ (1−α)(n+1)을 만족하는 가장 작은 값"으로 고르는 절차와 같습니다.

- **명제.** 앙상블을 보정·테스트 데이터와 독립인 학습 분할로 적합하고 (X_i, Y_i)가 교환 가능하면, H가 LA, 트리별 정규화 LA, LA^γ 중 무엇이든 P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1−α.
- **증명 요지.** 세 커널은 학습 분할로 적합한 트리에만 의존하는 대칭 함수이므로 보정·테스트 데이터에 대해 고정입니다. 따라서 (S_1, …, S_{n+1})은 교환 가능한 표본의 대칭 함수이고, S_{n+1}의 순위가 K 이하일 확률은 K/(n+1) ≥ 1−α 이상입니다.
- 구간은 인접한 보정 잔차 사이의 모든 구간에서 정확히 평가하고, 받아들여진 값의 상한을 반폭으로 씁니다(볼록 껍질이므로 coverage는 줄지 않음). 위에서부터 블록 단위로 훑어 처음 받아들여지는 슬롯에서 멈추므로, 전체 계산과 결과가 같으면서(무작위 30건에서 일치 확인) 대체로 상위 α 비율의 슬롯만 확인합니다.
- 제출본 WLACP(`wlacp_submitted`)는 테스트점 질량도 수준 보정도 없어 보장이 없습니다. `e10`에서 보정셋이 작을 때 coverage가 0.90 아래로 내려가는 것을 확인할 수 있습니다.
- 테스트점 질량만 넣고 수준 보정을 하지 않으면(`wlacp_testmass`) 테스트점 가중치 H(x*, x*)가 다른 점보다 커서 지나치게 보수적인 구간(무한 구간 포함)이 나옵니다. 리뷰어 2의 5(a)에 대한 답변에서 "테스트점 질량만으로는 충분하지 않고 수준 보정이 필요하다"는 근거로 쓸 수 있습니다.

## 4. 제출본 프로토콜과 달라진 점 (원래 코드와 대조 필요)

원래 실험 코드가 없어 논문 서술만으로 재현했습니다. 아래 항목은 원래 구현과 다를 수 있으니 확인 후 맞추거나, 개정 원고에 새 설정으로 명시해야 합니다. 이 코드로 메인 표까지 모두 다시 생성하면 원고 전체가 내부적으로 일관됩니다.

1. 60/15/15 분할에서 남던 10%를 검증셋(상한 1000)으로 명시했습니다. IBUG k, LoBoost 영역 크기, normalized CP 지수, RBF 대역폭, WLACP 커널 선택, LPDS 사분위 임계값이 모두 여기서 정해지므로 보정·테스트 데이터와 독립입니다(R2-7(b)의 답).
2. 외삽 집합은 검증셋 점을 이동해 만듭니다(제출본은 "held-out test points").
3. 하위 25% 부분집단은 검증셋 LPDS 사분위 임계값으로 정의한 `q1`을 기본으로 하고, 제출본 정의(테스트셋 LPDS 하위 25%)는 `low_testq`로 함께 보고합니다(minor 3의 답).
4. LightGBM은 `subsample_freq=1`로 bagging을 실제로 켰고 `max_depth=-1`(제한 없음)입니다. 깊이 4로 제한한 `lgbm_d4`도 함께 돌립니다(minor 5).
5. XGBoost는 `tree_method="hist"`, `min_child_weight=20`입니다.
6. 거리 기반 점수(IsoForest, Mahalanobis, kNN, LOF)는 학습셋으로 표준화한 특징에서 계산합니다. AA-Range는 원래 단위입니다.
7. CrossConf 대신 정확한 CV+(5-fold)를 씁니다(minor 8).
8. Ring 벤치마크와 noise sweep은 생성 규칙을 새로 정했습니다(`e08`, `e07` 문서 문자열). 원래 생성 코드가 있으면 상수를 맞춰 주세요.
9. IBUG와 LoBoost는 원 논문 설명대로 재구현했습니다. IBUG 공식 패키지(github.com/jjbrophy47/ibug)는 PyPI 배포판이 빌드되지 않아 내장 구현을 기본으로 합니다. LoBoost는 arXiv v1에 코드 링크가 비어 있어 Algorithm 1 서술을 따랐습니다(트리 가중치 Var[h_t], 영역 최소 크기는 검증셋으로 선택).

## 5. 수행 시간

아래 시간은 이 코드를 개발한 컨테이너(2코어, 단일 스레드 작업)에서 실제 크기로 작업 1건씩 잰 값입니다. 데스크톱·서버 CPU에서는 보통 1.5–2배 빠릅니다.

측정 환경: Intel Xeon 2.1 GHz 클라우드 vCPU, 작업 하나당 1코어. 9개 데이터셋 모두 시드 42로 1건씩 실측했고(e04는 Superconduct와 Bike_Sharing만 실측, 나머지는 특징 수에 비례해 보간), 5시드 합계는 그 5배입니다.

| 실험 | 작업 수 (9×5) | 작업 1건 (범위) | 합계 (코어 시간) |
|---|---|---|---|
| e01 순위 | 45 | 3–5초, Superconduct 27초 | 약 5분 |
| e02 conformal | 45 | 16–19초, Superconduct 41초 | 약 15분 |
| e03 재적합 참조·구현 비교 | 45 | 20–34초, Superconduct 148초 | 약 30분 |
| e04 민감도·튜닝 | 45 | Bike 2.6분, Superconduct 25분 (튜닝이 대부분) | 약 5.7시간 |
| e05 이동 크기 | 45 | 4–12초, Superconduct 23초 | 약 7분 |
| e06 보정셋 크기 | 45 | 5–12초, Superconduct 23초 | 약 8분 |
| e07 노이즈 | 10 | Bike 7초, 합성 33초 | 약 4분 |
| e08 ring | 5 | 1초 미만 | 수 초 |
| e09 시간 측정 | 1 | 약 1.5분 (다른 작업 없이 단독 실행) | 약 1.5분 |
| e10 coverage 검증 | 1 | 약 2분 (2000회 반복) | 약 2분 |
| a01·a02 집계·통계 | 1 | 약 2분 (bootstrap 1000회) | 약 2분 |
| **합계** | | | **약 7코어 시간 (e04 제외 시 약 1.2코어 시간)** |

- 16개 작업을 병렬로 돌리면 전체가 약 40분–1시간에 끝납니다. 가장 긴 단일 작업이 e04의 Superconduct(약 25분)이므로 코어를 더 늘려도 이보다 빨라지지는 않습니다.
- 2코어 노트북급이면 약 3.5–4시간입니다. e04를 빼고 먼저 돌리면 약 40분입니다.
- 19개 데이터셋 전체(DATASETS=all)는 대략 2배(약 14코어 시간)입니다. 추가 데이터셋은 특징 수가 3–42개로 Superconduct보다 작아 작업당 시간은 중간 데이터셋 수준입니다.
- e04가 부담되면 `--trials_sklearn 5 --trials_xgb 15 --trials_lgbm 15`로 튜닝 횟수를 줄이면 약 절반이 됩니다. Superconduct 기준 sklearn 튜닝 10회가 573초, XGBoost 30회가 410초, LightGBM 30회가 245초였습니다.

## 6. 결과

이 코드로 9개 데이터셋 × 5시드 전체와 19개 데이터셋(e01, e02, e03)을 실제로 돌린 결과는 별도 파일 `lpds_rebuttal_results.zip`에 있습니다. `results/RESULTS_SUMMARY.md`가 요약이고, `python -m analysis.a03_key_numbers --results results --results_all results_all`로 같은 표를 다시 만들 수 있습니다.

## 7. 원고 그림·표·수치 생성

- `python -m analysis.a04_paper_numbers --results results --results_all results_all` : 표 1–3 이외에 원고 본문에 쓰인 수치(저장 카운트 일치도, 데이터셋별 bootstrap CI 개수, 20% 거부 시 MAE, WLACP 대 모든 구간 방법의 데이터셋 단위 검정 등)를 `results/tables/paper_numbers.json`과 `wlacp_vs_all.csv`로 저장합니다.
- `python -m analysis.f01_figures --results results --results_all results_all --out figures` : 개정 원고의 그림 10개(600 dpi PNG)와 캡션(.txt)을 만듭니다. 원고의 캡션은 이 .txt와 같은 문장입니다.
- `python -m analysis.a05_latex_tables --out <원고 폴더>/tables` : 개정 원고의 LaTeX 표 9개를 결과 레코드에서 바로 생성합니다(표 2 설정표만 원고에 직접 기술).
- 논문에서 LPDS는 LPM(leaf population mass), `WLACP_LCP_la_norm_g1`은 WLACP(기본 커널), `WLACP_submitted`는 weighted quantile로 표기합니다.
