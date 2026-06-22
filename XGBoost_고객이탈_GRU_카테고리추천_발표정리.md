# XGBoost 고객 이탈 예측 및 Category-GRU 카테고리 추천 발표 정리

## 1. 프로젝트 전체 개요

본 프로젝트는 쇼핑몰 사용자의 장기 이탈 위험과 현재 세션의 관심사를 서로 다른 모델로 예측한다.

| 모델 | 해결하려는 질문 | 입력 | 출력 |
|---|---|---|---|
| XGBoost Churn | 이 사용자가 향후 7일 동안 이탈할 가능성은 얼마인가? | 사용자 누적 행동 피처 22개 | 향후 7일 이탈 확률 |
| Category-GRU | 이 사용자가 현재 세션에서 다음에 방문할 카테고리는 무엇인가? | 최근 세션 이벤트 최대 10개 | 다음 카테고리 Top-4 |

두 모델은 서로 경쟁하는 모델이 아니라 역할이 다른 보완 관계다.

```text
XGBoost → 이 사용자를 붙잡아야 하는가?
GRU      → 이 사용자에게 지금 무엇을 보여줘야 하는가?
```

최종 서비스에서는 XGBoost의 이탈 확률로 쿠폰·푸시·이메일 여부를 결정하고, GRU가 추천한 카테고리를 기반으로 노출 상품을 선택한다.

---

# 2. XGBoost 고객 이탈 예측 모델

## 2.1 모델의 목적

현재까지의 사용자 행동을 이용해 예측 기준 시점 이후 7일 동안 조회·장바구니·구매 등의 이벤트가 하나도 발생하지 않을 확률을 계산한다.

```text
예측 이후 7일 동안 이벤트 0개 → churn = 1
예측 이후 7일 안에 이벤트 존재 → churn = 0
```

현재 세션이 바로 종료되는지를 예측하는 Session Bounce와는 다른 계정 단위 장기 이탈 모델이다.

## 2.2 데이터 구성

- Train 기간: 2019년 10월~2020년 1월 25일
- OOT Test 기간: 2020년 2월
- 학습 사용자: 109,378명
- 테스트 사용자: 109,736명
- 대상 코호트: `cohort_recency7 == 1`, 최근 활동 이후 7일 이하 사용자
- 테스트 이탈 비율: 83.30%
- 사용자 ID는 식별에만 사용하며 학습 피처에는 포함하지 않는다.

과거 기간으로 학습하고 미래 기간인 2월로 평가했기 때문에 무작위 분할보다 실제 운영 상황에 가까운 시간 외 검증이다.

## 2.3 입력 피처 22개

### 최근성과 활동성

- `recency_days`: 마지막 이벤트 이후 경과 일수
- `tenure_days`: 최초 활동부터 예측 시점까지의 기간
- `ndays`: 실제 활동한 날짜 수
- `n_events`: 전체 이벤트 수
- `n_sessions`: 세션 수
- `events_per_session`: 세션당 이벤트 수

### 행동 피처

- `n_view`: 조회 횟수
- `n_cart`: 장바구니 추가 횟수
- `n_remove_from_cart`: 장바구니 삭제 횟수
- `n_purchase`: 구매 횟수
- `remove_ratio`: 장바구니 대비 삭제 비율
- `cart_purchase_ratio`: 장바구니 대비 구매 비율

### 가격과 매출 피처

- `avg_price`: 조회·행동 상품의 평균가격
- `purch_amt`: 총 구매금액
- `min_price`, `max_price`, `std_price`: 가격 분포
- `purchase_avg_price`: 평균 구매가격

### 취향 피처

- `n_categories`: 이용 카테고리 수
- `cat_entropy`: 카테고리 이용 다양성
- `n_brands`: 이용 브랜드 수
- `brand_loyalty`: 특정 브랜드에 집중된 정도

## 2.4 XGBoost를 사용한 이유

이 데이터는 사용자 한 명당 한 행으로 구성된 정형 데이터다. XGBoost는 다음과 같은 비선형 조건과 피처 상호작용을 잘 학습한다.

```text
미접속 일수가 증가하고
활동일은 적으며
조회는 많지만 구매 전환이 낮은 사용자
→ 이탈 위험 증가
```

주요 장점은 다음과 같다.

- 정형 데이터에서 안정적인 성능
- 복잡한 비선형 관계와 피처 조합 학습
- 빠른 학습과 추론
- TreeSHAP을 이용한 예측 원인 설명
- 결측치와 이상치에 비교적 강함

## 2.5 XGBoost 학습 방식

XGBoost는 딥러닝처럼 순전파·역전파로 모든 가중치를 동시에 수정하지 않는다. 여러 개의 트리를 순차적으로 추가하며 앞선 트리의 오류를 다음 트리가 보완하는 Gradient Boosting 방식이다.

```text
첫 번째 트리의 예측
→ 남은 오류 계산
→ 두 번째 트리가 오류 보완
→ 다시 남은 오류 계산
→ 총 250개 트리가 순차적으로 보완
```

따라서 딥러닝의 epoch 대신 `n_estimators`, 즉 Boosting Tree 개수가 학습 반복과 가까운 개념이다.

## 2.6 전처리

22개 원시 피처에 RobustScaler를 적용했다.

```text
Train 원시 피처
→ Train에 RobustScaler fit
→ Train과 Test에 동일한 scaler transform
→ XGBoost 입력
```

트리 모델은 원칙적으로 스케일링이 필수는 아니다. 이 프로젝트에서는 극단적인 이벤트 수와 구매금액의 영향을 완화하고, 학습·평가·서빙에서 동일한 전처리 계약을 보장하기 위해 사용했다. 저장된 `preprocessor.joblib`을 운영에서도 동일하게 사용해야 한다.

## 2.7 하이퍼파라미터 튜닝 방법

Train 데이터 내부에서 Stratified 5-Fold 교차검증을 수행했다. 각 Fold의 이탈·비이탈 비율을 유지하고, Fold별로 scaler를 별도 학습해 전처리 누수를 방지했다.

```text
Train 내부 5-Fold CV
→ RandomizedSearch
→ 평균 PR-AUC 기준 설정 선택
→ 전체 Train으로 최종 refit
→ OOT Test는 마지막에 한 번 평가
```

- CV PR-AUC: 약 0.9372
- OOT PR-AUC: 0.9361
- CV와 OOT 차이: 약 0.0011

두 값이 매우 비슷하므로 시간 외 데이터에서도 성능이 안정적으로 유지됐다고 해석할 수 있다.

## 2.8 하이퍼파라미터와 선정 이유

| 파라미터 | 값 | 의미와 선정 이유 |
|---|---:|---|
| `n_estimators` | 250 | 낮은 학습률로 조금씩 학습하기 위해 충분한 수의 트리 사용 |
| `max_depth` | 3 | 얕은 트리로 사용자별 세부 패턴 암기를 막고 일반화 강화 |
| `learning_rate` | 0.0359 | 각 트리의 영향력을 작게 제한해 과적합 억제 |
| `subsample` | 0.8 | 각 트리가 사용자 80%만 학습해 트리 다양성과 규제 효과 확보 |
| `colsample_bytree` | 1.0 | 피처가 22개로 많지 않아 각 트리에 전체 후보 피처 허용 |
| `min_child_weight` | 20 | 소수 사용자만 만족하는 지나치게 세부적인 리프 생성 억제 |
| `gamma` | 0 | 별도 분할 제약 없이 다른 규제 조건으로 복잡도 관리 |
| `reg_alpha` | 0 | L1 규제의 추가 이득이 작아 비활성화 |
| `reg_lambda` | 10 | 강한 L2 규제로 리프 예측값의 극단적 변화 억제 |
| `scale_pos_weight` | 1 | 이탈자가 이미 다수이므로 Positive class 추가 가중치 불필요 |
| `tree_method` | `hist` | Histogram 기반의 빠르고 메모리 효율적인 학습 |
| `seed` | 42 | 실험 재현성 확보 |

검토 과정에서 `n_estimators=220`, `learning_rate=0.06`, `min_child_weight=30` 후보도 확인했다. CV PR-AUC는 약 0.0004 높았지만 OOT PR-AUC는 사실상 동일했다. 현재 실제 저장·배포된 모델은 `250 / 0.0359 / min_child_weight 20` 설정이다.

## 2.9 임계값 0.43의 의미

XGBoost는 0과 1이 아니라 이탈 확률을 출력한다.

```text
churn_probability >= 0.43 → 이탈 위험 사용자
churn_probability < 0.43  → 비이탈 사용자
```

0.43은 Train 내부 5-Fold OOF 예측에서 F1이 가장 높았던 값이다. OOT Test 결과를 보고 선택하지 않았기 때문에 Test 누수를 방지했다.

임계값을 낮추면 Recall은 증가하지만 정상 사용자를 이탈자로 판단하는 False Positive가 늘어난다. 임계값을 높이면 쿠폰 대상자는 줄지만 실제 이탈자를 놓칠 수 있다.

## 2.10 성능과 지표 해석

| 지표 | 결과 |
|---|---:|
| ROC-AUC | 0.7908 |
| PR-AUC | 0.9361 |
| Precision | 0.8506 |
| Recall | 0.9929 |
| F1 | 0.9163 |
| 운영 임계값 | 0.43 |

### Recall

실제 이탈자 중 모델이 이탈자로 탐지한 비율이다.

```text
Recall = TP / (TP + FN)
       = 90,761 / (90,761 + 646)
       = 99.29%
```

실제 이탈자 100명 중 약 99명을 탐지했다는 의미다.

### Precision

모델이 이탈자라고 예측한 사용자 중 실제 이탈자의 비율이다.

```text
Precision = TP / (TP + FP)
          = 90,761 / (90,761 + 15,943)
          = 85.06%
```

이탈 위험으로 분류한 100명 중 약 85명이 실제 이탈자다.

### F1

Precision과 Recall의 조화평균이다. 한쪽만 높고 다른 쪽이 지나치게 낮은 모델을 방지한다.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### ROC-AUC

임계값 전체에서 이탈자와 비이탈자를 구분하는 순위 능력이다. 실제 이탈자 한 명과 비이탈자 한 명을 무작위로 선택했을 때 이탈자에게 더 높은 위험 점수를 줄 확률이 약 79.1%라고 해석할 수 있다.

### PR-AUC

임계값 전체에서 Precision과 Recall의 관계를 종합한 지표다. 이탈 기본 비율이 약 0.833이므로 무정보 기준선 PR-AUC도 약 0.833이다. 모델의 0.936은 기준선보다 약 0.103 높다.

## 2.11 혼동행렬과 운영 해석

|  | 실제 비이탈 | 실제 이탈 |
|---|---:|---:|
| 비이탈 예측 | TN 2,386 | FN 646 |
| 이탈 예측 | FP 15,943 | TP 90,761 |

이 모델은 실제 이탈자를 놓치는 FN을 줄이는 공격적인 리텐션 정책이다. 현재 임계값에서는 테스트 사용자의 약 97%를 이탈 위험으로 분류하므로 쿠폰 비용이 중요한 운영 환경에서는 임계값을 높이거나 상위 위험도 5~20%만 타기팅하는 정책이 필요하다.

## 2.12 주요 이탈 요인

Test TreeSHAP의 평균 절댓값 기준 중요 피처는 다음과 같다.

1. `tenure_days`
2. `ndays`
3. `recency_days`
4. `n_view`
5. `n_events`
6. `n_sessions`

단순히 미접속 일수만 보는 것이 아니라 사용 기간, 실제 활동일, 조회량, 전체 이벤트와 세션 수를 함께 사용한다.

## 2.13 실시간 서비스 연결

```text
user_id로 누적 행동 피처 22개 조회
→ preprocessor.joblib의 RobustScaler 적용
→ model.json XGBoost 추론
→ churn_probability 계산
→ threshold 또는 위험도 정책 적용
→ 쿠폰·앱 푸시·이메일 결정
```

예시 응답:

```json
{
  "user_id": 485174092,
  "churn_7d_probability": 0.78,
  "risk_level": "HIGH",
  "recommended_action": "할인 쿠폰 및 앱 푸시"
}
```

---

# 3. Category-GRU 카테고리 추천 모델

## 3.1 모델의 목적

사용자의 현재 세션에서 최근 행동 순서를 분석해 다음에 방문할 가능성이 높은 카테고리 4개를 예측한다.

```text
최근 조회·장바구니·삭제·구매 이벤트
→ Category-GRU
→ 다음 카테고리 확률
→ Top-4 카테고리 추천
```

이 모델은 사용자의 장기 선호보다 현재 세션의 단기 의도에 집중한다.

## 3.2 GRU를 사용한 이유

현재 세션에서는 이벤트 종류뿐 아니라 순서가 중요하다.

```text
상품 조회 → 상품 조회 → 장바구니
상품 조회 → 장바구니 → 장바구니 삭제
```

같은 카테고리가 포함되어도 두 행동 흐름의 관심도와 다음 행동은 다를 수 있다. GRU는 이전 이벤트 정보를 Hidden State에 누적해 행동 순서를 학습한다.

LSTM보다 구조가 단순하고 파라미터 수가 적어 짧은 세션 시퀀스에서 빠르게 학습·추론할 수 있다는 장점도 있다.

## 3.3 원본 데이터와 시간 분할

이벤트 타입:

- `view`: 상품 조회
- `cart`: 장바구니 추가
- `remove_from_cart`: 장바구니 삭제
- `purchase`: 구매

| 분할 | 기간 | 저장 표본 | 생성 가능한 Window |
|---|---|---:|---:|
| Train | 2019년 10~12월 | 300,000 | 5,276,131 |
| Validation | 2020년 1월 | 75,000 | 1,826,147 |
| Test | 2020년 2월 | 100,000 | 1,802,428 |

미래 기간을 Test로 사용해 시간 누수를 막았다. 전체 Window를 모두 메모리에 올리지 않고 Reservoir Sampling으로 표본을 추출해 CSV 앞부분에 편향되지 않게 구성했다.

- 최근 시퀀스 길이: 최대 10개 이벤트
- 학습 Window 생성 최소 이전 이벤트: 2개
- 세션별 최대 Window: 20개
- 학습 카테고리: 514개
- Padding index: 0
- Unknown index: 1
- 전체 Vocabulary: 516개

## 3.4 각 이벤트의 입력 피처

### 카테고리 피처

- 원본 `category_id`
- 학습용 정수 Index로 매핑
- Category Embedding으로 밀집 벡터 변환

### 행동·수치 피처 6개

- `is_view`
- `is_cart`
- `is_remove_from_cart`
- `is_purchase`
- `log1p_gap_seconds`
- `log1p_price`

행동 종류는 One-hot 형태로 표현한다. 시간 간격과 가격은 극단값의 영향을 줄이기 위해 `log1p`를 사용한다.

현재 Streamlit에서는 사용자가 가격과 경과 시간을 직접 입력하지 않는다.

- 가격: 카테고리 중앙가격 자동 적용
- 경과 시간: 실제 이벤트 클릭 시간 차이로 자동 계산

## 3.5 모델 구조

```text
category_id
→ Embedding 64차원

6개 수치 피처
→ Linear + ReLU
→ Numeric Projection 16차원

64차원 + 16차원
→ 이벤트당 80차원 벡터
→ GRU Hidden Size 128
→ Dropout
→ 516개 카테고리 Logit
→ Softmax
→ Top-4
```

Padding class는 추천 결과에서 제외하고, 추론 시 Unknown class도 최종 추천에서 제외한다.

## 3.6 GRU 내부 작동 방식

GRU는 현재 입력과 이전 Hidden State를 이용해 새로운 Hidden State를 만든다.

### Update Gate

과거 정보를 얼마나 유지하고 현재 정보를 얼마나 반영할지 결정한다.

### Reset Gate

과거 정보 중 현재 예측에 불필요한 부분을 얼마나 잊을지 결정한다.

```text
이전 Hidden State
+ 현재 이벤트
→ Reset/Update Gate
→ 새로운 세션 상태
```

최근 10개 이벤트를 모두 읽은 후 마지막 Hidden State가 현재 세션의 관심사를 요약한다.

## 3.7 GRU 학습 방식

GRU는 딥러닝 모델이므로 순전파와 역전파를 사용한다.

```text
순전파
최근 이벤트 입력 → GRU → 카테고리 Logit → CrossEntropy Loss

역전파
Loss → BPTT → GRU·Embedding·Linear 가중치 Gradient 계산
→ AdamW로 가중치 수정
```

시퀀스 길이가 서로 다른 샘플은 Padding을 사용하고, `pack_padded_sequence`로 실제 이벤트 길이까지만 GRU가 처리하도록 했다.

## 3.8 GRU 하이퍼파라미터와 선정 이유

| 파라미터 | 값 | 의미와 선정 이유 |
|---|---:|---|
| `sequence_length` | 10 | 최근 의도를 반영하면서 긴 시퀀스의 노이즈와 지연을 제한 |
| `embedding_dim` | 64 | 514개 카테고리 간 관계를 표현할 충분한 차원과 모델 크기의 균형 |
| `numeric_dim` | 16 | 6개 수치 피처를 카테고리 Embedding과 결합하기 위한 투영 차원 |
| `hidden_size` | 128 | 세션 행동 패턴 표현력과 실시간 추론 속도의 균형 |
| `num_layers` | 1 | 시퀀스가 최대 10으로 짧아 깊은 GRU의 추가 이득보다 과적합 위험이 큼 |
| `dropout` | 0.2 | Hidden State가 특정 패턴에 과도하게 의존하는 것을 방지 |
| `batch_size` | 512 | GPU 메모리를 활용해 안정적이고 빠르게 학습 |
| `learning_rate` | 0.001 | AdamW에서 널리 사용하는 안정적인 초기 학습률 |
| `weight_decay` | 0.0001 | 가중치가 과도하게 커지는 것을 억제하는 L2 계열 규제 |
| `epochs_max` | 15 | 충분한 학습 기회를 주되 불필요한 장기 학습 방지 |
| `patience` | 3 | Validation Hit@4가 3회 개선되지 않으면 조기 종료 |
| `gradient_clip` | 5.0 | 시퀀스 역전파에서 Gradient 폭주 방지 |
| `optimizer` | AdamW | 학습률 적응과 Weight Decay 분리를 통한 안정적 학습 |
| `seed` | 42 | 데이터와 학습 결과 재현성 확보 |

최적 Epoch는 12였다. 모델 선택 기준은 Validation Loss나 Top-1이 아니라 실제 화면의 추천 개수와 일치하는 Validation Hit@4였다.

## 3.9 왜 Top-4를 핵심 지표로 선택했는가

쇼핑몰 화면에서 추천상품을 4개 노출하기로 했기 때문이다. 따라서 첫 번째 카테고리만 정확히 맞히는 Top-1보다 추천 후보 4개 안에 실제 다음 카테고리가 포함되는지가 더 중요한 비즈니스 지표다.

## 3.10 평가 지표 의미

### Top-1 Accuracy

가장 높은 확률의 카테고리 하나가 실제 다음 카테고리와 일치한 비율이다.

### Hit@4

추천한 상위 4개 카테고리 안에 실제 다음 카테고리가 포함된 비율이다.

```text
정답이 Top-4 안에 존재 → 1
정답이 Top-4 밖에 존재 → 0
전체 평균 → Hit@4
```

### Hit@10

추천 상위 10개 안에 실제 다음 카테고리가 포함된 비율이다. 후보 생성 모델로서의 상한을 확인할 때 사용한다.

### MRR@10

실제 정답이 높은 순위에 있을수록 높은 점수를 준다.

```text
정답 1위 → 1
정답 2위 → 1/2
정답 3위 → 1/3
정답 10위 → 1/10
Top-10 밖 → 0
```

Hit@4가 정답 포함 여부만 본다면 MRR은 정답이 얼마나 앞쪽에 배치됐는지 평가한다.

### Coverage@10

전체 Test 추천 결과에서 실제로 추천된 서로 다른 카테고리 수다. 일부 인기 카테고리만 반복 추천하는 모델인지, 다양한 카테고리를 활용하는지 확인한다.

### CrossEntropy Loss

실제 정답 카테고리에 낮은 확률을 부여할수록 큰 페널티를 주는 학습 손실이다. 학습에는 사용하지만 서비스 선택 지표는 Hit@4로 설정했다.

## 3.11 Test 성능

| 지표 | Category-GRU | 마지막 카테고리+인기 기준선 | 차이 |
|---|---:|---:|---:|
| Top-1 | 62.407% | 63.416% | -1.009%p |
| Hit@4 | 76.653% | 66.788% | +9.865%p |
| Hit@10 | 83.119% | 72.163% | +10.956%p |
| MRR@10 | 0.6935 | 0.6539 | +0.0396 |
| Coverage@10 | 444개 | 393개 | +51개 |

기준선은 마지막 방문 카테고리를 첫 번째로 추천하고 나머지를 학습 데이터의 인기 카테고리로 채운다.

Top-1에서는 기준선이 약 1%p 높다. 카테고리 하나만 추천한다면 단순 기준선이 더 적합할 수 있다. 하지만 실제 서비스 조건인 4개 후보에서는 GRU가 약 9.9%p 높고 추천 다양성도 51개 많다.

발표에서는 이 내용을 숨기지 않고 다음처럼 설명한다.

> 카테고리 하나만 추천할 경우 마지막 카테고리 기준선이 조금 더 높았습니다. 하지만 저희 서비스는 상품 4개를 노출하므로 Hit@4를 핵심 지표로 선택했고, 이 조건에서는 GRU가 기준선보다 약 9.9%p 개선됐습니다.

## 3.12 실시간 추론 과정

프론트에서는 모델 가공값이 아니라 현재 사용자 이벤트만 전달한다.

```json
{
  "session_id": "session-abc123",
  "user_id": 485174092,
  "event_type": "cart",
  "product_id": 5844305,
  "category_id": 1487580006317032337,
  "event_time": "2026-06-22T18:30:15+09:00"
}
```

백엔드 처리:

```text
이벤트 수신
→ session_id별 최근 이벤트 최대 10개 저장
→ product/category catalog에서 가격 보완
→ 이벤트 시각 차이 계산
→ Category ID와 수치 피처 인코딩
→ model.pt 로드된 GRU 추론
→ Top-4 카테고리와 점수 반환
```

운영에서는 모델을 요청마다 다시 로드하지 않고 서버 시작 시 한 번 로드해 메모리에 유지해야 한다. 세션 이벤트는 Redis 같은 빠른 저장소에 유지하는 것이 적합하다.

## 3.13 세션 이벤트가 부족한 경우

학습 Window는 이전 이벤트가 최소 2개일 때 생성했다. 따라서 운영 정책은 다음이 안전하다.

```text
이벤트 0~1개 → 인기 카테고리 또는 XGBoost 장기 선호 추천
이벤트 2개 이상 → Category-GRU 추천
```

## 3.14 카테고리와 상품 추천의 차이

현재 GRU가 직접 추천하는 것은 상품이 아니라 카테고리다.

```text
Category-GRU Top-4
→ 각 카테고리에서 상품 후보 조회
→ 재고·인기도·가격·구매 이력 적용
→ 최종 상품 4개 선택
```

원본 데이터에는 상품 ID는 있지만 대부분 상품명과 이미지가 없다. 실제 쇼핑몰 화면에는 별도의 상품 마스터가 필요하다.

## 3.15 카테고리 이름 한계

- GRU 학습 카테고리: 514개
- 원본 `category_code`로 직접 확인 가능한 카테고리: 17개
- 동일 상품 ID 교차로 추정 가능한 카테고리: 2개
- 나머지는 정확한 이름 없이 숫자 ID만 존재

이름이 없는 카테고리는 임의의 실제 이름을 붙이기보다 다음과 같이 표시하는 것이 안전하다.

```text
CAT-0032
대표 브랜드: runail
중앙 가격: 2.67
상품 수: 643개
원본 category_id: 1487580005134238553
```

---

# 4. XGBoost와 GRU 비교

| 구분 | XGBoost Churn | Category-GRU |
|---|---|---|
| 목적 | 향후 7일 이탈 확률 | 현재 세션 다음 카테고리 Top-4 |
| 관찰 범위 | 사용자 장기 누적 행동 | 현재 세션 최근 행동 |
| 입력 형태 | 사용자당 22개 정형 피처 | 최대 길이 10의 이벤트 시퀀스 |
| 출력 | Binary 확률 | 514개 카테고리 확률 |
| 학습 방식 | 트리를 순차적으로 추가 | 순전파·BPTT·AdamW |
| 반복 개념 | 250개 Boosting Tree | 최대 15 Epoch |
| 핵심 지표 | PR-AUC, Recall, F1 | Hit@4, MRR, Coverage |
| 운영 활용 | 쿠폰·푸시·이메일 | 추천상품 후보 생성 |

## 4.1 두 모델을 함께 사용하는 예

```text
XGBoost 이탈 확률: 82%
GRU 추천 카테고리: A, B, C, D

→ 이탈 위험이 높으므로 리텐션 액션 필요
→ A~D 카테고리의 할인 가능 상품을 우선 노출
→ 쿠폰과 앱 푸시 전송
```

두 모델의 확률은 의미가 다르므로 직접 더하지 않는다. XGBoost는 액션 강도를 결정하고 GRU는 노출 콘텐츠를 결정한다.

---

# 5. 발표용 핵심 대본

## 5.1 XGBoost 대본

> XGBoost 모델은 사용자의 미접속 일수, 활동일, 조회·장바구니·구매 횟수와 가격·카테고리·브랜드 관련 피처 총 22개를 사용해 향후 7일 동안 추가 행동이 없을 확률을 예측합니다. 하이퍼파라미터는 Train 내부 Stratified 5-Fold 교차검증에서 PR-AUC를 기준으로 선택했습니다. 최대 깊이 3의 얕은 트리와 낮은 학습률 0.0359, 높은 L2 규제 10을 사용해 과적합을 억제했고, 250개의 트리가 앞선 트리의 오류를 순차적으로 보완합니다. 2020년 2월 OOT Test에서 PR-AUC 0.9361, ROC-AUC 0.7908을 기록했고, Train OOF에서 선택한 임계값 0.43에서는 실제 이탈자의 99.29%를 탐지했습니다.

## 5.2 GRU 대본

> Category-GRU는 현재 세션의 최근 행동 최대 10개를 순서대로 분석해 다음에 방문할 가능성이 높은 카테고리 4개를 추천합니다. 카테고리 ID는 64차원 Embedding으로 변환하고, 조회·장바구니·삭제·구매 행동과 시간 간격·가격 피처를 결합해 Hidden Size 128의 GRU에 입력했습니다. 데이터는 과거 3개월 학습, 1월 검증, 2월 테스트로 시간 분할했습니다. 서비스에서 4개의 상품을 노출하기 때문에 Validation Hit@4를 모델 선택 기준으로 사용했고, 최적 모델은 12 Epoch에서 선택됐습니다. 2월 Test Hit@4는 76.65%로 단순 기준선보다 약 9.9%p 높았습니다.

## 5.3 통합 마무리 대본

> 두 모델을 통해 고객이 떠날 가능성과 현재 원하는 콘텐츠를 함께 판단할 수 있습니다. XGBoost는 이 사용자를 붙잡아야 하는지를 판단하고, Category-GRU는 어떤 카테고리를 보여줘야 하는지를 판단합니다. 따라서 이탈 위험이 높은 사용자에게 현재 관심사에 맞는 상품과 쿠폰을 제공하는 개인화 리텐션 시스템으로 연결할 수 있습니다.

---

# 6. 예상 질문과 답변

## 왜 XGBoost 임계값이 0.5가 아니라 0.43인가?

Train 내부 5-Fold OOF 예측에서 F1이 가장 높은 임계값을 선택했다. Test 데이터로 임계값을 고르지 않아 평가 누수를 막았다.

## 왜 Accuracy를 강조하지 않는가?

테스트 이탈 비율이 약 83%로 높아 모든 사용자를 이탈자로 예측해도 Accuracy가 높게 나올 수 있다. 따라서 이탈 순위 능력을 보는 PR-AUC와 ROC-AUC, 이탈자를 놓치지 않는 Recall을 함께 사용했다.

## Recall이 99%인데 완벽한 모델인가?

아니다. 실제 이탈자를 거의 놓치지 않지만 False Positive가 15,943명 발생했다. 현재 임계값은 이탈자를 놓치지 않는 정책이며 쿠폰 비용이 중요하면 임계값을 높이거나 상위 위험 사용자만 타기팅해야 한다.

## XGBoost에 Epoch가 있는가?

딥러닝 Epoch는 없다. `n_estimators=250`개의 트리를 순차적으로 추가하며 앞선 트리의 오류를 보완한다.

## 왜 GRU인가?

현재 세션은 단순 행동 횟수보다 조회·장바구니·삭제가 발생한 순서가 중요하다. GRU는 Hidden State로 이전 행동 정보를 유지하고 불필요한 정보는 Gate로 조절한다.

## GRU와 LSTM의 차이는 무엇인가?

LSTM은 Input·Forget·Output Gate와 Cell State를 사용한다. GRU는 Reset·Update Gate로 구조가 더 단순하고 파라미터가 적다. 시퀀스가 최대 10으로 짧은 현재 문제에서는 GRU가 충분한 표현력과 빠른 추론 속도를 제공한다.

## GRU Top-1이 기준선보다 낮은데 왜 사용하는가?

서비스가 카테고리 하나가 아니라 추천상품 4개를 노출하기 때문이다. 핵심 지표인 Hit@4에서 GRU는 76.65%로 기준선보다 약 9.9%p 높다.

## GRU가 직접 상품을 추천하는가?

현재는 다음 카테고리 Top-4를 추천한다. 실제 상품은 각 카테고리에서 재고·인기도·가격·구매 이력을 적용해 별도로 선정한다.

## 이벤트가 하나뿐인 신규 세션은 어떻게 처리하는가?

학습은 이전 이벤트가 최소 2개인 Window를 사용했다. 이벤트가 부족하면 인기 카테고리 또는 사용자 누적 피처 기반 추천을 사용하고, 2개 이상부터 GRU를 적용한다.

## XGBoost와 GRU를 하나의 모델로 합친 것인가?

아니다. 예측 목적과 데이터 단위가 다른 독립 모델이다. 백엔드 정책 계층에서 XGBoost의 이탈 위험과 GRU의 추천 카테고리를 결합한다.

---

# 7. 관련 파일

## XGBoost Churn

```text
models/churn/xgboost/model.json
models/churn/xgboost/model_config.json
models/churn/xgboost/feature_schema.json
models/churn/xgboost/preprocessor.joblib
src/models/churn/xgboost_trainer.py
data/processed/evaluation/churn/xgboost/metrics_summary.json
data/processed/evaluation/churn/xgboost/model_run_manifest.json
data/processed/evaluation/churn/xgboost/shap_summary.json
```

## Category-GRU

```text
models/next_category/gru/model.pt
models/next_category/gru/model_config.json
models/next_category/gru/feature_schema.json
models/next_category/gru/category_index_map.json
src/models/next_category/gru_model.py
src/models/next_category/gru_trainer.py
src/models/next_category/gru_inference.py
streamlit_next_category.py
data/processed/evaluation/next_category/gru/metrics_summary.json
data/processed/evaluation/next_category/gru/training_history.json
```
