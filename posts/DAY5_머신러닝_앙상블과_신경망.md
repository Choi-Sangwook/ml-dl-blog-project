# 🌲 머신러닝 완전 입문 가이드 — DAY5. 앙상블 · ANN

> **시리즈**: 파이썬 기본만 있는 사람을 위한 머신러닝 입문
> **DAY1**: 머신러닝 핵심 개념과 데이터 관리 (이전 편)
> **DAY2**: 회귀 모델과 분류 모델 실습 (이전 편)
> **DAY3**: SVM · KNN · 나이브 베이즈 (이전 편)
> **DAY4**: 결정트리 · 회귀트리 · 선형 회귀 · 로지스틱 회귀 · 역전파 (이전 편)
> **DAY5**: 머신러닝 알고리즘 분류 · 앙상블 · ANN

---

## 1. 머신러닝 알고리즘 전체 지도

DAY1~4에서 배운 알고리즘들을 포함해 머신러닝의 전체 구조를 한번 정리하겠습니다.

```
머신러닝
├── 지도학습 (Supervised Learning) — 정답(Label)이 있는 데이터 학습
│   ├── 선형회귀, 로지스틱 회귀, 결정트리, SVM, KNN, 신경망
│   └── 앙상블: Random Forest, AdaBoost, Gradient Boosting,
│              XGBoost, LightGBM, CatBoost
│
├── 비지도학습 (Unsupervised Learning) — 정답 없이 패턴 발견
│   └── 클러스터링: K-Means, DBSCAN, Hierarchical Clustering ...
│
└── 차원축소 (Dimensionality Reduction) — 특성 수를 줄임
    └── PCA, LDA, t-SNE, UMAP
```

DAY5에서는 **앙상블**과 **인공 신경망(ANN)** 을 집중적으로 다룹니다.

---

## 2. 앙상블 (Ensemble Learning)

### 2.1 왜 앙상블이 필요한가?

DAY4에서 배운 결정트리는 해석이 쉽지만 **과적합에 약하고 데이터 변화에 민감**하다는 단점이 있습니다. 학습 데이터가 조금만 바뀌어도 트리 구조가 크게 달라지기 때문입니다.

앙상블은 이 문제를 **여러 모델의 의견을 종합**하는 방식으로 해결합니다.

```
[앙상블 직관]

단일 결정트리   한 명의 전문가 판단 → 실수 가능성 있음
                        ↓
앙상블          여러 전문가의 의견 종합 → 오류가 상쇄됨
```

> 💡 **실제 사례**: 독일 신용 데이터(German Credit Data)로 채무 불이행 예측 실험에서 단일 결정트리 정확도는 **73%** 였지만, AdaBoost 적용 후 **75%** 로 향상되었습니다. 작은 차이처럼 보이지만 금융처럼 위험 비용이 큰 도메인에서는 중요한 차이입니다.

---

### 2.2 앙상블의 3가지 핵심 전략

| 전략 | 대표 알고리즘 | 핵심 아이디어 |
|------|-------------|-------------|
| **Bagging** | Random Forest | 데이터를 다르게 샘플링 → 모델을 **병렬**로 학습 → 평균/다수결 |
| **Boosting** | AdaBoost, XGBoost | 이전 모델의 실수를 다음 모델이 집중 학습 → **순차** 학습 |
| **Stacking** | (메타 학습기) | 여러 모델의 예측값을 다시 새 모델의 입력으로 사용 |

**Stacking**은 Random Forest, Logistic Regression, SVM의 예측값을 모아서 최종 판단을 내리는 메타 모델을 학습시키는 방법입니다. 성능이 좋아질 수 있지만 구조가 복잡하고 데이터 누수에 주의해야 하므로, DAY5에서는 개념만 알아두면 충분합니다.

> 💡 **편향-분산 트레이드오프(Bias-Variance Tradeoff)**
> - **Bagging**은 모델의 **분산(Variance)** 을 줄입니다. 각 트리가 서로 다른 데이터로 학습하므로 한 트리의 실수가 전체에 미치는 영향이 줄어듭니다.
> - **Boosting**은 모델의 **편향(Bias)** 을 줄입니다. 어려운 데이터를 반복해서 집중 학습하므로 처음에 못 맞히던 패턴도 점점 잡아냅니다.

> ⚠️ **트리 앙상블과 스케일링**
> DAY3~4에서 SVM, KNN, 로지스틱 회귀에는 스케일링이 필수라고 배웠습니다. 반면 **Random Forest, Gradient Boosting 같은 트리 기반 앙상블은 값의 크기가 아니라 분할 기준을 사용하므로 스케일링이 거의 필요하지 않습니다.** MLP나 로지스틱 회귀와 혼합해서 쓸 때 혼동하지 않도록 주의하세요.

**공통 파라미터 정리:**

| 파라미터 | 의미 | 너무 크거나 작으면 |
|---------|------|-----------------|
| `n_estimators` | 생성할 트리 개수 | 많을수록 안정적이지만 학습이 느려짐 |
| `max_depth` | 각 트리의 최대 깊이 | 너무 크면 과적합, 너무 작으면 과소적합 |
| `learning_rate` | Boosting에서 새 모델의 기여도 | 너무 크면 학습 불안정, 너무 작으면 많은 트리 필요 |
| `random_state` | 재현성 고정 | 없으면 실행마다 결과가 달라짐 |

---

## 3. Bagging (Bootstrap Aggregating)

### 3.1 동작 원리

원본 데이터에서 **중복을 허용하면서 무작위로 샘플링(Bootstrap Sampling)** 해 여러 개의 학습 데이터를 만들고, 각 데이터로 독립적인 모델을 병렬로 학습시킵니다.

```
원본 데이터 D = {x1, x2, x3, ..., xn}

Bootstrap Sampling (중복 허용):
  D1 = {x1, x3, x3, x7, x2, ...}   → Tree 1 학습
  D2 = {x5, x1, x9, x2, x2, ...}   → Tree 2 학습
  D3 = {x4, x8, x1, x6, x3, ...}   → Tree 3 학습
  ...
  D100                               → Tree 100 학습

최종 예측 (분류): H(x) = argmax Σ I(hᵢ(x) = c)   ← 다수결

예시:
  Tree1→YES, Tree2→YES, Tree3→NO, Tree4→YES, Tree5→NO
  YES 3표 > NO 2표 → 최종 예측: YES
```

> 💡 **왜 중복을 허용하는가?**
> 중복을 허용한 복원 추출을 하면 각 데이터셋마다 포함되지 않는 약 **37%의 데이터**가 생깁니다. 이를 **OOB(Out-Of-Bag)** 샘플이라 하며, 별도 검증 세트 없이도 모델 성능을 추정하는 데 활용할 수 있습니다.

### 3.2 sklearn으로 Bagging 구현하기

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 데이터 준비
X, y = make_classification(
    n_samples=1000, n_features=17, n_informative=10,
    n_redundant=3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 단일 결정트리 (비교 기준)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print(f"단일 결정트리: {accuracy_score(y_test, dt.predict(X_test)):.4f}")

# Bagging
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)
bag.fit(X_train, y_train)
print(f"Bagging:       {accuracy_score(y_test, bag.predict(X_test)):.4f}")
```
```
단일 결정트리: 0.7350
Bagging:       0.8100
```

단일 트리 73.5% → Bagging 81.0%. **분산이 줄면서 과적합이 완화**된 효과입니다.

---

## 4. Random Forest

### 4.1 Bagging과의 차이 — 특성 무작위 선택

Random Forest는 Bagging에 한 가지를 더합니다. **각 트리를 만들 때 사용할 특성(Feature)도 무작위로 선택**합니다.

```
[일반 Bagging]
  모든 트리가 모든 특성 중 최선을 선택 → 트리들이 비슷해짐

[Random Forest]
  각 분할마다 전체 특성 중 √n개만 무작위 선택 → 트리들이 다양해짐

예: 특성이 나이·소득·신용등급·대출금액·계좌잔고·직업 (6개)
  Tree 1: 나이, 신용등급, 직업만 사용
  Tree 2: 소득, 대출금액, 계좌잔고만 사용
  Tree 3: 나이, 소득, 직업만 사용
  → 각 트리가 서로 다른 관점으로 학습 → 다양성 증가 → 성능 향상
```

```
분류 최종 예측: H(x) = mode{h₁(x), h₂(x), ..., hₘ(x)}   ← 다수결
회귀 최종 예측: H(x) = (1/M) Σ hₘ(x)                     ← 평균
```

### 4.2 sklearn으로 Random Forest 구현하기

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"Random Forest 정확도: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))
```
```
Random Forest 정확도: 0.8450

              precision    recall  f1-score   support
           0       0.84      0.86      0.85       100
           1       0.85      0.83      0.84       100
    accuracy                           0.85       200
```

```python
# OOB 점수 — 별도 검증 세트 없이 성능 추정
rf_oob = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    oob_score=True, random_state=42
)
rf_oob.fit(X_train, y_train)
print(f"OOB 점수:    {rf_oob.oob_score_:.4f}")
print(f"테스트 정확도: {accuracy_score(y_test, rf_oob.predict(X_test)):.4f}")
```
```
OOB 점수:    0.8300
테스트 정확도: 0.8450
```

> 💡 **OOB 점수의 한계**: OOB 점수는 훈련 중에 사용되지 않은 샘플로 성능을 추정하므로 편리합니다. 하지만 최종 성능 보고를 완전히 대체하지는 않습니다. 가능하면 별도 테스트 세트나 교차검증 결과도 함께 확인하는 것이 좋습니다.

```python
# 특성 중요도 시각화
import pandas as pd

importances = pd.Series(
    rf.feature_importances_,
    index=[f'Feature_{i}' for i in range(X.shape[1])]
).sort_values(ascending=False).head(8)

plt.figure(figsize=(8, 4))
importances.plot(kind='barh')
plt.title('Random Forest 특성 중요도 (Top 8)')
plt.xlabel('중요도')
plt.tight_layout(); plt.show()
```

> ⚠️ **feature importance 해석 주의**: 트리 기반 feature importance는 "모델이 분할에 얼마나 자주/효과적으로 사용했는가"를 나타냅니다. 연속형 변수나 값의 종류가 많은 변수에 중요도가 높게 나오는 **편향**이 있습니다. 인과관계를 의미하지 않으므로 중요도가 낮은 특성을 제거할 때는 교차검증으로 성능 변화를 꼭 확인하세요.

---

## 5. Boosting

### 5.1 Bagging vs Boosting — 핵심 차이

```
[Bagging]                         [Boosting]
트리들이 독립적으로 병렬 학습        트리들이 이전 결과를 보며 순차 학습
         ↓                                  ↓
결과를 단순 평균/다수결              이전 트리의 실수를 다음 트리가 집중 학습
         ↓                                  ↓
분산(Variance) 감소                 편향(Bias) 감소
```

Boosting의 학습 흐름:

```
1단계: 모든 데이터 동일 가중치로 Tree 1 학습
2단계: Tree 1이 틀린 데이터에 더 높은 가중치 부여
3단계: 높은 가중치 데이터를 집중 학습하는 Tree 2 학습
4단계: Tree 1+2가 틀린 데이터에 더 높은 가중치 부여
...반복
```

> ⚠️ **Boosting의 주의점**: Boosting은 틀린 데이터에 집중하는 특성상 **이상치(Outlier)** 에 민감할 수 있습니다. 데이터에 이상치가 많다면 Random Forest가 더 안정적입니다.

---

## 6. AdaBoost (Adaptive Boosting)

### 6.1 동작 원리

AdaBoost는 **"틀린 문제를 점점 더 중요하게 학습한다"** 는 아이디어로 동작합니다.

**① 초기 가중치 — 모든 데이터 동일**
```
wᵢ = 1/N   (N=100이면 모든 데이터 가중치 = 0.01)
```

**② 오류율 계산**
```
εₜ = Σ wᵢ · I(yᵢ ≠ hₜ(xᵢ))
```

**③ 모델 중요도 α 계산 — 잘 맞힌 모델일수록 큰 영향력**
```
αₜ = (1/2) · ln((1 - εₜ) / εₜ)

예) 오류율 0.2 → α ≈ 0.693  (영향력 큼)
    오류율 0.4 → α ≈ 0.203  (영향력 작음)
```

**④ 데이터 가중치 갱신**
```
wᵢ ← wᵢ · exp(-αₜ · yᵢ · hₜ(xᵢ))

틀린 데이터 → 가중치 증가
맞힌 데이터 → 가중치 감소
```

> 📌 위 수식은 클래스 라벨을 **-1과 +1** 로 표현할 때의 형태입니다. sklearn에서는 0과 1 라벨을 사용하지만 내부적으로 변환을 처리하므로 직접 구현할 필요는 없습니다.

**⑤ 최종 예측 — 단순 평균이 아닌 α 가중합**
```
H(x) = sign(Σ αₜ · hₜ(x))
```

### 6.2 sklearn으로 AdaBoost 구현하기

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,   # 각 모델의 기여 강도
    random_state=42
)
ada.fit(X_train, y_train)
print(f"AdaBoost 정확도: {accuracy_score(y_test, ada.predict(X_test)):.4f}")
```
```
AdaBoost 정확도: 0.8550
```

---

## 7. Gradient Boosting

### 7.1 AdaBoost와의 차이 — 잔차(Residual)를 학습

AdaBoost가 **틀린 데이터의 가중치를 높이는 방식**이라면, Gradient Boosting은 **오차(잔차) 자체를 다음 트리가 학습**합니다.

```
[Gradient Boosting 학습 흐름]

1단계: F₁(x)로 예측값 ŷ 계산
2단계: 잔차 계산
       rᵢ = yᵢ - ŷᵢ
       예) 실제값 80, 예측값 70 → 잔차 = 10

3단계: 다음 트리는 원래 정답(80)이 아니라 잔차(10)을 학습
4단계: 모델 업데이트
       Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)
5단계: 반복 → 잔차가 점점 작아짐

손실함수: L(y, F(x))
기울기:   gᵢ = -∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)
최종 모델: F(x) = Σₘ η · hₘ(x)
```

> 📌 **회귀 vs 분류에서의 차이**
> 회귀 문제에서는 "잔차(실제값 - 예측값)를 다음 트리가 학습한다"는 설명이 직관적으로 잘 맞습니다. 분류 문제에서는 엄밀히 말하면 **손실함수의 음의 기울기(의사잔차, pseudo residual)** 를 학습합니다. 초심자 단계에서는 "이전 모델이 줄이지 못한 오차 방향을 다음 모델이 보완한다"고 이해하면 충분합니다.

### 7.2 sklearn으로 Gradient Boosting 구현하기

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,   # 새 트리가 기존 모델에 10%만 영향
    max_depth=3,         # 얕은 트리 사용 — 깊을수록 과적합 위험
    random_state=42
)
gb.fit(X_train, y_train)
print(f"Gradient Boosting 정확도: {accuracy_score(y_test, gb.predict(X_test)):.4f}")
```
```
Gradient Boosting 정확도: 0.8600
```

> 💡 **`learning_rate=0.1`의 의미**: 새 트리가 기존 모델에 10%만 영향을 줍니다. 실전에서는 `learning_rate=0.05~0.1`, `n_estimators=200~500` 조합이 자주 사용됩니다. 학습률을 낮추면 더 많은 트리가 필요하지만 더 안정적으로 수렴합니다.

---

## 8. XGBoost / LightGBM — 실무의 표준 (선택 실습)

> 📌 **선택 실습**: XGBoost와 LightGBM은 scikit-learn 기본 내장 모델이 아닙니다. 실행 전에 별도 설치가 필요하며, 설치가 어렵다면 이 섹션은 개념과 표만 읽고 넘어가도 됩니다.
>
> ```bash
> pip install xgboost lightgbm
> ```

Gradient Boosting을 더욱 발전시킨 라이브러리들로, Kaggle 등 머신러닝 경진대회에서 가장 많이 우승한 알고리즘입니다.

| 라이브러리 | 특징 | 속도 |
|-----------|------|------|
| **XGBoost** | 정규화 추가, 병렬 처리, 결측치 자동 처리 | 빠름 |
| **LightGBM** | 리프 중심 트리 분할, 대용량 데이터에 강함 | **매우 빠름** |
| **CatBoost** | 범주형 변수 자동 처리, 과적합 방지 기법 | 빠름 |

> 📌 **CatBoost**는 범주형 변수가 많은 표 데이터에서 강점을 가지는 Boosting 계열 라이브러리입니다. DAY5에서는 개념만 언급하고 자세한 실습은 생략합니다.

```python
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    XGBLGBM_AVAILABLE = True
except ImportError:
    print("XGBoost/LightGBM이 설치되어 있지 않습니다.")
    print("pip install xgboost lightgbm 후 다시 실행하세요.")
    XGBLGBM_AVAILABLE = False

if XGBLGBM_AVAILABLE:
    xgb = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    xgb.fit(X_train, y_train)
    print(f"XGBoost  정확도: {accuracy_score(y_test, xgb.predict(X_test)):.4f}")

    lgbm = LGBMClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    print(f"LightGBM 정확도: {accuracy_score(y_test, lgbm.predict(X_test)):.4f}")
```
```
XGBoost  정확도: 0.8700
LightGBM 정확도: 0.8750
```

---

## 9. 앙상블 전체 성능 비교 — 트리 계열 + MLP

같은 데이터로 모든 앙상블 기법과 MLP를 직접 비교해봅시다.

> 💡 **여러 모델을 비교할 때의 원칙**: 테스트 데이터를 반복해서 보면서 모델을 고르면 테스트 데이터에 맞춘 선택이 됩니다. 실전에서는 교차검증이나 Validation set으로 모델을 선택하고, **Test set은 마지막 최종 평가에만 사용**해야 합니다.

```python
import time
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# MLP는 스케일링이 필요하므로 Pipeline으로 묶어야 합니다
models = {
    '결정트리 (단일)':    DecisionTreeClassifier(random_state=42),
    'Bagging':           BaggingClassifier(n_estimators=100, random_state=42),
    'Random Forest':     RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'AdaBoost':          AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'MLP':               Pipeline([
                             ('scaler', StandardScaler()),
                             ('mlp',   MLPClassifier(hidden_layer_sizes=(64, 32),
                                                     max_iter=500, random_state=42))
                         ]),
}

print(f"{'모델':<22} {'정확도':>8} {'F1':>8} {'학습(ms)':>10}")
print("-" * 52)
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = (time.time() - start) * 1000
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    print(f"{name:<22} {acc:>8.4f} {f1:>8.4f} {elapsed:>9.1f}ms")
```
```
모델                   정확도       F1   학습(ms)
----------------------------------------------------
결정트리 (단일)        0.7350   0.7283      1.2ms
Bagging                0.8100   0.8073     52.3ms
Random Forest          0.8450   0.8437    312.1ms
AdaBoost               0.8550   0.8541    184.7ms
Gradient Boosting      0.8600   0.8591    423.6ms
MLP                    0.8200   0.8195    312.4ms
```

> 📌 **성능 순서에 대한 주의**: 단일 결정트리보다 Bagging/Random Forest/Boosting 계열이 더 안정적인 성능을 내는 경우가 많습니다. 하지만 **어떤 모델이 가장 좋은지는 데이터 특성, 전처리, 하이퍼파라미터에 따라 달라지므로** 반드시 같은 평가 기준으로 직접 비교해야 합니다.

> 💡 **클래스 불균형 시 Accuracy만 믿으면 안 됩니다**
> 신용 위험 예측처럼 채무 불이행 고객이 소수인 문제에서는 Accuracy만 보면 위험합니다. 모든 고객을 "정상 상환"으로 예측해도 정확도가 높아 보일 수 있기 때문입니다. 이럴 때는 **Recall, Precision, F1-score를 함께 확인**해야 합니다.
>
> ```
> Precision = TP / (TP + FP)   ← 예측한 위험 고객 중 실제 위험 고객 비율
> Recall    = TP / (TP + FN)   ← 실제 위험 고객 중 모델이 잡아낸 비율
> F1-Score  = 2 × Precision × Recall / (Precision + Recall)
> ```

---

## 10. 인공 신경망 (ANN: Artificial Neural Network)

### 10.1 생물학적 뉴런 → 인공 뉴런

ANN은 **인간 뇌의 신경세포(뉴런) 구조**를 수학적으로 모델링한 것입니다.

```
[생물학적 뉴런]                  [인공 뉴런]

수상돌기(신호 입력)              입력 x₁, x₂, x₃
      ↓                               ↓
세포체(신호 누적)        가중합 Σ = w₁x₁ + w₂x₂ + w₃x₃ + b
      ↓                               ↓
임계값 도달 시 발화       활성함수 f(Σ) — 임계값 초과 시 신호 전달
      ↓                               ↓
축삭(신호 전달)                    출력 y
```

| 생물학 | 인공 신경망 |
|--------|------------|
| 수상돌기 | 입력값 xᵢ |
| 시냅스 강도 | 가중치 wᵢ |
| 세포체 | 가중합 Σ |
| 발화 여부 | 활성함수 f |
| 축삭 | 출력 y |

---

### 10.2 퍼셉트론에서 다층 신경망으로

**단층 퍼셉트론**: 입력층과 출력층만 있어 선형 분리 가능한 문제만 풀 수 있습니다.

**다층 퍼셉트론(MLP)**: 은닉층을 추가해 비선형 패턴도 학습 가능합니다.

```
[단층 퍼셉트론]         [다층 퍼셉트론 (MLP)]

입력층 → 출력층         입력층 → 은닉층₁ → 은닉층₂ → 출력층
 (선형 문제만 가능)      (비선형 패턴도 학습 가능)
```

은닉층이 2개 이상이면 **심층 신경망(Deep Neural Network)** 이라 하며, 이것이 딥러닝의 기반입니다.

---

### 10.3 네트워크 토폴로지

신경망의 학습 능력은 **구조(topology)** 에 따라 크게 달라집니다.

| 구성 요소 | 설명 |
|-----------|------|
| **입력 노드 수** | 데이터의 특성(Feature) 개수로 고정 |
| **출력 노드 수** | 예측할 클래스 수 (분류) 또는 1 (회귀) |
| **은닉 노드 수** | 설계자가 결정 — 너무 적으면 과소적합, 너무 많으면 과적합 |
| **층 수** | 깊을수록 복잡한 패턴 학습 가능, 학습 시간 증가 |

---

### 10.4 PyTorch로 MLP 구현하기 — 캘리포니아 주택 가격 예측

> 📌 PDF에서는 콘크리트 내압 강도 데이터(시멘트, 물, 골재, 양생 기간 등 8개 특성)로 ANN을 실습합니다. 입력과 출력의 관계가 단순 선형이 아닌 경우에 MLP 같은 비선형 모델을 적용해볼 수 있다는 것이 핵심입니다. 아래 예시는 별도 CSV 없이 바로 실행할 수 있도록 sklearn의 캘리포니아 주택 데이터로 같은 흐름을 실습합니다.
>
> `fetch_california_housing()`은 처음 실행할 때 데이터를 다운로드할 수 있습니다. 인터넷 연결이 안 되는 환경이라면 `load_diabetes()`나 `make_regression()`으로 대체할 수 있습니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 재현성 고정
torch.manual_seed(42)
np.random.seed(42)

# 1. 데이터 준비 — Train / Validation / Test 3분할
data = fetch_california_housing()
X, y = data.data, data.target.reshape(-1, 1)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"훈련: {len(X_train)}  검증: {len(X_val)}  테스트: {len(X_test)}")
```
```
훈련: 13209  검증: 3303  테스트: 4128
```

```python
# 2. 표준화 — fit은 훈련 데이터에만!
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train)
y_val_s   = scaler_y.transform(y_val)
y_test_s  = scaler_y.transform(y_test)

# 3. Tensor 변환
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

X_train_t = to_tensor(X_train_s); y_train_t = to_tensor(y_train_s)
X_val_t   = to_tensor(X_val_s);   y_val_t   = to_tensor(y_val_s)
X_test_t  = to_tensor(X_test_s);  y_test_t  = to_tensor(y_test_s)
```

```python
# 4. MLP 모델 정의 — Dropout으로 과적합 방지
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),   # 학습 중 20% 뉴런 무작위로 끔 → 과적합 방지
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X_train.shape[1])
print(model)
```
```
MLP(
  (net): Sequential(
    (0): Linear(in_features=8, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=64, out_features=32, bias=True)
    (4): ReLU()
    (5): Linear(in_features=32, out_features=1, bias=True)
  )
)
```

> 💡 **Dropout이란?** 학습 중 일부 뉴런을 무작위로 꺼서 모델이 특정 뉴런에 과하게 의존하지 않도록 만드는 방법입니다. 평가 시(`model.eval()`)에는 자동으로 비활성화됩니다.

```python
# 5. 학습 — Validation 손실로 모델 상태 모니터링
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 300
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience, no_improve = 20, 0   # Early Stopping

for epoch in range(epochs):
    # 학습
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

    # 검증 (가중치 업데이트 없음)
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()

    train_losses.append(loss.item())
    val_losses.append(val_loss)

    # Early Stopping: 검증 손실이 개선되지 않으면 학습 중단
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early Stopping: epoch {epoch+1}에서 학습 중단")
            break

# 손실 그래프
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='훈련 손실')
plt.plot(val_losses,   label='검증 손실', linestyle='--')
plt.title('학습/검증 손실 변화')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.legend(); plt.tight_layout(); plt.show()
```

> 💡 **Early Stopping이란?** 검증 손실이 더 이상 좋아지지 않으면 학습을 조기 종료하는 방법입니다. epoch를 너무 많이 돌리면 과적합이 발생하는데, 검증 세트에서 성능이 나빠지기 시작하는 시점을 감지해 불필요한 학습을 막습니다.

> 💡 **Validation set이 필요한 이유**: Test set은 최종 평가에만 사용해야 합니다. epoch 수나 은닉층 수 같은 하이퍼파라미터 결정은 **Validation 성능을 보며** 해야 Test set 정보가 학습에 새지 않습니다.

```python
# 6. 최종 평가 — Test set은 딱 한 번
model.eval()
with torch.no_grad():
    y_pred_s = model(X_test_t).numpy()

y_pred = scaler_y.inverse_transform(y_pred_s)   # 역표준화
y_true = scaler_y.inverse_transform(y_test_s)

mae = mean_absolute_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)
print(f"MAE: {mae:.4f}")
print(f"R²:  {r2:.4f}")
```
```
MAE: 0.4821
R²:  0.7643
```

```python
# 7. 잔차 분석
residuals = y_true - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel('예측값'); plt.ylabel('잔차 (실제 - 예측)')
plt.title('MLP 예측값 vs 잔차')
plt.tight_layout(); plt.show()
```

> 💡 잔차가 0을 기준으로 **무작위로 흩어져 있으면** 모델이 비교적 적절하게 학습된 것입니다. 패턴이 보이면 모델 구조나 특성 공학을 재검토해볼 수 있습니다.

---

### 10.5 Mini-batch 학습 (실무 방식)

현재 예시는 이해를 위해 전체 훈련 데이터를 한 번에 넣는 **Full-batch** 방식입니다. 실제 딥러닝에서는 데이터를 작은 묶음(batch)으로 나누어 학습하는 **Mini-batch** 방식을 많이 사용합니다.

```python
from torch.utils.data import TensorDataset, DataLoader

# TensorDataset으로 묶기
train_dataset = TensorDataset(X_train_t, y_train_t)

# DataLoader: 배치 단위로 데이터 자동 공급
train_loader = DataLoader(
    train_dataset,
    batch_size=64,   # 한 번에 64개씩 학습
    shuffle=True     # 에포크마다 순서 섞기
)

# Mini-batch 학습 루프
model_mb = MLP(X_train.shape[1])
optimizer_mb = optim.Adam(model_mb.parameters(), lr=0.01)

for epoch in range(50):
    model_mb.train()
    for X_batch, y_batch in train_loader:
        optimizer_mb.zero_grad()
        loss = criterion(model_mb(X_batch), y_batch)
        loss.backward()
        optimizer_mb.step()
```

> 💡 **Mini-batch의 장점**: 대용량 데이터를 한 번에 메모리에 올릴 수 없을 때 필수입니다. 또한 배치마다 기울기가 약간씩 달라져 **지역 최솟값에서 탈출**하는 데 도움이 됩니다.

---

### 10.6 경사 하강법 (Gradient Descent)

신경망 학습의 핵심은 **손실값을 줄이는 방향으로 가중치를 조금씩 수정**하는 것입니다.

```
w ← w - η · (∂L/∂w)
     ↑            ↑
  학습률       기울기(손실의 미분)
```

경사 하강법은 손실 지형이 복잡할 때 학습이 느려지거나 좋지 않은 지점에 머무를 수 있습니다. Adam, RMSProp 같은 최적화 알고리즘은 **학습률을 적응적으로 조절**해 학습을 더 안정적으로 만드는 데 도움을 줍니다.

```python
# 주요 최적화 알고리즘 비교
optimizer_sgd  = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# → 기본, 학습률 튜닝 필요. momentum으로 방향성 유지

optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
# → 학습률 자동 조절, 가장 많이 사용. 입문 시 기본값 추천

optimizer_rms  = optim.RMSprop(model.parameters(), lr=0.001)
# → 학습률 자동 조절, RNN에 자주 사용
```

---

### 10.7 ANN 장단점

| 구분 | 내용 |
|------|------|
| **장점** | 분류·회귀 모두 적용 가능 |
| **장점** | 비선형 패턴, 복잡한 관계 학습 |
| **장점** | 데이터의 근본 관계를 사전 가정하지 않음 |
| **단점** | 학습이 계산 집약적이고 느림 |
| **단점** | 과적합 위험 — 데이터가 충분해야 함 |
| **단점** | 해석이 어려운 블랙박스 모델 |
| **단점** | 층 수, 노드 수, 학습률 등 하이퍼파라미터 튜닝 필요 |

---

## 11. 알고리즘 선택 가이드

| 상황 | 추천 | 이유 |
|------|------|------|
| 빠른 프로토타이핑, 표 형태 데이터 | **Random Forest** | 튜닝 부담 적고 안정적 |
| 최고 성능 목표, 표 형태 데이터 | **XGBoost / LightGBM** | 현재 표 형태 데이터 최강자 |
| 이미지 · 음성 · 텍스트 | **딥러닝 (CNN/RNN)** | 비정형 데이터에 압도적 |
| 결과를 설명해야 할 때 | **결정트리 / 로지스틱 회귀** | 해석 가능성 우선 |
| 클래스 불균형 문제 | **어떤 모델이든 F1/Recall 함께 확인** | Accuracy만으로는 위험 |

```
선택 흐름

데이터가 이미지 / 텍스트 / 음성?
  → YES: 딥러닝 (CNN, RNN, Transformer)
  → NO (표 형태 데이터)
      ↓
결과를 설명해야 하나?
  → YES: 결정트리 / 로지스틱 회귀
  → NO
      ↓
데이터가 충분히 많은가? (수만 건+)
  → YES: XGBoost / LightGBM / MLP
  → NO: Random Forest (안정적)
```

---

## DAY5 정리

```
✅ 앙상블 = 여러 모델을 결합해 단일 모델보다 높은 성능을 얻는 방법
   - 트리 기반 앙상블은 스케일링이 불필요 (SVM, KNN, MLP와 다름)
   - 모델 선택은 교차검증/Validation — Test set은 최종 평가에만

✅ Bagging = Bootstrap Sampling(중복 허용) → 병렬 학습 → 다수결/평균
   - 분산(Variance) 감소 효과
   - OOB 점수로 간편 성능 추정 가능 (하지만 교차검증을 완전히 대체하지는 않음)

✅ Random Forest = Bagging + 랜덤 특성 선택
   - 각 분할마다 √n개 특성만 무작위 사용 → 트리 다양성 증가
   - feature importance는 편향 있음 — 해석 시 주의

✅ AdaBoost = 틀린 데이터 가중치 증가 → 다음 모델이 집중 학습
   - 잘 맞히는 모델(α 큼)이 최종 예측에 큰 영향력
   - 최종 예측: α 가중합 (단순 평균 아님)

✅ Gradient Boosting = 잔차(의사잔차)를 다음 트리가 학습
   - 회귀: 실제값-예측값(잔차) 학습
   - 분류: 손실함수의 음의 기울기(의사잔차) 학습
   - XGBoost, LightGBM이 실무 표준 (별도 설치 필요)

✅ ANN (인공 신경망)
   - 생물학적 뉴런의 수학적 모델: 가중합 + 활성함수
   - 단층 퍼셉트론 → MLP → 딥러닝
   - Train / Validation / Test 3분할이 좋은 습관
   - Dropout으로 과적합 방지, Early Stopping으로 학습 조기 종료
   - Mini-batch(DataLoader)가 실제 딥러닝 표준 방식
   - 경사 하강법으로 가중치 업데이트 (Adam 추천)
   - 블랙박스 특성 → 해석이 필요하면 앙상블이 더 적합

✅ 클래스 불균형 시 반드시 F1/Recall 확인
   - Accuracy만 보면 "모두 정상 예측"에 속을 수 있음
```

> 다음 단계로는 **은닉층이 깊은 딥러닝 MLP 심화**, 또는 이미지 분류를 위한 **CNN(합성곱 신경망)** 입문을 추천합니다.

---

## 🔗 참고 자료

- [scikit-learn Ensemble Methods 공식 문서](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost 공식 문서](https://xgboost.readthedocs.io/)
- [LightGBM 공식 문서](https://lightgbm.readthedocs.io/)
- [PyTorch nn.Sequential 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
- [PyTorch DataLoader 공식 문서](https://pytorch.org/docs/stable/data.html)
- [Google Colab](https://colab.research.google.com/)
- [Kaggle — 데이터셋 & 대회](https://www.kaggle.com/)
