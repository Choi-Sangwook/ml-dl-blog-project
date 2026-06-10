# 🌳 머신러닝 완전 입문 가이드 — DAY4. 결정트리 · 회귀트리 · 선형 회귀 · 로지스틱 회귀 · 역전파

> **시리즈**: 파이썬 기본만 있는 사람을 위한 머신러닝 입문
> **DAY1**: 머신러닝 핵심 개념과 데이터 관리 (이전 편)
> **DAY2**: 회귀 모델과 분류 모델 실습 (이전 편)
> **DAY3**: SVM · KNN · 나이브 베이즈 — 지도학습 분류 알고리즘 심화 (이전 편)
> **DAY4**: 결정트리 · 회귀트리 · 선형 회귀 · 로지스틱 회귀 · 역전파

---

## 1. 의사 결정 트리 (Decision Tree)

### 1.1 핵심 아이디어

의사 결정 트리는 데이터를 **조건(Yes/No)으로 계속 쪼개 나가는** 분류·예측 모델입니다. 나무(Tree)가 줄기에서 가지로 뻗어 나가듯, 루트 노드(Root Node)에서 시작해 결정 노드(Decision Node)를 거쳐 잎 노드(Leaf Node)에서 최종 예측을 내립니다.

```
[구직자 면접 합격 예측 — 결정트리 예시]

              [최소 급여 ≥ 55,000,000?]
               /                  \
             YES                   NO
         [한시간 이상            → 요청 거절
          출퇴근?]
         /      \
       YES        NO
  [무료음료     → 요청 거절
   제안?]
   /    \
 YES     NO
요청     요청
수락     거절

루트 노드 → 결정 노드 → 잎 노드(최종 결정)
```

**의사 결정 트리의 3가지 강점:**
1. **해석 가능성**: 사람이 모델 구조를 바로 읽을 수 있습니다
2. **범용성**: 분류·회귀 모두 적용 가능
3. **법적 설명 가능성**: 신용 평가, 의료 진단처럼 분류 근거를 사람에게 설명해야 하는 경우에 적합

> ⚠️ **sklearn 결정트리 전처리 주의사항**  
> scikit-learn의 `DecisionTreeClassifier`는 **입력값이 숫자여야** 합니다.  
> 문자열 범주형 변수는 원핫 인코딩이나 레이블 인코딩이 필요하고, 결측치(`NaN`)는 미리 채워야 합니다.  
> 다만 SVM, KNN, 로지스틱 회귀와 달리 **스케일링은 보통 필요하지 않습니다.** 트리는 특성의 크기가 아니라 분할 기준값만 보기 때문입니다.

---

### 1.2 분할 기준: 엔트로피와 정보 획득량

의사 결정 트리는 **어떤 특성으로 데이터를 쪼갤지**를 결정해야 합니다. 이때 분할 전후의 **불순도(무질서도)** 변화를 기준으로 삼습니다.

#### 엔트로피(Entropy)란?

집합이 얼마나 **무질서한지**를 0~1 사이 숫자로 표현합니다.

```
엔트로피 공식: E(S) = -Σ p_i * log₂(p_i)

- 엔트로피 = 0   → 모든 데이터가 같은 클래스 (완전히 순수)
- 엔트로피 = 1   → 두 클래스가 50:50으로 섞임 (최대 무질서)
- 0에 가까울수록 분류가 잘 된 상태
```

#### 정보 획득량(Information Gain)

분할 **전 엔트로피** - 분할 **후 엔트로피의 가중 평균** = 정보 획득량  
→ 이 값이 클수록 좋은 분할 기준입니다.

```python
import numpy as np

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0
    counts = np.bincount(labels)
    probs = counts[counts > 0] / n
    return -np.sum(probs * np.log2(probs))

def information_gain(parent, left, right):
    n = len(parent)
    weighted = (len(left)/n) * entropy(left) + (len(right)/n) * entropy(right)
    return entropy(parent) - weighted

parent = np.array([0, 0, 0, 1, 1, 1, 1])
left   = np.array([0, 0, 0])
right  = np.array([1, 1, 1, 1])

print(f"분할 전 엔트로피: {entropy(parent):.4f}")
print(f"정보 획득량:      {information_gain(parent, left, right):.4f}")
```
```
분할 전 엔트로피: 0.9852
정보 획득량:      0.9852
```

> 💡 **CART vs C5.0 알고리즘**  
> - **C4.5 / C5.0**: 엔트로피 기반 정보 획득량 사용  
> - **CART (scikit-learn 기본값)**: **지니 불순도(Gini Impurity)** 사용 → `Gini = 1 - Σ p_i²`  
>   지니는 log 계산이 없어 엔트로피보다 연산이 빠릅니다. 두 기준 모두 결과는 비슷하며, sklearn에서 `criterion='gini'`(기본)와 `criterion='entropy'` 중 선택할 수 있습니다.

---

### 1.3 가지치기(Pruning): 과적합 방지

아무 제한 없이 자라난 결정 트리는 훈련 데이터를 **완벽하게 외워버리는 과적합** 문제가 생깁니다.

| 방법 | 설명 | sklearn 적용 |
|------|------|-------------|
| **사전 가지치기 (Pre-pruning)** | 성장 중에 멈추는 조건 설정 | `max_depth`, `min_samples_split` 등 |
| **사후 가지치기 (Post-pruning)** | 크게 키운 뒤 잎 노드를 잘라냄 | `ccp_alpha` (비용 복잡도 가지치기) |

```python
from sklearn.tree import DecisionTreeClassifier

# 사전 가지치기 파라미터
model = DecisionTreeClassifier(
    max_depth=5,           # 트리 최대 깊이 (가장 중요한 파라미터)
    min_samples_split=20,  # 노드 분할에 필요한 최소 샘플 수
    min_samples_leaf=10,   # 잎 노드의 최소 샘플 수
    random_state=42
)

# 사후 가지치기: ccp_alpha 경로 확인
tree_full = DecisionTreeClassifier(random_state=42)
# path = tree_full.cost_complexity_pruning_path(X_train, y_train)
# ccp_alphas = path.ccp_alphas
# → ccp_alpha가 커질수록 트리가 더 강하게 가지치기되어 단순해집니다.
# → 교차검증으로 적절한 값을 선택할 수 있습니다.
```

> ⚠️ `max_depth`를 너무 작게 설정하면 과소적합, 너무 크게 설정하면 과적합이 됩니다. 교차검증으로 적절한 값을 찾는 것이 중요합니다.

---

### 1.4 sklearn으로 결정트리 구현하기 — 신용 위험 식별

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 신용 데이터 시뮬레이션 (1,000개 샘플, 17개 특성)
X, y = make_classification(
    n_samples=1000, n_features=17, n_informative=10,
    n_redundant=3, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 기본 모델 (과적합 위험)
dt_basic = DecisionTreeClassifier(random_state=42)
dt_basic.fit(X_train, y_train)
print(f"기본 모델  — 훈련 정확도: {dt_basic.score(X_train, y_train):.4f}")
print(f"           테스트 정확도: {dt_basic.score(X_test,  y_test):.4f}")
print(f"           트리 깊이:     {dt_basic.get_depth()}")

# ─── 가지치기 모델
dt_pruned = DecisionTreeClassifier(
    max_depth=5, min_samples_split=20, random_state=42
)
dt_pruned.fit(X_train, y_train)
print(f"\n가지치기 모델 — 훈련 정확도: {dt_pruned.score(X_train, y_train):.4f}")
print(f"              테스트 정확도: {dt_pruned.score(X_test,  y_test):.4f}")
print(f"              트리 깊이:     {dt_pruned.get_depth()}")
```
```
기본 모델  — 훈련 정확도: 1.0000
           테스트 정확도: 0.7350
           트리 깊이:     24

가지치기 모델 — 훈련 정확도: 0.8637
              테스트 정확도: 0.7850
              트리 깊이:     5
```

훈련 정확도 1.0000이지만 테스트 정확도 0.7350 → 전형적인 **과적합** 사례입니다. 가지치기 후 테스트 성능이 오히려 높아집니다.

```python
# 혼동 행렬 — 어떤 오류가 더 위험한지 파악
y_pred = dt_pruned.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('예측값'); plt.ylabel('실제값')
plt.title('결정트리 혼동 행렬')
plt.tight_layout(); plt.show()

print(classification_report(y_test, y_pred))
```

> 💡 **신용 위험 예측에서 혼동 행렬이 중요한 이유**  
> 단순 정확도(Accuracy)가 높아도 "실제 위험 고객을 안전하다고 잘못 예측(FN)"하면 은행 손실이 큽니다. 이런 문제에서는 **Recall(재현율)** 을 함께 확인해야 합니다.

```python
# 특성 중요도 시각화
importances = pd.Series(
    dt_pruned.feature_importances_,
    index=[f'Feature_{i}' for i in range(X.shape[1])]
).sort_values(ascending=False).head(8)

plt.figure(figsize=(8, 4))
importances.plot(kind='barh')
plt.title('결정트리 특성 중요도 (Top 8)')
plt.xlabel('중요도')
plt.tight_layout(); plt.show()
```

> ⚠️ **특성 중요도 해석 시 주의점**  
> 트리 기반 feature importance는 "모델이 분할에 얼마나 자주/효과적으로 사용했는가"를 나타냅니다. 값이 높다고 해서 현실에서 인과관계가 있다는 뜻은 아닙니다. 또한 연속형 변수나 값의 종류가 많은 변수에 중요도가 높게 나오는 **편향**이 생길 수 있습니다.

---

### 1.5 결정트리 단점과 앙상블로의 연결

단일 결정트리는 해석이 쉽지만 **과적합에 약하다**는 치명적 단점이 있습니다. 이 단점을 줄이기 위해 여러 트리를 함께 사용하는 **앙상블 기법**이 자주 사용됩니다.

| 방법 | 설명 |
|------|------|
| **Random Forest** | 여러 트리를 독립적으로 만들고 다수결/평균으로 예측 |
| **Boosting (AdaBoost, XGBoost)** | 이전 모델이 틀린 데이터를 다음 모델이 집중 학습 |
| **Bagging** | 데이터를 중복 샘플링해 여러 트리를 병렬 학습 |

결정트리를 제대로 이해하면 **Random Forest, XGBoost, LightGBM** 같은 강력한 모델도 쉽게 이해할 수 있습니다.

| 구분 | 내용 |
|------|------|
| **장점** | 해석·시각화 쉬움 |
| **장점** | 스케일링 불필요, 비선형 관계도 처리 |
| **단점** | 과적합되기 쉬움 |
| **단점** | 축 평행 분할에만 의존 → 대각선 패턴에 취약 |
| **단점** | 훈련 데이터의 작은 변화에 민감 |

---

## 2. 회귀 트리 (Regression Tree)

### 2.1 분류 트리 vs 회귀 트리

분류 트리는 **범주**를 예측하고, 회귀 트리는 **연속적인 숫자**를 예측합니다.

| 구분 | 분류 트리 | 회귀 트리 |
|------|-----------|-----------|
| 예측 대상 | 클래스 (0, 1, 2...) | 연속 숫자 (예: 집값 3.5만 달러) |
| 동질성 지표 | 엔트로피, 지니 불순도 | **표준편차, 분산, MAE** |
| 잎 노드 출력 | 다수결 클래스 | **해당 그룹의 평균값** |

> 💡 **회귀트리 예측값은 계단형입니다**  
> 회귀트리는 데이터를 여러 구간으로 나눈 뒤 각 구간의 평균값을 예측합니다. 그래서 선형회귀처럼 부드러운 직선이 아니라 **계단처럼 변하는** 예측값이 나옵니다.

### 2.2 모델 트리 (Model Tree) — 회귀 트리의 확장

**회귀 트리**: 잎 노드 → 평균값 하나 출력  
**모델 트리**: 잎 노드 → **선형 회귀 모델 자체**를 붙여 출력

```
[회귀 트리]              [모델 트리]
잎 노드 → 5.3            잎 노드 → y = 0.8*alcohol + 0.2*acidity + ...
(그룹 평균)              (해당 구간의 선형 회귀식)
```

데이터가 각 구간 안에서 선형 관계를 보인다면 단순 회귀 트리보다 더 좋은 성능을 낼 수 있습니다. 다만 구간이 많아질수록 복잡해져 과적합 위험도 있습니다. sklearn의 `DecisionTreeRegressor`가 회귀 트리에 해당하며, Cubist 알고리즘이 모델 트리의 대표 구현체입니다.

---

### 2.3 sklearn으로 회귀 트리 구현하기 — 캘리포니아 주택 가격 예측

> 📌 PDF에서는 화이트 와인 품질 데이터(`whitewines.csv`)를 사용해 회귀 트리를 실습합니다. 아래 예시는 실행 편의성을 위해 sklearn 내장 캘리포니아 주택 가격 데이터로 동일한 흐름을 실습합니다.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

# 캘리포니아 주택 데이터 준비 (여러 지역 특성 → 주택 가격 예측)
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 회귀 트리
rt = DecisionTreeRegressor(max_depth=5, random_state=42)
rt.fit(X_train, y_train)
y_pred_rt = rt.predict(X_test)
```

```python
# 모델 성능 비교
models = {
    '회귀 트리 (depth=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
    '랜덤 포레스트':       RandomForestRegressor(n_estimators=100, random_state=42),
    '선형 회귀':          LinearRegression()
}

print(f"{'모델':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 50)
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    print(f"{name:<22} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")
```
```
모델                   MAE       RMSE        R²
--------------------------------------------------
회귀 트리 (depth=5)   0.5123   0.7031   0.6187
랜덤 포레스트         0.3291   0.5019   0.8051
선형 회귀             0.5331   0.7254   0.6062
```

---

### 2.4 회귀 평가 지표 — MAE / MSE / RMSE / R²

| 지표 | 수식 | 특징 | 해석 방법 |
|------|------|------|----------|
| **MAE** | `Σ\|실제 - 예측\| / n` | 이상치에 덜 민감, **단위가 원래와 같음** | "평균적으로 예측이 실제값에서 X만큼 벗어남" |
| **MSE** | `Σ(실제 - 예측)² / n` | 큰 오차에 민감(제곱), 미분 가능 | 단위가 원래²여서 직관적이지 않음 |
| **RMSE** | `√MSE` | MSE와 동일하지만 단위 복원 | MAE보다 이상치에 더 민감 |
| **R²** | `1 - SSres/SStot` | 0~1 (높을수록 좋음) | "모델이 y 변화량의 X%를 설명함" |

> 💡 **MAE 해석 예시**  
> 와인 품질 예측에서 MAE = 0.50이라면, 모델의 예측 품질 점수가 실제 점수와 **평균적으로 약 0.5점 차이** 난다는 뜻입니다. 단위가 원래 변수와 같아 초심자가 해석하기 가장 쉽습니다.  
> R² = 0.62라면 "모델이 집값 변화량의 약 62%를 설명한다"는 의미입니다.

```python
mae  = mean_absolute_error(y_test, y_pred_rt)
mse  = mean_squared_error(y_test, y_pred_rt)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred_rt)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
```

---

## 3. 선형 회귀 (Linear Regression)

### 3.1 회귀의 기본 개념

회귀는 **연속적인 숫자 값을 예측**하는 문제를 다룹니다.

```
단순 선형 회귀: y = α + βx

다중 선형 회귀: y = α + β₁x₁ + β₂x₂ + ... + βᵢxᵢ + ε

- y  : 예측하려는 종속 변수
- α  : 절편 (x=0일 때 y값)
- β  : 기울기 (x가 1 증가할 때 y의 변화량)
- ε  : 오차항 (모델로 설명되지 않는 잔차)
```

### 3.2 보통 최소제곱법 (OLS: Ordinary Least Squares)

가장 좋은 직선을 찾는 기준은 **잔차 제곱합(SSE)** 을 최소화하는 것입니다.

```
SSE = Σ(실제값 - 예측값)²  → 이것을 최소로 만드는 α, β를 찾는다
```

수직 거리(잔차)가 클수록 SSE가 크게 증가(제곱 효과)하므로, 이상치에 민감하게 반응합니다.

### 3.3 선형 회귀의 5가지 가정

선형 회귀 결과를 믿으려면 데이터가 아래 가정을 만족하는지 확인하는 것이 좋습니다.

| 가정 | 의미 |
|------|------|
| **선형성** | 입력과 출력 사이 관계가 대체로 직선 형태 |
| **독립성** | 관측치들이 서로 독립 |
| **등분산성** | 오차의 퍼짐이 예측값 구간마다 비슷함 |
| **정규성** | 잔차가 대체로 정규분포를 따름 |
| **다중공선성 낮음** | 독립변수끼리 너무 강하게 중복되지 않음 |

> 💡 **다중공선성(Multicollinearity)**: 독립변수들 사이에 강한 상관 관계가 있으면 계수 추정이 불안정해집니다. VIF(분산팽창인수)로 진단하고, Ridge/Lasso 회귀로 해결할 수 있습니다.

---

### 3.4 sklearn으로 다중 선형 회귀 구현하기 — 의료비 예측

> 📌 아래 예시는 Kaggle 보험 데이터의 구조를 참고해 만든 **가상 데이터**입니다. 따라서 계수와 성능 수치는 실제 보험료 분석 결과가 아니라, 선형회귀 해석 방법을 익히기 위한 예시입니다.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 가상 보험 데이터 생성 (실제 데이터: Kaggle insurance dataset)
np.random.seed(42)
n = 1338
age      = np.random.randint(18, 65, n)
bmi      = np.random.uniform(18.5, 45.0, n)
children = np.random.randint(0, 5, n)
smoker   = np.random.choice([0, 1], n, p=[0.8, 0.2])

expense = (-11941 + 256.8 * age + 339.3 * bmi
           + 475.7 * children + 23847 * smoker
           + np.random.normal(0, 4000, n))

df = pd.DataFrame({'age': age, 'bmi': bmi,
                   'children': children, 'smoker': smoker, 'expense': expense})

X = df[['age', 'bmi', 'children', 'smoker']]
y = df['expense']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

# 계수 출력
coef_df = pd.DataFrame({'변수': X.columns, '계수(β)': lr.coef_})
print(f"절편(α): {lr.intercept_:.1f}")
print(coef_df.to_string(index=False))

y_pred = lr.predict(X_test)
print(f"\nMAE: ${mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
```
```
절편(α): -11832.7
 변수      계수(β)
  age       257.3
  bmi       340.1
  children  476.2
  smoker  23851.4

MAE: $4,021
R²:  0.7512
```

```python
# 결과 해석
print("나이 1세 증가 시 의료비:    +${:,.0f}".format(lr.coef_[0]))
print("BMI 1단위 증가 시 의료비:   +${:,.0f}".format(lr.coef_[1]))
print("자녀 1명 추가 시 의료비:    +${:,.0f}".format(lr.coef_[2]))
print("흡연자의 추가 의료비:       +${:,.0f}".format(lr.coef_[3]))
```

---

### 3.5 잔차 분석 — 모델 적합성 확인

선형회귀 후에는 **잔차(실제값 - 예측값)** 를 꼭 확인해야 합니다.

```python
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel('예측 의료비')
plt.ylabel('잔차 (실제 - 예측)')
plt.title('예측값 vs 잔차')
plt.tight_layout()
plt.show()
```

> 💡 **잔차 그래프 해석**  
> - 잔차가 0을 기준으로 **무작위로 흩어져 있으면** → 선형 모델이 비교적 적절  
> - **특정 패턴(곡선, 나팔 모양 등)이 보이면** → 비선형 관계 또는 누락된 변수가 있을 가능성

---

## 4. 로지스틱 회귀 (Logistic Regression)

### 4.1 왜 이름에 "회귀"가 붙는가?

로지스틱 회귀는 이름과 달리 **분류 알고리즘**입니다. 선형 회귀식의 출력값을 **시그모이드 함수**로 변환해 0~1 사이의 확률로 바꾼 뒤, 임계값(보통 0.5)을 기준으로 클래스를 결정합니다.

> 💡 **DAY2와의 차이**: DAY2에서는 sklearn으로 로지스틱 회귀를 간단하게 사용했습니다. DAY4에서는 **PyTorch의 `nn.Linear`와 `CrossEntropyLoss`로 동일한 아이디어를 직접 구현**하며, 이 과정이 역전파와 딥러닝 이해의 기초가 됩니다.

```
선형 회귀 출력:  z = α + β₁x₁ + β₂x₂ + ...

시그모이드:      σ(z) = 1 / (1 + e^(-z))  →  출력이 0~1 확률

분류 결정:       P(y=1) ≥ 0.5 → 클래스 1
                 P(y=1) <  0.5 → 클래스 0
```

### 4.2 이진 분류 vs 다중 클래스 분류

| 구분 | 활성함수 | 손실함수 | 예시 |
|------|----------|---------|------|
| **이진 분류** | Sigmoid | Binary Cross-Entropy | 스팸/정상 |
| **다중 클래스** | **Softmax** | **Cross-Entropy Loss** | 붓꽃 품종 3개 |

**Softmax** 는 여러 클래스에 대한 점수를 모두 합이 1이 되는 확률로 변환합니다.

```
Softmax 예시:
  raw score: [2.0, 1.0, 0.1]
  softmax:   [0.66, 0.24, 0.10]  → 합계 = 1.0 (확률처럼 해석 가능)
```

---

### 4.3 Iris 데이터 탐색 — 모델 전 반드시 확인하기

모델을 만들기 전에 입력 특성이 무엇인지, 정답 클래스가 어떤 비율로 들어 있는지 확인하는 습관이 중요합니다.

```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['target']  = iris.target
df_iris['species'] = [iris.target_names[i] for i in iris.target]

print("=== 데이터 앞 5행 ===")
print(df_iris.head())

print("\n=== 클래스별 개수 ===")
print(df_iris['species'].value_counts())

print("\n=== 기초 통계 ===")
print(df_iris.describe().round(2))
```
```
=== 클래스별 개수 ===
setosa        50
versicolor    50
virginica     50

특성: sepal length(cm), sepal width(cm), petal length(cm), petal width(cm)
라벨: 0=setosa, 1=versicolor, 2=virginica
```

---

### 4.4 PyTorch로 로지스틱 회귀 구현하기

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 재현성 고정
torch.manual_seed(42)
np.random.seed(42)

# 1. 데이터 준비
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 표준화 — fit은 훈련 데이터에만!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # transform만 (데이터 누수 방지)

# 3. Tensor 변환
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,        dtype=torch.long)   # CrossEntropyLoss는 long 요구
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,         dtype=torch.long)
```

```python
# 4. 모델 정의
class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)   # CrossEntropyLoss가 내부적으로 Softmax 처리

model = LogisticRegressionClassifier(
    input_dim=X_train.shape[1],   # 4 (꽃받침/꽃잎 길이·너비)
    output_dim=len(target_names)  # 3 (setosa, versicolor, virginica)
)
```

```python
# 5. 손실함수 & 최적화 & 학습
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

epochs = 500
loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()                          # ① 기울기 초기화
    outputs = model(X_train_tensor)                # ② 순전파
    loss = criterion(outputs, y_train_tensor)      # ③ 손실 계산
    loss.backward()                                # ④ 역전파
    optimizer.step()                               # ⑤ 가중치 업데이트
    loss_history.append(loss.item())

# 손실 그래프
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.title('학습 손실 변화 — 손실이 줄어들면 정상')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.tight_layout(); plt.show()
```

```python
# 6. 평가
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, dim=1)

y_true_names = [target_names[i] for i in y_test]
y_pred_names = [target_names[i] for i in predicted.numpy()]

print("=== 분류 리포트 ===")
print(classification_report(y_true_names, y_pred_names))

accuracy = (predicted == y_test_tensor).float().mean().item()
print(f"최종 테스트 정확도: {accuracy*100:.2f}%")
```
```
=== 분류 리포트 ===
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30

최종 테스트 정확도: 96.67%
```

```python
# 7. 혼동 행렬 시각화
cm = confusion_matrix(y_true_names, y_pred_names, labels=target_names)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('예측 품종'); plt.ylabel('실제 품종')
plt.title('Iris 분류 혼동 행렬')
plt.tight_layout(); plt.show()
```

```python
# 8. 새 데이터 예측 — 반드시 학습 때 사용한 scaler로 표준화
new_data = np.array([[5.1, 3.5, 1.4, 0.2],   # setosa 예상
                     [6.3, 3.3, 6.0, 2.5]])   # virginica 예상

new_scaled = scaler.transform(new_data)        # fit_transform ❌  transform ✅
new_tensor = torch.tensor(new_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    scores = model(new_tensor)
    _, pred_classes = torch.max(scores, dim=1)

for i, pred in enumerate(pred_classes):
    print(f"샘플 {i+1}: {target_names[pred.item()]}")
```

---

### 4.5 PyTorch 학습 루프 5단계 — 반드시 기억할 순서

```python
for epoch in range(epochs):
    model.train()                          # 학습 모드 (Dropout 등 활성화)
    optimizer.zero_grad()                  # ① 이전 기울기 초기화  ← 빠뜨리면 기울기 누적됨!
    outputs = model(X_train_tensor)        # ② 순전파 (Forward Pass)
    loss = criterion(outputs, y_labels)    # ③ 손실 계산
    loss.backward()                        # ④ 역전파 (Backward Pass)
    optimizer.step()                       # ⑤ 가중치 업데이트

model.eval()                               # 평가 모드 (Dropout 등 비활성화)
with torch.no_grad():                      # 기울기 계산 OFF → 메모리·속도 절약
    outputs = model(X_test_tensor)
```

> ⚠️ **`optimizer.zero_grad()`를 꼭 해야 하는 이유**  
> PyTorch는 기본적으로 기울기를 **누적**합니다. 초기화하지 않으면 이전 배치의 기울기가 더해져 가중치가 잘못 업데이트됩니다.

---

## 5. 역전파 (Backpropagation)

### 5.1 왜 역전파가 필요한가?

신경망은 여러 층으로 구성되어 있습니다. 출력층에서는 정답과 예측값을 직접 비교할 수 있지만, **은닉층(Hidden Layer)은 직접적인 정답이 없습니다.** 역전파는 이 문제를 해결합니다.

```
[기여도 할당 문제 — 시험 망친 원인 찾기에 비유]

오답이 나왔을 때:
  공부 시간이 부족했나?
  문제를 잘못 이해했나?
  계산 실수였나?

신경망도 마찬가지:
  출력이 틀렸을 때 → 어느 가중치가 얼마나 책임이 있는가?
```

**다층 퍼셉트론 구조:**
```
입력층           은닉층           출력층
 x₁  ─┐        ┌─ h₁ ─┐
 x₂  ─┤──→ W₁ ─┤  h₂  ├──→ W₂ ─┤─ y₁ (setosa)
 x₃  ─┤        │  h₃  │        │─ y₂ (versicolor)
 x₄  ─┘        └─ h₄ ─┘        └─ y₃ (virginica)
```

---

### 5.2 역전파 전체 흐름

역전파는 두 단계로 이루어집니다.

```
[1단계: 순전파 (Forward Pass)]
  입력 x → 은닉층 계산 → 출력층 계산 → 예측값 ŷ → 손실 L 계산

[2단계: 역전파 (Backward Pass)]
  손실 L → 출력층 기울기 계산 → 은닉층 기울기 계산 → 가중치 수정
           ← 오차가 거꾸로 흘러갑니다 ←
```

---

### 5.3 체인룰 (Chain Rule) — 직관적으로 이해하기

가중치는 손실값에 직접 연결되지 않고 여러 계산을 거칩니다. 체인룰은 이 연결고리를 하나씩 따라가며 미분합니다.

```
가중치 변화
  → 노드 입력값 변화
    → 노드 출력값 변화 (활성함수 통과)
      → 예측값 변화
        → 손실값 변화

dL/dW = (dL/dŷ) × (dŷ/dh) × (dh/dW)
```

**가중치 업데이트 요약:**

| 대상 | 오차 신호 | 업데이트 크기 결정 요소 |
|------|-----------|----------------------|
| 출력층 | `Δᵢ = (정답 - 예측) × 활성함수'` | 오차 크기, 학습률, 입력값 |
| 은닉층 | `δⱼ = 활성함수' × Σ(출력층 가중치 × Δᵢ)` | 출력층 오차 신호, 학습률, 입력값 |

---

### 5.4 활성함수와 기울기 소실 문제

가중치 업데이트 식에 **활성함수의 미분값**이 포함됩니다. 이 미분값이 작으면 오차가 앞쪽 층으로 전달되지 않아 학습이 멈춥니다.

```
[기울기 소실 과정]

출력층 오차 → 은닉층으로 전달
  → 활성함수 미분값이 작으면 (≈ 0)
    → 오차 신호가 점점 약해짐
      → 앞쪽 층 가중치가 거의 학습되지 않음
        → 깊은 신경망 학습 불가
```

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):      return 1 / (1 + np.exp(-z))
def sig_deriv(z):    s = sigmoid(z); return s * (1 - s)
def relu(z):         return np.maximum(0, z)
def relu_deriv(z):   return (z > 0).astype(float)

z = np.linspace(-6, 6, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(z, sigmoid(z),    label='sigmoid(z)')
axes[0].plot(z, sig_deriv(z),  label="sigmoid'(z) — 최대 0.25", linestyle='--')
axes[0].axhline(0.25, color='red', linestyle=':', alpha=0.5)
axes[0].set_title('시그모이드 — 기울기 소실 발생')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(z, relu(z),       label='relu(z)')
axes[1].plot(z, relu_deriv(z), label="relu'(z) — z>0에서 항상 1", linestyle='--')
axes[1].set_title('ReLU — 양수 구간에서 기울기 소실 완화')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout(); plt.show()
```

**활성함수 비교:**

| 활성함수 | 수식 | 기울기 소실 | 주의사항 | 사용처 |
|---------|------|------------|---------|--------|
| **Sigmoid** | `1/(1+e^-z)` | ⚠️ 발생 (최대 0.25) | — | 이진 분류 출력층 |
| **ReLU** | `max(0, z)` | ✅ 양수 구간에서 완화 | Dead Neuron (음수 구간 기울기=0) | **은닉층 기본값** |
| **Leaky ReLU** | `max(0.01z, z)` | ✅ 없음 | — | Dead Neuron 보완 |
| **Tanh** | `(e^z-e^-z)/(e^z+e^-z)` | ⚠️ 발생하나 Sigmoid보다 약함 | — | RNN 은닉층 |
| **Softmax** | `e^zᵢ / Σe^zⱼ` | — | 다중 클래스 전용 | 다중 분류 출력층 |

> 💡 **Dead Neuron 문제**: ReLU에서 z ≤ 0이면 기울기가 0이 되어 해당 뉴런이 영구적으로 학습되지 않는 현상입니다. **Leaky ReLU**나 **ELU**로 보완할 수 있습니다.

---

### 5.5 PyTorch에서 역전파는 한 줄

PyTorch는 **자동 미분(Autograd)** 기능으로 역전파를 자동 처리합니다. `loss.backward()` 한 줄이 체인룰 전체를 수행합니다.

```python
optimizer.zero_grad()          # 기울기 초기화
outputs = model(X_train)       # ① 순전파
loss = criterion(outputs, y)   # ② 손실 계산
loss.backward()                # ③ 역전파 — 모든 파라미터의 .grad 자동 계산
optimizer.step()               # ④ 가중치 업데이트

# 기울기 직접 확인 (디버깅용)
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
```

> 💡 **수동으로 체인룰 수식을 외울 필요는 없습니다.** PyTorch가 자동으로 계산해주기 때문입니다. 다만 "왜 `loss.backward()`가 필요한지", "기울기가 왜 작아지는지"를 이해하는 것이 중요합니다.

---

## 6. 알고리즘 비교 — 같은 데이터로 직접 비교

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    '결정 트리':      Pipeline([('m', DecisionTreeClassifier(max_depth=4, random_state=42))]),
    '로지스틱 회귀': Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=1000, random_state=42))]),
    '랜덤 포레스트': Pipeline([('m', RandomForestClassifier(n_estimators=100, random_state=42))]),
}

print(f"{'모델':<16} {'정확도':>8} {'F1(macro)':>10} {'5-Fold CV':>10}")
print("-" * 48)
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    cv  = cross_val_score(pipeline, X_train, y_train, cv=5).mean()
    print(f"{name:<16} {acc:>8.4f} {f1:>10.4f} {cv:>10.4f}")
```
```
모델             정확도  F1(macro)  5-Fold CV
------------------------------------------------
결정 트리        0.9667    0.9667     0.9500
로지스틱 회귀   1.0000    1.0000     0.9750
랜덤 포레스트   1.0000    1.0000     0.9667
```

---

## 7. 알고리즘 선택 가이드

| 상황 | 추천 | 이유 |
|------|------|------|
| 결과를 사람에게 설명해야 할 때 | **결정 트리** | 분기 조건 시각화로 설명 가능 |
| 분류 + 해석력 + 빠른 속도 | **로지스틱 회귀** | 계수가 영향력 방향·크기를 직접 제공 |
| 연속값 예측, 선형 관계 | **선형 회귀** | 계수 해석이 직관적 |
| 연속값 예측, 비선형 관계 | **회귀 트리 / 랜덤 포레스트 Regressor** | 비선형 패턴도 포착 |
| 과적합 걱정 + 고성능 | **랜덤 포레스트** | 앙상블로 분산 감소 |
| 복잡한 패턴, 대용량 데이터 | **PyTorch 다층 퍼셉트론** | 비선형, 대용량에 강함 |

```
선택 흐름

결과를 설명해야 하나?
  → YES → 결정 트리
  → NO
      ↓
숫자를 예측하나? (회귀)
  → YES → 선형 관계? → 선형 회귀
                비선형? → 회귀 트리 / RF Regressor
  → NO (분류)
      ↓
데이터가 많고 패턴이 복잡한가?
  → YES → 랜덤 포레스트 / XGBoost / 딥러닝
  → NO  → 로지스틱 회귀 (빠르고 해석 가능)
```

---

## DAY4 정리

```
✅ 결정 트리 = 조건 분기로 데이터를 쪼개는 트리 구조 분류기
   - sklearn은 숫자 입력만 처리 → 범주형은 인코딩, 결측치는 사전 처리 필요
   - 스케일링은 보통 불필요
   - 엔트로피/지니로 최적 분할 선택 (sklearn 기본값: gini)
   - max_depth, min_samples_split으로 과적합 방지
   - Random Forest, XGBoost로 단점 보완 가능

✅ 회귀 트리 = 연속값을 예측하는 결정 트리 변형
   - 잎 노드에서 그룹 평균값 출력 → 예측값이 계단형으로 변함
   - 모델 트리는 잎 노드에 선형 회귀식을 붙여 더 유연하게 예측
   - MAE, RMSE, R²로 성능 평가

✅ 선형 회귀 = 독립변수와 종속변수의 선형 관계 모델링
   - OLS(최소제곱법)으로 SSE를 최소화하는 기울기·절편 추정
   - 잔차 그래프로 모델 적합성 확인하는 습관이 중요
   - 다중공선성, 선형성 등 5가지 가정 확인 필요

✅ 로지스틱 회귀 = 선형 출력을 시그모이드/소프트맥스로 변환한 분류기
   - DAY4에서는 PyTorch로 직접 구현 (nn.Linear + CrossEntropyLoss)
   - 학습 루프 5단계: zero_grad → forward → loss → backward → step
   - 모델 전 데이터 탐색과 클래스 분포 확인 습관 필요

✅ 역전파 = 다층 신경망의 가중치를 업데이트하는 핵심 알고리즘
   - 순전파(예측) → 손실 계산 → 역전파(체인룰로 기울기 계산) → 업데이트
   - 시그모이드는 기울기 소실 문제 → 은닉층은 ReLU 사용 권장
   - ReLU도 음수 구간에서 Dead Neuron 문제 있음 → Leaky ReLU로 보완
   - PyTorch에서 loss.backward() 한 줄이 체인룰 전체를 수행

✅ 공통 핵심 습관
   - 표준화는 훈련 데이터에만 fit_transform(), 테스트는 transform()만
   - torch.manual_seed()로 재현성 고정
   - Pipeline으로 전처리 + 모델 묶어 데이터 누수 방지
   - 교차검증으로 하이퍼파라미터 선택, 테스트 데이터는 최종 평가에만
```

> 다음 단계로는 **Random Forest / XGBoost** 앙상블 심화, 또는 **은닉층이 있는 MLP(다층 퍼셉트론)** 를 PyTorch로 직접 구현하는 딥러닝 입문을 추천합니다.

---

## 🔗 참고 자료

- [scikit-learn DecisionTreeClassifier 공식 문서](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html)
- [PyTorch nn.Module 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Google Colab](https://colab.research.google.com/)
- [Kaggle — 데이터셋 & 대회](https://www.kaggle.com/)
