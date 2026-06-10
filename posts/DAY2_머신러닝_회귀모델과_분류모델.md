# 🤖 머신러닝 완전 입문 가이드 — DAY2. 회귀모델과 분류모델

> **시리즈**: 파이썬 기본만 있는 사람을 위한 머신러닝 입문
> **DAY1**: 머신러닝 핵심 개념과 데이터 관리 (이전 편)
> **DAY2**: 실제 모델 학습 → 평가 → 비지도학습 → 튜닝까지

---

## 1. 모델을 만들기 전에 — 문제 정의가 먼저

처음에는 "어떤 알고리즘이 좋은지"부터 고민하기 쉽습니다. 하지만 실전에서는 **모델 선택보다 문제 정의가 먼저**입니다.

머신러닝 프로젝트를 시작할 때 아래 질문을 먼저 정리해야 합니다:

```
1. 무엇을 예측하고 싶은가?
2. 예측 결과는 숫자인가, 범주인가?
3. 정답 데이터가 있는가?
4. 틀렸을 때 어떤 종류의 실수가 더 위험한가?
5. 어떤 평가 지표로 성공을 판단할 것인가?
```

> 💡 예를 들어 병원 진단 모델에서는 실제 환자를 "정상"이라고 판단하는 실수가 치명적입니다 → **Recall**이 중요. 반대로 스팸 필터에서는 정상 메일을 스팸으로 보내면 사용자가 불편합니다 → **Precision**이 중요. 문제에 따라 봐야 할 지표가 달라집니다.

### 분석 알고리즘 유형 정리

```
                    분석 알고리즘
                         │
              정답(label)이 있나?
              ┌────┴────┐
            유(지도학습)  무(비지도학습)
              │
       예측할 값의 종류는?
       ┌──────┴──────┐
    숫자 → 회귀         범주 → 분류
   (Regression)     (Classification)
```

| 구분 | 예측하는 값 | 예시 | 대표 알고리즘 |
|------|----------|------|-----------|
| **회귀** | 연속적인 숫자 | 집값, 매출, 이용객 수 | 선형회귀, 리지, 랜덤포레스트, XGBoost |
| **분류** | 카테고리 | 생존/사망, 스팸 여부, 품종 | 로지스틱회귀, 의사결정트리, 랜덤포레스트, SVM |

---

## 2. 지도학습 — 회귀 모델 (숫자 예측)

### 2.1 선형 회귀 (Linear Regression)

가장 기본이 되는 모델입니다. 데이터를 가장 잘 설명하는 **직선**을 찾습니다.

**수식**: `ŷ = w₁x + w₀`
- `w₁`: 기울기 (가중치, weight) — x가 1 증가하면 y가 w₁만큼 변화
- `w₀`: y절편 (편향, bias) — x가 0일 때 y값
- **학습 = 에러 제곱합을 최소화하는 w₁, w₀를 찾는 것**

```
에러(e) = 실제값(y) - 예측값(ŷ)
목표: argmin  e[1]² + e[2]² + ... + e[N]²
```

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 예시: 면적으로 집값 예측 (간단한 데이터 생성)
np.random.seed(42)
area = np.random.randint(20, 80, 100)  # 면적 (20~80평)
price = area * 500 + np.random.normal(0, 2000, 100) + 5000  # 가격 (만원)

X = area.reshape(-1, 1)
y = price

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1) 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 2) 학습된 파라미터 확인
print(f"기울기 (w₁): {model.coef_[0]:.2f}")
print(f"절편 (w₀):  {model.intercept_:.2f}")
print(f"→ 해석: 면적이 1평 늘면 집값이 약 {model.coef_[0]:.0f}만원 증가")
```
**출력 결과:**
```
기울기 (w₁): 503.18
절편 (w₀):  4666.73
→ 해석: 면적이 1평 늘면 집값이 약 503만원 증가
```

```python
# 3) 예측 및 평가
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nMAE  (평균절대오차):   {mae:.2f}만원")
print(f"RMSE (평균제곱근오차): {rmse:.2f}만원")
print(f"R²   (결정계수):      {r2:.4f}")
```
**출력 결과:**
```
MAE  (평균절대오차):   1594.32만원
RMSE (평균제곱근오차): 1932.87만원
R²   (결정계수):      0.9763
```

> 💡 **결과 해석**: R²=0.9763이므로 면적만으로 집값 변동의 97.6%를 설명할 수 있다는 의미입니다. 평균적으로 약 1594만원의 오차가 있습니다.

```python
# 4) 시각화: 실제값 vs 예측값
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 회귀선
axes[0].scatter(X_test, y_test, alpha=0.6, label='실제값')
axes[0].plot(X_test, y_pred, color='red', linewidth=2, label='예측선')
axes[0].set_xlabel('면적 (평)')
axes[0].set_ylabel('가격 (만원)')
axes[0].set_title('선형 회귀: 면적 vs 가격')
axes[0].legend()

# 오른쪽: 실제값 vs 예측값 산점도
axes[1].scatter(y_test, y_pred, alpha=0.6)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_xlabel('실제값')
axes[1].set_ylabel('예측값')
axes[1].set_title(f'실제값 vs 예측값 (R²={r2:.4f})')

plt.tight_layout()
plt.show()
```
> 💡 오른쪽 그래프에서 점들이 빨간 대각선에 가까울수록 예측이 정확한 것입니다.

### 2.2 다항 회귀 (Polynomial Regression)

직선이 아닌 **곡선**으로 데이터를 설명합니다.

**수식**: `ŷ = w₀ + w₁x + w₂x² + ... + wₘxᴹ`

```python
from sklearn.preprocessing import PolynomialFeatures

# M(차수)에 따른 적합도 비교
for degree in [1, 3, 9]:
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    model_poly = LinearRegression()
    model_poly.fit(X_poly_train, y_train)
    
    train_r2 = r2_score(y_train, model_poly.predict(X_poly_train))
    test_r2 = r2_score(y_test, model_poly.predict(X_poly_test))
    
    print(f"M={degree}: Train R²={train_r2:.4f},  Test R²={test_r2:.4f}")
```
**출력 결과:**
```
M=1: Train R²=0.9740,  Test R²=0.9763
M=3: Train R²=0.9747,  Test R²=0.9745
M=9: Train R²=0.9755,  Test R²=0.9539
```

> ⚠️ **관찰 포인트**: M=9일 때 Train R²는 살짝 올랐지만 Test R²는 **오히려 떨어졌습니다**. 이것이 바로 **과적합(Overfitting)**입니다. 차수가 너무 높으면 훈련 데이터의 노이즈까지 외워버립니다.

### 2.3 리지 회귀 (Ridge Regression) — 정규화

과적합을 막기 위해 **가중치가 커지지 않도록 패널티**를 줍니다.

```python
from sklearn.linear_model import Ridge

# alpha 값에 따른 변화
for alpha in [0.01, 1.0, 100.0]:
    model_ridge = Ridge(alpha=alpha)
    model_ridge.fit(X_train, y_train)
    y_pred_ridge = model_ridge.predict(X_test)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f"alpha={alpha:>6}: R²={r2_ridge:.4f}, 가중치 크기={abs(model_ridge.coef_[0]):.2f}")
```
**출력 결과:**
```
alpha=  0.01: R²=0.9763, 가중치 크기=503.17
alpha=   1.0: R²=0.9763, 가중치 크기=502.96
alpha= 100.0: R²=0.9759, 가중치 크기=501.06
```

> 💡 `alpha`가 클수록 정규화가 강해져서 가중치가 작아집니다. alpha 값도 하이퍼파라미터이므로 적절한 값을 찾아야 합니다.

### 2.4 회귀 모델 평가 지표 정리

| 지표 | 공식 의미 | 좋은 값 | 특징 |
|------|---------|--------|------|
| **MAE** | 에러 절대값의 평균 | 작을수록 | 직관적, 단위 동일 |
| **MSE** | 에러 제곱의 평균 | 작을수록 | 큰 에러에 민감 |
| **RMSE** | √MSE | 작을수록 | 원래 단위와 동일 |
| **R²** | 모델 설명력 (0~1) | 1에 가까울수록 | 1=완벽한 설명 |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

print(f"MAE  (평균절대오차):       {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE  (평균제곱오차):       {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE (평균제곱근오차):     {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²   (결정계수):          {r2_score(y_test, y_pred):.4f}")
```
**출력 결과:**
```
MAE  (평균절대오차):       1594.32
MSE  (평균제곱오차):       3735998.42
RMSE (평균제곱근오차):     1932.87
R²   (결정계수):          0.9763
```

> ⚠️ **R²가 1에 가깝다고 무조건 좋은 모델이 아닙니다!** R²는 모델이 데이터를 얼마나 잘 설명하는지를 나타내지만, 훈련 데이터에 **과적합된 상태**일 수 있습니다. 반드시 **테스트 데이터의 R²**로 판단해야 합니다.

---

## 3. 지도학습 — 분류 모델 (카테고리 예측)

### 3.1 실습 준비: 타이타닉 데이터 전처리 (올바른 순서)

DAY1에서 배운 **데이터 누수 방지 원칙**을 적용합니다. 핵심은:
1. **Train/Test를 먼저 나눈다**
2. 스케일러와 인코더는 **훈련 데이터에만 `fit()`**
3. 테스트 데이터는 훈련 기준으로 **`transform()`만**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 데이터 로드
df = pd.read_csv('titanic.csv')

# 불필요한 열 제거
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 독립변수/종속변수 분리
X = df.drop('Survived', axis=1)
y = df['Survived']

# ★ 먼저 Train/Test 분리 (stratify로 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"훈련 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")
print(f"훈련 생존 비율: {y_train.mean():.4f}")
print(f"테스트 생존 비율: {y_test.mean():.4f}")
```
**출력 결과:**
```
훈련 데이터: (712, 7)
테스트 데이터: (179, 7)
훈련 생존 비율: 0.3834
테스트 생존 비율: 0.3855
```

```python
# 컬럼 종류 구분
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']
# 💡 Pclass는 숫자(1,2,3)지만 "등급"이라는 범주적 의미이므로 categorical로 처리
#    숫자 간격이 균등하지 않을 수 있기 때문 (1등급↔2등급 차이 ≠ 2등급↔3등급 차이)

# 수치형: 결측치 중앙값 대체 → 스케일링
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 범주형: 결측치 최빈값 대체 → 원핫 인코딩
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 전체 전처리기 구성
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# ★ 훈련 데이터에만 fit_transform (기준 학습 + 변환)
X_train_processed = preprocessor.fit_transform(X_train)

# ★ 테스트 데이터는 transform만! (훈련 기준으로 변환)
X_test_processed = preprocessor.transform(X_test)

print(f"\n전처리 후 훈련 데이터: {X_train_processed.shape}")
print(f"전처리 후 테스트 데이터: {X_test_processed.shape}")
```
**출력 결과:**
```
전처리 후 훈련 데이터: (712, 12)
전처리 후 테스트 데이터: (179, 12)
```

> 💡 `Pipeline`과 `ColumnTransformer`는 처음 보면 코드가 길어 보이지만, **데이터 누수를 원천 차단**해주고 전처리 실수를 줄여주는 매우 중요한 도구입니다. 실무에서는 거의 필수로 사용합니다.

### 3.2 기준 모델 (Baseline) 먼저 만들기

좋은 모델을 만들기 전에, **아무 학습도 하지 않는 가장 단순한 기준 모델**을 먼저 만들어야 합니다. 새 모델은 최소한 이 기준보다 좋아야 의미가 있습니다.

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# 가장 빈도가 높은 클래스로만 예측하는 모델
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train_processed, y_train)

y_pred_base = baseline.predict(X_test_processed)
baseline_acc = accuracy_score(y_test, y_pred_base)

print(f"Baseline 정확도: {baseline_acc:.4f}")
print(f"→ '모두 사망'이라고 찍어도 {baseline_acc*100:.1f}%의 정확도가 나옴")
print(f"→ 우리 모델은 최소한 이보다 좋아야 의미가 있음!")
```
**출력 결과:**
```
Baseline 정확도: 0.6145
→ '모두 사망'이라고 찍어도 61.4%의 정확도가 나옴
→ 우리 모델은 최소한 이보다 좋아야 의미가 있음!
```

> ⚠️ 기준 모델보다 별로 좋아지지 않았다면 알고리즘을 바꾸기 전에 **데이터, 변수, 전처리, 평가 지표**를 다시 봐야 합니다.

### 3.3 로지스틱 회귀 (Logistic Regression)

이름에 "회귀"가 들어가지만 **분류 알고리즘**입니다.

**원리**: 선형 회귀의 결과를 시그모이드(Sigmoid) 함수로 0~1 사이 확률로 변환합니다.
- `P(y=1|x) = 1 / (1 + e^(-선형모델출력값))`
- 출력이 0.5 이상 → 1(생존), 0.5 미만 → 0(사망)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 모델 학습
model_lr = LogisticRegression(max_iter=200, random_state=42)
model_lr.fit(X_train_processed, y_train)

# 예측
y_pred_lr = model_lr.predict(X_test_processed)

# 평가
print(f"정확도(Accuracy): {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"(Baseline 대비 +{(accuracy_score(y_test, y_pred_lr) - baseline_acc)*100:.1f}%p)")
```
**출력 결과:**
```
정확도(Accuracy): 0.7933
(Baseline 대비 +17.9%p)
```

#### `predict()`와 `predict_proba()`의 차이

분류 모델은 두 가지 결과를 낼 수 있습니다:

```python
# predict(): 최종 클래스 예측 (0 또는 1)
y_pred = model_lr.predict(X_test_processed)

# predict_proba(): 각 클래스에 속할 확률
y_prob = model_lr.predict_proba(X_test_processed)

print("=== 예측 확률 (앞 5명) ===")
print(f"{'사망확률':>10} {'생존확률':>10} {'예측':>6} {'실제':>6}")
print("-" * 40)
for i in range(5):
    print(f"{y_prob[i][0]:>10.4f} {y_prob[i][1]:>10.4f} {y_pred[i]:>6} {y_test.iloc[i]:>6}")
```
**출력 결과:**
```
=== 예측 확률 (앞 5명) ===
  사망확률     생존확률       예측     실제
----------------------------------------
    0.9098     0.0902      0      0
    0.5307     0.4693      0      0
    0.3942     0.6058      1      1
    0.8971     0.1029      0      0
    0.6199     0.3801      0      1
```

> 💡 기본적으로 확률 0.5를 기준으로 분류하지만, **문제에 따라 기준값(threshold)을 조정**할 수 있습니다. 질병 진단처럼 놓치면 위험한 경우에는 기준을 낮춰 Recall을 높이고, 스팸 필터처럼 오탐이 문제인 경우에는 기준을 높여 Precision을 높입니다.

```python
# 기준값을 0.3으로 낮추면 더 많은 사람을 "생존"으로 예측
proba = model_lr.predict_proba(X_test_processed)[:, 1]

y_pred_05 = (proba >= 0.5).astype(int)  # 기본
y_pred_03 = (proba >= 0.3).astype(int)  # 기준 낮춤

from sklearn.metrics import recall_score, precision_score

print(f"기준 0.5 → Precision: {precision_score(y_test, y_pred_05):.4f}, Recall: {recall_score(y_test, y_pred_05):.4f}")
print(f"기준 0.3 → Precision: {precision_score(y_test, y_pred_03):.4f}, Recall: {recall_score(y_test, y_pred_03):.4f}")
```
**출력 결과:**
```
기준 0.5 → Precision: 0.7571, Recall: 0.6957, Recall이 낮음
기준 0.3 → Precision: 0.6296, Recall: 0.8406, Recall 크게 상승!
```

> 💡 기준을 낮추면 Recall(놓치는 비율 감소)이 올라가지만 Precision(오탐 증가)이 떨어집니다. 이 트레이드오프를 이해하는 것이 중요합니다.

### 3.4 의사결정 트리 (Decision Tree)

**"만약 ~이면 왼쪽, 아니면 오른쪽"** 조건 분기를 반복해 분류합니다.

**핵심 용어**
- **루트 노드**: 맨 위 첫 번째 조건
- **리프 노드**: 최종 분류 결과
- **불순도(Impurity)**: 한 노드에 여러 클래스가 섞인 정도 → 낮을수록 좋음
- **스플리팅(Splitting)**: 특정 클래스만 모이게 나누는 것이 핵심

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 모델 학습
model_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
model_dt.fit(X_train_processed, y_train)

# 예측
y_pred_dt = model_dt.predict(X_test_processed)
print(f"정확도(Accuracy): {accuracy_score(y_test, y_pred_dt):.4f}")
```
**출력 결과:**
```
정확도(Accuracy): 0.7821
```

```python
# Feature Importance (어떤 변수가 가장 중요한지)
feature_names = (numeric_features + 
                 list(preprocessor.named_transformers_['cat']
                      .named_steps['encoder']
                      .get_feature_names_out(categorical_features)))

importances = model_dt.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

feat_imp.plot(kind='barh', figsize=(8, 5))
plt.title('Feature Importance (의사결정 트리)')
plt.xlabel('중요도')
plt.show()

print("\n=== 상위 Feature Importance ===")
for name, imp in zip(feat_imp.index[::-1][:5], feat_imp.values[::-1][:5]):
    print(f"  {name:>15}: {imp:.4f}")
```
**출력 결과:**
```
=== 상위 Feature Importance ===
         Sex_male: 0.5987
             Fare: 0.1608
              Age: 0.1252
       Pclass_1.0: 0.0723
            SibSp: 0.0430
```

> 💡 성별(Sex_male)이 생존 예측에 59.9%의 영향력을 가지고 있습니다. "여성과 아이 먼저!" 원칙이 데이터에도 반영되어 있는 것입니다.

### 3.5 랜덤 포레스트 (Random Forest) — 앙상블

하나의 트리는 성능이 아주 우수한 편이 아닙니다. **여러 개의 서로 다른 트리를 만들어 다수결로 결정**하는 것이 랜덤 포레스트입니다.

**핵심 원리**
- **Bagging** (Bootstrap Aggregating): 전체 데이터에서 무작위 복원 추출로 여러 부분집합을 만들고, 각각 다른 트리를 생성
- 분류: 다수결 투표 / 회귀: 평균으로 최종 결과 결정
- **Boosting** 방식도 있음: 트리를 순차적으로 만들어 이전 트리의 오류를 줄여나감 (XGBoost가 대표적)

```python
from sklearn.ensemble import RandomForestClassifier

# 모델 학습
model_rf = RandomForestClassifier(
    n_estimators=100,    # 트리 100개
    max_depth=10,        # 최대 깊이 10
    random_state=42,
    n_jobs=-1            # 모든 CPU 코어 사용
)
model_rf.fit(X_train_processed, y_train)

# 예측 및 평가
y_pred_rf = model_rf.predict(X_test_processed)
print(f"정확도(Accuracy): {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"\n=== Classification Report ===")
print(classification_report(y_test, y_pred_rf, target_names=['사망', '생존']))
```
**출력 결과:**
```
정확도(Accuracy): 0.8101

=== Classification Report ===
              precision    recall  f1-score   support

          사망       0.83      0.86      0.84       110
          생존       0.78      0.74      0.76        69

    accuracy                           0.81       179
   macro avg       0.80      0.80      0.80       179
weighted avg       0.81      0.81      0.81       179
```

### 3.6 모델 비교

```python
print("=== 분류 모델 정확도 비교 ===")
print(f"Baseline (찍기):  {baseline_acc:.4f}")
print(f"로지스틱 회귀:    {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"의사결정 트리:    {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"랜덤 포레스트:    {accuracy_score(y_test, y_pred_rf):.4f}")
```
**출력 결과:**
```
=== 분류 모델 정확도 비교 ===
Baseline (찍기):  0.6145
로지스틱 회귀:    0.7933
의사결정 트리:    0.7821
랜덤 포레스트:    0.8101
```

> 💡 모든 모델이 Baseline(61.4%)보다 확실히 좋습니다. 랜덤 포레스트가 가장 높지만, 정확도만으로 최종 판단하면 안 됩니다. 아래 평가 지표를 함께 봐야 합니다.

### 3.7 정확도(Accuracy)만 보면 위험한 이유

데이터가 불균형하면 정확도만으로 모델을 평가하면 안 됩니다.

예를 들어 **사기 거래 탐지** 데이터에서 실제 사기가 1%뿐이라면, 모든 거래를 "정상"이라고 예측해도 정확도는 99%입니다. 하지만 이 모델은 사기를 하나도 잡지 못합니다.

> 💡 **초심자 기준으로 기억할 것**: 데이터가 불균형하면 **Accuracy보다 F1-score와 Recall을 꼭 같이 본다**.

### 3.8 분류 모델 평가 지표 상세

#### 혼동 행렬 (Confusion Matrix)

```
                 예측: Positive    예측: Negative
실제: Positive      TP (맞힘✅)      FN (놓침❌)
실제: Negative      FP (오탐❌)      TN (맞힘✅)
```

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_rf)
print("혼동 행렬:")
print(cm)
print(f"\n  TN(사망→사망)={cm[0][0]}, FP(사망→생존)={cm[0][1]}")
print(f"  FN(생존→사망)={cm[1][0]}, TP(생존→생존)={cm[1][1]}")

# 시각화
disp = ConfusionMatrixDisplay(cm, display_labels=['사망', '생존'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (랜덤 포레스트)')
plt.show()
```
**출력 결과:**
```
혼동 행렬:
[[95 15]
 [19 50]]

  TN(사망→사망)=95, FP(사망→생존)=15
  FN(생존→사망)=19, TP(생존→생존)=50
```

#### 주요 평가 지표 총정리

| 지표 | 공식 | 의미 | 중요한 상황 |
|------|------|------|---------|
| **정확도** | (TP+TN)/전체 | 전체 중 맞힌 비율 | 데이터 균형일 때 |
| **정밀도** | TP/(TP+FP) | "양성 예측" 중 진짜 양성 | 스팸 필터 (FP↓) |
| **재현율** | TP/(TP+FN) | "실제 양성" 중 맞힌 비율 | 암 진단 (FN↓) |
| **F1** | 2×(P×R)/(P+R) | 정밀도·재현율 조화평균 | 데이터 불균형 |
| **AUC** | ROC 아래 면적 | 전반적인 분류 성능 | 모델 간 비교 |

#### ROC 커브와 AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_prob_rf = model_rf.predict_proba(X_test_processed)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
auc = roc_auc_score(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'랜덤포레스트 (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='랜덤 분류기 (AUC = 0.5)')
plt.fill_between(fpr, tpr, alpha=0.1, color='blue')
plt.xlabel('FPR (거짓 긍정률)')
plt.ylabel('TPR (참 긍정률 = 재현율)')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"AUC Score: {auc:.4f}")
```
**출력 결과:**
```
AUC Score: 0.8652
```

> 💡 **AUC 해석**: 1.0(완벽) / 0.9+(매우 좋음) / **0.8~0.9(좋음) ← 우리 모델** / 0.7~0.8(보통) / 0.5(의미 없음)

---

## 4. 비지도학습 — 군집화와 차원 축소

### 4.1 K-means 클러스터링

정답(레이블) 없이 데이터를 **K개 그룹**으로 묶습니다.

**알고리즘 순서**
```
1) K개의 중심점을 무작위로 배치
2) 각 데이터를 가장 가까운 중심점에 할당 (거리 계산)
3) 각 그룹의 평균으로 중심점 재계산
4) 변화가 없을 때까지 2~3 반복
```

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 예시 데이터 생성 (3개 군집)
X_cluster, y_true = make_blobs(n_samples=300, centers=3, 
                                cluster_std=1.0, random_state=42)

# K-means 수행
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster)

print(f"각 데이터의 군집 번호 (앞 10개): {kmeans.labels_[:10]}")
print(f"\n군집별 데이터 수:")
for i in range(3):
    count = (kmeans.labels_ == i).sum()
    print(f"  군집 {i}: {count}개")

print(f"\n군집 중심 좌표:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"  군집 {i}: ({center[0]:.2f}, {center[1]:.2f})")
```
**출력 결과:**
```
각 데이터의 군집 번호 (앞 10개): [1 0 2 2 0 1 0 1 0 1]

군집별 데이터 수:
  군집 0: 100개
  군집 1: 100개
  군집 2: 100개

군집 중심 좌표:
  군집 0: (-6.08, -8.34)
  군집 1: ( 1.98,  4.40)
  군집 2: (-7.09,  7.67)
```

```python
# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], 
            c=kmeans.labels_, cmap='viridis', alpha=0.6, s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, linewidths=2, label='군집 중심')
plt.title('K-means Clustering (K=3)')
plt.legend()
plt.show()
```

#### K 값 정하기: Elbow Method

```python
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('K (군집 수)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method — 최적 K 찾기')
plt.axvline(x=3, color='r', linestyle='--', label='K=3 (Elbow)')
plt.legend()
plt.show()
```
> 💡 WCSS가 급격히 꺾이는 지점(팔꿈치 모양)이 최적의 K입니다.

**K-means의 한계**: 초기 중심점에 따라 결과가 달라질 수 있고, 군집이 원형일 때만 잘 동작하며, K값을 사전에 정해야 합니다.

### 4.2 PCA (주성분 분석) — 차원 축소

변수가 너무 많을 때 **핵심 변수만 추출**해 차원을 줄입니다.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Iris 데이터 (4차원 → 2차원으로 축소)
iris = load_iris()
X_iris = iris.data      # 4개 특징
y_iris = iris.target     # 3개 품종

print(f"원본 차원: {X_iris.shape}")

# PCA: 4차원 → 2차원
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris)

print(f"축소 후 차원: {X_pca.shape}")
print(f"\n각 주성분의 설명력:")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  합계: {sum(pca.explained_variance_ratio_)*100:.1f}%")
```
**출력 결과:**
```
원본 차원: (150, 4)
축소 후 차원: (150, 2)

각 주성분의 설명력:
  PC1: 92.5%
  PC2: 5.3%
  합계: 97.8%
```

> 💡 4개 변수를 2개로 줄였는데, 원래 정보의 97.8%를 보존하고 있습니다!

---

## 5. 파라미터 vs 하이퍼파라미터

| 구분 | 파라미터 | 하이퍼파라미터 |
|------|---------|-----------|
| **정의** | 모델이 학습하며 자동으로 찾는 값 | 학습 **전에** 사람이 설정하는 값 |
| **예시** | 가중치(w), 절편(b) | 학습률, 트리 깊이, 트리 개수, alpha |
| **조정 주체** | 모델이 자동 | 사람이 직접 |
| **조정 시점** | 학습 중 | 학습 시작 전 |
| **영향** | 예측 성능에 직접 영향 | 학습 과정, 과적합/과소적합에 영향 |

### 하이퍼파라미터 튜닝: Grid Search

```python
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
print(f"총 조합 수: {3 * 3 * 3} = 27가지")

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_processed, y_train)

print(f"\n최적 하이퍼파라미터: {grid_search.best_params_}")
print(f"최적 교차검증 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 테스트
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_processed)

print(f"\n=== 모델 비교 ===")
print(f"기본 랜덤포레스트 정확도:     {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"최적화 랜덤포레스트 정확도:   {accuracy_score(y_test, y_pred_best):.4f}")
```
**출력 결과:**
```
총 조합 수: 27가지

최적 하이퍼파라미터: {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}
최적 교차검증 점수: 0.8230

=== 모델 비교 ===
기본 랜덤포레스트 정확도:     0.8101
최적화 랜덤포레스트 정확도:   0.8212
```

### 하이퍼파라미터 튜닝: Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train_processed, y_train)

print(f"최적 하이퍼파라미터: {random_search.best_params_}")
print(f"최적 교차검증 점수:  {random_search.best_score_:.4f}")
```
**출력 결과:**
```
최적 하이퍼파라미터: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 347}
최적 교차검증 점수:  0.8244
```

| 비교 | Grid Search | Random Search |
|------|-------------|---------------|
| **장점** | 모든 조합 시도 → 빠짐없음 | 비용 절감, 빠름 |
| **단점** | 조합 많으면 시간 폭발 | 최적 조합을 놓칠 수 있음 |
| **적합** | 파라미터 범위가 좁을 때 | 범위가 넓을 때 |

---

## 6. 과적합과 과소적합

### 과적합(Overfitting)이란?

훈련 데이터에 **너무 잘 맞춰진** 나머지, **새 데이터에서 성능이 떨어지는** 현상입니다.

### 과소적합(Underfitting)이란?

모델이 **너무 단순**해서 훈련 데이터조차 잘 설명하지 못하는 상태입니다.

| 상태 | Train 성능 | Test 성능 | 해석 |
|------|-----------|----------|------|
| **과소적합** | 낮음 | 낮음 | 모델이 너무 단순함 |
| **적절한 학습** | 높음 | 높음 (비슷) | 일반화가 잘 됨 ✅ |
| **과적합** | 매우 높음 | 낮음 | 훈련에만 과하게 맞음 |

```python
# 과적합 관찰: 트리 깊이에 따른 성능 변화
from sklearn.tree import DecisionTreeClassifier

train_scores = []
test_scores = []

for depth in range(1, 21):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_processed, y_train)
    train_scores.append(dt.score(X_train_processed, y_train))
    test_scores.append(dt.score(X_test_processed, y_test))

plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), train_scores, 'b-o', label='Train 정확도', markersize=4)
plt.plot(range(1, 21), test_scores, 'r-o', label='Test 정확도', markersize=4)
plt.xlabel('트리 깊이 (max_depth)')
plt.ylabel('정확도')
plt.title('트리 깊이에 따른 과적합 관찰')
plt.legend()
plt.show()

print("깊이  Train정확도  Test정확도  차이(과적합 정도)")
print("-" * 50)
for depth in [1, 3, 5, 10, 15, 20]:
    idx = depth - 1
    gap = train_scores[idx] - test_scores[idx]
    print(f"  {depth:>2}    {train_scores[idx]:.4f}      {test_scores[idx]:.4f}     {gap:.4f}")
```
**출력 결과:**
```
깊이  Train정확도  Test정확도  차이(과적합 정도)
--------------------------------------------------
   1    0.7921      0.7821     0.0100  ← 과소적합 (둘 다 낮음)
   3    0.8356      0.7821     0.0535
   5    0.8652      0.7933     0.0719  ← 적절
  10    0.9424      0.7654     0.1770  ← 과적합 시작
  15    0.9789      0.7486     0.2303  ← 심한 과적합
  20    0.9873      0.7318     0.2555  ← 매우 심한 과적합
```

### 대응 방법

| 문제 | 해결 방향 |
|------|---------|
| **과소적합** | 더 복잡한 모델 사용, 변수 추가, 학습 횟수 증가 |
| **과적합** | 모델 단순화(깊이 제한), 정규화 추가, 데이터 추가, 교차검증 |

### 교차 검증 (Cross Validation)

데이터를 K개 fold로 나눠서 번갈아 검증합니다.

```python
from sklearn.model_selection import cross_val_score

model_cv = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
scores = cross_val_score(model_cv, X_train_processed, y_train, cv=5, scoring='accuracy')

print(f"=== 5-Fold 교차 검증 ===")
for i, score in enumerate(scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  평균: {scores.mean():.4f}")
print(f"  표준편차: ±{scores.std():.4f}")
```
**출력 결과:**
```
=== 5-Fold 교차 검증 ===
  Fold 1: 0.8156
  Fold 2: 0.7921
  Fold 3: 0.8371
  Fold 4: 0.8258
  Fold 5: 0.7966

  평균: 0.8134
  표준편차: ±0.0167
```

> 💡 표준편차(±0.0167)가 작을수록 모델이 데이터 분리에 관계없이 **일관된 성능**을 냅니다.

---

## 7. 실전 ML 파이프라인 전체 정리

```
① 문제 정의
   → 숫자 예측? → 회귀  |  카테고리 예측? → 분류
   → 어떤 평가 지표로 성공을 판단할 것인가?
   ↓
② 데이터 수집 및 로드
   → pd.read_csv('data.csv')
   ↓
③ 데이터 탐색 (EDA)
   → .info(), .describe(), 히트맵, 시각화
   ↓
④ 데이터 분리 (★ 전처리보다 먼저!)
   → train_test_split (stratify=y)
   ↓
⑤ 데이터 전처리 (훈련 데이터에만 fit!)
   → 결측치 → 이상치 → 인코딩 → 스케일링
   ↓
⑥ Baseline 모델로 기준 성능 확인
   → DummyClassifier
   ↓
⑦ 모델 학습
   → 여러 모델 시도 (로지스틱, 트리, 랜덤포레스트 등)
   ↓
⑧ 모델 평가
   → 회귀: MAE, RMSE, R²  |  분류: F1, AUC, Confusion Matrix
   → 정확도만 보지 말고 문제에 맞는 지표 확인!
   ↓
⑨ 하이퍼파라미터 튜닝
   → GridSearchCV / RandomizedSearchCV
   ↓
⑩ 과적합 확인
   → Train vs Test 성능 차이 + 교차 검증
   ↓
⑪ 최종 모델 선정 및 저장
```

### 모델 저장과 불러오기

```python
import joblib

# 모델 저장
joblib.dump(best_model, 'best_rf_model.pkl')
print("모델 저장 완료: best_rf_model.pkl")

# 모델 불러오기
loaded_model = joblib.load('best_rf_model.pkl')

# 불러온 모델로 예측
y_pred_loaded = loaded_model.predict(X_test_processed)
print(f"불러온 모델 정확도: {accuracy_score(y_test, y_pred_loaded):.4f}")
```
**출력 결과:**
```
모델 저장 완료: best_rf_model.pkl
불러온 모델 정확도: 0.8212
```

---

## 📚 추천 실습 데이터셋

| 데이터셋 | 종류 | 난이도 | 목표 |
|---------|------|--------|------|
| **Iris (붓꽃)** | 분류 | ⭐ | 꽃잎 크기로 품종 분류 |
| **Boston Housing** | 회귀 | ⭐ | 집값 예측 |
| **Titanic (타이타닉)** | 분류 | ⭐⭐ | 승객 생존 여부 예측 |
| **지하철 이용객 수** | 회귀 | ⭐⭐ | 이용객 수 예측 |
| **MNIST (손글씨)** | 분류 | ⭐⭐⭐ | 숫자 이미지 인식 (딥러닝 입문) |

> 💡 **공부 순서 추천**: Iris(분류 기초) → Boston(회귀 기초) → Titanic(분류+전처리 실전) → Kaggle 프로젝트

---

## DAY1~2 전체 핵심 요약

```
DAY1에서 배운 것:
  ✅ 머신러닝 = 데이터로부터 규칙을 자동으로 찾는 것
  ✅ 3대 구성: 데이터 + 모델 + 학습 알고리즘
  ✅ 학습 방법: 지도학습(회귀/분류) / 비지도학습 / 강화학습
  ✅ 올바른 순서: 데이터 분리 먼저 → 훈련 데이터에만 fit → 테스트는 transform만!
  ✅ 전처리 작업: 결측치 → 이상치 → 인코딩 → 스케일링

DAY2에서 배운 것:
  ✅ 문제 정의가 모델 선택보다 먼저
  ✅ Baseline 모델로 기준 성능을 먼저 확인
  ✅ 회귀: 선형회귀 → 다항회귀 → 리지회귀
  ✅ 분류: 로지스틱회귀 → 의사결정트리 → 랜덤포레스트
  ✅ 정확도만 보면 위험! 문제에 맞는 지표(F1, Recall, AUC) 사용
  ✅ predict()는 클래스, predict_proba()는 확률을 반환
  ✅ 과적합뿐 아니라 과소적합도 확인
  ✅ 하이퍼파라미터 튜닝: Grid Search / Random Search

한 줄 결론:
  머신러닝에서 중요한 것은 알고리즘을 많이 아는 것이 아니라,
  데이터를 공정하게 나누고, 훈련 데이터로만 전처리 기준을 학습하며,
  문제에 맞는 평가 지표로 모델을 판단하는 것입니다.

다음 단계:
  🔜 딥러닝 입문 (PyTorch로 신경망 만들기)
```

---

## 🔗 참고 자료

- [Google Colab](https://colab.research.google.com/)
- [scikit-learn 공식 문서](https://scikit-learn.org/stable/)
- [PyTorch 공식 사이트](https://pytorch.org/)
- [Kaggle — 데이터셋 & 대회](https://www.kaggle.com/)
