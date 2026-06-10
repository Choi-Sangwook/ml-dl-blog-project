# 🤖 머신러닝 완전 입문 가이드 — DAY3. SVM · KNN · 나이브 베이즈

> **시리즈**: 파이썬 기본만 있는 사람을 위한 머신러닝 입문
> **DAY1**: 머신러닝 핵심 개념과 데이터 관리 (이전 편)
> **DAY2**: 회귀 모델과 분류 모델 실습 (이전 편)
> **DAY3**: SVM · KNN · 나이브 베이즈 — 지도학습 분류 알고리즘 심화

---

## 1. 지도학습 분류 알고리즘 한눈에 보기

DAY2에서 로지스틱 회귀, 의사결정 트리, 랜덤 포레스트를 배웠습니다. DAY3에서는 **SVM, KNN, 나이브 베이즈** 세 가지를 더 깊이 다룹니다.

### 분류 알고리즘 비교 요약

| 알고리즘 | 핵심 방식 | 장점 | 단점 |
|---------|---------|------|------|
| **Logistic Regression** | 확률 (시그모이드) | 빠름, 해석 쉬움 | 비선형 패턴 한계 |
| **KNN** | 거리 기반 | 직관적, 구현 쉬움 | 데이터 많으면 느림 |
| **Naive Bayes** | 확률 (베이즈 정리) | 텍스트 분류 강함 | 독립 가정 비현실적 |
| **Decision Tree** | 조건 분기 | 해석 쉬움, 시각화 | 과적합 쉬움 |
| **Random Forest** | 앙상블 (투표) | 성능 우수, 과적합 감소 | 해석 어려움 |
| **SVM** | 마진 최대화 | 고차원 강함, 일반화 우수 | 속도 느림, 튜닝 필요 |

> 💡 **어떤 알고리즘을 골라야 할까?** 데이터 크기, 차원 수, 해석 필요 여부, 학습/예측 속도 등을 고려해야 합니다. 하나의 "정답" 알고리즘은 없으니, DAY2에서 배운 것처럼 Baseline → 여러 모델 비교 → 평가 지표로 판단하는 흐름이 중요합니다.

---

## 2. SVM (서포트 벡터 머신)

### 2.1 핵심 아이디어: 마진 최대화

SVM의 목표는 두 클래스를 가장 잘 구분하는 **초평면(Hyperplane)** 을 찾는 것입니다. 여기서 핵심은 단순히 "분류가 되는" 경계선이 아니라, **마진(Margin)이 최대인** 경계선을 찾는 것입니다.

```
초평면(분류 경계선): w^T * x + b = 0

마진 = 각 클래스에서 경계선까지의 거리 합
         → 이 마진을 최대로 만드는 w, b를 찾는 것이 SVM의 학습
```

**왜 마진이 중요한가?**
- 경계선이 어느 한쪽에 치우쳐 있으면 새 데이터가 들어왔을 때 오분류 가능성이 높아집니다
- 마진이 넓을수록 양 클래스 사이에 여유 공간이 확보되어 **일반화 성능**이 높아집니다

```
[마진 개념]

■ ■ ■         ● ● ●
  ■     ← margin → ●
  ■               ●
     [ 초평면 ]

서포트 벡터: 각 클래스에서 초평면에 가장 가까운 점들
            이 점들만이 SVM 모델을 결정합니다
            나머지 점을 삭제해도 모델이 동일합니다
```

### 2.2 서포트 벡터란?

각 클래스에서 초평면에 **가장 가까운 점들**을 서포트 벡터라고 합니다. 서포트 벡터만이 모델 정의에 관여하므로, 특징 수가 많은 **고차원 데이터**에서도 효율적으로 동작합니다.

> 💡 SVM의 이름 자체가 "경계를 결정하는 벡터(서포트 벡터)를 사용하는 머신"에서 왔습니다.

### 2.3 선형의 한계와 커널 트릭

**선형 SVM**은 직선 또는 평면으로 데이터를 나눕니다. 하지만 실제 데이터는 원형, 곡선형, 복잡한 형태로 섞여 있는 경우가 많습니다.

```
[선형 분리가 가능한 경우]        [선형 분리가 불가능한 경우]

■ ■           ● ●              ● ● ●
  ■    ───    ●                ■   ■
■ ■           ● ●              ● ● ●
          직선 가능!                직선 불가!
```

이때 SVM은 **커널 트릭(Kernel Trick)** 을 사용합니다.

데이터를 더 높은 차원의 공간으로 보낸 것처럼 계산해, 원래 공간에서는 곡선처럼 보이는 결정 경계를 만들 수 있습니다. 중요한 점은 **실제로 고차원 변환을 계산하지 않고 내적(inner product)만으로 동일한 효과**를 얻는다는 것입니다.

```
입력 공간 (2차원, 선형 분리 불가)
         ↓  커널 함수 φ(x) 적용
고차원 공간 (선형 분리 가능!)
→ 계산 비용 절감: 실제 변환 없이 내적만으로 처리
```

**대표적인 커널 종류**

| 커널 | 특징 | 사용 상황 |
|------|------|---------|
| **Linear** | 선형 분리 | 차원이 매우 높을 때, 텍스트 분류 |
| **RBF (Gaussian)** | 방사형 기저 함수 | 가장 범용적, **기본값 추천** |
| **Polynomial** | 다항식 | 피처 간 교호작용이 있을 때 |

**SVC vs LinearSVC — 어떤 걸 써야 할까?**

| 모델 | 특징 | 추천 상황 |
|------|------|---------|
| `SVC(kernel='rbf')` | 비선형 경계 가능 | 데이터가 크지 않고 비선형 패턴이 있을 때 |
| `SVC(kernel='linear')` | 커널 방식의 선형 SVM | 비교적 작은 데이터에서 선형 경계 |
| `LinearSVC` | 선형 SVM 최적화 구현 | 피처가 많고 데이터가 큰 텍스트 분류 |

> 💡 텍스트처럼 차원이 매우 큰 데이터에서는 RBF SVM보다 `LinearSVC`나 로지스틱 회귀가 더 실용적인 경우가 많습니다.

### 2.4 sklearn으로 SVM 구현하기

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt

# 유방암 데이터 로드 (이진 분류)
data = load_breast_cancer()
X, y = data.data, data.target

# Train/Test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ★ SVM은 스케일에 민감 → StandardScaler 필수!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # 훈련 데이터에만 fit
X_test_scaled  = scaler.transform(X_test)         # 테스트는 transform만

# SVM 모델 학습 (기본 커널: RBF)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = svm_model.predict(X_test_scaled)

print("=== SVM 분류 결과 ===")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```
**출력 결과:**
```
=== SVM 분류 결과 ===
              precision    recall  f1-score   support

   malignant       0.98      0.95      0.96        42
      benign       0.97      0.99      0.98        72

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

```python
# 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.xlabel('예측값')
plt.ylabel('실제값')
plt.title('SVM 혼동 행렬')
plt.tight_layout()
plt.show()

print(f"\n서포트 벡터 수: {svm_model.support_vectors_.shape[0]}개")
print(f"전체 훈련 데이터: {X_train.shape[0]}개")
print(f"→ 전체의 {svm_model.support_vectors_.shape[0]/X_train.shape[0]*100:.1f}%만이 모델 결정에 관여")
```
**출력 결과:**
```
서포트 벡터 수: 78개
전체 훈련 데이터: 455개
→ 전체의 17.1%만이 모델 결정에 관여
```

**예측 확률 얻기 — `probability=True`**

SVC는 기본적으로 확률을 바로 제공하지 않습니다. `predict_proba()`를 사용하려면 `probability=True`를 설정해야 합니다. 다만 이 옵션을 켜면 내부적으로 교차검증이 추가되어 학습 시간이 더 길어질 수 있습니다.

```python
svm_prob = SVC(kernel='rbf', C=1.0, gamma='scale',
               probability=True, random_state=42)
svm_prob.fit(X_train_scaled, y_train)

proba = svm_prob.predict_proba(X_test_scaled)
print(f"첫 번째 샘플 — malignant 확률: {proba[0][0]:.4f}, benign 확률: {proba[0][1]:.4f}")
```
**출력 결과:**
```
첫 번째 샘플 — malignant 확률: 0.0312, benign 확률: 0.9688
```

### 2.5 SVM 핵심 하이퍼파라미터: C와 gamma

#### C (비용 파라미터) — 마진 vs 오분류 트레이드오프

```
C가 작을 때 → 넓은 마진 허용, 일부 오분류 허용 → 과소적합 위험
C가 클 때  → 좁은 마진, 모든 점 정확 분류 시도 → 과적합 위험
```

#### gamma — 하나의 데이터가 영향을 미치는 범위

`gamma`는 RBF 커널에서 하나의 데이터 포인트가 주변에 얼마나 넓게 영향을 미치는지 조절합니다.

```
gamma가 작을 때 → 영향 범위가 넓음 → 결정 경계가 부드러움 → 과소적합 위험
gamma가 클 때  → 영향 범위가 좁음 → 결정 경계가 복잡함 → 과적합 위험
```

정리하면, **`C`는 오분류를 얼마나 허용할지** 조절하고, **`gamma`는 결정 경계를 얼마나 복잡하게 만들지** 조절합니다.

### 2.6 Pipeline으로 올바르게 튜닝하기

교차검증이나 GridSearchCV를 사용할 때는 **Pipeline 안에 스케일러와 모델을 함께 묶어야** 합니다. 스케일러를 바깥에서 먼저 적용하면, 각 fold의 검증 데이터 정보가 스케일러에 미리 반영되어 **데이터 누수**가 생깁니다.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ✅ 올바른 방법: Pipeline으로 스케일러와 모델을 묶기
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42))
])

param_grid = {
    'svm__C':     [0.01, 0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1]
}

grid_search = GridSearchCV(
    svm_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# X_train (스케일링 전 원본) 그대로 넣으면 됩니다
grid_search.fit(X_train, y_train)

print(f"최적 파라미터:   {grid_search.best_params_}")
print(f"교차검증 정확도: {grid_search.best_score_:.4f}")
print(f"테스트 정확도:   {grid_search.score(X_test, y_test):.4f}")
```
**출력 결과:**
```
최적 파라미터:   {'svm__C': 10, 'svm__gamma': 'scale'}
교차검증 정확도: 0.9780
테스트 정확도:   0.9825
```

> 💡 `Pipeline`은 전처리와 모델을 하나의 묶음으로 관리하는 도구입니다. GridSearchCV와 함께 쓰면 각 fold마다 스케일러가 훈련 데이터에만 `fit()`되어, DAY1에서 배운 데이터 누수를 완전히 막을 수 있습니다.

> ⚠️ 성능을 제대로 끌어내려면 기본값보다 교차검증으로 C, gamma 후보를 비교하는 것을 권장합니다.

**커널별 성능 비교**

```python
import time

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
print(f"{'커널':<12} {'정확도':>8} {'학습시간(ms)':>14}")
print("-" * 38)

for kernel in kernels:
    start = time.time()
    model = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    model.fit(X_train_scaled, y_train)
    elapsed = (time.time() - start) * 1000
    acc = model.score(X_test_scaled, y_test)
    print(f"{kernel:<12} {acc:>8.4f} {elapsed:>12.1f}ms")
```
**출력 결과:**
```
커널         정확도    학습시간(ms)
--------------------------------------
linear       0.9561          4.2ms
poly         0.9737         12.8ms
rbf          0.9737          6.1ms
sigmoid      0.9123          5.3ms
```

### 2.7 SVM 실무 활용 분야

SVM은 특히 **피처 수가 많고 데이터 수는 상대적으로 적은 문제**에서 자주 사용됩니다.

| 분야 | 적용 예시 |
|------|---------|
| **의료** | 암세포 vs 정상세포 판별 |
| **생체정보학** | 마이크로어레이 유전자 발현 데이터 분류 |
| **텍스트** | 주제별 문서 분류, 언어 식별 |
| **이미지/OCR** | 손글씨 문자 인식 (이미지를 통계 특성으로 변환 후 분류) |
| **이상 탐지** | 연소 엔진 고장, 보안 결함, 지진 탐지 |

예를 들어 유전자 발현 데이터는 샘플 수는 많지 않지만 유전자 피처가 수만 개에 달합니다. 이런 고차원 소규모 데이터에서 SVM은 안정적으로 동작합니다.

OCR처럼 이미지를 픽셀 통계(가로/세로 차원, 흑백 비율, 평균 위치 등)로 변환한 뒤 문자를 분류하는 문제에서도 SVM을 적용할 수 있습니다.

### 2.8 데이터가 많을 때 SVM 대안

`SVC`는 데이터가 수만 건을 넘어가면 학습 시간이 급격히 느려집니다. 이 경우 아래 두 가지 대안을 고려할 수 있습니다.

| 모델 | 특징 | 적합한 상황 |
|------|------|---------|
| `LinearSVC` | 선형 SVM에 최적화된 구현 | 피처가 많고 데이터가 큰 텍스트 분류 |
| `SGDClassifier(loss='hinge')` | 확률적 경사하강법 기반 선형 SVM | 수십만 건 이상의 대용량 데이터 |

```python
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 방법 1: LinearSVC (SVC보다 빠른 선형 SVM)
linear_svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, max_iter=10000, random_state=42))
])

# 방법 2: SGDClassifier — 수십만 건 이상의 대용량 데이터에 적합
sgd_svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SGDClassifier(loss='hinge', random_state=42))
    # loss='hinge' → 선형 SVM과 동일한 목적함수
])
```

> 💡 `SVC`는 소규모 고차원 데이터에서 강력하지만, 데이터가 많아지면 실서비스 적용 시 응답 속도 문제가 생길 수 있습니다. 데이터 규모에 따라 알고리즘을 바꾸는 것도 중요한 실무 판단입니다.

---

## 3. KNN (K-최근접 이웃)

### 3.1 핵심 아이디어: "비슷한 건 비슷한 것끼리"

KNN은 새로운 데이터가 들어오면 **훈련 데이터에서 가장 가까운 K개의 이웃**을 찾아 다수결로 클래스를 결정합니다.

```
[분류 예시 — K=5]
새 데이터 ★에 가장 가까운 5개:
  ● ● ● ■ ●

  ● 3개 > ■ 2개
  → ★는 ● 클래스로 분류
```

**KNN의 특징: "게으른 학습(Lazy Learning)"**
- 학습 단계에서는 **데이터를 저장만** 합니다 (파라미터 학습 없음)
- 예측할 때마다 거리를 계산하므로 **예측 단계가 느립니다**
- 훈련 단계 자체는 매우 빠릅니다

> ⚠️ **실서비스 적용 시 주의**: KNN은 훈련이 빠른 대신 **예측이 느립니다**. 예측할 때마다 전체 훈련 데이터와의 거리를 계산하기 때문에, 데이터가 수만 건을 넘어가면 실시간 서비스에 붙이기 어려울 수 있습니다. 프로토타이핑과 오프라인 분석에는 적합하지만, 응답 속도가 중요한 서비스에는 SVM·로지스틱 회귀처럼 예측이 빠른 모델을 고려하세요.

### 3.2 거리 계산 방법과 `weights` 옵션

```
유클리드 거리 (기본값):  d = √Σ(xi - yi)²
맨해튼 거리:             d = Σ|xi - yi|
```

KNN에서 "투표" 방식도 선택할 수 있습니다.

| 옵션 | 의미 |
|------|------|
| `weights='uniform'` | 가까운 이웃과 먼 이웃을 동등하게 투표 (기본값) |
| `weights='distance'` | 가까운 이웃에게 더 큰 가중치 부여 |
| `metric='euclidean'` | 직선 거리 |
| `metric='manhattan'` | 축을 따라 이동한 거리 |

```python
# 거리 기반 가중치 적용 예시
knn = KNeighborsClassifier(
    n_neighbors=7,
    metric='euclidean',
    weights='distance'   # 가까울수록 더 큰 영향력
)
```

### 3.3 sklearn으로 KNN 구현하기

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ★ KNN도 스케일에 매우 민감 → 스케일링 필수!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# k=5로 기본 학습
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
print("=== KNN (k=5) 분류 결과 ===")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```
**출력 결과:**
```
=== KNN (k=5) 분류 결과 ===
              precision    recall  f1-score   support

   malignant       0.97      0.93      0.95        42
      benign       0.96      0.99      0.97        72

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.96      0.96      0.96       114
```

```python
# 혼동 행렬 확인
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.xlabel('예측값')
plt.ylabel('실제값')
plt.title('KNN 혼동 행렬')
plt.tight_layout()
plt.show()
```

> 💡 혼동 행렬을 보면 어떤 클래스를 어떤 클래스로 자주 헷갈리는지 확인할 수 있습니다. 단순 정확도만으로는 오분류의 **방향**을 알 수 없습니다.

### 3.4 최적의 K값 찾기 — Pipeline + GridSearchCV

K값 선택이 KNN 성능의 핵심입니다.

```
K가 작으면 → 훈련 데이터 세세한 구조 반영 → 과적합 위험 (편향↓, 분산↑)
K가 크면  → 전체적인 추세 반영 → 과소적합 위험 (편향↑, 분산↓)
```

K값은 **훈련 데이터 내부의 교차검증으로 선택**해야 합니다. 테스트 데이터는 최종 평가에만 사용합니다.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# ✅ Pipeline으로 스케일러와 KNN 묶기
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': range(1, 31)
}

grid = GridSearchCV(
    knn_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# X_train 원본 그대로 넣으면 됩니다
grid.fit(X_train, y_train)

best_k  = grid.best_params_['knn__n_neighbors']
cv_acc  = grid.best_score_
test_acc = grid.score(X_test, y_test)

print(f"최적 K값:        {best_k}")
print(f"교차검증 정확도: {cv_acc:.4f}")
print(f"테스트 정확도:   {test_acc:.4f}")
```
**출력 결과:**
```
최적 K값:        7
교차검증 정확도: 0.9692
테스트 정확도:   0.9737
```

```python
# K값에 따른 교차검증 점수 시각화
cv_scores = grid.cv_results_['mean_test_score']
k_values  = range(1, 31)

plt.figure(figsize=(12, 5))
plt.plot(k_values, cv_scores, 'g-o', markersize=5, label='CV 정확도 (5-Fold)')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'최적 K={best_k}')
plt.xlabel('K (이웃 수)')
plt.ylabel('교차검증 정확도')
plt.title('K값에 따른 교차검증 성능 (테스트 데이터 미사용)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

> ⚠️ 이전 버전의 코드처럼 K값을 고를 때 테스트 데이터 점수를 반복해서 보면, K 선택이 테스트 데이터에 맞춰져 버립니다. 이것도 일종의 데이터 누수입니다. **하이퍼파라미터는 훈련 데이터 내부의 교차검증으로, 테스트 데이터는 최종 평가에만 딱 한 번** 사용하세요.

### 3.5 KNN에 적합한 데이터 준비

KNN은 거리 계산에 기반하므로 **스케일 차이에 매우 민감**합니다. 나이(0~100)와 연봉(0~10,000,000원)이 섞인 데이터라면, 연봉이 거리 계산을 지배해버립니다.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 일반적인 거리 기반 모델  → StandardScaler 자주 사용
# 값의 범위를 0~1로 맞추고 싶다 → MinMaxScaler
# 이상치가 많다              → RobustScaler 고려
#   (평균 대신 중앙값, 표준편차 대신 IQR을 사용해 이상치 영향 감소)
```

**차원의 저주 (Curse of Dimensionality)**

피처가 너무 많으면 모든 데이터가 서로 비슷하게 멀어지는 문제가 생깁니다. KNN은 거리 기반 모델이기 때문에, 불필요한 피처가 많으면 성능이 떨어질 수 있습니다. 이럴 때는 **중요한 피처만 선택**하거나, DAY2에서 배운 **PCA 같은 차원 축소**를 고려할 수 있습니다.

**KNN 실전 진행 체크리스트**

KNN은 원리가 단순하지만, "학습"보다 **"데이터 준비"가 성능에 더 큰 영향**을 줍니다.

1. 사용할 피처를 선택합니다
2. 결측치와 이상치를 처리합니다
3. Train/Test 데이터를 먼저 나눕니다
4. Pipeline으로 스케일링을 묶습니다
5. GridSearchCV로 K값을 교차검증으로 비교합니다
6. 최종 모델을 테스트 데이터로 평가합니다

### 3.6 KNN 활용 분야

| 분야 | 적용 예시 |
|------|---------|
| **추천 시스템** | 비슷한 취향의 사용자 찾기 (영화, 음악 추천) |
| **의료** | 유방암 진단, 유사 증상 환자 탐색 |
| **이미지** | 광학 문자 인식, 얼굴 인식 |
| **생물정보학** | 단백질·유전자 데이터 패턴 인식 |

---

## 4. 나이브 베이즈 (Naive Bayes)

### 4.1 핵심 아이디어: 베이즈 정리

나이브 베이즈는 **베이즈 정리**를 이용한 확률 기반 분류 알고리즘입니다. 수학자 토마스 베이즈의 연구에서 유래했으며, "나이브(Naive)"라는 이름은 **모든 피처가 서로 독립이라고 가정**하기 때문에 붙었습니다.

**베이즈 정리:**
```
P(A|B) = P(B|A) × P(A) / P(B)

분류 문제에 적용:
P(클래스 | 피처들) ∝ P(피처들 | 클래스) × P(클래스)
     ↑ 사후확률          ↑ 조건부확률       ↑ 사전확률
```

**확률 용어 정리**

| 용어 | 의미 | 스팸 필터 예시 |
|------|------|--------------|
| **사전확률 P(클래스)** | 아무 정보가 없을 때의 기본 확률 | 전체 메일 중 스팸 비율 (10%) |
| **조건부확률 P(피처\|클래스)** | 클래스가 주어졌을 때 피처의 확률 | 스팸일 때 `free`가 등장할 확률 |
| **사후확률 P(클래스\|피처)** | 관측 정보를 반영한 뒤의 확률 | `free`가 포함된 메일이 스팸일 확률 |

나이브 베이즈 분류기는 여러 단어가 주어졌을 때 각 클래스의 사후확률을 계산하고, **확률이 가장 높은 클래스를 선택**합니다.

**나이브 가정 (독립 가정):**
```python
# 피처가 서로 독립이라고 가정하면:
P(단어1, 단어2, ..., 단어n | 클래스)
  = P(단어1 | 클래스) × P(단어2 | 클래스) × ... × P(단어n | 클래스)
# 이 단순화 덕분에 계산이 매우 빠르고 텍스트 분류에서 잘 작동합니다
```

> 💡 실제로는 단어들이 완전히 독립이지 않지만, 이 단순한 가정에도 불구하고 텍스트 분류에서 놀랍도록 잘 작동합니다.

### 4.2 나이브 베이즈의 종류

| 종류 | 사용 상황 | 예시 |
|------|---------|------|
| **GaussianNB** | 연속형 피처, 정규 분포 가정 | 의료 데이터, 센서 데이터 |
| **MultinomialNB** | 정수 카운트 피처 | 텍스트 분류 (단어 빈도) |
| **BernoulliNB** | 이진 피처 (0/1) | 단어 출현 여부 |

### 4.3 텍스트를 숫자로: CountVectorizer와 TF-IDF

나이브 베이즈 스팸 필터를 만들기 전에, 텍스트를 숫자 벡터로 변환하는 방법부터 살펴봅시다.

| 방식 | 의미 | 특징 |
|------|------|------|
| `CountVectorizer` | 단어가 몇 번 나왔는지 세기 | 단순하고 직관적 |
| `TfidfVectorizer` | 자주 나오지만 덜 중요한 단어의 영향 감소 | 실전에서 자주 사용 |

처음에는 CountVectorizer로 "문장이 숫자 벡터로 바뀐다"는 감각을 잡고, 그 다음 TF-IDF를 사용하면 이해가 쉽습니다.

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "free money now",
    "meeting tomorrow",
    "free prize"
]

vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(texts)

print("단어 목록:", vectorizer.get_feature_names_out())
print("\n단어 카운트 행렬:")
print(X_count.toarray())
```
**출력 결과:**
```
단어 목록: ['free' 'meeting' 'money' 'now' 'prize' 'tomorrow']

단어 카운트 행렬:
[[1 0 1 1 0 0]   ← "free money now"
 [0 1 0 0 0 1]   ← "meeting tomorrow"
 [1 0 0 0 1 0]]  ← "free prize"
```

> 💡 각 문장이 숫자 벡터로 변환되었습니다. `free`는 1번째 열, `meeting`은 2번째 열처럼 단어마다 열이 하나씩 생깁니다.

### 4.4 sklearn으로 나이브 베이즈 구현 — 스팸 필터링

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import pandas as pd

# 예시 SMS 스팸 데이터 (알고리즘 흐름 확인용 샘플)
messages = [
    ("ham",  "Hey, what time are we meeting tomorrow?"),
    ("ham",  "I'll be home by 7pm"),
    ("spam", "FREE! Win £1000 cash now! Call 09061 501234"),
    ("ham",  "Ok I'll check it out later"),
    ("spam", "Congratulations! You've won a free phone. Claim now!"),
    ("ham",  "Can you send me the report by end of day?"),
    ("spam", "URGENT: Your mobile number has won £5000!"),
    ("ham",  "Let's grab coffee this weekend"),
    ("spam", "Get free cash prizes! Reply WIN to 80085"),
    ("ham",  "Are you coming to the meeting?"),
    ("spam", "You have been selected! Call now to claim your prize"),
    ("ham",  "Thanks for your help today"),
    ("spam", "Free entry to our competition! Text now"),
    ("ham",  "I'm running a bit late, see you soon"),
    ("spam", "Limited offer! Buy 1 get 2 free. Call 08000")
]

df_sms = pd.DataFrame(messages, columns=['label', 'text'])
df_sms['label_enc'] = (df_sms['label'] == 'spam').astype(int)

X = df_sms['text']
y = df_sms['label_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Pipeline: 벡터화 → 모델 학습을 한 번에
nb_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),  # 단어 + 2-gram
    ('classifier', MultinomialNB(alpha=1.0))               # 라플라스 스무딩
])

nb_pipeline.fit(X_train, y_train)
y_pred = nb_pipeline.predict(X_test)

print("=== 나이브 베이즈 스팸 필터 결과 ===")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
```
**출력 결과:**
```
=== 나이브 베이즈 스팸 필터 결과 ===
              precision    recall  f1-score   support

         ham       0.75      1.00      0.86         3
        spam       1.00      0.75      0.86         4

    accuracy                           0.86         7
   macro avg       0.88      0.88      0.86         7
weighted avg       0.89      0.86      0.86         7
```

> ⚠️ **주의**: 이 예시는 알고리즘 흐름을 보여주기 위한 15개짜리 샘플입니다. 실제 성능을 확인하려면 수백~수천 개 이상의 SMS 데이터셋으로 평가해야 합니다. 작은 데이터에서는 train/test 분리에 따라 점수가 크게 달라질 수 있습니다.

> 💡 **실제 데이터로 더 실습하고 싶다면**: Kaggle의 [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)을 추천합니다. ham 4,827건 + spam 747건, 총 5,574건으로 구성되어 있어 실제 모델 성능을 확인하기에 적합합니다. 위 코드 구조 그대로 데이터만 교체하면 바로 실행할 수 있습니다.

```python
# 새 메시지 스팸 판별 테스트
new_messages = [
    "You've won a FREE prize! Call now to claim!",
    "Can we reschedule the meeting to 3pm?",
    "URGENT: Claim your £2000 reward today!"
]

predictions  = nb_pipeline.predict(new_messages)
probabilities = nb_pipeline.predict_proba(new_messages)

print("\n=== 새 메시지 분류 ===")
for msg, pred, prob in zip(new_messages, predictions, probabilities):
    label = "🚫 SPAM" if pred == 1 else "✅ HAM"
    print(f"\n메시지: {msg[:50]}")
    print(f"  판정: {label}  (스팸 확률: {prob[1]*100:.1f}%)")
```
**출력 결과:**
```
=== 새 메시지 분류 ===

메시지: You've won a FREE prize! Call now to claim!
  판정: 🚫 SPAM  (스팸 확률: 87.3%)

메시지: Can we reschedule the meeting to 3pm?
  판정: ✅ HAM  (스팸 확률: 4.2%)

메시지: URGENT: Claim your £2000 reward today!
  판정: 🚫 SPAM  (스팸 확률: 91.6%)
```

### 4.5 alpha — 라플라스 스무딩

```python
# alpha 파라미터: 훈련 데이터에 없는 단어 처리
# alpha=0 → 새 단어 등장 시 확률이 0이 됨 → 전체 예측이 망가짐
# alpha=1 → 각 단어에 카운트 1을 추가 (기본값, 권장)
# alpha>1 → 더 강한 스무딩 (데이터가 매우 적을 때)

from sklearn.model_selection import cross_val_score

alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
print(f"{'alpha':>6}  {'교차검증 정확도':>16}")
print("-" * 28)

for alpha in alphas:
    pipeline = Pipeline([
        ('vec', TfidfVectorizer()),
        ('clf', MultinomialNB(alpha=alpha))
    ])
    scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')
    print(f"{alpha:>6.2f}  {scores.mean():>10.4f} ± {scores.std():.4f}")
```
**출력 결과:**
```
 alpha  교차검증 정확도
----------------------------
  0.01    0.7333 ± 0.0943
  0.10    0.8000 ± 0.0816
  0.50    0.8667 ± 0.0667
  1.00    0.8667 ± 0.0667
  2.00    0.8000 ± 0.0816
  5.00    0.8000 ± 0.0816
```

### 4.6 나이브 베이즈 장단점 정리

| 구분 | 내용 |
|------|------|
| **장점** | 학습·예측 속도가 매우 빠름 |
| **장점** | 학습 데이터가 적어도 잘 작동 |
| **장점** | 텍스트처럼 피처가 많고 희소한 데이터에서 빠르고 강력함 |
| **장점** | 노이즈가 있는 고차원 희소 데이터에서도 비교적 잘 작동 |
| **단점** | 피처 독립 가정이 비현실적 |
| **단점** | 수치 피처가 많은 데이터에는 부적합 |
| **단점** | 피처 간 상호작용을 무시함 |

**스케일링 관련 주의사항**

```python
# SVM, KNN   → 스케일링 필수
# GaussianNB → 각 피처별 평균·분산을 사용하므로 스케일링이 필수는 아님
# MultinomialNB → 음수 값 불가, Count/TF-IDF 같은 비음수 피처 사용
```

---

## 5. SVM vs KNN vs Naive Bayes — 실전 비교

같은 데이터(유방암 데이터)로 알고리즘을 직접 비교해봅시다.

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import time

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 각 모델을 Pipeline으로 묶어 스케일링까지 포함
models = {
    'SVM (RBF)':     Pipeline([('sc', StandardScaler()), ('m', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))]),
    'KNN (k=7)':     Pipeline([('sc', StandardScaler()), ('m', KNeighborsClassifier(n_neighbors=7))]),
    'Naive Bayes':   Pipeline([('sc', StandardScaler()), ('m', GaussianNB())]),
    'Logistic Reg':  Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=10000, random_state=42))]),
    'Random Forest': Pipeline([('sc', StandardScaler()), ('m', RandomForestClassifier(n_estimators=100, random_state=42))])
}

print(f"{'모델':<18} {'정확도':>8} {'F1':>8} {'5-Fold CV':>10} {'학습(ms)':>10}")
print("-" * 60)

for name, pipeline in models.items():
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = (time.time() - start) * 1000

    y_pred  = pipeline.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    cv_mean = cross_val_score(pipeline, X_train, y_train, cv=5).mean()

    print(f"{name:<18} {acc:>8.4f} {f1:>8.4f} {cv_mean:>10.4f} {train_time:>9.1f}ms")
```
**출력 결과:**
```
모델               정확도       F1   5-Fold CV   학습(ms)
------------------------------------------------------------
SVM (RBF)          0.9825   0.9863     0.9780      23.4ms
KNN (k=7)          0.9737   0.9797     0.9648       2.1ms
Naive Bayes        0.9386   0.9517     0.9363       1.8ms
Logistic Reg       0.9649   0.9730     0.9648      14.7ms
Random Forest      0.9649   0.9730     0.9538     312.8ms
```

> 💡 **결과 해석**:
> - SVM이 가장 높은 정확도지만 학습 시간도 상대적으로 깁니다
> - Naive Bayes가 가장 빠르게 학습했고 성능도 나쁘지 않습니다
> - "항상 좋은 알고리즘"은 없습니다 — 데이터와 목적에 맞게 선택하세요

---

## 6. 알고리즘 선택 가이드

| 상황 | 추천 알고리즘 | 이유 |
|------|------------|------|
| 텍스트 분류 (스팸, 감성 분석) | **Naive Bayes** | 희소 고차원 데이터에 빠르고 강력 |
| 고차원 소규모 데이터 | **SVM** | 고차원에서 일반화 우수 |
| 대용량 데이터 (수만 건+) | **Logistic Regression** | SVM·KNN은 대용량에서 느림 |
| 빠른 프로토타이핑 | **KNN** | 구현 단순, 해석 직관적 |
| 범용 고성능 모델 | **Random Forest** | 전처리 부담 적고 안정적 |

```
상황별 선택 흐름

텍스트(문서, 메시지)?  → YES → Naive Bayes
    ↓ NO
피처가 매우 많고 데이터는 적음? → YES → SVM
    ↓ NO
데이터가 수만 건 이상? → YES → Logistic Regression
    ↓ NO
빠른 프로토타이핑 필요? → YES → KNN (스케일링 필수)
    ↓ NO
범용 최고 성능 목표? → Random Forest / XGBoost 비교
```

---

## 7. 실전 팁 — 공통 주의사항 정리

### Pipeline은 선택이 아닌 습관

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ✅ 올바른 방법: Pipeline으로 전처리 + 모델 묶기
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  SVC(kernel='rbf'))
])

# 이렇게 하면:
# - 교차검증 시 각 fold마다 스케일러가 훈련 데이터에만 fit()
# - GridSearchCV와 함께 써도 데이터 누수 없음
# - 나중에 새 데이터가 들어와도 일관된 전처리 적용
```

### 교차 검증으로 안정적인 성능 측정

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, scoring='f1')
    print(f"{name:<18}: {scores.mean():.4f} ± {scores.std():.4f}")
```
**출력 결과:**
```
SVM (RBF)         : 0.9805 ± 0.0121
KNN (k=7)         : 0.9740 ± 0.0143
Naive Bayes       : 0.9542 ± 0.0215
Logistic Reg      : 0.9726 ± 0.0158
Random Forest     : 0.9712 ± 0.0094
```

> 💡 표준편차(±)가 작을수록 데이터 분리에 관계없이 **일관된 성능**을 냅니다. 표준편차가 크다면 모델이 데이터 분리에 민감하다는 신호입니다.

---

## DAY3 정리

오늘 배운 핵심을 정리하면:

```
✅ SVM = 마진을 최대화하는 초평면을 찾는 알고리즘
   - 서포트 벡터: 모델 결정에 관여하는 소수의 데이터 포인트
   - 커널 트릭: 고차원 변환으로 비선형 분류 가능 (RBF 커널이 기본값)
   - C: 오분류 허용 수준 / gamma: 결정 경계 복잡도
   - probability=True 설정 시 predict_proba() 사용 가능

✅ KNN = 가장 가까운 K개 이웃의 다수결로 분류
   - 게으른 학습: 학습 단계엔 저장만, 예측 시 거리 계산
   - K값 선택이 핵심: 교차검증으로 선택, 테스트 데이터 사용 금지
   - 차원의 저주: 피처가 너무 많으면 거리 계산이 무의미해짐

✅ Naive Bayes = 베이즈 정리 + 피처 독립 가정으로 빠른 분류
   - 사전확률 → 조건부확률 → 사후확률 계산
   - 텍스트 분류에서 탁월 (CountVectorizer → TF-IDF 순서로 학습)
   - alpha(라플라스 스무딩)로 미관측 단어 문제 해결

✅ 공통 핵심 습관
   - Pipeline으로 전처리 + 모델을 묶어 데이터 누수 방지
   - 하이퍼파라미터(K, C, gamma 등)는 교차검증으로 선택
   - 테스트 데이터는 최종 평가에만 딱 한 번 사용

✅ 알고리즘 선택 기준
   - 텍스트 데이터 → Naive Bayes
   - 고차원 소규모 데이터 → SVM
   - 대용량 데이터 → Logistic Regression
   - 범용 고성능 → Random Forest
```

> 다음 단계로는 **XGBoost / LightGBM** 같은 그래디언트 부스팅 계열 앙상블 모델이나, **PyTorch로 신경망(ANN)** 을 직접 구현해보는 딥러닝 입문을 추천합니다.

---

## 🔗 참고 자료

- [scikit-learn SVC 공식 문서](https://scikit-learn.org/stable/modules/svm.html)
- [scikit-learn KNeighborsClassifier](https://scikit-learn.org/stable/modules/neighbors.html)
- [scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/pipeline.html)
- [Google Colab](https://colab.research.google.com/)
- [Kaggle — 데이터셋 & 대회](https://www.kaggle.com/)
