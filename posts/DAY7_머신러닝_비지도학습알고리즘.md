# 🔍 머신러닝 완전 입문 가이드 — DAY7. 비지도학습 알고리즘

> **시리즈**: 파이썬 기본만 있는 사람을 위한 머신러닝 입문
> **DAY1**: 머신러닝 핵심 개념과 데이터 관리 (이전 편)
> **DAY2**: 회귀 모델과 분류 모델 실습 (이전 편)
> **DAY3**: SVM · KNN · 나이브 베이즈 (이전 편)
> **DAY4**: 결정트리 · 회귀트리 · 선형 회귀 · 로지스틱 회귀 · 역전파 (이전 편)
> **DAY5**: 앙상블 · ANN (이전 편)
> **DAY6**: 차원 축소 & 시각화 (이전 편)
> **DAY7**: 비지도학습 알고리즘

---

## 1. 비지도학습이란?

**비지도학습(Unsupervised Learning)** 은 정답(Label)이 없는 데이터를 이용해 데이터 내부에 숨어 있는 패턴, 구조, 관계를 스스로 찾아내는 머신러닝 기법입니다.

DAY1~6까지 다룬 대부분의 알고리즘은 **지도학습** — 즉 "정답이 있는 문제"를 학습하는 방식이었습니다. 비지도학습은 반대로 정답 없이 데이터만 주어진 상태에서 데이터의 특성과 유사성을 분석해 의미 있는 정보를 발견하는 것이 목적입니다.

```
[지도학습 vs 비지도학습]

지도학습:
  입력 X  +  정답 y  →  모델 학습
  예) "이 이메일은 스팸이다 / 아니다" 라고 라벨이 붙은 데이터

비지도학습:
  입력 X  (정답 없음)  →  패턴 발견
  예) "이 고객들 중 비슷한 사람끼리 묶어봐라"
```

예를 들어 다음 고객 데이터가 있다고 하자.

```
| 고객번호 | 나이 | 구매금액  |
|---------|------|---------|
| A       |  20  | 50만원  |
| B       |  22  | 45만원  |
| C       |  55  | 300만원 |
| D       |  58  | 280만원 |
```

누군가 "이 고객의 등급은 무엇인가?"라는 정답을 붙이지 않아도, 비지도학습은 다음 질문에 스스로 답합니다.

- 어떤 고객들이 비슷한가?
- 고객 그룹은 몇 개로 나눌 수 있는가?
- 이상한 데이터는 존재하는가?

---

### 1.1 비지도학습의 4가지 목적

```
비지도학습
├── 군집화 (Clustering)              — 비슷한 데이터끼리 그룹으로 묶기
├── 차원 축소 (Dimensionality Reduction) — 많은 특성을 적은 특성으로 압축
├── 연관 규칙 분석 (Association Rule)    — 함께 발생하는 패턴 발견
└── 이상치 탐지 (Anomaly Detection)      — 정상 패턴에서 벗어난 데이터 탐지
```

| 목적 | 설명 | 대표 활용 |
|------|------|---------|
| **군집화** | 비슷한 데이터끼리 자동으로 그룹화 | 고객 세분화, 상품 추천 |
| **차원 축소** | 고차원 → 저차원으로 압축 | 시각화, 노이즈 제거, 학습 속도 향상 |
| **연관 규칙** | 함께 등장하는 항목의 패턴 발견 | 장바구니 분석, 추천 시스템 |
| **이상치 탐지** | 정상 패턴과 다른 데이터 탐지 | 신용카드 사기, 네트워크 침입 탐지 |

> 💡 **차원 축소는 DAY6에서 PCA·t-SNE·UMAP·Autoencoder로 이미 깊이 다뤘습니다.** DAY7에서는 군집화·연관 규칙·이상치 탐지에 집중합니다.

---

### 1.2 비지도학습은 요즘 어디에 많이 쓰일까?

비지도학습은 정답이 없는 데이터에서 구조를 찾는 방식이기 때문에, 최근 AI 서비스에서도 자주 사용됩니다.

- **고객 세분화**: 구매 패턴이 비슷한 고객을 자동으로 묶어 마케팅 전략 수립
- **문서 그룹화**: 비슷한 주제의 뉴스, 블로그, 문서를 자동 분류
- **이미지·영상 분석**: 비슷한 시각적 특징을 가진 이미지끼리 묶기
- **이상 거래 탐지**: 평소와 다른 카드 사용, 네트워크 접속, 센서 값 탐지
- **추천 시스템**: 함께 소비되는 상품·콘텐츠 패턴 발견
- **RAG·벡터 검색**: 문장을 임베딩 벡터로 바꾼 뒤, 의미적으로 가까운 문서를 클러스터링하거나 검색

즉, 비지도학습은 단순히 "정답 없는 학습"이 아니라, 데이터 안에 숨어 있는 패턴을 찾아 이후 분석이나 서비스에 활용하는 **기반 기술**입니다.

---

### 1.3 비지도학습 알고리즘 전체 구조

```
비지도학습 알고리즘
│
├── 군집화 (Clustering)
│   ├── K-Means            — 중심점 거리 기반, 가장 빠름
│   ├── 계층적 클러스터링    — 트리 구조로 단계적 병합
│   └── DBSCAN             — 밀도 기반, 이상치 자동 탐지
│
├── 연관 규칙 분석
│   └── Apriori            — 장바구니 분석 대표 알고리즘
│
└── 이상치 탐지
    ├── Isolation Forest   — 트리 기반, 빠른 이상치 분리
    └── One-Class SVM      — 정상 영역 학습 후 이상치 판별
```

> 참고: PDF에는 Autoencoder도 비지도학습 알고리즘의 한 종류로 소개됩니다. Autoencoder는 신경망 기반 차원 축소·특징 추출 방법이므로 DAY6 차원 축소 파트에서 자세히 다뤘습니다. DAY7에서는 군집화, 연관 규칙, 이상치 탐지에 집중합니다.

---

### 1.4 비지도학습 결과는 "정답"이 아니라 "해석 가능한 가설"이다

비지도학습 입문자가 가장 많이 하는 오해가 있습니다.

> **"클러스터링 결과가 나오면 그 군집이 정답이라고 생각한다."**

하지만 비지도학습은 라벨 없이 패턴을 찾는 방식이기 때문에, 결과를 그대로 정답처럼 받아들이면 위험합니다.

예를 들어 고객 데이터를 3개 군집으로 나눴다고 해서, 그 3개 군집이 반드시 현실의 고객 유형을 완벽히 의미하는 것은 아닙니다. 분석가는 각 군집의 평균값, 분포, 대표 샘플을 확인하면서 다음과 같이 **직접 해석**해야 합니다.

```
0번 군집 → 구매 금액이 높고 방문 빈도가 낮다  → "고가 구매 고객"
1번 군집 → 구매 금액은 낮지만 방문 빈도가 높다 → "자주 오는 소액 고객"
2번 군집 → 최근 가입한 신규 고객이 많다       → "신규 유입 고객"
```

즉, 비지도학습은 모델 결과 자체보다 **결과를 어떻게 해석하고 이름 붙이는지**가 매우 중요합니다. 클러스터링은 그룹을 자동으로 나눠주는 단계이고, 그 그룹에 의미를 부여하는 것은 **분석가의 역할**입니다.

---

## 2. 클러스터링 기초: 거리 계산

모든 클러스터링 알고리즘의 핵심은 **"두 데이터가 얼마나 가까운가"** 입니다. 가장 많이 사용하는 거리 공식은 **유클리드 거리(Euclidean Distance)** 입니다.

### 2.1 유클리드 거리

```
두 점 A = (x1, y1),  B = (x2, y2) 사이의 거리:

d(A, B) = √((x2-x1)² + (y2-y1)²)
```

숫자 예시로 바로 이해해봅시다.

```
A = (1, 2),  B = (4, 6)

x 차이 = 4 - 1 = 3
y 차이 = 6 - 2 = 4

거리 = √(3² + 4²)
     = √(9 + 16)
     = √25
     = 5
```

> 💡 **피타고라스 정리**와 같습니다. 직각삼각형의 빗변 = 두 점 사이의 거리입니다.

특성이 여러 개일 때는 다음처럼 확장됩니다.

```
d(x, y) = √(Σ (xj - yj)²)   (j = 1 ... m, m = 특성 개수)

예: Iris 데이터 (특성 4개)
   A = [5.1, 3.5, 1.4, 0.2]
   B = [4.9, 3.0, 1.4, 0.2]
   거리 = √((5.1-4.9)² + (3.5-3.0)² + 0² + 0²)
        = √(0.04 + 0.25)
        ≈ 0.54
```

---

### 2.2 클러스터링 전 표준화가 필수인 이유

거리 계산 기반 알고리즘은 단위가 다른 변수가 섞이면 결과가 왜곡됩니다.

```
나이:  20 ~ 70         → 범위: 약 50
연봉:  2,000만 ~ 1억   → 범위: 약 8,000만
```

표준화 없이 거리를 계산하면 연봉 차이가 압도적으로 커서 나이는 사실상 무시됩니다. **`StandardScaler`로 먼저 표준화한 뒤 클러스터링을 적용해야 합니다.**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# → 이후 X_scaled를 클러스터링 알고리즘에 입력
```

> ⚠️ K-Means, 계층적 클러스터링처럼 거리 기반 알고리즘은 **표준화 없이 쓰면 스케일이 큰 변수가 결과를 좌우**합니다. DBSCAN도 마찬가지입니다. Isolation Forest는 트리 기반이라 거리 스케일에 직접 민감하지 않지만, 여러 모델을 비교하거나 이후 거리 기반 모델과 함께 쓸 계획이면 표준화를 적용해도 무방합니다.

---

### 2.3 범주형 데이터에 K-Means를 바로 적용하면 위험하다

K-Means는 평균과 유클리드 거리를 기반으로 동작합니다. 따라서 다음과 같은 **범주형 변수에는 바로 적용하기 어렵습니다.**

```
성별, 지역, 직업, 상품 카테고리, 가입 경로 ...
```

범주형 변수를 숫자로 단순 치환하면 모델이 잘못된 크기 관계를 학습합니다.

```python
# ❌ 이렇게 치환하면 모델이 "대구(2) > 부산(1) > 서울(0)"이라고 해석
지역 인코딩:  서울=0, 부산=1, 대구=2
```

실제로 지역 이름에는 이런 크기 관계가 없습니다. 범주형 데이터가 많다면 다음 방법을 고려하세요.

- **One-Hot Encoding** 후 거리 기반 클러스터링 적용
- 숫자형 변수 중심으로 먼저 군집화한 뒤, 범주형 변수는 **해석 단계에서** 사용
- 혼합형 데이터에 맞는 Gower Distance 같은 거리 척도 사용

---

## 3. K-Means 클러스터링

### 3.1 핵심 개념

K-Means는 가장 대표적인 클러스터링 알고리즘입니다. **K**개의 중심점(Centroid)을 기준으로 각 데이터를 가장 가까운 중심점에 배정하고, 중심점을 반복적으로 갱신하면서 군집을 완성합니다.

```
[K-Means 동작 과정 — K=2 예시]

Step 1. 중심점 2개를 임의로 선택
         ★           ★
         (중심1)       (중심2)

Step 2. 각 데이터를 더 가까운 중심점에 배정
         ●●●  ★  ●     ★  ○○○

Step 3. 배정된 데이터들의 평균 → 중심점 이동
         ●●●  ★→           →★  ○○○

Step 4. 중심점이 변하지 않을 때까지 반복 → 수렴 시 학습 종료
```

> 💡 **"K-Means"라는 이름**: K개의 그룹을 각 그룹의 평균(mean)으로 표현하기 때문입니다. 중심점은 항상 해당 군집 데이터들의 평균 좌표입니다.

---

### 3.2 목적 함수

K-Means는 각 데이터와 자신이 속한 군집 중심점 사이의 거리 제곱합을 최소화합니다.

```
J = Σ(k=1..K) Σ(xi ∈ Ck) ||xi - μk||²

J:   전체 오차 (작을수록 군집이 잘 형성됨)
xi:  하나의 데이터
μk:  k번째 군집의 중심점
Ck:  k번째 군집에 속한 데이터 집합
```

---

### 3.3 손계산 예시

```
데이터:
  A = (1, 1),  B = (2, 1),  C = (8, 8),  D = (9, 8)
군집 수: K = 2
초기 중심점: 중심1 = A = (1,1),  중심2 = C = (8,8)

--- Step 1. 각 데이터를 가까운 중심점에 배정 ---

B와 중심1 거리: √((2-1)² + (1-1)²) = 1
B와 중심2 거리: √((8-2)² + (8-1)²) = √85 ≈ 9.2
→ B는 중심1에 더 가까움

C, D는 중심2에 더 가까움

결과:
  군집 1 = {A, B}
  군집 2 = {C, D}

--- Step 2. 중심점 갱신 ---

군집 1 중심점 = ((1+2)/2, (1+1)/2) = (1.5, 1.0)
군집 2 중심점 = ((8+9)/2, (8+8)/2) = (8.5, 8.0)

--- Step 3. 다시 배정 → 변화 없음 → 수렴 종료 ---
```

---

### 3.4 K-Means가 잘 맞는 경우와 잘 안 맞는 경우

K-Means는 빠르고 이해하기 쉽지만 모든 데이터에 잘 맞지는 않습니다.

**잘 맞는 경우**
- 군집이 둥근 구형 형태에 가깝다
- 군집 간 크기 차이가 크지 않다
- 이상치가 많지 않다
- 숫자형 변수 중심의 데이터다

**성능이 떨어지는 경우**
- 초승달 모양처럼 비선형 구조를 가진 데이터
- 군집마다 밀도가 크게 다른 데이터
- 이상치가 많은 데이터
- 범주형 변수가 많은 데이터

이런 상황에서는 **DBSCAN, 계층적 클러스터링, 차원 축소 후 클러스터링**을 고려하세요.

---

### 3.5 K 값 선택: 엘보우 방법 (Elbow Method)

K-Means의 가장 큰 문제는 **K를 미리 정해야 한다**는 점입니다. 엘보우 방법으로 적절한 K를 찾을 수 있습니다.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.xlabel('K (군집 수)')
plt.ylabel('Inertia (군집 내 오차 제곱합)')
plt.title('Elbow Method — 적절한 K 찾기')
plt.xticks(K_range)
plt.grid(True)
plt.show()
```

**결과 해석:**
```
Inertia 값이 K가 늘어날수록 감소합니다.
감소 폭이 갑자기 줄어드는 "꺾이는 지점(팔꿈치)"이 최적 K입니다.

Iris 데이터 예: K=3 부근에서 꺾임 → K=3 선택
```

> ⚠️ 엘보우 방법은 항상 뚜렷한 꺾임을 보장하지 않습니다. 꺾임이 불분명하면 뒤에서 배울 **실루엣 계수**로 보완하세요.

---

### 3.6 전체 코드 예제 (Iris 데이터)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# 1. 데이터 로드 & 표준화
iris = load_iris()
X = iris.data
y = iris.target  # 정답 (평가 참고용으로만 사용)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. K-Means 적용 (K=3)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

# 3. 시각화 (PCA로 2차원 축소 후 시각화)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', alpha=0.7)
plt.title("K-Means 군집 결과")
plt.xlabel("PC1"); plt.ylabel("PC2")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Set1', alpha=0.7)
plt.title("실제 품종 (참고용)")
plt.xlabel("PC1"); plt.ylabel("PC2")

plt.tight_layout()
plt.show()
```

> ⚠️ **Iris 데이터는 학습용 예제이기 때문에 실제 품종 라벨이 함께 제공됩니다.** 하지만 실제 비지도학습 문제에서는 정답 라벨이 없는 경우가 대부분입니다. 여기서 실제 라벨과 비교하는 것은 "K-Means가 데이터를 어떻게 나눴는지 확인하기 위한 참고용"입니다. 또한 클러스터 번호는 임의로 붙기 때문에 `cluster=0`이 반드시 `setosa=0`과 대응된다는 뜻은 아닙니다.

---

### 3.7 군집 결과 해석하기: 군집별 평균 확인 (프로파일링)

클러스터링은 `fit_predict()`에서 끝나는 것이 아닙니다. **각 군집이 어떤 특성을 가지는지 확인하는 것**이 실무에서 가장 중요한 단계입니다.

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data
feature_names = iris.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

df = pd.DataFrame(X, columns=feature_names)
df["cluster"] = labels

# 군집별 데이터 개수
print("=== 군집별 데이터 수 ===")
print(df["cluster"].value_counts().sort_index())

# 군집별 평균값 (원본 스케일로 확인)
print("\n=== 군집별 평균값 ===")
print(df.groupby("cluster").mean().round(2))
```

**출력 예시:**
```
=== 군집별 데이터 수 ===
0    50
1    62
2    38

=== 군집별 평균값 ===
         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
cluster
0                     5.01              3.43               1.46              0.25
1                     5.90              2.75               4.39              1.43
2                     6.85              3.07               5.74              2.07
```

이렇게 각 군집의 평균을 보면, 직접 해석해서 이름을 붙일 수 있습니다.

| 군집 | 꽃잎 길이·너비 | 해석 예시 |
|------|------------|--------|
| 0번 | 매우 작음 | setosa 계열 (작은 꽃) |
| 1번 | 중간 | versicolor 계열 |
| 2번 | 매우 큼 | virginica 계열 (큰 꽃) |

> 💡 **클러스터 번호는 등급이나 크기를 의미하지 않습니다.** `cluster=0`이 `cluster=1`보다 더 좋은 고객이라는 뜻이 아니고, `cluster=2`가 가장 크다는 뜻도 아닙니다. 클러스터 번호는 모델이 임의로 붙인 그룹 ID입니다. 분석할 때는 반드시 각 군집의 특성을 따로 확인한 뒤 의미를 부여해야 합니다.

---

### 3.8 비지도학습도 데이터 누수에 주의해야 한다

클러스터링이나 차원 축소를 **단순 탐색 목적**으로 사용할 때는 전체 데이터를 대상으로 분석하는 경우가 많습니다. 그러나 클러스터 라벨이나 PCA 결과를 이후 **지도학습 모델의 입력 변수로 사용**할 때는 데이터 누수에 주의해야 합니다.

```python
# ❌ 잘못된 예: 전체 데이터로 스케일링과 클러스터링을 먼저 수행
X_scaled = scaler.fit_transform(X)         # 테스트 데이터 정보 포함
labels   = km.fit_predict(X_scaled)        # 테스트 데이터 영향 포함

X_train, X_test, y_train, y_test = train_test_split(...)  # 이미 누수

# ✅ 올바른 예: train/test 분리 → 훈련 데이터에만 fit
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

km.fit(X_train_s)
train_labels = km.predict(X_train_s)   # K-Means는 predict 지원
test_labels  = km.predict(X_test_s)
```

> ⚠️ 비지도학습 알고리즘도 나중에 예측 모델의 일부로 사용한다면 **train/test 분리 원칙을 지켜야 합니다.** (DAY1, DAY6에서 다룬 데이터 누수와 같은 맥락입니다.)

| 항목 | 내용 |
|------|------|
| **장점** | 구현 쉬움, 계산 빠름, 대용량 데이터 처리 가능 |
| **단점** | K를 미리 지정해야 함, 이상치에 민감, 구형 군집만 잘 찾음, 범주형 변수 직접 적용 어려움 |
| **언제 쓰나** | 군집 수를 예측할 수 있을 때, 빠른 탐색적 분석 |

---

## 4. 계층적 클러스터링 (Hierarchical Clustering)

### 4.1 핵심 개념

계층적 클러스터링은 데이터를 **단계적으로 묶어 나가는** 방식입니다. K-Means와 달리 군집 수를 미리 정하지 않아도 되며, 결과를 **덴드로그램(Dendrogram)** 이라는 트리 형태로 시각화할 수 있습니다.

```
Agglomerative (병합형, 아래→위)    Divisive (분할형, 위→아래)
가장 많이 쓰임                      계산 비용이 커서 잘 안 씀

Step 1: {A}, {B}, {C}, {D}        Step 1: {A, B, C, D}
Step 2: {A,B}, {C}, {D}           Step 2: {A,B}, {C,D}
Step 3: {A,B}, {C,D}              Step 3: {A,B}, {C}, {D}
Step 4: {A,B,C,D}                 Step 4: {A}, {B}, {C}, {D}
```

> 💡 **덴드로그램을 잘라서 군집 수를 결정합니다.** 트리의 어느 높이에서 수평으로 잘라내느냐에 따라 군집 수가 달라집니다. 이 방식 덕분에 "K를 먼저 정하는" 과정 없이도 결과를 보고 나중에 적절한 군집 수를 선택할 수 있습니다. 계층적 클러스터링은 데이터가 많아지면 덴드로그램이 매우 복잡해집니다. 실무에서는 전체 데이터를 모두 그리기보다 일부 샘플만 보거나, `truncate_mode`로 상위 구조만 확인하는 경우가 많습니다.

---

### 4.2 군집 간 거리 기준 (Linkage)

어떤 방법으로 두 군집 사이의 거리를 계산하느냐에 따라 결과가 달라집니다.

| 방법 | 거리 기준 | 특징 |
|------|---------|------|
| **단일 연결법** (Single) | 두 군집 중 가장 가까운 점 사이의 거리 | 긴 사슬 모양 군집 생성 가능 |
| **완전 연결법** (Complete) | 두 군집 중 가장 먼 점 사이의 거리 | 비교적 조밀하고 균형 잡힌 군집 |
| **평균 연결법** (Average) | 두 군집 모든 점 쌍의 거리 평균 | 단일·완전 연결법의 중간 성격 |
| **Ward 연결법** | 병합 후 군집 내 오차 증가량 최소화 | 일반적으로 좋은 결과, **기본 추천** |

> 💡 처음 시도할 때는 `linkage='ward'`를 기본값으로 사용하세요. K-Means의 목적 함수(군집 내 오차 제곱합 최소화)와 비슷한 기준이라 직관적이고 결과가 안정적입니다. 단, Ward 연결법은 **유클리드 거리와 함께 사용**하는 것이 일반적입니다. 다른 거리 척도를 사용하고 싶다면 `average`, `complete` 같은 연결법을 고려하세요.

---

### 4.3 코드 예제

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 덴드로그램 시각화 (군집 수 결정 참고용) ─────────────────────────────────
plt.figure(figsize=(12, 5))
Z = linkage(X_scaled, method='ward')
dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=45)
plt.title("덴드로그램 — 군집 수 결정에 활용")
plt.xlabel("데이터 (또는 군집)")
plt.ylabel("거리 (Ward)")
plt.show()
# → 긴 수직선이 끊기지 않고 이어지는 구간을 찾아 절단 높이 결정
# → 절단 높이에서 수평선과 교차하는 수직선 수 = 군집 수

# ── 군집화 & 시각화 ───────────────────────────────────────────────────────
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X_scaled)

X_pca = PCA(n_components=2).fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', alpha=0.7)
plt.title("계층적 클러스터링 결과 (n_clusters=3)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()
```

**덴드로그램 읽는 법:**
```
y축(높이) = 두 군집이 병합될 때의 거리
높이가 높을수록 → 두 군집이 서로 많이 다름
가장 긴 수직선에 수평 절단선을 그으면 그 수직선 수 = 군집 수

예: 높이 5에서 절단 → 수직선 3개 교차 → 3개 군집
```

| 항목 | 내용 |
|------|------|
| **장점** | 군집 수 미리 결정 불필요, 계층 구조 시각화 가능 |
| **단점** | 계산 비용 큼(O(n³)), 대용량 데이터 비적합, 병합된 군집 수정 불가 |
| **언제 쓰나** | 데이터가 어떻게 계층적으로 묶이는지 탐색할 때, 소규모 데이터 |

---

## 5. DBSCAN

### 5.1 핵심 개념

DBSCAN(Density-Based Spatial Clustering of Applications with Noise)은 **밀도** 기반 클러스터링입니다. K-Means처럼 군집 수를 미리 정하지 않아도 되고, **이상치(Noise)를 자동으로 탐지**한다는 큰 장점이 있습니다.

```
"어떤 점 주변에 데이터가 충분히 많은 지역 = 하나의 군집"
"주변에 데이터가 거의 없는 점 = 이상치(Noise)"
```

> ⚠️ **DBSCAN은 고차원 데이터에서 성능이 떨어질 수 있습니다.** 차원이 많아지면 데이터 간 거리가 서로 비슷해지는 현상이 생겨 밀도 차이를 구분하기 어려워집니다. 이런 경우 **PCA나 UMAP으로 차원을 줄인 뒤 DBSCAN을 적용**하는 방법을 고려하세요.

---

### 5.2 두 가지 파라미터와 세 가지 점

```
eps (ε):         이웃으로 볼 반경
min_samples:     eps 반경 안에 있어야 하는 최소 데이터 수
```

```
핵심점 (Core Point):
  eps 반경 안에 min_samples 개 이상의 데이터 → 핵심점
  핵심점끼리 연결되면서 군집이 확장됨

경계점 (Border Point):
  자기 주변엔 데이터가 부족하지만,
  어떤 핵심점의 반경 안에 포함되는 점 → 그 군집에 속함

잡음점 (Noise Point):
  어떤 군집에도 속하지 않는 점 → 이상치 (label = -1)
```

---

### 5.3 eps 후보 찾기: k-distance graph

DBSCAN에서 가장 어려운 부분은 `eps` 값을 정하는 것입니다. **k-distance graph**를 활용하면 체계적으로 후보를 찾을 수 있습니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

min_samples = 5

# 각 데이터에서 k번째 가까운 이웃까지의 거리 계산
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)

# k번째 이웃 거리를 정렬해서 시각화
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(7, 4))
plt.plot(k_distances)
plt.xlabel("데이터 포인트 (거리 오름차순 정렬)")
plt.ylabel(f"{min_samples}번째 이웃까지 거리")
plt.title("k-distance graph — eps 후보 탐색")
plt.grid(True)
plt.show()
# → 그래프가 완만하다가 급격히 상승하는 지점 근처를 eps 후보로 잡기
```

> 💡 이 방법도 절대적인 정답은 아닙니다. 여러 `eps` 값을 비교하면서 군집 수와 잡음점 비율을 함께 확인하며 조정해야 합니다.

---

### 5.4 코드 예제

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_noise    = np.sum(labels == -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"탐지된 군집 수: {n_clusters}")
print(f"이상치(Noise) 수: {n_noise}")

plt.figure(figsize=(7, 5))
mask = labels != -1
plt.scatter(X_scaled[mask,  0], X_scaled[mask,  1],
            c=labels[mask], cmap='Set1', alpha=0.7, label='군집')
plt.scatter(X_scaled[~mask, 0], X_scaled[~mask, 1],
            c='black', marker='x', s=60, label='이상치(Noise)')
plt.title(f"DBSCAN — eps={dbscan.eps}, min_samples={dbscan.min_samples}")
plt.legend()
plt.show()
```

**결과 해석:**
```
label =  0, 1, 2...  → 해당 군집에 속함
label = -1           → 이상치 (어떤 군집에도 속하지 않음)

초승달 데이터처럼 비구형 분포에서도 군집을 잘 찾아냄
K-Means는 이 데이터를 직선으로 나누려 해서 군집을 잘못 분류함
```

> ⚠️ **DBSCAN의 군집 번호(label)는 실행마다 달라질 수 있습니다.** 중요한 것은 같은 레이블끼리 같은 군집에 속한다는 점이며, `-1`은 항상 이상치를 의미합니다.

| 항목 | 내용 |
|------|------|
| **장점** | 군집 수 지정 불필요, 이상치 자동 탐지, 비구형 군집도 탐지 가능 |
| **단점** | eps·min_samples 설정 어려움, 밀도 차이가 큰 군집 혼재 시 성능 저하, 고차원 데이터 주의 |
| **언제 쓰나** | 이상치가 포함된 데이터, 비구형 군집, 위치·공간 데이터 분석 |

---

## 6. K-Means vs 계층적 클러스터링 vs DBSCAN 비교

| 알고리즘 | 기준 | 군집 수 지정 | 이상치 처리 | 특징 |
|---------|------|-----------|-----------|------|
| **K-Means** | 중심점 거리 | 필요 | 약함 | 빠르고 단순, 구형 군집에 강함 |
| **계층적** | 군집 간 거리 | 선택 가능 | 보통 | 덴드로그램으로 계층 구조 확인 |
| **DBSCAN** | 밀도 | 불필요 | 강함 | 비구형 군집 탐지, 이상치 자동 분류 |

```python
# 두 종류의 데이터로 세 알고리즘을 한눈에 비교해보기
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt

X_blobs, _ = make_blobs(n_samples=300, centers=3, random_state=42)
X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
datasets  = [("구형 군집 (Blobs)", X_blobs), ("비구형 군집 (Moons)", X_moons)]

for row, (title, X) in enumerate(datasets):
    Xs = StandardScaler().fit_transform(X)
    results = [
        ("K-Means",  KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Xs)),
        ("계층적",   AgglomerativeClustering(n_clusters=3).fit_predict(Xs)),
        ("DBSCAN",   DBSCAN(eps=0.3, min_samples=5).fit_predict(Xs)),
    ]
    for col, (algo, labels) in enumerate(results):
        ax = axes[row][col]
        ax.scatter(Xs[:, 0], Xs[:, 1], c=labels, cmap='Set1', s=10, alpha=0.7)
        ax.set_title(f"{title}\n{algo}")

plt.tight_layout()
plt.show()
```

---

## 7. 클러스터링 평가: 실루엣 계수

클러스터링은 정답이 없기 때문에 일반적인 정확도(Accuracy)로 평가할 수 없습니다. 대표적인 내부 평가 지표로 **실루엣 계수(Silhouette Score)** 를 사용합니다.

### 7.1 실루엣 계수 공식

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

a(i): 데이터 i와 같은 군집의 다른 데이터들과의 평균 거리 (응집도)
b(i): 데이터 i와 가장 가까운 다른 군집과의 평균 거리 (분리도)
```

| 값 | 의미 |
|----|------|
| **1에 가까움** | 자기 군집과 가깝고 다른 군집과 멀다 → 좋은 군집 |
| **0에 가까움** | 두 군집의 경계에 있다 |
| **-1에 가까움** | 다른 군집에 더 가깝다 → 잘못 배정됐을 가능성 |

**손계산 예시:**
```
a(i) = 2,  b(i) = 6

s(i) = (6 - 2) / max(2, 6) = 4 / 6 ≈ 0.67 → 비교적 좋은 군집 배정
```

---

### 7.2 실루엣 계수로 최적 K 찾기

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)

silhouette_scores = []
K_range = range(2, 10)   # 실루엣 계수는 K >= 2 필요

for k in K_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score  = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: 실루엣 계수 = {score:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel('K (군집 수)')
plt.ylabel('실루엣 계수')
plt.title('실루엣 계수로 최적 K 탐색')
plt.grid(True)
plt.show()
```

---

### 7.3 실루엣 계수의 한계

실루엣 계수는 클러스터링 품질을 정량적으로 볼 수 있는 좋은 지표지만, 모든 상황에서 완벽한 기준은 아닙니다.

- 둥근 구형 군집을 선호하는 경향이 있어 DBSCAN처럼 비선형 형태 군집에서는 해석이 애매할 수 있습니다
- 점수가 높아도 실무적으로 의미 없는 군집일 수 있습니다
- 데이터의 실제 업무적 의미를 자동으로 반영하지는 못합니다

따라서 실루엣 계수는 **엘보우 방법 + 시각화 + 군집별 프로파일링 + 도메인 지식과 함께 사용**하는 것이 좋습니다.

> 💡 **엘보우 + 실루엣 계수를 함께 쓰세요.** 엘보우로 후보 K 범위를 좁히고, 실루엣 계수로 최종 선택한 뒤, 군집별 평균으로 실무적 의미를 검증하는 3단계 흐름이 실무에서 일반적입니다.

---

## 8. 연관 규칙 분석 (Association Rule Learning)

### 8.1 핵심 개념

연관 규칙 분석은 **"A를 사면 B도 함께 산다"** 같은 패턴을 데이터에서 발견하는 방법입니다. 대표적인 활용이 **장바구니 분석(Market Basket Analysis)** 입니다.

연관 규칙 분석을 설명할 때 자주 등장하는 예시로 "기저귀를 산 사람이 맥주도 함께 산다"는 이야기가 있습니다. 여기서는 실제 사례 여부보다, **함께 구매되는 패턴을 찾는다는 개념**을 이해하는 예시로 보면 됩니다.

---

### 8.2 연관 규칙은 인과관계가 아니다

> ⚠️ **연관 규칙 분석에서 가장 중요한 주의사항입니다.**

연관 규칙 분석은 "A를 산 사람이 B도 사는 경향이 있다"를 찾는 방법입니다. 하지만 이것이 "A를 사면 B를 사게 된다(인과관계)"는 뜻은 아닙니다.

```
{라면} → {김치}  규칙의 lift가 높더라도
→ "라면 구매가 김치 구매의 원인이다"고 단정할 수 없습니다.
→ 연관 규칙은 동시에 나타나는 상관관계만 발견합니다.
```

---

### 8.3 세 가지 핵심 지표

#### 지지도 (Support)
전체 거래 중 해당 항목 집합이 등장한 비율입니다.

```
Support(A → B) = P(A ∩ B) = A와 B가 함께 등장한 거래 수 / 전체 거래 수

예: 전체 1,000건 중 빵과 우유가 함께 담긴 경우 200건
    Support(빵 → 우유) = 200 / 1,000 = 0.2 (20%)
```

지지도는 비율로 계산하지만, **실제 건수도 반드시 함께 확인**해야 합니다.

```
support = 0.01 이라도...
  전체 100건   → 1건 (통계적으로 불안정)
  전체 1,000,000건 → 10,000건 (충분히 신뢰 가능)
```

#### 신뢰도 (Confidence)
A를 구매한 사람 중 B도 구매한 비율입니다.

```
Confidence(A → B) = P(A ∩ B) / P(A)

예: 빵을 산 400명 중 우유도 산 사람이 200명
    Confidence(빵 → 우유) = 200 / 400 = 0.5 (50%)
→ "빵을 산 사람의 50%는 우유도 산다"
```

#### 향상도 (Lift)
A와 B가 독립적일 때에 비해 얼마나 더 자주 함께 구매되는지를 나타냅니다.

```
Lift(A → B) = P(A ∩ B) / (P(A) × P(B))
            = Confidence(A → B) / P(B)
```

| Lift 값 | 의미 |
|---------|------|
| **= 1** | A와 B는 독립 (연관 없음) |
| **> 1** | 양의 상관관계 (함께 구매 경향) |
| **< 1** | 음의 상관관계 (A 구매 시 B 덜 구매) |

> 💡 **신뢰도만 보면 안 됩니다.** 우유처럼 원래 구매율이 높은 상품은 신뢰도가 자동으로 높게 나옵니다. 향상도(Lift)는 이를 보정해 "원래 구매율보다 얼마나 더 자주 함께 사는가"를 보여주므로 반드시 함께 확인해야 합니다.

---

### 8.4 Apriori 알고리즘

Apriori는 연관 규칙 분석의 대표 알고리즘입니다.

```
Apriori 원칙:
  빈번한 항목 집합의 모든 부분집합도 빈번해야 한다.
  반대로, 빈번하지 않은 집합을 포함하는 더 큰 집합은 검사하지 않아도 됨
  → 검색 공간을 대폭 줄이는 가지치기 전략
```

> 참고: Apriori는 이해하기 쉽지만 상품 수가 많아지면 후보 조합이 급격히 늘어나 느려질 수 있습니다. 대용량 장바구니 데이터에서는 **FP-Growth** 같은 더 효율적인 알고리즘을 사용하기도 합니다.

---

### 8.5 코드 예제

```python
# pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

transactions = [
    ['빵', '우유', '버터'],
    ['빵', '기저귀', '맥주', '달걀'],
    ['우유', '기저귀', '맥주', '콜라'],
    ['빵', '우유', '기저귀', '맥주'],
    ['빵', '우유', '기저귀', '콜라'],
    ['빵', '우유'],
    ['우유', '버터'],
    ['빵', '버터'],
]

# 원핫 인코딩 (항목 존재 여부 행렬로 변환)
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

# 빈번한 항목 집합 찾기
frequent_items = apriori(df, min_support=0.4, use_colnames=True)
print(frequent_items.sort_values('support', ascending=False))

# 연관 규칙 생성
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.6)
rules = rules.sort_values('lift', ascending=False)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

**결과 해석:**
```
지지도(support):   낮으면 드물게 발생 → 신뢰도가 높아도 실제 영향 미미, 건수도 함께 확인
신뢰도(confidence): A 구매 시 B 구매 확률
향상도(lift):      1보다 크면 의미 있는 연관 관계

실무 기준:
  lift > 1.5 이상인 규칙을 우선 검토
  min_support, min_confidence는 0.01~0.1 / 0.5~0.8 범위에서 시작해 조정
```

> ⚠️ **min_support와 min_confidence를 너무 낮게 설정하면** 규칙이 수천 개 이상 나와 해석이 어렵습니다. 너무 높게 설정하면 의미 있는 규칙을 놓칩니다. 데이터 규모에 따라 탐색적으로 조정해야 합니다.

| 항목 | 내용 |
|------|------|
| **장점** | 이해하기 쉬운 규칙, 대규모 거래 데이터 처리 가능 |
| **단점** | 데이터가 작으면 유용하지 않음, 랜덤 패턴에서 잘못된 결론 도출 위험 |
| **언제 쓰나** | 쇼핑몰 추천 시스템, 교차판매(Cross-selling) 전략 수립 |

---

## 9. 이상치 탐지 (Anomaly Detection)

### 9.1 핵심 개념

이상치 탐지는 **정상 패턴에서 벗어난 데이터를 찾는** 방법입니다.

```
활용 예:
  금융:    신용카드 부정 사용 탐지 (평소와 다른 소비 패턴)
  보안:    네트워크 침입 탐지 (비정상적인 트래픽)
  제조:    센서 데이터 이상 감지 (기계 고장 예측)
  의료:    비정상 검사 결과 탐지
```

---

### 9.2 이상치 탐지는 평가가 어렵다

비지도 이상치 탐지는 정답 라벨이 없는 경우가 많습니다. 모델이 `-1`로 예측했다고 해서 무조건 실제 이상치라고 단정할 수도 없습니다. 실무에서는 다음 방법을 함께 사용합니다.

- 이상치로 탐지된 샘플을 **사람이 직접 검토**
- **도메인 전문가**에게 확인
- 기존 규칙 기반 탐지 결과와 비교
- 여러 파라미터 값을 바꿔가며 결과 비교
- 실제 장애·사기·불량 발생 기록이 있다면 사후 검증

즉, 이상치 탐지는 **모델 성능 숫자만 보는 것이 아니라 실제 업무 검토와 함께** 진행해야 합니다.

---

### 9.3 Isolation Forest

Isolation Forest는 **"이상치는 정상 데이터보다 고립시키기 쉽다"** 는 아이디어를 기반으로 합니다.

```
무작위로 특성을 선택하고 무작위 임계값으로 데이터를 분할
→ 이상치는 정상 데이터보다 적은 분할(짧은 경로)로 고립됨

정상 데이터:    많은 분할 필요 (긴 경로)
이상치 데이터:  적은 분할로 고립 (짧은 경로)
```

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X_normal  = np.random.randn(200, 2)
X_outlier = np.random.uniform(low=-5, high=5, size=(20, 2))
X = np.vstack([X_normal, X_outlier])

# Isolation Forest는 트리 기반이라 거리 스케일에 직접 민감하지 않지만
# 여러 모델 비교 시 일관성을 위해 표준화를 적용해도 무방합니다.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iforest = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # 전체 데이터 중 이상치 비율 추정값
    random_state=42
)
pred = iforest.fit_predict(X_scaled)
# pred =  1: 정상
# pred = -1: 이상치

print(f"탐지된 이상치 수: {np.sum(pred == -1)}")

plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[pred == 1,  0], X_scaled[pred == 1,  1],
            c='steelblue', alpha=0.6, label='정상')
plt.scatter(X_scaled[pred == -1, 0], X_scaled[pred == -1, 1],
            c='red', marker='x', s=80, label='이상치')
plt.title("Isolation Forest 이상치 탐지")
plt.legend()
plt.show()
```

> ⚠️ **contamination은 이상치 확률이 아닙니다.** `contamination=0.1`은 "각 데이터가 이상치일 확률이 10%"라는 뜻이 아닙니다. 모델이 전체 데이터 중 약 10%를 이상치로 판단하도록 **기준선을 잡는 설정값**에 가깝습니다. 실제 이상치 비율을 모르는 상황에서는 여러 값을 비교해보고, 도메인 지식이나 사람이 검토한 샘플과 함께 판단해야 합니다.

---

### 9.4 One-Class SVM

One-Class SVM은 **정상 데이터만 학습**한 뒤, 정상 영역 밖에 있는 데이터를 이상치로 판별합니다.

```
정상 데이터만으로 학습
→ 정상 데이터를 감싸는 경계(decision boundary) 학습
→ 경계 밖에 있는 새 데이터 = 이상치

DAY3에서 배운 SVM의 "마진 최대화" 아이디어를
정상 vs 이상치 구분에 적용한 것
```

One-Class SVM은 거리와 커널 계산에 영향을 받기 때문에 **표준화를 반드시 적용**해야 합니다.

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_train        = np.random.randn(200, 2)
X_test_normal  = np.random.randn(50, 2)
X_test_outlier = np.random.uniform(-5, 5, size=(20, 2))

scaler     = StandardScaler()
X_train_s  = scaler.fit_transform(X_train)
X_test_n   = scaler.transform(X_test_normal)
X_test_o   = scaler.transform(X_test_outlier)

oc_svm = OneClassSVM(
    kernel='rbf',
    nu=0.05,       # 이상치 허용 비율의 상한 (0~1)
    gamma='scale'
)
oc_svm.fit(X_train_s)   # 정상 데이터만으로 학습

pred_n = oc_svm.predict(X_test_n)
pred_o = oc_svm.predict(X_test_o)

print(f"정상 데이터 탐지율: {np.mean(pred_n == 1):.2%}")
print(f"이상치 탐지율:     {np.mean(pred_o == -1):.2%}")
```

> 💡 **nu 파라미터**: 직관적으로는 "정상 데이터 중 이상치로 허용하는 정도"입니다. `nu`가 작을수록 더 엄격하게 정상 영역을 잡고, `nu`가 클수록 더 많은 데이터를 이상치로 볼 수 있습니다. 정확히는 학습 데이터에서 이상치로 허용할 비율의 상한이자, support vector 비율의 하한입니다.

**Isolation Forest vs One-Class SVM:**

| 항목 | Isolation Forest | One-Class SVM |
|------|-----------------|---------------|
| **기반** | 트리 앙상블 | SVM 커널 |
| **속도** | 빠름 | 느림 (대용량 주의) |
| **표준화** | 선택적 | 필수 |
| **파라미터** | contamination | nu, gamma, kernel |
| **언제** | 범용적 이상치 탐지 | 정상 데이터만 확보된 경우 |

---

## 10. 비지도학습 결과를 실무에서 해석하는 방법

비지도학습은 정답 라벨 없이 데이터의 숨은 구조를 찾는 방법입니다. 모델이 만든 결과를 그대로 정답이라고 보기보다는, 데이터를 이해하기 위한 **해석 가능한 가설**로 보는 것이 좋습니다.

### 클러스터 번호는 의미 있는 숫자가 아니다

K-Means나 DBSCAN을 실행하면 `0`, `1`, `2` 같은 클러스터 번호가 나오지만, 이 숫자는 단순한 **그룹 ID**입니다.

```python
labels = km.fit_predict(X_scaled)
print(labels[:10])
# → [0 0 1 2 1 0 2 2 1 0]
# 0번 군집이 1번보다 우수하거나 더 큰 값을 가진다는 뜻이 아님
```

클러스터 번호 자체보다 중요한 것은 **각 클러스터가 어떤 특성을 가지는지 확인**하는 것입니다.

### 군집별 특징 확인 → 이름 붙이기

```python
df_cluster = pd.DataFrame(X, columns=feature_names)
df_cluster["cluster"] = labels

# 군집별 데이터 개수
print(df_cluster["cluster"].value_counts().sort_index())

# 군집별 평균값
print(df_cluster.groupby("cluster").mean().round(2))
```

확인한 결과를 바탕으로 각 군집에 이름을 붙일 수 있습니다.

| 군집 | 특성 요약 | 해석 예시 |
|------|---------|--------|
| 0번 | 구매 금액 높음, 방문 빈도 낮음 | 핵심 고가 구매 고객 |
| 1번 | 구매 금액 낮음, 방문 빈도 높음 | 자주 오는 소액 고객 |
| 2번 | 최근 가입 고객 비중 높음 | 신규 유입 고객 |

### 평가 지표 + 시각화 + 도메인 지식을 함께

실루엣 계수는 유용하지만 모든 상황에서 완벽한 기준은 아닙니다. 다음을 함께 확인하는 것이 좋습니다.

```
✓ 실루엣 계수             — 군집 품질 정량 평가
✓ 엘보우 그래프           — 최적 K 탐색
✓ 시각화 결과             — 군집 형태 직관적 확인
✓ 군집별 평균·분포        — 군집 의미 해석
✓ 도메인 지식             — 실제 업무 맥락 반영
✓ 이상치 탐지 시 사람 검토 — 모델 결과의 신뢰도 보완
```

---

## 11. 비지도학습 알고리즘 전체 비교

| 알고리즘 | 분류 | 목적 | 대표 활용 | 특징 |
|---------|------|------|---------|------|
| **K-Means** | 군집화 | 중심점 기반 그룹화 | 고객 세분화 | 빠름, K 미리 지정 필요 |
| **계층적** | 군집화 | 트리 구조 그룹화 | 유전자 분석 | 덴드로그램 시각화 |
| **DBSCAN** | 군집화 | 밀도 기반 그룹화 | 위치 데이터 | 이상치 자동 탐지 |
| **Apriori** | 연관 규칙 | 동시 구매 패턴 | 장바구니 분석 | 지지도·신뢰도·향상도 |
| **Isolation Forest** | 이상치 탐지 | 비정상 데이터 탐지 | 금융 사기 탐지 | 빠름, 대용량 적합 |
| **One-Class SVM** | 이상치 탐지 | 정상 영역 학습 | 제조 이상 감지 | 정상 데이터만으로 학습 |

---

## 12. 실무 선택 가이드

| 상황 | 추천 알고리즘 | 이유 |
|------|-------------|------|
| 군집 수를 어느 정도 알고 있음 | **K-Means** | 빠르고 단순, 대용량 처리 |
| 군집 수를 모름, 계층 구조 탐색 | **계층적 클러스터링** | 덴드로그램으로 탐색 |
| 비구형 군집 또는 이상치 포함 | **DBSCAN** | 밀도 기반, 이상치 자동 분류 |
| 쇼핑몰 추천·교차판매 전략 | **Apriori** | 장바구니 패턴 발견 |
| 신용카드 사기·네트워크 침입 탐지 | **Isolation Forest** | 빠른 이상치 탐지 |
| 정상 데이터만 확보됨 | **One-Class SVM** | 정상 영역 기반 이상치 판별 |

> 💡 **탐색적 분석 순서**: 처음 접하는 데이터라면 **K-Means(엘보우 방법)로 빠르게 군집 파악** → **실루엣 계수로 품질 확인** → **군집별 프로파일링으로 해석** → 군집이 비구형이거나 이상치가 많다고 판단되면 **DBSCAN으로 재시도** 하는 흐름이 실무에서 많이 쓰입니다.

---

## 13. 주의사항 & 자주 하는 실수

### ⚠️ 표준화 없이 거리 기반 클러스터링 적용

```python
# ❌ 표준화 없이 적용 → 스케일이 큰 특성이 결과 왜곡
km = KMeans(n_clusters=3).fit(X)

# ✅ 반드시 표준화 후 적용
X_scaled = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
```

### ⚠️ K-Means 결과가 실행마다 달라짐

```python
# ❌ random_state 없음 → 초기 중심점이 달라져 결과 매번 변화
km = KMeans(n_clusters=3)

# ✅ random_state 고정 + n_init 설정
km = KMeans(n_clusters=3, random_state=42, n_init=10)
# n_init: 다른 초기 중심점으로 n번 시도 후 최선 선택 (기본값 10)
```

### ⚠️ DBSCAN label=-1 처리 누락

```python
# ❌ -1을 일반 군집처럼 처리
n_clusters = len(set(labels))   # -1도 포함됨

# ✅ -1 (이상치) 제외하고 군집 수 계산
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"이상치 수: {np.sum(labels == -1)}")
```

### ⚠️ 연관 규칙에서 향상도(Lift) 확인 생략

```python
# ❌ 신뢰도만 보고 규칙 선택 → 원래 구매율이 높은 상품은 자동으로 높게 나옴
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.8)

# ✅ 향상도(lift)로 정렬해 실제 연관 관계 확인
rules = rules.sort_values('lift', ascending=False)
rules = rules[rules['lift'] > 1.0]
```

### ⚠️ 실루엣 계수는 K=1일 때 사용 불가

```python
# ❌ K=1 → 다른 군집이 없어 b(i) 계산 불가 → 오류
for k in range(1, 10):  # 1부터 시작 → 오류

# ✅ K >= 2 일 때만 사용
for k in range(2, 10):  # 반드시 2부터
    labels = KMeans(n_clusters=k, n_init=10).fit_predict(X_scaled)
    score  = silhouette_score(X_scaled, labels)
```

### ⚠️ 비지도학습 결과를 정답처럼 사용

```python
# ❌ 군집 결과를 해석 없이 바로 활용
# "cluster=0이 VIP 고객이니까 cluster=0에 할인 쿠폰을 보내자"

# ✅ 군집별 평균·분포 확인 후 해석
cluster_profile = df.groupby("cluster").mean()
# → 각 군집의 특성을 직접 확인한 뒤 의미를 부여
```

---

## DAY7 정리

```
✅ 비지도학습 = 정답(Label) 없이 데이터의 숨겨진 구조 발견
   목적: 군집화 / 차원 축소(DAY6) / 연관 규칙 / 이상치 탐지
   최근 활용: 고객 세분화, 문서 그룹화, 이상 거래 탐지, RAG·벡터 검색

✅ 비지도학습 결과는 "정답"이 아닌 "해석 가능한 가설"
   → 클러스터 번호는 단순 그룹 ID (등급이나 크기 의미 없음)
   → 군집별 평균·분포 확인 → 분석가가 직접 해석·이름 붙이기

✅ 거리 기반 클러스터링은 표준화 필수 (StandardScaler 먼저!)
   → 범주형 변수는 K-Means에 바로 적용하면 위험

✅ K-Means — 중심점 거리 기반, 가장 빠름
   → 엘보우 방법 + 실루엣 계수로 최적 K 탐색
   → random_state + n_init 설정으로 결과 안정화
   → 구형 군집·이상치 없는 데이터에 적합

✅ 계층적 클러스터링 — 덴드로그램으로 계층 구조 시각화
   → 군집 수 몰라도 시작 가능
   → Ward + 유클리드 거리가 일반적으로 좋은 결과
   → 대용량 데이터에는 계산 비용이 큼

✅ DBSCAN — 밀도 기반, 이상치 자동 탐지
   → 군집 수 지정 불필요, 비구형 군집 탐지
   → label=-1 → 이상치 / k-distance graph로 eps 탐색
   → 고차원 데이터에서는 차원 축소 후 적용 권장

✅ 실루엣 계수 — 군집 품질 평가 (정답 없이)
   → 1에 가까울수록 좋음 / K >= 2 일 때만 사용
   → 엘보우 + 시각화 + 프로파일링 + 도메인 지식과 함께 사용

✅ 연관 규칙 (Apriori)
   → 인과관계가 아닌 동시 발생 상관관계
   → 지지도(support), 신뢰도(confidence), 향상도(lift) 모두 확인
   → lift > 1 + 실제 건수(count)도 함께 확인
   → 대용량 데이터에는 FP-Growth 고려

✅ 이상치 탐지
   → Isolation Forest: 빠름, 대용량, contamination = 기준선 설정값
   → One-Class SVM: 정상 데이터만 확보됐을 때, 표준화 필수, nu 파라미터 주의
   → 모델 결과 단독이 아닌 사람의 검토와 함께 진행
```

---

## 🔗 참고 자료

- [scikit-learn KMeans 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn AgglomerativeClustering 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [scikit-learn DBSCAN 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [scikit-learn IsolationForest 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [mlxtend 연관 규칙 공식 문서](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
- [Google Colab](https://colab.research.google.com/)
- [Kaggle — 데이터셋 & 대회](https://www.kaggle.com/)
