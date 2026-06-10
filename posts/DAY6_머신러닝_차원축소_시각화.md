# 📉 머신러닝 완전 입문 가이드 — DAY6. 차원 축소 & 시각화

> **시리즈**: 파이썬 기본만 있는 사람을 위한 머신러닝 입문
> **DAY1**: 머신러닝 핵심 개념과 데이터 관리 (이전 편)
> **DAY2**: 회귀 모델과 분류 모델 실습 (이전 편)
> **DAY3**: SVM · KNN · 나이브 베이즈 (이전 편)
> **DAY4**: 결정트리 · 회귀트리 · 선형 회귀 · 로지스틱 회귀 · 역전파 (이전 편)
> **DAY5**: 앙상블 · ANN (이전 편)
> **DAY6**: 차원 축소 & 시각화

---

## 1. 차원 축소 (Dimensionality Reduction) 란?

**차원 축소**는 고차원 데이터를 저차원 데이터로 변환하면서 중요한 정보는 최대한 유지하는 기법입니다.

DAY1에서 전처리를 다룰 때 잠깐 언급했는데, 이번 DAY6에서 제대로 파고듭니다.

---

### 1.1 차원이란?

머신러닝에서 차원은 보통 **입력 변수(Feature)의 개수**를 말합니다.

```
| 이름 | 키  | 몸무게 | 나이 | 공부시간 | 출석률 |
|------|-----|--------|------|---------|------|
|  A   | 170 |   65   |  20  |    5    |  95  |
```

여기서 입력 변수는 `키, 몸무게, 나이, 공부시간, 출석률` — 총 **5개**, 즉 5차원 데이터입니다.

> 💡 **숫자로 느껴보기**: 1차원은 선 위의 점, 2차원은 평면 위의 점, 3차원은 공간 속 점입니다. 100차원은 사람이 직접 볼 수도, 그릴 수도 없습니다. 차원 축소는 이 "볼 수 없는 공간"을 우리가 이해할 수 있는 공간으로 끌어내리는 작업입니다.

---

### 1.2 차원이 높아지면 생기는 문제들

Wine Quality 데이터는 다음과 같은 11개 특성을 가집니다.

```
fixed acidity, volatile acidity, citric acid, residual sugar,
chlorides, free sulfur dioxide, total sulfur dioxide,
density, pH, sulphates, alcohol
```

11차원만 돼도 이미 문제가 생깁니다.

| 문제 | 설명 |
|------|------|
| **학습 속도 저하** | 특성 1,000개 → 학습 느림 / 특성 50개 → 학습 빠름 |
| **불필요한 변수** | 성적 예측에 신발 사이즈는 필요 없음 → 오히려 노이즈 |
| **과적합** | 변수가 너무 많으면 학습 데이터의 세세한 특징까지 외워버림 |
| **시각화 불가** | 사람은 3차원까지만 직관적으로 이해 가능 |

---

### 1.3 차원의 저주 (Curse of Dimensionality)

차원이 늘어날수록 **데이터가 채워야 할 공간**이 기하급수적으로 커집니다.

```
1차원: 선 → 점 10개면 어느 정도 채울 수 있음
2차원: 평면 → 같은 밀도를 유지하려면 점 100개 필요
3차원: 공간 → 점 1,000개 필요
...
n차원: 공간이 지수 함수적으로 커짐
```

같은 데이터 수로는 고차원 공간을 충분히 채울 수 없게 되고, 데이터가 듬성듬성 분포하면서 모델이 패턴을 찾기 어려워집니다.

> ⚠️ "데이터를 더 많이 모으면 해결되지 않나요?" — 차원이 1개 늘어날 때마다 필요한 데이터가 지수 함수적으로 증가하기 때문에, 현실적으로 데이터를 더 모아서 해결하기는 어렵습니다. 차원 자체를 줄이는 것이 더 현실적인 접근입니다.

---

### 1.4 차원 축소의 두 가지 방법

차원 축소는 크게 두 가지 방식으로 나뉩니다.

```
차원 축소
 ├── 특성 선택 (Feature Selection) — 기존 변수 중 중요한 것만 선택
 └── 특성 추출 (Feature Extraction) — 기존 변수들을 조합해 새로운 변수 생성
```

#### 특성 선택 (Feature Selection)

기존 변수 그대로를 살려서 쓰는 방식입니다. 변수의 의미가 유지됩니다.

```
원본: x1, x2, x3, x4, x5, x6, x7, x8, x9, x10
선택: x1, x3, x5, x8   ← 4개만 골라서 사용
```

| 방식 | 설명 | 예시 | 장/단점 |
|------|------|------|--------|
| **필터 (Filter)** | 통계적 기준으로 선택 | 상관계수, 카이제곱, 분산 | 빠름 / 모델 성능과 직결 안 됨 |
| **래퍼 (Wrapper)** | 변수 조합을 직접 모델에 넣어봄 | RFE, 전진선택, 후진제거 | 성능 기준 선택 / 계산 비용 큼 |
| **임베디드 (Embedded)** | 모델 학습 중 자동 선택 | Lasso, 결정트리, XGBoost | 동시 수행 / 모델 종류에 의존 |

> 💡 DAY4에서 배운 **결정트리/랜덤포레스트의 `feature_importances_`** 가 바로 임베디드 방식의 특성 선택입니다. 모델이 학습하면서 "이 변수가 얼마나 중요했는지"를 자동으로 계산해줍니다.

#### 특성 추출 (Feature Extraction)

기존 변수들을 조합해 **새로운 변수**를 만드는 방식입니다.

```
원본 변수: x1, x2, x3, x4, x5 (5개)
      ↓ 조합 · 변환
새 변수:   z1, z2              (2개)
```

`z1`, `z2`는 원래 변수들의 조합이라 직관적 해석이 어려울 수 있지만, 데이터의 핵심 구조를 더 효율적으로 압축할 수 있습니다.

**PCA, t-SNE, UMAP, Autoencoder** 등 DAY6에서 다루는 알고리즘 대부분이 특성 추출 방식입니다.

---

### 1.5 시각화용 vs 모델 전처리용: 목적이 다르면 방법도 다릅니다

> ⚠️ DAY6 전체에서 가장 중요한 구분입니다. 차원 축소를 어떤 목적으로 쓰느냐에 따라 알고리즘 선택과 코드 작성 방식이 달라집니다.

| 목적 | 추천 알고리즘 | 특징 |
|------|-------------|------|
| **사람이 보기 위한 시각화** | PCA, t-SNE, UMAP | 전체 데이터에 한 번에 적용해도 무방 |
| **모델 학습 전 전처리** | PCA, TruncatedSVD, (경우에 따라) LDA | Train에만 `fit()`, Test에는 `transform()` 만 |
| **비선형 구조 압축** | Autoencoder | 딥러닝 기반, 가장 강력한 표현 가능 |

t-SNE는 시각화에는 훌륭하지만, 새 데이터에 같은 변환을 적용하는 `transform()`이 없기 때문에 **모델 입력용 전처리로는 사용할 수 없습니다.**

---

### 1.6 차원 축소 알고리즘 전체 구조

```
차원 축소
├── 선형 차원 축소
│   ├── PCA   — 분산이 큰 방향 찾기 (비지도)
│   ├── LDA   — 클래스가 잘 구분되는 방향 찾기 (지도)
│   ├── SVD   — 행렬 분해로 핵심 구조 추출
│   ├── ICA   — 독립 성분 분리
│   └── NMF   — 비음수 행렬 분해
│
└── 비선형 차원 축소
    ├── t-SNE      — 가까운 이웃 관계 보존 시각화
    ├── UMAP       — t-SNE보다 빠른 시각화
    ├── Isomap     — 곡면(매니폴드) 구조 보존
    └── Autoencoder — 딥러닝 기반 압축·복원
```

---

## 2. 주요 알고리즘 한눈에 비교

| 알고리즘 | 지도/비지도 | 선형/비선형 | 주요 목적 | 특징 |
|---------|-----------|-----------|---------|------|
| **PCA** | 비지도 | 선형 | 차원 축소 | 가장 기본적, 빠름 |
| **LDA** | 지도 | 선형 | 분류 성능 개선 | 정답 라벨 사용 |
| **t-SNE** | 비지도 | 비선형 | 시각화 | 군집 표현 우수, 느림 |
| **UMAP** | 비지도 | 비선형 | 시각화·탐색 | 빠름, 대용량 적합 |
| **Autoencoder** | 비지도 | 비선형 | 압축·복원 | 딥러닝 기반 |
| **SVD** | 비지도 | 선형 | 행렬 압축 | 텍스트·추천 시스템 |
| **ICA** | 비지도 | 선형 | 독립 성분 분리 | 신호 처리 |
| **NMF** | 비지도 | 선형 | 토픽·패턴 추출 | 비음수 데이터 |
| **Isomap** | 비지도 | 비선형 | 곡면 구조 보존 | 계산 비용 큼 |

> 💡 **어떤 알고리즘을 선택해야 할까?**
> - 기본 전처리 → **PCA**
> - 분류 전처리·시각화 → **LDA**
> - 탐색 시각화 → **t-SNE → UMAP** 순으로 시도
> - 텍스트 TF-IDF 축소 → **TruncatedSVD**
> - 이미지·복잡한 데이터 압축 → **Autoencoder**
> - 신호(뇌파, 음성) 분리 → **ICA**
> - 토픽 추출 (양수 데이터) → **NMF**

---

## 3. PCA (주성분 분석)

### 3.1 핵심 개념

PCA(Principal Component Analysis)는 **분산이 가장 큰 방향**을 새로운 축으로 삼아 차원을 줄이는 알고리즘입니다.

정답 라벨(y)을 사용하지 않는 **비지도학습** 기반입니다.

```
[직관적 이해]

원본 데이터: 점들이 타원 모양으로 분포

     y
     │    ●  ●
     │  ●   ●
     │    ●
     └──────── x

PCA: 이 점들이 가장 많이 퍼진 방향 → PC1 (제1 주성분)
     그것과 수직인 방향             → PC2 (제2 주성분)
```

> 💡 **분산 = 정보량**으로 이해하면 편합니다. 분산이 크다는 건 그 방향으로 데이터가 많이 퍼져 있다는 것이고, 퍼져 있을수록 데이터의 차이(=정보)를 더 잘 담고 있습니다. PCA는 "정보가 가장 많은 방향"부터 순서대로 축을 잡아나가는 알고리즘입니다.

---

### 3.2 계산 과정

#### Step 1. 표준화

스케일이 다른 변수들을 동등하게 비교하기 위해 먼저 표준화합니다.

> ⚠️ **PCA 전 표준화는 거의 필수입니다.** 키(170cm)와 몸무게(65kg)를 같이 넣으면, 단위가 큰 키가 분산도 크게 나와 PCA 결과를 왜곡합니다. `StandardScaler`로 평균=0, 표준편차=1로 맞춰주세요.

#### Step 2. 중심화 (Centering) — 작은 숫자로 이해하기

표준화 이후 PCA는 내부적으로 평균을 제거해 데이터를 원점 중심으로 옮깁니다.

```
원본 데이터:         평균 제거 후 (중심화):
[[2, 3],            [[-2, -2],
 [4, 5],    →        [ 0,  0],
 [6, 7]]             [ 2,  2]]

평균 μ = [4, 5]
Xc = X - μ
```

이렇게 평균을 제거한 뒤, 데이터가 가장 많이 퍼진 방향을 찾습니다.

#### Step 3. 공분산 행렬 계산

두 변수가 얼마나 같은 방향으로 움직이는지(공분산)를 행렬로 표현합니다.

```
C = (1 / n-1) × Xc^T × Xc
```

#### Step 4. 고유값 분해

공분산 행렬을 분해해 고유값(eigenvalue)과 고유벡터(eigenvector)를 구합니다.

```
Cv = λv

고유값 λ  → 이 방향으로 데이터가 얼마나 퍼져 있는지 (크기)
고유벡터 v → 그 방향 (방향 벡터)
```

> 💡 **고유벡터 = 주성분의 방향**, **고유값 = 그 방향의 중요도**로 기억하면 됩니다. 고유값이 클수록 더 많은 정보를 담은 축입니다.

#### Step 5. 주성분 선택 & 투영

고유값이 큰 순서대로 주성분을 선택하고, 원본 데이터를 그 축으로 투영합니다.

```
Z = Xc × V

10차원 데이터  →  PC1, PC2  →  2차원 데이터
```

#### Step 6. 설명 분산 비율 확인

```python
print(pca.explained_variance_ratio_)

# 예: λ = [8, 2, 1] 이면
# PC1 = 8/11 ≈ 72.7%
# PC2 = 2/11 ≈ 18.2%
# PC3 = 1/11 ≈  9.1%
# 누적 = PC1 + PC2 ≈ 90.9% → PC1, PC2만으로 충분
```

---

### 3.3 주성분 개수(n_components) 정하는 법

시각화 목적이면 `n_components=2` 또는 `3`으로 고정해도 됩니다. 하지만 **모델 학습용 전처리**로 쓸 때는 정보 손실을 고려해 누적 설명 분산을 보고 결정해야 합니다.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # 훈련 데이터에만 fit!

# n_components 지정 없이 전체 분석
pca_full = PCA()
pca_full.fit(X_scaled)

# 누적 설명 분산 계산
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# 95% 분산을 보존하는 최소 주성분 수 찾기
n_components = np.argmax(cum_var >= 0.95) + 1
print(f"95% 분산 보존에 필요한 주성분 개수: {n_components}")
```

> 💡 일반적으로 누적 설명 분산 **80~95%** 가 되는 지점까지 주성분을 선택합니다. 100%를 맞추려면 원본 차원이 전부 필요하니 의미가 없습니다.

---

### 3.4 주성분이 원본 변수와 어떤 관계인지 보기 (`components_`)

PCA로 만든 축이 원본 변수들과 어떤 관계인지 확인할 수 있습니다.

```python
import pandas as pd

loadings = pd.DataFrame(
    pca.components_,
    columns=iris.feature_names,
    index=['PC1', 'PC2']
)
print(loadings)
```

**출력 예시:**
```
     sepal length  sepal width  petal length  petal width
PC1      0.522       -0.263         0.581        0.565
PC2      0.372        0.925         0.021        0.065
```

절댓값이 클수록 해당 주성분에 더 크게 기여한 원본 변수입니다. PC1은 꽃잎 길이·너비의 영향을 많이 받고, PC2는 꽃받침 너비의 영향을 크게 받는 것을 확인할 수 있습니다.

---

### 3.5 전체 코드 예제 (Iris 데이터 — 시각화)

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# 1. 데이터 로드
iris = load_iris()
X = iris.data   # 4차원: 꽃받침 길이/너비, 꽃잎 길이/너비
y = iris.target

# 2. 표준화 (PCA 전 필수!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ※ 시각화 목적이므로 전체 X에 fit_transform. 
#    모델 학습용이라면 Train/Test를 먼저 나눈 뒤 훈련 데이터에만 fit() 해야 합니다.

# 3. PCA 생성 & 적용
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. 설명 분산 비율 확인
print(pca.explained_variance_ratio_)
# 출력 예: [0.7296 0.2285]
print(f"누적 설명 분산: {pca.explained_variance_ratio_.sum():.4f}")
# 출력 예: 누적 설명 분산: 0.9581  → 4차원 정보의 95.8%를 2차원으로 표현

# 5. 시각화
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["target"] = y

colors = ['red', 'green', 'blue']
labels = iris.target_names

plt.figure(figsize=(8, 5))
for i, (color, label) in enumerate(zip(colors, labels)):
    subset = pca_df[pca_df["target"] == i]
    plt.scatter(subset["PC1"], subset["PC2"], c=color, label=label, alpha=0.7)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA — Iris 데이터 2차원 시각화")
plt.legend()
plt.show()
```

**결과 해석:**
```
누적 설명 분산 0.9581 → 4개 변수의 정보를 95.8% 보존하면서 2차원으로 축소 성공
시각화 결과:
  - setosa(빨강)는 다른 두 품종과 완전히 분리됨
  - versicolor(초록)와 virginica(파랑)는 일부 겹침
  → PC1, PC2만으로도 품종 구분이 상당히 가능함을 확인
```

---

### 3.6 모델 전처리로 PCA 사용할 때: Pipeline으로 데이터 누수 방지

PCA 결과를 모델 입력으로 쓸 때는 반드시 훈련 데이터에만 `fit()`해야 합니다. `Pipeline`을 사용하면 이 순서가 자동으로 보장됩니다.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# 1. 먼저 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Pipeline으로 연결 (fit은 훈련 데이터에만 자동 적용됨)
model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=2)),
    ('clf',    LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
print(f"테스트 정확도: {model.score(X_test, y_test):.4f}")
```

> ⚠️ **데이터 누수 주의**: PCA도 스케일러와 마찬가지로 **전체 X에 먼저 fit 후 train/test 분리**하면 테스트 데이터의 분포 정보가 PCA 축을 만드는 데 들어갑니다. DAY1에서 배운 데이터 누수(Data Leakage)와 같은 문제입니다. `Pipeline`을 쓰거나, 직접 할 때는 **분리 → fit(train) → transform(test)** 순서를 지키세요.

| 항목 | 내용 |
|------|------|
| **장점** | 빠름, 노이즈 제거, 다중공선성 감소 |
| **단점** | 비선형 구조 반영 어려움, 클래스 정보 미사용 |
| **언제 쓰나** | 전처리 단계 차원 축소, 시각화, 노이즈 제거 |

---

## 4. LDA (선형 판별 분석)

### 4.1 핵심 개념 — PCA와의 차이

LDA(Linear Discriminant Analysis)는 PCA처럼 선형 변환으로 차원을 줄이지만, **정답 라벨(y)을 사용**한다는 점에서 근본적으로 다릅니다.

```
PCA 목적: 데이터가 가장 많이 퍼진 방향 → 정보 보존 극대화
LDA 목적: 클래스가 가장 잘 구분되는 방향 → 분류 성능 극대화
```

```
[직관적 차이]

PCA: ●●○○  (두 클래스가 겹침)  ← 분산이 큰 방향으로 투영
LDA: ●●   ○○               ← 클래스 간 거리가 최대가 되는 방향으로 투영
```

LDA는 단순히 데이터가 많이 퍼진 방향을 찾지 않습니다. 각 클래스의 중심이 서로 멀어지고, 같은 클래스 내부의 점들은 가까워지는 방향을 찾습니다.

즉 좋은 LDA 축은 다음 조건을 만족합니다.

- 클래스 0의 점들은 한쪽에 촘촘히 모임
- 클래스 1의 점들은 다른 쪽에 촘촘히 모임
- 각 클래스 내부는 좁게 모임, 클래스 간 간격은 넓게

> 💡 **시험 채점 비유**: PCA는 "점수 분포가 가장 넓게 퍼지는 방향"을 찾고, LDA는 "상위반과 하위반이 가장 잘 구분되는 방향"을 찾습니다. 분류 문제가 목적이라면 LDA가 더 적합합니다.

---

### 4.2 LDA가 최적화하는 목표

LDA는 다음 두 가지를 **동시에** 만족하는 축을 찾습니다.

```
1. 같은 클래스끼리 가깝게  → 클래스 내 분산(Sw) 최소화
2. 다른 클래스끼리 멀게    → 클래스 간 분산(Sb) 최대화

J(W) = |W^T × Sb × W| / |W^T × Sw × W|  를 최대화

Sb는 크게 + Sw는 작게 → 클래스 분리가 잘 된 축
```

---

### 4.3 LDA의 중요한 제약

LDA는 최대 `클래스 수 - 1`개의 축만 만들 수 있습니다.

```
클래스가 3개 → LDA 축 최대 2개 (3-1=2)
클래스가 2개 → LDA 축 최대 1개 (2-1=1)
```

> ⚠️ 클래스가 2개인 이진 분류 문제에서 `n_components=2`로 설정하면 오류가 납니다. `n_components=1`이 최대입니다. PCA는 이런 제약이 없습니다.

---

### 4.4 코드 예제

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# --- 시각화 목적: 전체 데이터에 적용 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)  # ← y를 반드시 넣어야 함

print(lda.explained_variance_ratio_)
# 출력 예: [0.9912 0.0088]  → LDA1이 99.1% 설명

# 시각화
colors = ['red', 'green', 'blue']
labels = iris.target_names

plt.figure(figsize=(8, 5))
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y == i
    plt.scatter(X_lda[mask, 0], X_lda[mask, 1], c=color, label=label, alpha=0.7)

plt.xlabel("LDA1")
plt.ylabel("LDA2")
plt.title("LDA — Iris 데이터 2차원 시각화")
plt.legend()
plt.show()
```

**결과 해석:**
```
PCA 결과: setosa 분리 / versicolor·virginica 일부 겹침
LDA 결과: 3개 클래스 모두 훨씬 명확하게 분리됨
  → 라벨 정보를 활용한 LDA가 분류 시각화에 더 유리
```

---

### 4.5 LDA는 분류 모델로도 사용 가능

LDA는 차원 축소 도구이면서, 그 자체로 **분류 모델**로도 쓸 수 있습니다.

```python
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 분류 모델로 사용 (n_components 지정 없이)
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)
print(f"LDA 분류 정확도: {lda_clf.score(X_test, y_test):.4f}")
```

`fit_transform()`으로 차원 축소에 사용할 수도 있고, `fit()` 후 `predict()`로 분류 모델처럼 사용할 수도 있습니다.

> ⚠️ LDA를 모델 성능 평가에 사용할 전처리로 쓸 때는 PCA와 마찬가지로 반드시 훈련 데이터에만 `fit()` 해야 합니다. 라벨 정보를 사용하는 LDA 특성상, 테스트 데이터 라벨이 전처리에 새어나가는 것에 주의하세요.

| 항목 | 내용 |
|------|------|
| **장점** | 분류 문제에서 PCA보다 효과적인 분리 가능 |
| **단점** | 선형 분리 가정, 축 수 = 클래스 수 - 1 제한 |
| **언제 쓰나** | 분류 전처리, 클래스 분리 시각화, 분류 모델 자체 |

---

## 5. t-SNE

### 5.1 핵심 개념

t-SNE(t-distributed Stochastic Neighbor Embedding)는 고차원 데이터에서 **가까운 데이터끼리의 관계**를 저차원에서도 유지하도록 배치하는 비선형 알고리즘입니다.

주로 **데이터 시각화**에 특화되어 있습니다.

```
t-SNE의 철학:
"고차원에서 가까운 점들은 2D에서도 가까워야 한다"

100차원 이미지 데이터
↓ t-SNE
2차원 산점도
  → 숫자 0끼리 한 군집, 숫자 1끼리 다른 군집...
```

> ⚠️ **t-SNE는 시각화 전용입니다.** `transform()` 메서드가 없어 학습에 사용하지 않은 새 데이터에 같은 변환을 적용할 수 없습니다. 모델 입력용 전처리로는 쓰지 마세요.

---

### 5.2 동작 원리

PCA나 LDA는 직선 축을 찾지만, t-SNE는 **거리를 확률**로 변환해서 다룹니다.

**Step 1. 고차원 거리 → 확률로 변환**

```
거리가 가까운 두 점 A-B → 높은 유사도 확률 (pij ≈ 1)
거리가 먼 두 점 A-C     → 낮은 유사도 확률 (pij ≈ 0)

예:
  A와 B 거리 = 1  →  pB|A 크다 (비슷하다)
  A와 C 거리 = 5  →  pC|A 작다 (비슷하지 않다)
```

**Step 2. 저차원에서도 유사도 계산 (t-분포 사용)**

저차원에서의 유사도 qij를 계산합니다. 이때 **Student t-분포**를 사용합니다.

> 💡 **왜 t-분포?** 정규분포는 꼬리가 얇아서, 고차원에서 "중간 거리"에 있는 점들이 저차원에서 너무 뭉치는 문제가 생깁니다. t-분포는 꼬리가 두꺼워서, 고차원의 중간 거리를 저차원에서 더 멀리 밀어내 군집 간 간격을 잘 표현합니다.

**Step 3. KL Divergence 최소화**

고차원 분포 P와 저차원 분포 Q가 비슷해지도록 점들의 위치를 반복적으로 조정합니다.

```
KL(P || Q) = Σ Σ pij × log(pij / qij)

KL 값이 작을수록 → 고차원 구조가 저차원에 잘 보존됨
```

---

### 5.3 코드 예제

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 손글씨 숫자 데이터 (64차원)
digits = load_digits()
X = digits.data
y = digits.target

# t-SNE 적용
# ※ scikit-learn 버전에 따라 n_iter 대신 max_iter를 사용해야 할 수 있습니다.
#   오류가 나면 n_iter=1000 → max_iter=1000 으로 바꿔서 실행하세요.
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,       # scikit-learn < 1.4: n_iter / >= 1.4: max_iter
    random_state=42
)
X_tsne = tsne.fit_transform(X)

# 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=y, cmap='tab10', alpha=0.7, s=10
)
plt.colorbar(scatter, label="숫자 (0~9)")
plt.title("t-SNE — 손글씨 숫자 시각화 (64차원 → 2차원)")
plt.show()
```

**결과 해석:**
```
같은 숫자끼리 군집을 이루는가?   → 군집이 선명하면 특성 구분이 잘 됨
다른 숫자와 많이 섞이는가?        → 섞인 숫자는 서로 혼동되기 쉬운 숫자
```

> ⚠️ **t-SNE/UMAP 결과를 해석할 때 주의할 점**: 축의 숫자(t-SNE 1, t-SNE 2)나 군집 간 거리 자체를 너무 강하게 해석하면 안 됩니다. t-SNE는 가까운 이웃 관계를 보존하지만, 군집 사이의 상대적 거리를 정확히 보존하는 알고리즘이 아닙니다. "A 군집과 B 군집이 C 군집보다 2배 멀다"처럼 해석하면 안 됩니다. 주로 **같은 클래스가 모이는지, 서로 섞이는 클래스가 있는지** 탐색하는 용도로 봐야 합니다.

**perplexity 파라미터 가이드:**

| perplexity | 특징 |
|-----------|------|
| 5~10 | 지역 구조 강조, 군집이 잘게 쪼개짐 |
| 30~50 | 일반적으로 무난 (기본값 30) |
| 100+ | 전체 구조 반영, 군집이 합쳐짐 |

> ⚠️ perplexity는 반드시 데이터 수보다 작아야 합니다.

---

### 5.4 실무 팁: t-SNE 전에 PCA로 먼저 줄이기

t-SNE는 고차원 데이터에서 속도가 느릴 수 있습니다. 실무에서는 **PCA로 먼저 30~50차원 정도로 줄인 뒤 t-SNE를 적용**하는 경우가 많습니다.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1단계: PCA로 노이즈 제거 + 차원 축소
X_50 = PCA(n_components=50, random_state=42).fit_transform(X_scaled)

# 2단계: t-SNE로 2차원 시각화
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_50)
```

PCA로 노이즈와 차원을 먼저 줄이면 t-SNE 속도가 빨라지고 결과가 더 안정적일 수 있습니다.

| 항목 | 내용 |
|------|------|
| **장점** | 복잡한 비선형 구조 표현, 군집 시각화 탁월 |
| **단점** | 느림, 재현성 낮음, transform 없음, 전역 구조 보존 약함 |
| **언제 쓰나** | 고차원 데이터의 군집 패턴 탐색, 탐색적 시각화 전용 |

---

## 6. UMAP

### 6.1 핵심 개념

UMAP(Uniform Manifold Approximation and Projection)은 t-SNE와 비슷하게 고차원 데이터를 2~3차원으로 줄여 시각화하지만, **일반적으로 더 빠르고 대용량 데이터에 적합**합니다.

```
t-SNE vs UMAP 간단 비교:

t-SNE: 지역 구조(군집 내부) 보존에 집중, 느림, transform 없음
UMAP:  지역 구조 + 전역 구조 모두 어느 정도 보존, 빠름, transform 있음
```

> 💡 **매니폴드(Manifold)란?** 고차원 공간에 있지만 실제로는 저차원 구조를 갖는 데이터 집합을 말합니다. 예를 들어 지구 표면은 3차원 공간에 있지만 본질적으로 2차원 구조(위도·경도)입니다. UMAP은 이런 숨겨진 저차원 구조를 찾아내는 알고리즘입니다.

---

### 6.2 동작 원리

UMAP은 데이터를 **그래프**로 봅니다.

```
Step 1. KNN 이웃 찾기
  각 데이터에 대해 가장 가까운 k개 이웃 탐색

Step 2. 그래프 가중치 계산
  가까운 점 → 강한 연결 (wij ≈ 1)
  먼 점     → 약한 연결 (wij ≈ 0)
  단순화: wij = e^(-거리)
  예) 거리 1 → w = 0.368 / 거리 5 → w = 0.007

Step 3. 저차원에서도 같은 연결 구조 재현
  고차원 wij ≈ 저차원 ŵij 가 되도록 점들의 위치 조정

Step 4. Cross Entropy 손실 최소화로 반복 학습
```

---

### 6.3 코드 예제

```python
# 설치: pip install umap-learn
# ※ 설치 이름은 umap-learn 이지만 import할 때는 import umap 을 사용합니다.
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target

# UMAP 적용
# ※ 시각화 목적이므로 전체 데이터에 적용. 모델 전처리로 쓸 때는 반드시 train/test 분리 먼저.
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,   # KNN 이웃 수 (크면 전역 구조 반영)
    min_dist=0.1,     # 저차원에서 점들의 최소 거리 (크면 퍼짐)
    random_state=42
)
X_umap = umap_model.fit_transform(X)

# 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_umap[:, 0], X_umap[:, 1],
    c=y, cmap='tab10', alpha=0.7, s=10
)
plt.colorbar(scatter, label="숫자 (0~9)")
plt.title("UMAP — 손글씨 숫자 시각화 (64차원 → 2차원)")
plt.show()
```

**주요 파라미터 가이드:**

| 파라미터 | 의미 | 작을 때 | 클 때 |
|---------|------|--------|------|
| `n_neighbors` | 이웃 수 | 지역 구조 강조, 군집이 잘게 쪼개짐 | 전역 구조 반영, 군집이 합쳐짐 |
| `min_dist` | 저차원 최소 거리 | 같은 군집끼리 촘촘히 모임 | 점들이 고르게 퍼짐 |

> ⚠️ UMAP도 t-SNE와 마찬가지로 **UMAP 1, UMAP 2 좌표 자체보다 점들의 배치(군집 형태)** 를 봐야 합니다. 군집 간 상대적 거리를 수치로 해석하는 것은 주의가 필요합니다.

---

### 6.4 UMAP을 모델 입력으로 실험해볼 때

UMAP은 `transform()`을 지원하므로 t-SNE와 달리 새 데이터에도 같은 변환을 적용할 수 있습니다.

```python
# 모델 입력용 차원 축소 실험 예시 (교차검증으로 성능 반드시 확인)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

umap_model = umap.UMAP(n_components=10, random_state=42)
X_train_umap = umap_model.fit_transform(X_train)  # 훈련 데이터에만 fit
X_test_umap  = umap_model.transform(X_test)        # 테스트는 transform만
```

모델 입력용 차원 축소에는 보통 **PCA처럼 안정적이고 해석 가능한 방법을 먼저 사용**합니다. UMAP은 `transform()`을 지원하므로 실험해볼 수 있지만, 파라미터와 랜덤성에 따라 결과가 크게 바뀔 수 있습니다. 반드시 **교차검증으로 PCA 대비 성능을 비교**한 뒤 사용 여부를 결정하세요.

| 항목 | 내용 |
|------|------|
| **장점** | t-SNE보다 빠름, 대용량 적합, 전역 구조도 어느 정도 보존, transform 지원 |
| **단점** | 파라미터에 따라 결과 차이 큼, 별도 설치 필요 (`pip install umap-learn`) |
| **언제 쓰나** | 대용량 군집 시각화, 탐색적 임베딩 분석 |

---

## 7. Autoencoder (오토인코더)

### 7.1 핵심 개념

Autoencoder는 **딥러닝 기반 차원 축소** 알고리즘입니다.

입력 데이터를 **압축(Encoder)** 했다가 다시 **복원(Decoder)** 하도록 학습하면서, 그 과정에서 중간의 압축된 표현(잠재 벡터)을 차원 축소 결과로 사용합니다.

```
입력 x
  ↓ Encoder
잠재 벡터 z  ← 이것이 차원 축소 결과
  ↓ Decoder
복원값 x̂

학습 목표: x ≈ x̂  (원본과 복원값의 차이 최소화)
```

> 💡 **비유**: 친구에게 복잡한 영화 줄거리를 한 문장으로 요약(Encoder)했다가, 그 한 문장으로 다시 원래 줄거리를 재구성(Decoder)하는 것과 같습니다. 요약이 잘 될수록 핵심 정보를 잘 담은 것입니다. Autoencoder는 "핵심만 남기는 요약"을 데이터에서 스스로 학습합니다.

---

### 7.2 구조 이해

```
[13차원 → 2차원 → 13차원 Autoencoder 예시]

Input (13)
  ↓ Linear(13→64) + ReLU
Hidden (64)
  ↓ Linear(64→32) + ReLU
Hidden (32)
  ↓ Linear(32→2)
Bottleneck / 잠재 벡터 z (2)  ← 차원 축소 결과
  ↓ Linear(2→32) + ReLU
Hidden (32)
  ↓ Linear(32→64) + ReLU
Hidden (64)
  ↓ Linear(64→13)
Output (13)
```

`Bottleneck Layer`가 원본보다 작은 차원을 가지므로, 네트워크가 데이터의 핵심 구조를 그 좁은 공간에 압축하는 법을 학습하게 됩니다.

---

### 7.3 코드 예제 (PyTorch — 실행 가능한 전체 코드)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── 1. 데이터 준비 ─────────────────────────────────────────────────────────
wine = load_wine()
X = wine.data   # 178개 샘플, 13개 특성

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test 분리 (과적합 확인용)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

input_dim = X_train_t.shape[1]   # 13

# ── 2. 모델 정의 ────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        # 출력층에 Sigmoid를 쓰지 않음
        # ← StandardScaler는 음수를 포함하므로 Sigmoid(0~1 출력)와 맞지 않음
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)   # 활성함수 없음 → 범위 제한 없는 실수 출력
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

# ── 3. 학습 ─────────────────────────────────────────────────────────────────
model     = Autoencoder(input_dim=input_dim, latent_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(150):
    model.train()
    output = model(X_train_t)
    loss   = criterion(output, X_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f}")

# ── 4. Train / Test 복원 손실 비교 (과적합 확인) ───────────────────────────
model.eval()
with torch.no_grad():
    train_loss = criterion(model(X_train_t), X_train_t).item()
    test_loss  = criterion(model(X_test_t),  X_test_t ).item()

print(f"\nTrain reconstruction loss: {train_loss:.4f}")
print(f"Test  reconstruction loss: {test_loss:.4f}")

# ── 5. 차원 축소 결과 추출 ──────────────────────────────────────────────────
with torch.no_grad():
    z_all = model.encode(
        torch.tensor(X_scaled, dtype=torch.float32)
    ).numpy()   # shape: (178, 2)

print(f"\n차원 축소 완료: {input_dim}차원 → {z_all.shape[1]}차원")
```

**학습 과정 및 결과 해석:**
```
Epoch  30 | Train Loss: 0.8500
Epoch  60 | Train Loss: 0.5300
Epoch  90 | Train Loss: 0.3800
Epoch 120 | Train Loss: 0.2900
Epoch 150 | Train Loss: 0.2300

Train reconstruction loss: 0.2300
Test  reconstruction loss: 0.2600

Train Loss가 감소한다          → 복원 성능 향상 (압축이 잘 되고 있음)
Train/Test 손실이 비슷하다     → 과적합 없이 학습됨
Train Loss↓, Test Loss만 높다 → 과적합 → 구조나 학습률 조정 필요
```

---

### 7.4 출력층 활성화 함수 선택 주의

| 입력 데이터 전처리 | 출력층 | 이유 |
|-----------------|-------|------|
| `StandardScaler` (평균=0, 표준편차=1) | **활성함수 없음** | 음수 포함 → Sigmoid 사용하면 복원 불가 |
| `MinMaxScaler` (0~1 정규화) | `Sigmoid` 사용 가능 | 출력 범위가 0~1과 일치 |

> ⚠️ 가장 흔한 실수 중 하나입니다. `StandardScaler`로 표준화한 데이터에 `Sigmoid` 출력층을 달면 음수 부분을 아예 복원할 수 없습니다. 위 예제처럼 `StandardScaler`를 쓸 때는 마지막 `Linear` 뒤에 활성함수를 붙이지 마세요.

**PCA vs Autoencoder 비교:**

| 항목 | PCA | Autoencoder |
|------|-----|-------------|
| 기반 | 선형 변환 | 딥러닝 (비선형) |
| 속도 | 매우 빠름 | 학습 시간 필요 |
| 비선형 구조 | 표현 못함 | 표현 가능 |
| 과적합 확인 | 불필요 | train/test loss 비교 필요 |
| 주 용도 | 전처리, 빠른 축소 | 이미지 압축, 이상 탐지 |

| 항목 | 내용 |
|------|------|
| **장점** | 비선형 구조 학습, 다양한 데이터 타입 적용 가능 |
| **단점** | 학습 시간, 구조 설계 필요, 해석 어려움 |
| **언제 쓰나** | 이미지 압축, 이상 탐지, 복잡한 패턴 표현 |

---

## 8. 그 외 알고리즘 간략 정리

### 8.1 SVD (특이값 분해)

행렬을 세 행렬로 분해해서 중요한 정보만 남기는 방법입니다.

```
A = U × Σ × Vᵀ

큰 특이값(Σ의 대각 원소)만 사용 → 원본 압축
```

> 💡 텍스트 데이터에서 TF-IDF 행렬을 `TruncatedSVD`로 줄이는 방식(LSA)이 실제로 많이 쓰입니다. 추천 시스템(협업 필터링)에서도 핵심 기술입니다.

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100)
X_svd = svd.fit_transform(X_tfidf)  # TF-IDF 행렬에 적용
```

### 8.2 ICA (독립 성분 분석)

섞인 신호를 서로 독립적인 성분으로 분리합니다.

```
[칵테일 파티 문제]
여러 사람이 동시에 말하는 소리 → 각자의 목소리 분리

섞인 신호 → ICA → 독립된 신호 1 (사람 A), 신호 2 (사람 B), 신호 3 (배경음)
```

> 💡 PCA는 "분산이 큰 방향"을 찾고, ICA는 "서로 독립적인 방향"을 찾습니다. 뇌파(EEG) 분석에서 눈 깜빡임 노이즈를 제거할 때 많이 씁니다.

```python
from sklearn.decomposition import FastICA

ica = FastICA(n_components=3, random_state=42)
X_ica = ica.fit_transform(X)
```

### 8.3 NMF (비음수 행렬 분해)

데이터가 0 이상인 경우에만 사용할 수 있는 행렬 분해입니다.

```
V ≈ W × H   (단, 모든 원소 ≥ 0)

문서-단어 행렬에서 토픽 추출:
  토픽 1: "파이썬", "코드", "함수" 관련 단어들이 높은 가중치
  토픽 2: "주가", "투자", "배당" 관련 단어들이 높은 가중치
```

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=5, random_state=42)
X_nmf = nmf.fit_transform(X)  # X의 모든 값은 반드시 0 이상이어야 함
```

> ⚠️ **NMF에 StandardScaler를 쓰면 안 됩니다.** `StandardScaler`는 음수를 만들 수 있어 NMF 조건(모든 값 ≥ 0)을 위반합니다. NMF를 쓸 때는 `MinMaxScaler`를 사용하거나, Count/TF-IDF처럼 애초에 비음수인 데이터에 적용하세요.

### 8.4 Isomap

데이터가 곡면(매니폴드) 위에 분포할 때, 그 곡면 구조를 보존하면서 차원을 줄입니다.

```
[직관]
말려 있는 종이(Swiss Roll) 위의 점들
  ↓ Isomap
종이를 펼쳤을 때의 좌표로 변환
```

> ⚠️ 계산 비용이 크고 이상치에 민감합니다. 실무보다는 매니폴드 구조 연구에서 더 많이 쓰입니다.

```python
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2, n_neighbors=5)
X_isomap = isomap.fit_transform(X)
```

---

## 9. PCA vs LDA vs t-SNE vs UMAP 실전 비교

같은 데이터에 4가지 알고리즘을 한 번에 적용해 비교해봅시다.

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# 데이터 로드 (1,797개 샘플, 64차원)
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ※ 시각화 목적이므로 전체 데이터에 한 번에 적용

# --- 각 알고리즘 적용 ---
X_pca   = PCA(n_components=2).fit_transform(X_scaled)
X_lda   = LinearDiscriminantAnalysis(n_components=2).fit_transform(X_scaled, y)
X_tsne  = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_scaled)
X_umap  = umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)

# --- 시각화 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
methods = [
    (X_pca,  "PCA"),
    (X_lda,  "LDA"),
    (X_tsne, "t-SNE"),
    (X_umap, "UMAP"),
]

for ax, (data, title) in zip(axes.flatten(), methods):
    scatter = ax.scatter(data[:, 0], data[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{title}1")
    ax.set_ylabel(f"{title}2")

fig.colorbar(scatter, ax=axes, label="숫자 (0~9)")
plt.suptitle("차원 축소 알고리즘 비교 — 손글씨 숫자 (64차원 → 2차원)", fontsize=14)
plt.tight_layout()
plt.show()
```

**결과에서 보통 보이는 패턴:**
```
PCA:    군집이 어느 정도 보이지만 겹치는 부분이 많음
LDA:    10개 숫자 클래스가 방향적으로 분리됨 (라벨 활용)
t-SNE:  군집이 가장 선명하게 분리됨 (시각화 특화)
UMAP:   t-SNE와 유사하나 군집 간 상대적 위치도 어느 정도 유지됨
```

---

## 10. 실무 선택 가이드

| 상황 | 추천 알고리즘 | 이유 |
|------|-------------|------|
| 가장 기본적인 전처리 | **PCA** | 빠르고 안정적, 대부분 상황에서 무난 |
| 라벨이 있는 분류 전처리 | **LDA** | 클래스 분리 기준으로 축 생성 |
| 고차원 데이터 탐색 시각화 | **t-SNE** | 군집 구조를 가장 선명하게 표현 |
| 대용량 시각화 | **UMAP** | t-SNE보다 빠르고 큰 데이터에도 적합 |
| 텍스트 TF-IDF 행렬 축소 | **TruncatedSVD** | 희소 행렬에 특화 |
| 이미지·복잡한 데이터 압축 | **Autoencoder** | 비선형 구조 학습 가능 |
| 신호 분리 (뇌파, 음성) | **ICA** | 독립 성분 분리에 특화 |
| 토픽 추출 (양수 데이터) | **NMF** | 결과 해석이 비교적 직관적 |

> 💡 **처음 접하는 데이터라면 PCA → t-SNE 순서로 해보세요.** PCA로 빠르게 구조를 파악하고, 군집 패턴이 더 궁금하면 t-SNE나 UMAP으로 세밀하게 보는 방식이 실무에서도 자주 쓰입니다.

---

## 11. 주의사항 & 자주 하는 실수

### ⚠️ 모델 전처리용 차원 축소에서 데이터 누수

```python
# ❌ 잘못된 방법 (PCA가 테스트 데이터 정보를 간접적으로 봄)
X_pca = PCA(n_components=2).fit_transform(X_scaled)  # 전체 데이터에 fit
X_train, X_test, y_train, y_test = train_test_split(X_pca, y)

# ✅ 올바른 방법 1: 직접
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)  # 훈련에만 fit
X_test_pca  = pca.transform(X_test)       # 테스트는 transform만

# ✅ 올바른 방법 2: Pipeline 활용 (권장)
from sklearn.pipeline import Pipeline
model = Pipeline([('scaler', StandardScaler()), ('pca', PCA(2)), ('clf', LogisticRegression())])
model.fit(X_train, y_train)
```

### ⚠️ t-SNE 결과를 모델 입력으로 쓰기

```python
# ❌ t-SNE는 transform()이 없어 새 데이터에 적용 불가
X_tsne = TSNE(n_components=2).fit_transform(X_train)
# → X_test에 같은 변환을 적용할 방법이 없음

# ✅ 시각화 목적으로만 사용
# ✅ 모델 입력용 전처리는 PCA 또는 (실험 목적으로) UMAP
```

### ⚠️ StandardScaler + Autoencoder Sigmoid 조합

```python
# ❌ 표준화 데이터에 Sigmoid 출력층 사용
self.decoder = nn.Sequential(..., nn.Linear(64, input_dim), nn.Sigmoid())
# → Sigmoid는 0~1 출력 / 표준화 데이터는 음수 포함 → 복원 불가

# ✅ StandardScaler 사용 시 마지막 Linear에 활성함수 없음
self.decoder = nn.Sequential(..., nn.Linear(64, input_dim))
```

### ⚠️ NMF에 StandardScaler 적용

```python
# ❌ StandardScaler → 음수 생성 → NMF 오류
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
nmf = NMF(n_components=5)
X_nmf = nmf.fit_transform(X_scaled)  # 오류: NMF 입력에 음수 있음

# ✅ MinMaxScaler 또는 원래 비음수 데이터 사용
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(X)
```

### ⚠️ LDA를 이진 분류에서 n_components=2로 설정

```python
# ❌ 이진 분류(클래스 2개) → 축은 최대 1개
lda = LinearDiscriminantAnalysis(n_components=2)  # ValueError 발생

# ✅
lda = LinearDiscriminantAnalysis(n_components=1)
```

### ⚠️ t-SNE/UMAP 좌표값을 직접 수치로 해석

```
t-SNE 1: -12.5   t-SNE 2: 3.1
→ 이 숫자 자체는 의미 없음
→ 군집 A와 군집 B 사이의 거리로 "A가 B보다 2배 멀다"는 해석도 피해야 함
→ 같은 색(같은 클래스)끼리 모여 있는가, 다른 색과 잘 분리되는가만 봐야 함
```

---

## DAY6 정리

```
✅ 차원 축소 = 고차원 → 저차원, 정보 최대 보존
   → 학습 속도↑, 과적합↓, 시각화 가능

✅ 차원의 저주: 차원이 늘수록 필요 데이터 지수 함수적으로 증가

✅ 특성 선택 vs 특성 추출
   → 선택: 기존 변수 그대로 일부만 사용
   → 추출: 기존 변수 조합해 새 변수 생성 (PCA, t-SNE 등)

✅ 목적에 따라 알고리즘과 코드 방식이 달라짐
   → 시각화: 전체 데이터에 한 번에 적용 가능 (PCA, t-SNE, UMAP)
   → 모델 전처리: 반드시 Train/Test 분리 후 훈련 데이터에만 fit()

✅ PCA — 분산이 큰 방향 찾기 (비지도, 선형)
   → 표준화 필수, explained_variance_ratio_ 확인
   → n_components는 누적 설명 분산 80~95% 기준으로 결정
   → Pipeline으로 데이터 누수 방지

✅ LDA — 클래스 분리 방향 찾기 (지도, 선형)
   → y 라벨 필요, 축 수 = 클래스 수 - 1
   → 차원 축소이면서 분류 모델로도 사용 가능

✅ t-SNE — 가까운 이웃 관계 보존 시각화 (비지도, 비선형)
   → 군집 표현 탁월, 느림, transform 없음 → 시각화 전용
   → 좌표값보다 군집 배치 패턴을 봐야 함
   → 빠르게 쓰려면 PCA로 먼저 30~50차원으로 줄인 뒤 적용

✅ UMAP — t-SNE보다 빠른 시각화 (비지도, 비선형)
   → transform 지원, 대용량 적합
   → 모델 입력으로 실험 가능하나 교차검증으로 성능 확인 필수

✅ Autoencoder — 딥러닝 기반 압축·복원 (비지도, 비선형)
   → StandardScaler 사용 시 출력층에 Sigmoid 쓰지 말 것
   → Train/Test 복원 손실 비교로 과적합 확인

✅ NMF는 반드시 비음수 데이터에만 사용 (StandardScaler와 같이 쓰지 말 것)

✅ 실전 선택 원칙
   → 기본 전처리: PCA
   → 분류 전처리: LDA
   → 탐색 시각화: t-SNE → UMAP 순으로 시도
   → 복잡한 구조 압축: Autoencoder
```

---

## 🔗 참고 자료

- [scikit-learn PCA 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn LDA 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [scikit-learn t-SNE 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [UMAP 공식 문서](https://umap-learn.readthedocs.io/)
- [PyTorch nn.Module 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [Google Colab](https://colab.research.google.com/)
- [Kaggle — 데이터셋 & 대회](https://www.kaggle.com/)
