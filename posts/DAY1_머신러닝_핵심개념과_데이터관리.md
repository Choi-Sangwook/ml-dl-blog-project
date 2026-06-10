# 🤖 머신러닝 완전 입문 가이드 — DAY1. 핵심 개념과 데이터 관리

> **시리즈**: 파이썬 기본만 있는 사람을 위한 머신러닝 입문
> **DAY1**: 머신러닝이 뭔지 → 개발환경 세팅 → 데이터를 다루는 법까지
> **DAY2**: 실제 모델 학습 → 평가 → 튜닝 (다음 편)

---

## 1. 인공지능 · 머신러닝 · 딥러닝의 관계

세 가지 용어를 혼용하는 경우가 많은데, 포함 관계를 먼저 정리하겠습니다.

```
인공지능 (AI)  ← 가장 큰 범위
 └── 머신러닝 (ML)  ← 데이터 기반으로 학습
      └── 딥러닝 (DL)  ← 심층 신경망 사용
```

| 구분 | 설명 | 예시 |
|------|------|------|
| **인공지능** | 사람처럼 지능적인 행동을 하는 시스템 전체 | 시리, 자율주행 |
| **머신러닝** | 데이터를 입력하면 컴퓨터가 **스스로 규칙을 찾는** 방식 | 스팸 필터, 추천 시스템 |
| **딥러닝** | 머신러닝 중 **심층 신경망(Deep Neural Network)**을 사용하는 방식 | 이미지 인식, ChatGPT |

### 핵심 차이: 규칙 기반 vs 데이터 기반

**규칙 기반 AI** (전통적 방식)
```
사용자가 규칙 입력 → 컴퓨터가 규칙 적용 → 해답 도출
```
- 장점: 새로운 규칙에 빠른 대응
- 단점: 규칙에 없는 경우는 대처 불가

**데이터 기반 AI** (머신러닝)
```
사용자가 데이터 입력 → 컴퓨터가 스스로 규칙 발견 → 해답 도출
```
- 장점: 사람이 미처 발견하지 못한 패턴도 찾아냄
- 단점: 양질의 데이터가 많이 필요

> 💡 **쉬운 비유**: 전통 프로그래밍은 "레시피 책을 주고 요리하라"는 것이고, 머신러닝은 "완성된 요리를 수천 번 맛보게 한 뒤 스스로 레시피를 알아내게" 하는 것입니다.

---

## 2. 개발 환경 준비

### 추천: Google Colab (초보자에게 최적)

[Google Colab](https://colab.research.google.com/)은 브라우저에서 바로 파이썬 코드를 실행할 수 있는 무료 클라우드 환경입니다.

**Colab의 장점**
- 설치 필요 없음 (브라우저만 있으면 됨)
- GPU 무료 제공 (딥러닝 실습 가능)
- 구글 드라이브에 자동 저장
- pandas, numpy, sklearn 등 주요 라이브러리가 이미 설치되어 있음

**Colab 기본 사용법**
1. [colab.research.google.com](https://colab.research.google.com/) 접속
2. `[파일]` → `[새 노트]`로 새 노트북 생성
3. 코드 셀에 코드 입력 후 `Ctrl + Enter`로 실행
4. 텍스트 셀에는 마크다운으로 메모 가능
5. 노트북은 구글 드라이브 `Colab Notebooks` 폴더에 자동 저장

**GPU 설정** (딥러닝 실습 시 필요)
```
[런타임] → [런타임 유형 변경] → 하드웨어 가속기: GPU 선택
```

**Colab 제약 사항**
- 무료 계정: 동시에 노트북 최대 5개
- 세션 하나당 최대 12시간 연결
- 세션 종료 시 설치한 패키지·업로드한 파일 초기화

### 핵심 라이브러리 한눈에 보기

```python
# 이 import문들은 앞으로 계속 쓰게 됩니다
import pandas as pd                 # 데이터 조작 (표 형태)
import numpy as np                  # 수치 연산 (배열/행렬)
import matplotlib.pyplot as plt     # 기본 시각화
import seaborn as sns               # 고급 시각화
from sklearn.model_selection import train_test_split  # 데이터 분리
```

| 라이브러리 | 용도 | 별명 |
|-----------|------|------|
| **pandas** | 데이터 조작 (표 형태) | `pd` |
| **numpy** | 수치 연산 (배열/행렬) | `np` |
| **matplotlib** | 기본 시각화 | `plt` |
| **seaborn** | 고급 시각화 | `sns` |
| **scikit-learn** | ML 알고리즘 모음 | `sklearn` |

### PyTorch vs TensorFlow

딥러닝 프레임워크는 두 가지가 대표적입니다. 입문 단계에서는 scikit-learn만 써도 충분하지만, 나중에 딥러닝으로 넘어갈 때를 위해 차이점을 알아둡시다.

| 비교 | PyTorch | TensorFlow |
|------|---------|------------|
| 개발사 | Meta (Facebook) AI | Google |
| 방식 | **Define by Run** (동적 그래프) | Define and Run (정적 그래프) |
| 특징 | 파이썬 친화적, 디버깅 쉬움 | 배포·상용화에 강점 |
| 추세 | 학계·연구에서 주류 | 산업 현장에서 많이 사용 |
| GPU 지원 | `.cuda()` 메서드 | 자동 감지 |

> 💡 최근에는 PyTorch 사용자가 빠르게 늘고 있으며, 입문 학습용으로도 PyTorch를 많이 추천합니다. 다만 지금 단계에서는 scikit-learn으로 충분합니다.

---

## 3. 머신러닝의 핵심 개념

### 3.1 머신러닝 3대 구성 요소

```
데이터(Data)  +  모델(Model)  +  학습 알고리즘(Training Algorithm)
     ↓              ↓                    ↓
  학습 재료      수학적 함수         파라미터를 최적화하는 방법
```

**데이터**: 모델이 학습할 재료입니다.
**모델**: 입력을 받아 출력을 내는 수학적 함수입니다. `y = w₁x + w₀` 같은 선형 함수가 가장 단순한 모델입니다.
**학습 알고리즘**: 데이터에 가장 잘 맞는 파라미터(w₀, w₁ 등)를 찾아주는 방법입니다.

> 💡 **학습이란?** 데이터가 제공됐을 때 모델의 파라미터(가중치)를 찾아주는 과정입니다. 예를 들어 `집값 = w₁ × 면적 + w₀`에서 최적의 w₁과 w₀를 찾는 것이 학습입니다.

### 3.2 일반화 (Generalization) — 가장 중요한 개념

- 훈련 데이터를 잘 맞히는 것 → 쉬움
- **처음 보는 새 데이터**도 잘 맞히는 것 → 어려움 (= **일반화**)
- 일반화 능력이 머신러닝 모델의 진짜 성능입니다

> 💡 시험 준비에 비유하면, 기출문제를 완벽히 외우는 것(과적합)보다 **어떤 문제가 나와도 풀 수 있는 실력**(일반화)이 중요한 것과 같습니다.

### 3.3 학습 방법의 3가지 분류

| 분류 | 정답 여부 | 핵심 | 대표 알고리즘 |
|------|---------|------|-------------|
| **지도학습** | 있음 ✅ | 정답을 보며 학습 | 선형회귀, 로지스틱회귀, 의사결정트리, 랜덤포레스트 |
| **비지도학습** | 없음 ❌ | 패턴/구조 발견 | K-means, PCA, t-SNE |
| **강화학습** | 보상만 | 행동→보상/페널티 | Q-learning, DQN |

**지도학습**은 다시 두 가지로 나뉩니다:

```
지도학습
 ├── 예측하는 값이 숫자 → 회귀 (Regression)
 │    예: 내일 기온은 몇 도일까?
 │
 └── 예측하는 값이 범주 → 분류 (Classification)
      예: 이 메일은 스팸일까 아닐까?
```

### 3.4 머신러닝의 실전 학습 단계

```
학습 데이터 준비
    ↓
분석모형(알고리즘) 선정  ←──────────┐
    ↓                              │ 반복 (Agile Iteration)
학습 (모델 개발)                    │
    ↓                              │
평가 (모델 평가)  ─── 성능 부족? ──→┘
    ↓
모델 선정
    ↓
운영 적용 (서비스 배포)
```

데이터를 준비하고 → 모델을 학습시키고 → 성능을 평가하고 → 부족하면 다시 돌아가 개선하는 **반복 과정**입니다.

---

## 4. 데이터 이해하기

### 4.1 정형 데이터 vs 비정형 데이터

**정형 데이터 (Tabular Data)** — 표 형태

```
| PassengerId | Pclass | Name            | Sex    | Age  | Survived |
|-------------|--------|-----------------|--------|------|----------|
| 1           | 3      | Braund, Mr. Owen| male   | 22.0 | 0        |
| 2           | 1      | Cumings, Mrs... | female | 38.0 | 1        |
```

- **행(row)** = 하나의 데이터 포인트 (1명의 승객)
- **열(column)** = 특징/변수 (feature)
- **독립변수 (X, Feature)**: 예측에 사용하는 입력 → Pclass, Sex, Age
- **종속변수 (y, Target)**: 예측하고 싶은 값 → Survived
- 정형 데이터에는 의사결정 트리 계열이 적합

**비정형 데이터 (Informal Data)** — 이미지, 텍스트, 음성, 비디오
- 표 형태가 아니며, 데이터의 특징이 불명확
- 특징 추출 후 분류하는 방식으로 처리
- 딥러닝 계열이 적합

### 4.2 변수(Feature)의 종류

| 유형 | 설명 | 예시 | 처리 방법 |
|------|------|------|---------|
| **수치형 (연속)** | 숫자로 측정 가능 | 키, 몸무게, 온도, 요금 | 스케일링 |
| **범주형 (명목)** | 순서 없는 카테고리 | 성별, 혈액형, 도시명 | 원핫 인코딩 |
| **범주형 (서열)** | 순서 있는 카테고리 | 학점(A>B>C), 등급(상>중>하) | 레이블 인코딩 |

> ⚠️ 변수 유형에 따라 전처리 방법과 사용할 알고리즘이 달라지므로 반드시 구분해야 합니다.

---

## 5. 데이터 전처리 — 실전에서 시간의 70~80%

> ⚠️ **"쓰레기를 넣으면 쓰레기가 나온다 (Garbage In, Garbage Out)"**
> 아무리 좋은 모델이라도 데이터가 나쁘면 결과도 나쁩니다. 머신러닝 프로젝트의 품질은 데이터의 품질에 달려있습니다.

### 5.1 전처리의 전체 흐름

```
원시 데이터 (raw data)
  ↓
① 정제(Cleansing) — 결측치, 이상치, 중복 제거
  ↓
② 통합(Integration) — 여러 데이터 소스 합치기
  ↓
③ 변환(Transformation) — 스케일링, 인코딩
  ↓
④ 축소(Reduction) — 차원 축소, 샘플링
  ↓
⑤ 특징 선택 및 생성 — 주요 변수 선택, 파생 변수 생성
  ↓
분석에 적합한 데이터 완성!
```

> 💡 실무에서는 데이터 탐색과 전처리가 동시에 진행되는 경우가 많습니다. 결측치 여부 확인은 탐색이고, 결측치 제거는 전처리인데 보통 이어서 합니다.

> ⚠️ 위 흐름은 "어떤 전처리 작업들이 있는지"를 정리한 것입니다. 실제 코딩 순서는 **데이터 분리를 먼저** 한 뒤, 훈련 데이터에만 전처리를 적용해야 합니다. 이유는 5.9절(데이터 누수)에서 다룹니다.

### 5.2 데이터 탐색 (EDA) — 먼저 데이터를 "봐야" 합니다

타이타닉 데이터셋으로 실습해봅시다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('titanic.csv')

# 1) 상위 5개 행 확인
df.head()
```
**출력 결과:**
```
   PassengerId  Survived  Pclass                            Name   Sex   Age  SibSp  Parch     Fare Embarked
0            1         0       3         Braund, Mr. Owen Harris  male  22.0      1      0   7.2500        S
1            2         1       1  Cumings, Mrs. John Bradley...  female  38.0      1      0  71.2833        C
2            3         1       3         Heikkinen, Miss. Laina  female  26.0      0      0   7.9250        S
3            4         1       1  Futrelle, Mrs. Jacques Heath   female  35.0      1      0  53.1000        S
4            5         0       3      Allen, Mr. William Henry    male  35.0      0      0   8.0500        S
```

```python
# 2) 데이터 구조 확인
df.info()
```
**출력 결과:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 10 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object   ← 문자열
 4   Sex          891 non-null    object   ← 문자열 (인코딩 필요)
 5   Age          714 non-null    float64  ← 177개 결측치!
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Fare         891 non-null    float64
 9   Embarked     889 non-null    object   ← 2개 결측치
dtypes: float64(2), int64(5), object(3)
```

```python
# 3) 기술통계 (수치형 변수만)
df.describe()
```
**출력 결과:**
```
       PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
```

> 💡 **읽는 법**: Age의 count가 714 → 891-714=177개 결측. mean=29.7이니 평균 약 30세. Fare의 max=512인데 mean=32 → 이상치가 있을 수 있음.

```python
# 4) 결측치 개수 확인
df.isnull().sum()
```
**출력 결과:**
```
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177   ← 결측치 177개 (19.9%)
SibSp            0
Parch            0
Fare             0
Embarked         2   ← 결측치 2개
dtype: int64
```

```python
# 5) 상관관계 히트맵 (수치형 변수들 간의 관계 파악)
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), 
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('변수 간 상관관계')
plt.show()
```
> Survived와 Fare는 양의 상관관계(0.26), Survived와 Pclass는 음의 상관관계(-0.34)가 보입니다. 즉 "요금이 비쌀수록, 객실 등급이 높을수록 생존율이 높다"는 패턴을 숫자로 확인할 수 있습니다.

### 5.3 결측치 (Missing Value) 처리

**결측치가 왜 문제인가?**
대부분의 ML 알고리즘은 NaN이 있으면 에러가 나거나 잘못된 결과를 냅니다.

#### 방법 1: 삭제

```python
# 결측치 있는 행 전체 삭제
df_dropped = df.dropna()
print(f"삭제 전: {len(df)}행 → 삭제 후: {len(df_dropped)}행")
```
**출력 결과:**
```
삭제 전: 891행 → 삭제 후: 712행
```
> ⚠️ 179행(20%)이 사라졌습니다. 데이터가 적을 때 이렇게 많이 삭제하면 정보 손실이 큽니다.

#### 방법 2: 평균/중앙값/최빈값으로 채우기

```python
# 수치형: 중앙값으로 채우기 (이상치에 영향을 덜 받음)
df['Age'].fillna(df['Age'].median(), inplace=True)

# 범주형: 최빈값으로 채우기
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 확인
print(df.isnull().sum())
```
**출력 결과:**
```
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0   ← 결측치 해결!
SibSp          0
Parch          0
Fare           0
Embarked       0   ← 결측치 해결!
dtype: int64
```

#### 방법 3: 앞/뒤 값으로 채우기 (시계열 데이터에 유용)

```python
# 시계열 예시: 온도 데이터
temps = pd.Series([20.1, np.nan, np.nan, 22.5, 23.0, np.nan, 24.5])

print("원본:         ", temps.values)
print("앞값으로(ffill):", temps.fillna(method='ffill').values)
print("뒤값으로(bfill):", temps.fillna(method='bfill').values)
```
**출력 결과:**
```
원본:          [20.1  nan  nan 22.5 23.   nan 24.5]
앞값으로(ffill): [20.1 20.1 20.1 22.5 23.  23.  24.5]
뒤값으로(bfill): [20.1 22.5 22.5 22.5 23.  24.5 24.5]
```

**결측치 처리 선택 가이드**

| 상황 | 추천 방법 |
|------|---------|
| 결측치가 전체의 5% 미만 | 삭제 |
| 수치형 변수 + 이상치 많음 | **중앙값** |
| 수치형 변수 + 이상치 적음 | 평균값 |
| 범주형 변수 | **최빈값** |
| 시계열 데이터 | 앞/뒤 채우기 (ffill/bfill) |

### 5.4 이상치 (Outlier) 처리

다른 데이터 포인트들과 현저히 차이 나는 값을 처리합니다.

#### IQR 방법으로 이상치 탐지

```python
# Fare(요금) 컬럼의 이상치 확인
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"Q1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"하한선: {lower:.2f}")
print(f"상한선: {upper:.2f}")

outliers = df[(df['Fare'] < lower) | (df['Fare'] > upper)]
print(f"\n이상치 개수: {len(outliers)}개")
```
**출력 결과:**
```
Q1: 7.91
Q3: 31.00
IQR: 23.09
하한선: -26.72
상한선: 65.63

이상치 개수: 116개
```

#### 박스플롯으로 시각적 확인

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 처리 전
axes[0].boxplot(df['Fare'])
axes[0].set_title('처리 전: Fare')

# 이상치 제거 후
df_clean = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]
axes[1].boxplot(df_clean['Fare'])
axes[1].set_title('처리 후: Fare')

plt.show()
print(f"제거 전: {len(df)}행 → 제거 후: {len(df_clean)}행")
```
**출력 결과:**
```
제거 전: 891행 → 제거 후: 775행
```
> 💡 이상치 처리 방법은 제거만 있는 것이 아닙니다. 평균/중앙값 대체, 로그 변환, 클리핑(상한/하한으로 고정) 등도 많이 쓰입니다.

### 5.5 인코딩 — 문자를 숫자로

ML 알고리즘은 숫자만 이해합니다. 문자 데이터를 숫자로 변환해야 합니다.

#### 레이블 인코딩: 카테고리에 번호 부여

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])

# 확인
print(df[['Sex', 'Sex_encoded']].drop_duplicates())
```
**출력 결과:**
```
      Sex  Sex_encoded
0    male            1
1  female            0
```

#### 원핫 인코딩: 카테고리마다 별도 열 생성

```python
# 순서가 없는 범주형에 적합
df_encoded = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

print(df_encoded[['Embarked_C', 'Embarked_Q', 'Embarked_S']].head())
```
**출력 결과:**
```
   Embarked_C  Embarked_Q  Embarked_S
0       False       False        True
1        True       False       False
2       False       False        True
3       False       False        True
4       False       False        True
```

#### 인코딩 선택 기준 — 순서가 있는지 먼저 확인

단순히 "문자를 숫자로 바꾼다"가 아니라, **범주 사이에 순서가 있는지**를 먼저 봐야 합니다.

| 상황 | 예시 | 추천 방식 |
|------|------|---------|
| 순서가 있는 범주 | 낮음/중간/높음, 초급/중급/고급 | **레이블 인코딩** 또는 직접 매핑 |
| 순서가 없는 범주 | 성별, 지역, 혈액형, 승선항 | **원핫 인코딩** |

예를 들어 `Embarked`가 `C`, `Q`, `S`라면 숫자 0, 1, 2로 바꾸는 순간 모델이 "S(2) > Q(1) > C(0)"라는 순서가 있다고 오해할 수 있습니다. 이런 경우에는 원핫 인코딩이 더 자연스럽습니다.

```python
# drop_first=True로 다중공선성 방지 (C를 기준으로 Q, S만 남김)
df_encoded = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

> ⚠️ **트리 계열 모델**(의사결정트리, 랜덤포레스트)은 조건 분기 방식이라 레이블 인코딩을 써도 큰 문제가 없습니다. 하지만 **선형 모델, SVM, KNN** 등에서는 원핫 인코딩이 더 안전합니다.

### 5.6 스케일링 — 단위 통일

키(170cm)와 몸무게(65kg)처럼 범위가 다른 변수가 있으면, 값이 큰 변수가 모델에 과도한 영향을 줍니다.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 예시 데이터
data = pd.DataFrame({
    '키(cm)': [165, 170, 175, 180, 185],
    '몸무게(kg)': [55, 60, 65, 70, 75]
})

print("=== 원본 데이터 ===")
print(data)
```
**출력 결과:**
```
=== 원본 데이터 ===
   키(cm)  몸무게(kg)
0    165        55
1    170        60
2    175        65
3    180        70
4    185        75
```

```python
# StandardScaler: 평균=0, 표준편차=1로 변환
scaler = StandardScaler()
data_standard = pd.DataFrame(
    scaler.fit_transform(data), 
    columns=data.columns
)
print("\n=== StandardScaler 적용 후 ===")
print(data_standard.round(2))
```
**출력 결과:**
```
=== StandardScaler 적용 후 ===
   키(cm)  몸무게(kg)
0  -1.41     -1.41
1  -0.71     -0.71
2   0.00      0.00
3   0.71      0.71
4   1.41      1.41
```

```python
# MinMaxScaler: 0~1 범위로 변환
scaler = MinMaxScaler()
data_minmax = pd.DataFrame(
    scaler.fit_transform(data), 
    columns=data.columns
)
print("\n=== MinMaxScaler 적용 후 ===")
print(data_minmax.round(2))
```
**출력 결과:**
```
=== MinMaxScaler 적용 후 ===
   키(cm)  몸무게(kg)
0   0.00      0.00
1   0.25      0.25
2   0.50      0.50
3   0.75      0.75
4   1.00      1.00
```

**스케일러 선택 가이드**

| 상황 | 추천 스케일러 |
|------|-----------|
| 이상치가 많다 | `StandardScaler` (이상치에 비교적 강건) |
| 값의 범위를 0~1로 맞추고 싶다 | `MinMaxScaler` |
| 의사결정 트리 계열 모델 | **스케일링 안 해도 됨** (트리는 크기 비교만 함) |
| 선형 모델, KNN, SVM | **반드시 스케일링 필요** |

### 5.7 scikit-learn의 `fit`, `transform`, `predict` 이해하기

scikit-learn을 쓰다 보면 `fit()`, `transform()`, `predict()`가 계속 등장합니다. 처음에는 그냥 외워서 쓰기 쉽지만, 의미를 알고 나면 코드 흐름이 훨씬 잘 보입니다.

| 메서드 | 의미 | 예시 |
|------|------|------|
| `fit()` | 데이터에서 필요한 **규칙이나 기준을 학습** | 평균·표준편차 계산, 모델 파라미터 학습 |
| `transform()` | `fit()`에서 배운 기준으로 **데이터를 변환** | 스케일링, 인코딩 적용 |
| `fit_transform()` | `fit()`과 `transform()`을 **한 번에** 수행 | 훈련 데이터 전처리할 때 |
| `predict()` | 학습된 모델로 **새 입력의 결과를 예측** | 생존 여부, 집값 예측 |

예를 들어 `StandardScaler`는 데이터를 표준화할 때 평균과 표준편차가 필요합니다.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 훈련 데이터에서 평균과 표준편차를 학습(fit)하고 변환(transform)
X_train_scaled = scaler.fit_transform(X_train)

# 테스트 데이터는 훈련 데이터에서 학습한 기준으로만 변환(transform)
X_test_scaled = scaler.transform(X_test)
```

> ⚠️ **핵심: 테스트 데이터에는 절대 `fit()`을 하면 안 됩니다.** 테스트 데이터는 "처음 보는 데이터"처럼 남겨두어야 모델 성능을 공정하게 평가할 수 있습니다. 이유는 바로 다음 섹션에서 설명합니다.

### 5.8 데이터 분리 — Train / Test

학습에 쓴 데이터로 평가하면 시험 답을 미리 본 것과 같습니다. 반드시 학습/테스트 데이터를 분리해야 합니다.

```python
from sklearn.model_selection import train_test_split

# 타이타닉 데이터에서 독립변수(X), 종속변수(y) 설정
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# 8:2로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 테스트 20%
    random_state=42      # 재현성을 위한 시드값
)

print(f"전체 데이터: {len(X)}개")
print(f"훈련 세트:  {X_train.shape[0]}개 ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"테스트 세트: {X_test.shape[0]}개 ({X_test.shape[0]/len(X)*100:.0f}%)")
```
**출력 결과:**
```
전체 데이터: 891개
훈련 세트:  712개 (80%)
테스트 세트: 179개 (20%)
```

#### `stratify`를 쓰는 이유

분류 문제에서는 Train/Test를 나눌 때 **클래스 비율이 비슷하게 유지**되는 것이 중요합니다. 전체에서 생존자가 38%, 사망자가 62%라면 훈련·테스트 데이터도 비슷한 비율이어야 합니다.

```python
# stratify=y를 넣으면 y의 클래스 비율을 유지하면서 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y          # ← 이 한 줄 추가!
)

print(f"전체 생존 비율:    {y.mean():.4f}")
print(f"훈련 생존 비율:    {y_train.mean():.4f}")
print(f"테스트 생존 비율:  {y_test.mean():.4f}")
```
**출력 결과:**
```
전체 생존 비율:    0.3838
훈련 생존 비율:    0.3834
테스트 생존 비율:  0.3855
```

> 💡 `stratify=y`를 넣으면 비율이 거의 동일하게 유지됩니다. 특히 **데이터가 적거나 클래스 불균형**이 있는 분류 문제에서 꼭 사용하세요.

#### 실전에서의 데이터 분리: Train / Validate / Test

실전에서는 보통 3개로 나눕니다:

```
전체 데이터 (100%)
 ├── Train set (60~70%) — 모델 학습용
 ├── Validation set (15~20%) — 하이퍼파라미터 튜닝용
 └── Test set (15~20%) — 최종 평가용 (딱 1번만 사용)
```

> ⚠️ Test set은 **최종 성능 측정에만** 딱 한 번 사용합니다. 여러 번 테스트하면서 모델을 수정하면, 그것도 일종의 "답을 보고 수정하는 것"이 됩니다.

### 5.9 데이터 누수 (Data Leakage) — 초보자가 가장 많이 하는 실수

데이터 누수란, 모델이 **실제로는 알 수 없어야 하는 정보를 학습 과정에서 미리 보게 되는 상황**입니다. 초보자가 가장 많이 하는 실수 중 하나입니다.

#### ❌ 잘못된 예시: 전체에 스케일링 → 분리

```python
# 이렇게 하면 안 됩니다!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← 전체 데이터에 fit (테스트 포함!)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

이렇게 하면 스케일링할 때 **테스트 데이터의 평균과 표준편차 정보가 훈련 과정에 간접적으로 들어갑니다.** 모델이 "아직 못 본" 데이터의 정보를 미리 엿본 셈입니다.

#### ✅ 올바른 예시: 분리 → 훈련 데이터에만 fit

```python
# 올바른 순서
# 1) 먼저 데이터를 나누고
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) 훈련 데이터에만 fit (기준 학습)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform

# 3) 테스트 데이터는 훈련 기준으로 transform만!
X_test_scaled = scaler.transform(X_test)         # transform만
```

> ⚠️ 데이터 누수가 생기면 테스트 성능이 **실제보다 좋게** 나옵니다. 모델을 잘 만든 것처럼 보이지만, 실제 서비스나 새로운 데이터에서는 성능이 떨어집니다. **분리 → fit(train만) → transform(test)** 순서를 꼭 지키세요.

---

## 6. 머신러닝의 한계와 주의점

기술적인 내용은 아니지만, 머신러닝을 쓸 때 반드시 알아야 하는 점들입니다.

**알고리즘의 한계**
- 머신러닝은 인간 뇌의 상대적인 영역만 모방합니다
- 학습한 파라미터 범위 내에서만 추정이 가능하며, 범위 밖의 예측은 신뢰하기 어렵습니다
- 빅데이터에서 패턴을 찾는 것은 사람보다 컴퓨터가 더 적합하지만, 결과의 의미 해석과 실행은 사람이 필요합니다

**데이터 편향 문제**
- 학습 데이터에 편향이 있으면 모델도 편향된 결과를 냅니다
- 예: 특정 성별·인종에 편중된 채용 데이터로 학습하면 차별적 모델이 만들어짐
- 데이터 수집 단계부터 편향을 인식하고 주의해야 합니다

**개인정보 보호**
- 건강 데이터, 위치 데이터 등 민감한 정보를 다룰 때는 특히 주의해야 합니다
- 분석 결과 적용에 앞서 상식적이고 윤리적인 면을 반드시 고려해야 합니다

---

## DAY1 정리

오늘 배운 핵심을 정리하면:

```
✅ 머신러닝 = 데이터로부터 규칙을 자동으로 찾는 것
✅ 3대 구성: 데이터 + 모델 + 학습 알고리즘
✅ 학습 방법: 지도학습(회귀/분류) / 비지도학습 / 강화학습
✅ 일반화 = 새로운 데이터에서도 잘 작동하는 능력
✅ 전처리가 전체 작업의 70~80%
✅ 전처리 작업: 결측치 → 이상치 → 인코딩 → 스케일링
✅ 올바른 순서: 데이터 분리 먼저 → 훈련 데이터에만 fit → 테스트에는 transform만!
✅ 분류 문제에서는 stratify로 클래스 비율 유지
```

> 다음 DAY2에서는 이 데이터를 가지고 실제 모델을 학습시키고, 평가하고, 튜닝하는 과정을 다룹니다.
