# DAY1 딥러닝 개념과 프레임워크 설치 - Codex 통합 검토안

## 1. 검토 대상과 전체 평가

- PDF: `sources/8_딥러닝개념_프레임워크설치.pdf`
  - 요청에 적힌 `sources/DAY1_딥러닝_개념과_프레임워크설치.pdf`는 실제로 존재하지 않아, 내용상 대응되는 위 52쪽 PDF를 검토했다.
- 초안: `drafts/DAY1_딥러닝_개념과_프레임워크설치_초안_v1.md`
- 검토 방식:
  - 1~23쪽은 텍스트 레이어와 삽입 그림을 함께 확인했다.
  - 24~52쪽은 텍스트가 거의 없는 연속 이미지 슬라이드이므로 모든 페이지의 그림을 직접 확인했다.
  - 초안의 Python 코드 블록 3개는 문법 파싱을 통과했다.
  - 현재 환경에는 `tensorflow`, `torch`, `scikit-learn`, `ucimlrepo`가 없어 실제 학습 실행은 검증하지 못했다.

초안은 PDF의 큰 흐름을 충실하게 복원했고, PDF의 위험한 GPU 코드와 보스턴 집값 데이터셋 문제를 그대로 답습하지 않은 점이 좋다. 회귀의 `선형 출력 + MSE`, 이진 분류의 `sigmoid 출력 + binary_crossentropy` 조합도 방향이 맞다.

다만 최종본 전에는 다음 항목을 반드시 고쳐야 한다.

1. `StandardScaler`를 검증 데이터까지 포함한 `X_train` 전체에 맞춘 뒤 `validation_split`을 적용하므로 **검증 데이터 누수**가 발생한다.
2. Keras를 "TensorFlow 위에서만 동작하는 API"라고 설명한 부분은 **Keras 3 기준으로 오래된 설명**이다.
3. `Python → Anaconda → CUDA → cuDNN → PyTorch`를 일반적인 필수 설치 순서처럼 보이게 하면 현재 사용자에게 오해를 준다.
4. UCI Heart Disease의 범주형 코드를 전부 연속형 수치처럼 표준화하는 실습은 교육용 단순화임을 밝혀야 한다.
5. PDF의 역사 연표를 그대로 사실화한 "1995년 은닉층/DNN 부활" 설명은 지나치게 단순하고 일부 부정확하다.
6. 303개 표본에 1000-1000 은닉층을 쓰면서 조기 종료와 학습 곡선 해석이 없어 과적합 위험이 매우 크다.

---

## 2. PDF 페이지별 주제와 초안 반영 상태

| PDF 페이지 | 주요 내용 | 초안 상태 | 권장 조치 |
|---|---|---|---|
| 1~2 | 표지, 목차 | 반영 | 유지 |
| 3~7 | AI·ML·DL 정의, 규칙 기반 방식과 학습 방식 비교 | 대체로 반영 | "딥러닝은 특징을 스스로 추출한다"는 절대 표현 완화 |
| 8 | 시대별 AI 역할 변화 | 일부만 반영 | 필수는 아님. 활용 분야 표와 중복되므로 생략 가능 |
| 9~15 | McCulloch-Pitts, Rosenblatt, XOR, Hinton, AlexNet, 생성형 AI 연표 | 반영 | PDF의 단순 연표와 실제 연구사를 구분해 정정 |
| 16 | 퍼셉트론의 가중합, 편향, 활성화 함수 | 반영 | 순전파 개념과 연결하면 더 명확 |
| 17~18 | 빅데이터, GPU, 알고리즘 발전 | 반영 | 유지 |
| 19~21 | 비전, NLP, 음성, 생성형 AI, 모델별 활용 | 반영 | 모델과 분야를 1:1 대응처럼 단정하지 않기 |
| 22~23 | TensorFlow·Keras·PyTorch, CPU/GPU, CUDA/cuDNN, 환경 점검 | 반영 | Keras 3, Windows TensorFlow GPU, 현재 설치 방식 정정 |
| 24~25 | 보스턴 집값 데이터와 열 설명 | 반영 | PDF 내부 행·열 수 오류와 12개 입력 기준을 명시 |
| 26~27 | 상자그림, 산점도, 상관관계 | 상자그림만 반영 | 산점도와 상관관계 해석을 짧게 추가 |
| 28~30 | DataFrame, Z-점수, 학습/평가 분할 | 대부분 반영 | DataFrame은 한 문장 보충, 검증 세트까지 분리 |
| 31~34 | 회귀 DNN, ReLU, SGD, MSE, `evaluate`/`predict`, `regplot` | 대부분 반영 | `evaluate`와 `predict` 차이, 회귀 예측 그래프 해석 추가 |
| 35~36 | 보스턴 실습 문제 7개 | 대부분 요약 | `epochs=0`은 "학습 없음"으로 정정 |
| 37~38 | Heart Disease 303행, 13개 특성+정답, 변수 설명 | 반영 | 원래 표적 0~4를 코드에서 이진화한다는 설명 강화 |
| 39~40 | DNN 순전파, 역전파, 기울기 소실 | 일부만 반영 | 순전파·역전파가 사실상 누락되어 추가 필요 |
| 41~44 | 표준화, 상자그림 | 반영 | 범주형 코드까지 표준화하는 한계 명시 |
| 45~47 | 1000-1000 tanh, sigmoid, SGD, MSE | 반영 및 일부 정정 | 분류 손실 정정은 좋음. 큰 모델·MSE 오용을 더 분명히 구분 |
| 48 | PyTorch tensor와 NumPy 비교 | 반영 | 모든 tensor가 자동 미분되는 것은 아니라는 정정 필요 |
| 49~50 | 혼동행렬, precision, recall, accuracy, F1 | 반영 | 임상적 유효성을 암시하지 않도록 주의 문구 추가 |
| 51~52 | 심장병 실습 문제 7개 | 대부분 요약 | 임의 시드 제거 실험과 검증 세트 사용 원칙 보충 |

---

## 3. 높은 우선순위 수정

### 3.1 검증 데이터 누수 수정

**위치:** 초안 6.3의 데이터 분할·표준화·`validation_split=0.2`

현재 코드는 먼저 `X_train` 전체로 `StandardScaler.fit_transform()`을 수행한 다음, Keras가 그 배열의 일부를 검증 데이터로 떼어 낸다. 따라서 검증 데이터의 평균과 표준편차가 전처리 통계에 이미 포함된다. 테스트 데이터 누수는 막았지만 **검증 데이터 누수는 남아 있다.**

다음 순서로 바꿔야 한다.

1. 원본 데이터를 `train/validation/test`로 먼저 분리한다.
2. 결측치 대체, 인코딩, 표준화는 `train`에만 `fit`한다.
3. `validation`과 `test`에는 `transform`만 적용한다.
4. 모델 선택과 실습 비교는 validation 결과로 하고 test는 마지막에 한 번만 평가한다.

### 3.2 Keras 설명을 Keras 3 기준으로 수정

**위치:** 초안 5.1

초안의 "Keras는 TensorFlow 위에서 동작하는 고수준 API"는 과거 `tf.keras`를 설명할 때는 맞았지만 현재 Keras 3 전체를 설명하기에는 부정확하다. Keras 3는 TensorFlow, JAX, PyTorch 백엔드를 지원한다. 다만 초안의 실습 코드는 `tf.keras`를 사용하므로 **이 예제는 TensorFlow 백엔드의 Keras를 쓴다**고 쓰면 된다.

공식 근거: [Keras 3 소개](https://keras.io/keras_3/)

### 3.3 설치 순서를 "강의 당시 방식"과 "현재 권장 방식"으로 분리

**위치:** 초안 5.3

Anaconda, CUDA Toolkit, cuDNN을 항상 수동으로 먼저 설치해야 PyTorch를 쓸 수 있는 것처럼 보이면 안 된다. 현재는 운영체제·GPU에 맞춰 PyTorch 공식 설치 선택기가 제시하는 명령을 사용하는 것이 우선이다. CPU 사용자는 CUDA·cuDNN이 필요 없다.

또한 Windows에서 TensorFlow GPU를 다룰 경우 중요한 제한이 빠져 있다. TensorFlow 공식 문서상 네이티브 Windows GPU 지원은 2.10이 마지막이며, 2.11 이상은 WSL2 경로를 안내한다.

- [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)
- [TensorFlow pip 설치와 Windows GPU 제한](https://www.tensorflow.org/install/pip)

### 3.4 Heart Disease 데이터 설명과 전처리 보완

**위치:** 초안 6.3

PDF의 "14가지 항목"은 **13개 입력 특성 + 정답 1개**를 뜻한다. 초안의 13개 특성 설명은 맞지만 다음 두 사항을 본문에 명시해야 한다.

- UCI 원본의 표적 `num`은 단순한 0/1이 아니라 원래 0~4이며, 초안 코드가 `0`과 `1~4`를 묶어 이진화한다.
- `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal` 등은 범주형 코드다. 이를 모두 연속형 수치처럼 Z-점수 표준화하면 코드 사이의 거리와 순서에 불필요한 의미를 부여한다.

실전형 예제에서는 범주형 열을 원-핫 인코딩하고, 연속형 열만 표준화하는 편이 낫다. PDF 구조를 그대로 재현하려면 "강의 흐름을 단순 재현하기 위해 13개 값을 수치 입력으로 사용한다"는 한계를 밝혀야 한다.

공식 데이터 설명: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

### 3.5 신경망 역사 설명 정정

**위치:** 초안 3.2~3.3과 연표

다음 문장은 PDF의 단순화된 서술을 사실처럼 확정한다.

- XOR 한계 하나 때문에 약 20년의 AI 겨울이 왔다.
- 1995년에 Hinton이 은닉층을 넣어 DNN을 부활시켰다.

은닉층은 1995년에 처음 등장한 개념이 아니며, 다층 신경망 학습의 중요한 전환점으로는 1980년대 역전파의 확산, 2006년 전후의 심층 신경망 학습 연구, 2012년 AlexNet 등을 함께 보는 편이 정확하다. AI 겨울도 기술적 한계, 과도한 기대, 투자 축소 등 여러 원인이 겹친 결과다.

PDF 내용을 버릴 필요는 없지만 다음처럼 표시해야 한다.

> 강의 PDF는 흐름을 쉽게 보여 주기 위해 `1969 XOR 한계 → 1995 Hinton → 2012 AlexNet`으로 단순화한다. 실제 연구사는 역전파, 계산 자원, 데이터, 학습 기법과 여러 연구자의 기여가 누적된 과정이다.

### 3.6 모델 규모와 과적합 위험 수정

**위치:** 초안 6.3

약 300개 표본에 `Dense(1000) → Dense(1000)`을 쓰면 파라미터가 100만 개를 넘는다. PDF 설계도 소개 자체는 유지할 수 있지만, 이를 권장 실전 모델처럼 제시하면 안 된다.

- PDF 설계도: 1000-1000 tanh 구조를 그대로 설명
- 실행 예제: 64-32 정도의 작은 기준 모델 사용
- `EarlyStopping(restore_best_weights=True)` 추가
- `loss`와 `val_loss` 곡선을 함께 보여 과적합을 해석

---

## 4. 텐서 shape, dtype, 출력층과 손실함수

### 4.1 현재 조합 판정

| 과제 | 출력층 | 손실 | 판정 |
|---|---|---|---|
| 회귀 | `Dense(1)`, 활성화 없음 | `mse` | 적합 |
| 이진 분류 | `Dense(1, activation="sigmoid")` | `binary_crossentropy` | 적합 |

초안이 PDF의 심장병 분류용 MSE 설명을 그대로 따르지 않고 binary cross-entropy로 바꾼 것은 반드시 유지해야 한다.

### 4.2 shape와 dtype 보완

현재 설명은 출력이 `(batch, 1)`, 정답이 `(batch,)`라고 되어 있다. Keras가 일반적인 경우 이를 처리할 수 있더라도, 입문 글에서는 다음처럼 명시적으로 맞추는 편이 안전하고 이해하기 쉽다.

```python
y_train = y_train.to_numpy(dtype="float32").reshape(-1, 1)
y_val = y_val.to_numpy(dtype="float32").reshape(-1, 1)
y_test = y_test.to_numpy(dtype="float32").reshape(-1, 1)
```

```text
입력 X          : (batch_size, num_features), float32
모델 출력       : (batch_size, 1), float32
정답 y          : (batch_size, 1), float32
```

`sigmoid`를 제거해 raw logit 하나를 출력하는 방식도 가능하지만, 그때는 다음처럼 손실을 함께 바꿔야 한다.

```python
tf.keras.layers.Dense(1)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```

두 방식을 섞어 `sigmoid + from_logits=True`로 작성하면 안 된다.

### 4.3 PyTorch tensor 설명 정정

초안의 "PyTorch tensor는 자동 미분을 지원한다"는 표현은 조건을 붙여야 한다. 모든 tensor가 자동으로 gradient를 추적하는 것은 아니다. 특히 현재 예제의 `torch.tensor([1, 2, 3])`은 정수형 `torch.int64`이고 `requires_grad=True`를 사용할 수 없다.

권장 예:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()

print(x.dtype)  # torch.float32
print(x.grad)   # tensor([2., 4., 6.])
```

---

## 5. 부족하거나 누락된 설명

### 5.1 순전파와 역전파

PDF 39~40쪽의 핵심인데 초안에서는 SGD 설명에 흡수되어 거의 사라졌다. 다음 정도는 추가해야 한다.

- 순전파: 입력이 각 층을 지나 예측값이 되는 과정
- 손실 계산: 예측과 정답의 차이를 수치화
- 역전파: 손실이 각 가중치에 얼마나 영향을 받는지 chain rule로 계산
- optimizer step: 계산된 gradient를 사용해 가중치를 갱신

기울기 소실도 "sigmoid/tanh를 ReLU로 바꾸면 해결된다"가 아니라 **완화할 수 있다**고 써야 한다. 초안의 해당 단서는 적절하므로 순전파·역전파 문단과 연결하면 된다.

### 5.2 산점도와 회귀선

PDF 27쪽과 34쪽에는 산점도, 상관관계, `seaborn.regplot`이 나온다. 초안은 상자그림만 자세히 다룬다.

다음 요지를 짧게 추가하면 PDF 누락을 메울 수 있다.

> 상자그림은 각 변수의 분포와 이상치를 보고, 산점도는 한 특성과 목표값 사이의 관계를 본다. 회귀선이 보여도 인과관계가 증명되는 것은 아니며, 휘어진 관계나 집단별 차이는 한 직선으로 충분히 설명되지 않을 수 있다.

### 5.3 `evaluate()`와 `predict()` 구분

- `evaluate(X_test, y_test)`: 정답과 비교해 loss와 metric을 계산
- `predict(X_new)`: 정답 없이 새 입력의 예측값이나 점수를 생성

PDF 34쪽의 "새로운 데이터" 설명과 달리 초안 코드는 `X_test`에 `predict()`를 적용한다. 이는 혼동행렬을 만들기 위한 평가 예측이므로 정상이다. 다만 실제 서비스 추론과 테스트 평가 예측을 구분해 설명해야 한다.

### 5.4 상자그림의 whisker와 이상치

초안은 상자그림을 "최솟값·Q1·중앙값·Q3·최댓값"으로 설명하지만, 일반적인 Tukey boxplot에서 whisker 끝은 이상치를 제외한 범위의 끝이지 항상 실제 최솟값과 최댓값은 아니다.

또한 상자그림의 점을 발견했다고 자동 삭제하면 안 된다. 측정 오류일 수도 있지만 실제로 중요한 희귀 사례일 수도 있다.

### 5.5 학습 곡선 해석

`history`를 저장하지만 사용하지 않는다. 다음을 보여 주어야 "학습/검증 지표를 따로 추적"한다는 설명이 완성된다.

```python
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.xlabel("epoch")
plt.ylabel("binary cross-entropy")
plt.legend()
plt.show()
```

- 두 loss가 모두 높음: 과소적합 가능성
- train loss는 감소하지만 validation loss가 증가: 과적합 가능성

### 5.6 의료 데이터 주의

UCI Heart Disease 예제는 학습용 데이터다. 정확도나 F1이 높아도 실제 진단 도구의 임상적 유효성을 뜻하지 않는다. 표본이 작고 오래됐으며, 실제 의료 적용에는 외부 검증, 민감도·특이도, calibration, 임계값과 오류 비용 검토가 필요하다는 문장을 추가해야 한다.

---

## 6. 불필요하거나 줄일 부분

1. 최종본에서는 맨 아래의 `작성 메모(초안 v1)` 전체를 삭제한다. 제작 메모이지 독자용 본문이 아니다.
2. "강의 PDF 코드 전체가 아니다", "프레임워크가 바뀐다"는 경고가 여러 번 반복된다. 6장 시작의 한 개 callout과 코드 제목으로 통합할 수 있다.
3. 역사 연표는 PDF의 핵심이라 삭제할 필요는 없지만, 연구자별 서사를 길게 늘리기보다 "강의의 단순 연표 + 정확성 보충"으로 압축한다.
4. `Transformer = 생성형 AI`, `CNN = 이미지`처럼 모델과 분야가 고정된 것처럼 보이는 표는 "대표적으로 잘 알려진 활용"이라고 제목을 바꾼다.
5. "특이점이 많은 특성은 학습에 방해가 된다"는 문장은 삭제하거나 완화한다. 이상치는 원인 확인이 먼저다.

---

## 7. 붙여 넣기용 교체·추가 블록

### 7.1 Keras와 설치 설명 교체안

```markdown
### Keras는 무엇인가

강의 PDF는 Keras를 TensorFlow와 함께 소개합니다. 과거에는 `tf.keras`가
TensorFlow의 대표적인 고수준 API로 사용되었고, 이 글의 실습도
`tf.keras`를 사용합니다.

다만 현재의 **Keras 3**는 TensorFlow뿐 아니라 JAX와 PyTorch도 백엔드로
사용할 수 있는 멀티 백엔드 API입니다. 따라서 이 글에서는
"Keras 전체 = TensorFlow 전용"이라고 일반화하지 않고,
"아래 예제는 TensorFlow 백엔드의 Keras를 사용한다"고 구분하겠습니다.

> 설치 주의: Anaconda, CUDA, cuDNN을 모든 사용자가 같은 순서로 직접
> 설치해야 하는 것은 아닙니다. CPU/GPU와 운영체제에 맞춰 프레임워크
> 공식 설치 페이지가 제시하는 명령을 따르세요. Windows에서 최신
> TensorFlow GPU를 사용하려면 네이티브 Windows가 아니라 WSL2 경로를
> 먼저 확인해야 합니다.
```

### 7.2 역사 설명 교체안

```markdown
> **역사 설명 주의**
>
> 강의 PDF는 입문 흐름을 위해 `퍼셉트론 → XOR 한계 → Hinton →
> AlexNet`으로 단순화합니다. 실제 딥러닝의 발전은 한 사건이나 한
> 연구자만으로 설명하기 어렵습니다. 다층 신경망, 역전파, 학습 기법,
> 데이터, GPU와 여러 연구자의 성과가 누적되었고, AI 겨울 역시 기술적
> 한계와 과도한 기대, 투자 축소 등이 함께 작용했습니다.
```

### 7.3 상자그림 설명 교체안

```markdown
상자그림은 Q1, 중앙값, Q3와 데이터의 퍼짐을 요약합니다. 일반적인
Tukey 방식에서 whisker는 `Q1 - 1.5×IQR`과 `Q3 + 1.5×IQR` 안쪽의
가장 먼 관측값까지 이어지므로, 항상 실제 최솟값·최댓값을 뜻하지는
않습니다. whisker 밖의 점도 자동 삭제 대상이 아니라 측정 오류인지,
실제 희귀 사례인지 먼저 확인해야 합니다.
```

### 7.4 누수·범주형 처리·shape를 반영한 심장병 예제 핵심 교체안

아래 코드는 PDF의 1000-1000 구조를 그대로 복제하는 코드가 아니라, 같은 분류 흐름을 더 안전하게 보여 주는 **실전형 보충 예제**로 표시한다.

```python
import numpy as np
import tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# 1) 데이터 로드
heart = fetch_ucirepo(id=45)
X = heart.data.features.copy()
y = (heart.data.targets.iloc[:, 0] > 0).astype("float32")

# 2) train/validation/test를 전처리보다 먼저 분리
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

# 3) 연속형과 범주형을 다르게 처리
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_cols = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_cols),
])

X_train = preprocessor.fit_transform(X_train).astype("float32")
X_val = preprocessor.transform(X_val).astype("float32")
X_test = preprocessor.transform(X_test).astype("float32")

y_train = y_train.to_numpy(dtype="float32").reshape(-1, 1)
y_val = y_val.to_numpy(dtype="float32").reshape(-1, 1)
y_test = y_test.to_numpy(dtype="float32").reshape(-1, 1)

# 4) 작은 기준 모델
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="roc_auc"),
    ],
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0,
)

# 모델 선택이 끝난 뒤 test를 한 번만 평가
results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
print(results)

proba = model.predict(X_test, verbose=0).ravel()
pred = (proba >= 0.5).astype(int)
true = y_test.ravel().astype(int)

print(confusion_matrix(true, pred))
print(classification_report(true, pred, digits=3, zero_division=0))
```

### 7.5 순전파·역전파 추가안

```markdown
### 신경망은 어떻게 학습할까

1. **순전파(Forward propagation)**: 입력이 은닉층과 출력층을 지나
   예측값이 됩니다.
2. **손실 계산**: 예측값과 정답의 차이를 손실함수로 계산합니다.
3. **역전파(Backpropagation)**: 각 가중치가 손실에 미친 영향을
   chain rule로 계산해 gradient를 구합니다.
4. **가중치 갱신**: SGD 같은 optimizer가 gradient를 이용해 손실이
   작아지는 방향으로 가중치를 조금 이동시킵니다.

sigmoid와 tanh를 깊게 쌓으면 gradient가 매우 작아지는 기울기 소실이
생길 수 있습니다. ReLU 계열은 이를 완화하는 데 도움이 되지만 모든
경우에 문제를 완전히 없애는 것은 아닙니다.
```

---

## 8. 실습 문제 표현 수정

- `MY_EPOCH = 0`
  - 기존: "학습이 거의 안 된 상태"
  - 수정: "가중치 업데이트를 전혀 하지 않은 초기 모델의 평가 결과"
- 여러 설정 비교:
  - test 결과를 반복 비교하지 말고 validation 결과로 비교한다.
- 임의 시드 제거:
  - PDF 51쪽의 실습 문제다. 결과가 매번 달라질 수 있음을 보여 주되, 공정한 모델 비교에서는 같은 시드를 유지한다고 설명한다.
- 특성 제거:
  - 한 번의 성능 변화만 보고 특성 중요도를 확정하지 않는다.
- 처음 100개만 사용:
  - 정렬된 원본의 앞부분만 자르면 클래스 분포가 달라질 수 있다. `train_test_split(..., stratify=y)` 또는 층화 표본 추출을 사용한다.

---

## 9. 우선순위 표

| 우선순위 | 수정 항목 | 이유 |
|---|---|---|
| 높음 | train/validation/test 선분리 후 전처리 | 현재 코드에 검증 누수 존재 |
| 높음 | Keras 3 설명 정정 | 현재 프레임워크 설명과 불일치 |
| 높음 | 설치 순서와 Windows TensorFlow GPU 제한 보완 | 설치 실패 가능성이 큰 핵심 주제 |
| 높음 | 원본 표적 0~4와 이진화 명시 | 데이터 의미가 코드에서 바뀜 |
| 높음 | 범주형 열 인코딩 또는 교육용 단순화 표시 | 수치 코드의 잘못된 거리·순서 가정 |
| 높음 | 역사 연표 정정 | 은닉층·AI 겨울 설명이 지나치게 단순 |
| 높음 | 대형 모델 축소 또는 과적합 경고·조기 종료 | 표본 수 대비 파라미터가 지나치게 많음 |
| 중간 | 출력과 정답 shape·dtype 일치 | 초보자의 shape 오류 예방 |
| 중간 | 순전파·역전파 추가 | PDF 핵심 내용 누락 |
| 중간 | 상자그림 whisker 설명 정정 | 최솟값·최댓값 설명이 부정확 |
| 중간 | tensor 자동 미분 조건 정정 | 정수 tensor 예제와 설명 불일치 |
| 중간 | 학습·검증 곡선 추가 | `history`를 만들고 해석하지 않음 |
| 중간 | 의료 데이터의 비임상용 주의 | 교실 성능을 진단 성능으로 오해할 위험 |
| 낮음 | DataFrame 한 문장 설명 | PDF 28쪽 보완 |
| 낮음 | 시대별 AI 역할 표 생략 여부 명시 | 다른 활용 분야 내용과 중복 |
| 낮음 | 작성 메모 삭제와 경고문 통합 | 최종 글의 가독성 개선 |

---

## 10. 최종 권고

현재 초안은 **구조와 PDF 범위는 좋지만, 그대로 최종 발행하기에는 환경 설치 설명과 검증 절차가 위험하다.** 최종본에서는 다음 순서로 반영하는 것이 좋다.

1. 검증 누수와 범주형 전처리 수정
2. Keras 3 및 Windows 설치 정보 수정
3. 역사 연표와 boxplot 설명 정정
4. 순전파·역전파, shape·dtype, 학습 곡선 보충
5. PDF의 1000-1000 모델은 "강의 설계도"로 남기고 실행 코드는 작은 기준 모델로 분리
6. 의료 데이터의 교육용 한계와 테스트 세트 1회 평가 원칙 명시
7. 초안 작성 메모와 반복 경고문 정리

위 수정 후에는 PDF 내용을 충실히 다루면서도, 초보자가 현재 환경에서 설치·실행하고 결과를 올바르게 해석할 수 있는 DAY1 글이 된다.
