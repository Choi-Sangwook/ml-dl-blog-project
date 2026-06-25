# 🧠 딥러닝 완전 입문 가이드 — DAY3. 딥러닝 데이터 처리와 데이터셋

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: 딥러닝 DAY1(개념·프레임워크 설치) · DAY2(신경망 알고리즘·계층구조)
> **이번 편 한 줄 요약**: "딥러닝은 결국 **데이터**로 배운다." 데이터셋을 어떻게 나누고, 이미지를 어떻게 숫자로 바꿔 모델에 먹이며, 그 숫자 계산의 바탕인 **행렬**이 무엇인지까지 한 번에 정리합니다.

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 내용
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 사전지식·실무 주의점
> - 실습 코드는 **두 갈래**입니다.
>   - 이미지 분류(MNIST·CIFAR-10) → **TensorFlow / Keras** (PDF의 `Sequential·Dense·to_categorical` API와 동일)
>   - 행렬 계산(7장) → **NumPy** (딥러닝 프레임워크가 아니라 순수 배열·선형대수 도구)
> - **7장 NumPy 코드의 출력은 모두 실제 실행으로 확인**한 값입니다. 반면 **MNIST·CIFAR Keras 학습 코드는 작성 환경에 TensorFlow가 없어 실행하지 못했습니다.** 정확도 같은 수치는 비워 두었으니 직접 실행해 확인하세요. (임의의 결과를 지어내지 않았습니다.)

---

## 1. 이번 DAY에서 배우는 것

- 딥러닝에서 **데이터가 왜 가장 중요한가**, 그리고 데이터를 **훈련/검증/시험**으로 나누는 이유
- **언더피팅 vs 오버피팅**을 학습 곡선으로 읽는 법
- 딥러닝 모델을 만드는 **5단계 흐름** (데이터 준비 → 모델 설계 → 학습 설정 → 학습 → 평가)
- **MNIST** 손글씨 숫자를 끝까지 푸는 **하나의 완전한 실습** (전처리 → 모델 → Softmax → 평가)
- **Softmax vs Sigmoid** — 출력층과 손실함수를 짝 맞추기
- **CIFAR-10**이 MNIST보다 어려운 이유와 컬러 이미지(32×32×3)의 의미
- 딥러닝 계산의 바탕인 **행렬(다차원 배열) 연산**을 NumPy로 직접 확인

---

## 2. 데이터셋: 딥러닝의 교과서 🟦

### 2.1 왜 데이터가 중요한가

딥러닝 모델도 사람처럼 **많이 공부할수록** 잘하는 경향이 있습니다. 컴퓨터는 **입력 데이터 + 정답 데이터**를 함께 받아 그 안의 규칙·패턴을 스스로 찾고, 학습한 내용으로 새로운 데이터를 예측·분류합니다. 2016년 알파고 이후 음성·얼굴 인식, 자율주행, 의료 진단 등으로 쓰임새가 넓어졌습니다.

모델을 학습시키기 전에는 보통 **데이터를 수집하고 → 분포·오류·결측값을 분석하고 → 모델이 처리할 수 있는 숫자와 일정한 shape으로 가공**하는 단계를 거칩니다. 🟦

> 🟦 PDF의 예: 드론으로 **물에 빠진 사람**을 탐지하는 AI를 만들려면, 익수자 사진뿐 아니라 어린이·성인·노인·수영하는 사람 등 **다양한 상황**의 사진이 필요합니다.

> 🟩 **(보충) "많을수록·다양할수록 좋다"는 보장이 아니라 경향입니다** — 라벨이 틀렸거나, 실제 환경과 다른 데이터, 중복이 많으면 양이 늘어도 성능이 좋아지지 않을 수 있습니다. 정확히는 **실제 환경을 잘 대표하고 라벨 품질이 좋은 데이터를 충분히 확보**할 때 일반화 성능이 좋아질 가능성이 큽니다.

### 2.2 데이터를 셋으로 나눈다 — 훈련 / 검증 / 시험

딥러닝은 데이터를 **목적에 따라 세 가지**로 나눠 씁니다.

| 데이터셋 | 역할 | 학생 공부에 비유하면 |
|---|---|---|
| **훈련(Training)** | 모델이 가중치를 **학습** | 문제집으로 공부 |
| **검증(Validation)** | 학습 도중 **잘 되고 있는지 점검**, 설정 선택 | 모의고사로 실력 점검 |
| **시험(Test)** | 학습이 끝난 뒤 **최종 성능 평가(딱 한 번)** | 실제 수능 |

모든 데이터를 학습에만 쓰면 실력을 **객관적으로** 평가할 수 없습니다. 시험용 데이터를 따로 남겨 둬야 "처음 보는 문제"에 대한 진짜 실력을 잴 수 있습니다.

> 🟩 **(보충) 데이터 누수(leakage) 주의** — 시험 데이터는 학습·설정 선택 과정에서 **절대 미리 보면 안 됩니다.** 또한 표준화·결측치 대체처럼 **데이터에서 통계를 "배우는" 전처리는 훈련 데이터에서만 계산**하고, 검증·시험에는 그 기준을 적용만 해야 합니다. (이번 MNIST 예제의 `÷255`는 데이터에서 배우는 값이 아니라 **고정 상수**라 모든 데이터에 똑같이 써도 누수가 아닙니다.)

### 2.3 언더피팅과 오버피팅

| 상태 | 훈련 성능 | 검증 성능 | 해석 |
|---|---|---|---|
| **언더피팅(Underfitting)** | 낮음 | 낮음 | 모델이 패턴을 충분히 학습하지 못한 상태 |
| **적절한 학습** | 높음 | 높음 | 훈련에서 배운 패턴이 검증 데이터에도 비교적 잘 적용됨 |
| **오버피팅(Overfitting)** | 매우 높음 | 상대적으로 낮거나 악화 | 훈련 데이터에 지나치게 맞춰져 새 데이터에 일반화되지 못함 |

언더피팅·오버피팅은 **훈련·검증 성능의 차이와 변화 추세**로 진단합니다. **시험(test) 데이터는 이 진단이나 설정 선택에 사용하지 않고**, 모든 선택이 끝난 뒤 최종 성능을 확인할 때만 씁니다.

> 🟩 **(보충) 대응 방법(어느 것도 항상 해결을 보장하지는 않음)**
> - **언더피팅**: 모델 용량(층·노드) 늘리기, 특징·학습률·학습 시간 점검
> - **오버피팅**: 더 다양한 훈련 데이터, 데이터 증강, 정규화, 드롭아웃(dropout), 조기 종료(early stopping) 등을 검토

---

## 3. 딥러닝 모델을 만드는 5단계 흐름 🟦

PDF는 딥러닝 프로젝트를 다섯 단계로 정리합니다.

```
① 데이터 준비  →  ② 모델 설계  →  ③ 학습 방법 설정  →  ④ 학습  →  ⑤ 평가·활용
```

### 3.1 ② 모델 설계 — MLP, Sequential, Dense

이번 강의의 모델은 **MLP(다층 퍼셉트론)** 입니다. 여러 층을 순서대로 연결한 가장 기본적인 신경망으로, Keras에서는 **`Sequential`** 모델에 **`Dense`(완전 연결) 층**을 차례로 쌓아 만듭니다. (단층 퍼셉트론은 XOR도 못 풀기 때문에 층을 여러 개 쌓는다는 점은 DAY2에서 다뤘습니다.)

### 3.2 ③ 학습 방법 설정 — 손실함수·옵티마이저·평가지표

모델 구조를 만든 뒤에는 **어떻게 배울지**를 정합니다(`compile`).

| 구성 요소 | 역할 |
|---|---|
| **손실함수(Loss)** | 모델이 얼마나 **틀렸는지** 수치화 (작을수록 좋음) |
| **옵티마이저(Optimizer)** | 틀린 만큼 **가중치를 조정** |
| **평가지표(Metric)** | 정확도(Accuracy) 등 사람이 보는 **성능 수치** |

### 3.3 ④ 학습 — fit(), Epoch, Batch Size

학습은 `fit()` 으로 수행합니다.

- **Epoch**: 전체 데이터를 **몇 번 반복** 학습할지
- **Batch Size**: 한 번에 **몇 개씩** 처리하고 가중치를 수정할지
  - 예: Batch Size 10 → 10개 처리할 때마다 가중치 수정
  - 예: Batch Size 100 → 100개를 모두 처리한 뒤 한 번 수정

배치 크기는 **메모리 사용량뿐 아니라 학습의 안정성·일반화에도 영향**을 줍니다. 흔히 "큰 배치는 빠르고 작은 배치는 느리다"고 하지만, 실제 속도는 하드웨어·데이터 파이프라인·모델 크기에 따라 달라집니다.

> 🟦 PDF는 흔한 시작값으로 **32, 64, 128**을 제시합니다(보편적 정답은 아니며 출발점입니다).

### 3.4 ⑤ 평가 — Loss와 Accuracy, 그리고 학습 곡선

학습이 끝나면 **시험 데이터**로 평가합니다. 대표 지표는 두 가지입니다.

- **Loss(손실)**: 모델의 오차 → **낮을수록 좋음**
- **Accuracy(정확도)**: 전체 예측 중 **정답을 맞힌 비율** → **높을수록 좋음**

에폭별 값을 그래프로 그리면 상태가 한눈에 보입니다. 학습이 잘 되면 **Loss 곡선은 내려가고 Accuracy 곡선은 올라가는** 추세를 보입니다. (구체적인 그래프 코드는 4.6에서 실제 모델과 함께 다룹니다.)

> 🟩 **(보충)** Accuracy는 MNIST처럼 클래스 수가 균형적인 문제에 적합합니다. 클래스가 한쪽으로 치우쳤거나 오류 비용이 다른 문제에서는 정밀도(precision)·재현율(recall)·F1·혼동행렬 등을 함께 봐야 할 수 있습니다.

---

## 4. MNIST 실습 — 손글씨 숫자 분류 (완전 예제) 🟦 + 🟩

### 4.1 MNIST는 어떤 데이터인가

**MNIST**는 0~9 손글씨 숫자 이미지를 모은, 입문용으로 가장 유명한 데이터셋입니다.

- 총 **70,000장** = 훈련 **60,000** + 시험 **10,000**
- 각 이미지 = **28 × 28 = 784 픽셀**, 픽셀값 **0(검정)~255(흰색)**
- 입문에 적합한 이유: 크기가 적당해 일반 PC에서도 학습 가능하고, 결과를 **눈으로 확인**하기 쉬움
- 기본적인 신경망만으로도 **약 95% 이상**의 높은 정확도를 얻을 수 있다고 알려져 있음 🟦

> 🟩 **(보충)** 컴퓨터는 숫자를 "그림"이 아니라 **숫자들의 배열**로 봅니다. 글씨가 있는 부분은 밝기 값이 크고, 배경은 0에 가깝습니다. 학습 전에 직접 출력해 보면 이 점이 바로 와닿습니다.

```python
# (선택) 데이터가 '숫자 배열'임을 눈으로 확인 — 실행에는 TensorFlow 필요
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(X_train, y_train), _ = mnist.load_data()
print(X_train.shape, X_train.dtype, X_train.min(), X_train.max())
# 기대: (60000, 28, 28) uint8 0 255

plt.figure(figsize=(8, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"label={y_train[i]}"); plt.axis("off")
plt.tight_layout(); plt.show()
```

### 4.2 왜 전처리가 필요한가 — Flatten · 정규화 · 라벨 인코딩

이번 실습의 MLP는 입력 shape을 `(784,)` 로 설계하므로, 28×28 이미지를 **784개 특징의 1차원 벡터로 펼쳐** 넣습니다. 세 가지를 손봅니다.

**(1) Flatten (평탄화)** — 28×28 2차원을 **784개짜리 1차원 벡터**로 펼칩니다. `[[1,2,3],[4,5,6],[7,8,9]]` 를 `[1,2,3,4,5,6,7,8,9]` 로 펴는 것과 같습니다.

> 🟩 **(보충)** Flatten은 픽셀값 자체를 버리지 않습니다. 다만 **높이·너비의 2차원 이웃 구조를 입력 shape에 명시적으로 보존하지 않습니다.** 완전 연결(Dense)층은 CNN과 달리 지역성·이동 특성을 활용하는 구조적 가정이 없어서, 일반적인 이미지 분류에는 **CNN이 더 적합**합니다. (CNN은 이후 단계의 주제입니다.)

**(2) 정규화** — 0~255 픽셀을 **255로 나눠 0~1 범위**로 맞추면 학습이 안정적입니다. 🟩 (PDF엔 명시적 강조는 적지만, 입문 실습의 사실상 표준이라 포함합니다.)

**(3) 라벨 인코딩 — One-Hot** — 모델 출력은 "각 숫자일 점수 10개"입니다. **선택한 손실함수가 One-Hot 라벨을 요구할 때** 정답도 같은 형태로 바꿉니다. 숫자 5는:

```
5  →  [0,0,0,0,0,1,0,0,0,0]   (여섯 번째 자리 = 숫자 5)
3  →  [0,0,0,1,0,0,0,0,0,0]
```

PDF의 코드는 `Y_train = utils.to_categorical(Y_train)` 입니다.

> 🟩 **(보충) One-Hot은 "출력이 10개라서" 무조건 필요한 게 아닙니다** — 어떤 **손실함수**를 고르느냐에 따라 라벨 형식을 맞추는 것입니다. 이 글은 `categorical_crossentropy`를 쓰므로 One-Hot으로 바꿉니다. 정수 라벨(0~9)을 그대로 두고 싶으면, 출력층은 동일하게 Softmax를 쓰되 손실을 `sparse_categorical_crossentropy`로 선택하면 됩니다.

> 🟩 **(보충) Flatten·One-Hot을 NumPy로 직접 확인** (실제 실행 결과)
> ```python
> import numpy as np
>
> a = np.array([[1,2,3],[4,5,6],[7,8,9]])
> print(a.shape, "->", a.reshape(-1).shape)   # (3, 3) -> (9,)
> print(a.reshape(-1))                          # [1 2 3 4 5 6 7 8 9]
>
> def one_hot(y, n=10):
>     v = np.zeros(n, dtype=int); v[y] = 1; return v
> print(one_hot(5))   # [0 0 0 0 0 1 0 0 0 0]
> print(one_hot(3))   # [0 0 0 1 0 0 0 0 0 0]
> ```

### 4.3 모델 구조 — 784 → 512 → 256 → 128 → 10

```
입력층 784  →  은닉1 512(ReLU)  →  은닉2 256(ReLU)  →  은닉3 128(ReLU)  →  출력층 10(Softmax)
```

- 은닉층 활성화는 **ReLU** (학습이 빠르고 무난한 기본값)
- 출력층은 **Softmax** — 10개 값을 **0~1 확률**로 바꾸고 **합이 1**이 되게 함 → 다중 분류에 적합

> 🟩 **(보충)** 위 층 수·노드 수(512·256·128)는 **하나의 예시일 뿐 확정된 정답이 아닙니다.** PDF도 구조에 정해진 답이 없다고 강조합니다. 문제·데이터에 따라 실험적으로 조정하는 값으로 이해하세요.

### 4.4 Softmax 자세히 보기

모델의 마지막 층이 내놓는 값은 처음엔 **확률이 아니라 그냥 점수(logit)** 입니다. 예를 들어:

```
logit = [-2.3, 1.5, 0.8, 3.2, 0.1, 5.6, 2.1, 1.2, -1.0, 0.5]
```

이 값들을 **0~1 사이로 정규화**하고 **전체 합을 1**로 만드는 함수가 **Softmax**입니다.

$$
P(y_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

- $z_i$: 출력층의 i번째 점수, $e$: 자연상수(≈2.718), $P(y_i)$: i번째 클래스에 배정된 값

> 🟩 **(보충) 위 logit에 Softmax를 실제로 적용한 결과** (NumPy 실행 확인)
> ```python
> import numpy as np
>
> z = np.array([-2.3, 1.5, 0.8, 3.2, 0.1, 5.6, 2.1, 1.2, -1.0, 0.5])
> e = np.exp(z - z.max())     # max를 빼는 건 수치 안정화(결과는 동일)
> softmax = e / e.sum()
> print(np.round(softmax, 4))
> # [0.0003 0.0142 0.007  0.0775 0.0035 0.8548 0.0258 0.0105 0.0012 0.0052]
> print("합 =", round(softmax.sum(), 6))      # 합 = 1.0
> print("예측 =", int(softmax.argmax()))      # 예측 = 5
> ```
> 여섯 번째(숫자 5)가 **0.8548로 가장 크므로 예측은 5**입니다.
>
> ⚠️ PDF 슬라이드에 적힌 확률 숫자(예: `0.950`)는 위 logit에서 **실제로 계산된 값이 아니라 설명용 예시 수치**입니다. 직접 계산하면 약 **0.855**가 나옵니다. 개념(가장 큰 값을 고른다, 합이 1이다)은 동일합니다.

> 🟩 **(보충) Softmax 값 해석의 한계** — Softmax 출력의 합은 1이지만, 그 값이 **실제 정답 확률로 잘 보정(calibration)되어 있다는 보장은 없습니다.** 입문 단계에서는 "클래스별 상대 점수를 0~1로 정규화한 값이며, **가장 큰 값을 예측 클래스로 선택**한다" 정도로 이해하면 충분합니다.

### 4.5 전체 실행 코드 (TensorFlow / Keras)

> ⚠️ **이 코드는 작성 환경에 TensorFlow가 없어 실행하지 못했습니다.** 구조·전처리·출력층·손실함수 짝은 점검했으니, **정확도 등 실제 수치는 직접 실행**해 확인하세요.

> ⚠️ **설치 안내는 실행 시점의 공식 문서를 확인하세요**
> CPU 실습 기본 설치는 `python -m pip install tensorflow` 입니다. 지원 Python 버전과 GPU 설치 방법은 운영체제·TensorFlow 버전에 따라 다릅니다. 특히 **최신 TensorFlow의 NVIDIA GPU를 네이티브 Windows에서 바로 쓸 수 있다고 가정하면 안 되며**, WSL2 등 공식 지원 구성을 확인해야 합니다. ([TensorFlow 설치](https://www.tensorflow.org/install/pip))

```python
# 프레임워크: TensorFlow / Keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical

tf.keras.utils.set_random_seed(42)          # 재현성(완벽한 일치는 아닐 수 있음)

# 1) 데이터 로드
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train: (60000, 28, 28) uint8, 픽셀 0~255 / y_train: (60000,) 정수 0~9

# 2) 전처리 — Flatten + 정규화(÷255) + One-Hot
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0          # (60000, 784)
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0           # (10000, 784)
y_train = to_categorical(y_train, 10).astype("float32")              # (60000, 10)
y_test  = to_categorical(y_test, 10).astype("float32")               # (10000, 10)

# 3) validation_split은 배열의 '마지막 일부'를 떼어 검증으로 쓰므로,
#    먼저 재현 가능하게 순서를 섞어 둡니다(클래스 순서 편향 방지).
rng = np.random.default_rng(42)
order = rng.permutation(len(X_train))
X_train, y_train = X_train[order], y_train[order]

# 4) 모델 설계 (784 → 512 → 256 → 128 → 10)
model = Sequential([
    Input(shape=(784,)),
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax"),        # 다중 분류 → Softmax
])

# 5) 학습 설정 — Softmax + One-Hot 정답이면 categorical_crossentropy
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# 6) 학습 — 훈련 데이터의 일부를 검증으로 분리
history = model.fit(
    X_train, y_train,
    validation_split=0.2,                   # 훈련의 마지막 20%를 검증으로
    epochs=10, batch_size=128,
)

# 7) 평가 — 시험 데이터는 모든 선택이 끝난 뒤 마지막에 한 번만
test_loss, test_acc = model.evaluate(X_test, y_test)
print("test loss =", test_loss, " test acc =", test_acc)

# 8) TensorFlow가 실제로 인식한 장치 확인
print("인식된 GPU:", tf.config.list_physical_devices("GPU"))
```

**입력·출력 텐서 모양(shape)·자료형(dtype) 점검**

```
입력 X      : (batch, 784), float32, 값 0~1
출력        : (batch, 10),  float32, Softmax 확률(합=1)
정답 y      : (batch, 10),  float32, One-Hot
손실        : categorical_crossentropy  ← Softmax + One-Hot 짝
```

> 🟩 **(보충) 장치(device)에 대한 정확한 설명** — TensorFlow는 **현재 환경에서 인식한 호환 장치**에 연산을 배치합니다. GPU 하드웨어가 있어도 **드라이버와 TensorFlow 설치 구성이 맞지 않으면 CPU**를 씁니다. 위 `tf.config.list_physical_devices("GPU")`로 실제 인식 여부를 확인하세요. 네이티브 Windows의 최신 TensorFlow는 NVIDIA GPU를 직접 지원하지 않으므로, GPU 실습은 WSL2 등 공식 설치 조건을 확인해야 합니다.

> 🟩 **(보충) 라벨 형식과 손실함수 짝** — 정답을 **One-Hot**으로 줬으면 `categorical_crossentropy`, 정답을 **정수 라벨(0~9)** 그대로 줬으면 `sparse_categorical_crossentropy`를 씁니다. 출력층 Softmax는 동일합니다.

> 🟩 **(보충) `validation_split`을 더 명확히 쓰는 법** — 일반 분류 데이터에서는 가능하면 **클래스 비율을 유지하는(stratified) 분할**로 별도의 `X_val`, `y_val`을 만들어 `validation_data=(X_val, y_val)` 로 넘기는 편이 더 명확합니다. `set_random_seed`만으로는 `validation_split`의 "뒤에서 떼는" 동작이 무작위 분할로 바뀌지 않습니다.

> 🟩 **(보충) 파라미터 수 읽기** — `model.summary()`의 각 층 파라미터 수는 **(입력 수 + 편향 1) × 출력 노드 수**로 계산됩니다. 예를 들어 첫 층은 `(784 + 1) × 512 = 401,920`개입니다. DAY2에서 본 "은닉층을 쌓으면 파라미터가 빠르게 늘어난다"는 점과 이어집니다.

### 4.6 학습 곡선 확인 — Loss와 Accuracy 함께

```python
import matplotlib.pyplot as plt        # pip install matplotlib

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()

plt.tight_layout(); plt.show()
```

- 곡선은 매 epoch마다 완벽히 한 방향으로 움직이지 않고 조금씩 흔들릴 수 있으니 **전체 추세**를 봅니다.
- 훈련 loss는 계속 내려가는데 **validation loss가 다시 올라가거나**, 훈련 accuracy만 높아지고 **validation accuracy가 정체**되면 오버피팅을 의심합니다.

---

## 5. Softmax vs Sigmoid — 출력층·손실 짝 맞추기 🟦 + 🟩

PDF는 "여러 개 중 하나면 Softmax, 둘 중 하나면 Sigmoid"라고 설명합니다. 출력층·라벨·손실은 **서로 짝이 맞아야** 합니다.

| 문제 유형 | 출력층 활성화 | 정답(라벨) 형식 | Keras 손실함수 |
|---|---|---|---|
| **다중 분류**(MNIST 0~9) | **Softmax** (출력 10) | One-Hot | `categorical_crossentropy` |
| 다중 분류(정수 라벨) | Softmax | 정수 인덱스 | `sparse_categorical_crossentropy` |
| **이진 분류**(스팸/정상, 고양이/강아지) | **Sigmoid** (출력 1) | 0/1 | `binary_crossentropy` |

> 🟩 **(보충)** "둘 중 하나"인 이진 분류는 PDF 설명대로 **Sigmoid 출력 1개**로 풀 수 있습니다(고양이=1/강아지=0 식). 같은 문제를 Softmax(출력 2개)로 풀 수도 있어 **절대 규칙은 아니지만**, 입문 단계에서는 "이진=Sigmoid, 다중=Softmax"로 외워도 무방합니다. 라벨이 **여러 개 동시에 참**일 수 있는 다중 라벨(multi-label) 문제는 Softmax가 아니라 **라벨마다 Sigmoid**를 씁니다.

---

## 6. CIFAR-10 — MNIST보다 한 단계 어려운 컬러 이미지 🟦 + 🟩

### 6.1 CIFAR-10이란

캐나다 토론토 대학교가 만든 이미지 데이터셋으로 **10개 클래스**(이미지 종류)를 가집니다.

| 번호 | 클래스 | 번호 | 클래스 |
|---|---|---|---|
| 0 | 비행기(Airplane) | 5 | 개(Dog) |
| 1 | 자동차(Automobile) | 6 | 개구리(Frog) |
| 2 | 새(Bird) | 7 | 말(Horse) |
| 3 | 고양이(Cat) | 8 | 배(Ship) |
| 4 | 사슴(Deer) | 9 | 트럭(Truck) |

- 크기: **32 × 32 × 3** — 마지막 3은 **RGB 색상 채널**(Red·Green·Blue)
- 즉 입력값 = 32 × 32 × 3 = **3,072개** (MNIST의 784개보다 훨씬 많은 정보)
- 개수: 총 60,000장 = 훈련 **50,000** + 시험 **10,000**

### 6.2 왜 MNIST보다 어려운가

MNIST는 단순한 흑백 손글씨였지만, CIFAR-10은 **실제 사물 사진**이라:

- 고양이·개·말·사슴처럼 **서로 비슷한 대상**을 구분해야 하고,
- 같은 자동차라도 **촬영 위치·크기·각도·밝기·배경**이 제각각입니다.

그래서 훨씬 복잡한 특징을 학습해야 합니다. PDF는 더 잘게 나눈 **CIFAR-100(100개 세부 클래스)** 도 언급합니다.

> 🟩 **(보충) CIFAR-100 예시에 대한 메모** — PDF는 CIFAR-100의 예로 "푸들·시베리안 허스키·골든 리트리버"를 듭니다. **CIFAR-100의 공식 클래스 목록에는 그런 개 품종이 없으므로**, 이는 "분류가 더 잘게 쪼개진다"는 **세분화 개념을 설명하기 위한 예시**로 이해하는 편이 안전합니다.

> 🟩 **(보충) 공개 데이터셋도 완벽하지 않습니다** — Keras 공식 문서는 CIFAR-10에 **소량의 잘못된 라벨**이 알려져 있다고 안내합니다. "데이터 품질은 늘 점검 대상"이라는 좋은 사례입니다.

### 6.3 코드에서 무엇이 달라지나 (부분 코드)

MNIST 흐름을 그대로 쓰되 **입력 크기만** 바뀝니다.

```python
# [부분 코드] CIFAR-10 로드와 MLP 입력 맞추기 — 전체 흐름은 4.5의 MNIST와 동일
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train: (50000, 32, 32, 3) uint8 / y_train: (50000, 1) 정수 0~9

X_train = X_train.reshape(-1, 32*32*3).astype("float32") / 255.0   # (50000, 3072)
X_test  = X_test.reshape(-1, 32*32*3).astype("float32") / 255.0
y_train = to_categorical(y_train, 10).astype("float32")
y_test  = to_categorical(y_test, 10).astype("float32")
# 이후 Dense(...) 모델의 첫 Input(shape=(3072,))만 맞춰 주면 됩니다.
```

> 🟩 **(보충)** 6.1에서 본 대로 이미지를 1차원으로 펼치면 2차원 이웃 구조가 입력에 명시되지 않습니다. CIFAR-10 같은 컬러 사진은 보통 **CNN**이 더 적합하므로, 위 MLP로 돌리면 MNIST만큼 높은 정확도는 기대하기 어렵습니다 — 정확한 수치는 직접 실행해 비교해 보세요.

---

## 7. 다차원 배열의 계산 = 행렬 (NumPy) 🟦 + 🟩

딥러닝의 많은 핵심 계산, 특히 `Dense` 층의 **가중합(입력 784개 → 512개 변환 등)** 은 **행렬 연산**으로 표현됩니다. PDF 4장은 그 바탕인 행렬을 정리합니다. 여기서는 **NumPy로 직접 계산**해 봅니다 (아래 출력은 모두 **실제 실행으로 확인**한 값이며, 코드 블록들은 위에서부터 이어서 실행합니다 — 첫 블록의 `import numpy as np`를 재사용).

### 7.1 행렬과 shape

$m$개의 행과 $n$개의 열을 가진 행렬 $A$를 다음처럼 씁니다.

$$
A=[a_{ij}]\in\mathbb{R}^{m\times n}
$$

$a_{ij}$ 는 $i$번째 행, $j$번째 열의 **성분(Entry/Element)** 입니다. 두 행렬은 **shape이 같고 모든 대응 성분이 같을 때** 같은 행렬입니다. 자주 쓰는 종류로 **행벡터(1×m)·열벡터(n×1)**, **정사각행렬(n×n)**, **대각행렬**(대각선 외 0), **단위행렬 $I$**(대각선 1)가 있습니다.

### 7.2 덧셈·스칼라곱·영행렬·전치

같은 shape의 행렬 $A=[a_{ij}]$, $B=[b_{ij}]$ 와 스칼라 $c$ 에 대해:

$$
A+B=[a_{ij}+b_{ij}],\qquad cA=[ca_{ij}]
$$

모든 성분이 0인 **영행렬 $O$** 는 $A+O=A,\; A+(-A)=O$ 를 만족합니다. **전치 $A^{\mathsf{T}}$** 는 행과 열을 뒤바꾼 행렬입니다. PDF의 예제 행렬로 확인해 봅시다.

```python
import numpy as np

A = np.array([[1, -3, 0], [7, 5, 9]])
B = np.array([[0, 1, 2], [4, 0, 8]])

print(A + B)     # [[ 1 -2  2]  [11  5 17]]
print(2 * A)     # [[ 2 -6  0]  [14 10 18]]   (스칼라곱)
print(A - B)     # [[ 1 -4 -2]  [ 3  5  1]]
print(A.T)       # [[ 1  7] [-3  5] [ 0  9]]  (전치)
```

### 7.3 행렬 곱 — 딥러닝 계산의 핵심

$A\in\mathbb{R}^{m\times n}$, $B\in\mathbb{R}^{n\times r}$ 처럼 **앞 행렬의 열 수 = 뒤 행렬의 행 수**가 맞을 때만 곱할 수 있고, 결과 shape은 $(m,r)$ 입니다.

$$
C=AB\in\mathbb{R}^{m\times r},\qquad c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}
$$

즉 결과의 $c_{ij}$ 는 **$A$의 $i$번째 행과 $B$의 $j$번째 열을 내적**한 값입니다. NumPy에서는 `@` 를 씁니다.

```python
A2 = np.array([[1, -3, 0], [7, 5, 9]])   # (2, 3)
B2 = np.array([[0, 4], [1, 0], [2, 8]])  # (3, 2)
print(A2 @ B2)   # [[ -3   4]  [ 23 100]]   → 결과 shape (2, 2)
```

행렬 곱은 일반적으로 **교환법칙이 성립하지 않습니다**($AB \neq BA$). 하지만 shape이 맞으면 결합·분배법칙과 단위행렬 성질이 성립합니다.

$$
A(BC)=(AB)C,\qquad A(B+C)=AB+AC,\qquad I_mA=A=AI_n
$$

> 🟩 **(보충) Dense 층과의 연결** — DAY2에서 본 `Z = X @ W + b` 가 바로 이 행렬 곱입니다.
> $$
> X\in\mathbb{R}^{B\times 784},\quad W\in\mathbb{R}^{784\times 512},\quad b\in\mathbb{R}^{512}\;\Rightarrow\; Z=XW+b\in\mathbb{R}^{B\times 512}
> $$
> 여기서 $B$는 batch size이고, $b$는 각 샘플에 broadcasting됩니다. 입력 `(batch, 784)` 에 가중치 `(784, 512)` 를 곱해 `(batch, 512)` 가 나오는 것도 "앞의 열 수(784) = 뒤의 행 수(784)" 규칙 덕분입니다.

**거듭제곱**(정사각행렬):

```python
P = np.array([[6, 7], [0, 1]])
print(P @ P)     # [[36 49] [0 1]] = P²
```

### 7.4 내적과 전치의 성질

두 열벡터 $u,v\in\mathbb{R}^{n}$ 의 **내적(점곱)** 은 다음과 같고, 전치를 이용해 $u^{\mathsf{T}}v$ 로 씁니다.

$$
u^{\mathsf{T}}v=\sum_{k=1}^{n}u_kv_k
$$

```python
u = np.array([1, 2, 3]); v = np.array([4, 5, 6])
print(u @ v)     # 32   →  1*4 + 2*5 + 3*6
```

전치는 다음 성질을 가집니다. 특히 마지막 식에서 **곱의 순서가 반대로 바뀐다**는 점이 신경망 수식 이해에 자주 쓰입니다.

$$
(A^{\mathsf{T}})^{\mathsf{T}}=A,\qquad (A+B)^{\mathsf{T}}=A^{\mathsf{T}}+B^{\mathsf{T}},\qquad (AB)^{\mathsf{T}}=B^{\mathsf{T}}A^{\mathsf{T}}
$$

### 7.5 분할(블록) 행렬 🟦

큰 행렬에 가로·세로 선을 그어 **여러 블록으로 나눠** 계산할 수 있습니다. 각 블록의 덧셈·곱셈이 정의되도록 내부 shape만 맞으면, **블록을 숫자처럼 취급**해 계산합니다.

$$
X=\begin{bmatrix}A&B\\C&D\end{bmatrix},\quad
Y=\begin{bmatrix}E&F\\G&H\end{bmatrix}
\;\Rightarrow\;
X+Y=\begin{bmatrix}A+E&B+F\\C+G&D+H\end{bmatrix},\quad
XY=\begin{bmatrix}AE+BG&AF+BH\\CE+DG&CF+DH\end{bmatrix}
$$

PDF는 행렬 곱을 보는 여러 관점도 소개합니다. 예를 들어 $B=[\,b_1\ b_2\ \cdots\ b_r\,]$ 처럼 열로 나누면 $AB=[\,Ab_1\ Ab_2\ \cdots\ Ab_r\,]$ 이고, 열벡터·행벡터의 **외적(outer product)들의 합**으로도 볼 수 있습니다.

$$
AB=\sum_{k=1}^{n}a_k\,b_k^{\mathsf{T}}
$$

> 🟩 **(보충)** 입문 단계에서는 "큰 행렬 계산을 블록 단위로 쪼개 볼 수 있다"는 직관만 알아도 충분합니다. 실제 코드에서는 NumPy가 알아서 처리하므로 블록 분할을 손으로 할 일은 드뭅니다.

### 7.6 역행렬과 연립방정식

**정사각행렬** $A\in\mathbb{R}^{n\times n}$ 에 대해 $AA^{-1}=A^{-1}A=I_n$ 을 만족하는 $A^{-1}$ 이 존재하면 $A$ 를 **가역(invertible)** 이라 하고, **그 역행렬은 유일**합니다. 2×2 행렬은 다음 공식을 씁니다($\det(A)\neq 0$ 일 때만 존재).

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix},\quad
\det(A)=ad-bc,\quad
A^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}
$$

가역행렬의 곱은 $(AB)^{-1}=B^{-1}A^{-1}$ 성질을 가집니다(전치처럼 순서가 바뀝니다).

```python
M = np.array([[1., 2.], [3., 5.]])
print(round(np.linalg.det(M)))   # -1
print(np.linalg.inv(M))          # [[-5.  2.]  [ 3. -1.]]
```

선형연립방정식은 **행렬방정식 $Ax=b$** 로 바꿀 수 있고, $A$ 가 가역이면 $x = A^{-1}b$ 로 풉니다. 예: $x+2y=1,\; 3x+5y=4$.

```python
A = np.array([[1., 2.], [3., 5.]])
b = np.array([1., 4.])

# 방법1: 역행렬 (PDF 방식)
print(np.linalg.inv(A) @ b)   # [ 3. -1.]
# 방법2: 전용 solver (실무 권장 — 더 빠르고 수치적으로 안정적)
print(np.linalg.solve(A, b))  # [ 3. -1.]
```

→ $x=3,\; y=-1$.

> 🟩 **(보충)** PDF는 개념 학습을 위해 **역행렬로 직접 푸는 방법**을 보여 줍니다. 다만 실무에서는 역행렬을 명시적으로 구하기보다 **`np.linalg.solve`** 를 쓰는 것이 더 빠르고 안정적입니다.

### 7.7 영공간(Null Space) — 맛보기 🟦

$A\in\mathbb{R}^{m\times n}$ 에 대해 $Ax=0$ 을 만족하는 모든 $x$ 의 집합을 **영공간**이라고 하며, 이는 $\mathbb{R}^{n}$ 의 부분공간입니다.

$$
\operatorname{Null}(A)=\left\{\,x\in\mathbb{R}^{n}\mid Ax=0\,\right\}\subseteq\mathbb{R}^{n}
$$

예를 들어 $A=\begin{bmatrix}1&2\\-2&-4\end{bmatrix}$ 이면 두 행이 비례해 $x_1+2x_2=0$ 이고, 해가 직선을 이룹니다.

$$
\operatorname{Null}(A)=\left\{\begin{bmatrix}-2t\\ t\end{bmatrix}\;\middle|\; t\in\mathbb{R}\right\}
$$

> 🟩 **(보충)** 영공간은 선형대수의 중요한 개념이지만 딥러닝 입문 단계에서 직접 다룰 일은 적습니다. "행렬을 0으로 만드는 입력들의 모임"이라는 개념만 알아 두세요.

---

## 8. 자주 하는 실수

```text
❌ 모든 데이터를 학습에만 사용          → train/validation/test로 분리, test는 마지막 1회
❌ test로 오버피팅 진단·설정 선택        → 진단·비교는 validation, test는 최종 평가 전용
❌ 2차원 이미지를 (784,) 입력에 그대로   → Flatten으로 1차원(784/3072) 벡터로 펼치기
❌ 픽셀 0~255를 그대로 학습            → ÷255로 0~1 정규화(고정 상수라 누수 아님)
❌ One-Hot이 출력 10개라서 필수라 오해   → 손실함수에 맞춤(One-Hot→categorical, 정수→sparse_categorical)
❌ 출력층 활성화·손실 짝 안 맞추기       → 다중=Softmax+(sparse_)categorical, 이진=Sigmoid+binary
❌ Softmax 값을 보정된 진짜 확률로 신뢰   → 합=1인 상대 점수, 가장 큰 값을 예측으로 선택
❌ GPU 하드웨어만 있으면 자동 사용된다 가정 → 드라이버·TF 구성 필요, 최신 Windows는 WSL2 확인
❌ to_categorical 결과가 float32라 단정   → 기본은 float64, 필요하면 .astype("float32")
❌ 컬러 이미지에 MLP가 최선이라 생각      → 2차원 구조 미보존, 이미지는 보통 CNN이 더 적합
❌ 연립방정식을 늘 역행렬로 계산         → 실무는 np.linalg.solve 권장
```

---

## 9. DAY3 핵심 정리

```text
데이터 분할
  - train(학습) / validation(점검·설정선택) / test(최종 1회)
  - 언더·오버피팅은 train·validation 추세로 진단 (test 사용 X)

MNIST 전처리 shape
  - 입력 (batch, 784) float32 0~1  /  정답 (batch, 10) One-Hot
  - 구조 784→512→256→128→10 (예시일 뿐 정답 아님), 은닉 ReLU·출력 Softmax

출력층·라벨·손실 조합
  - 다중분류: Softmax + (One-Hot→categorical / 정수→sparse_categorical)
  - 이진분류: Sigmoid + binary_crossentropy

CIFAR-10 vs MNIST
  - 32×32×3=3072 컬러·실사진이라 더 어려움 / 이미지는 보통 CNN이 적합

Dense와 행렬 곱의 연결
  - Z = X @ W + b 가 곧 행렬 곱 (앞 열 = 뒤 행, 결과 (batch, out))
  - (AB)ᵀ=BᵀAᵀ, 2×2 역행렬·det, AX=B→x=A⁻¹b (실무는 np.linalg.solve)
```

---

## 🔗 참고 자료

- 강의 PDF: `DAY3_딥러닝_데이터처리와데이터셋.pdf` (본문의 1차 출처)
- [TensorFlow 설치와 플랫폼별 GPU 조건](https://www.tensorflow.org/install/pip)
- [Keras 공식 문서 — Sequential 모델](https://keras.io/guides/sequential_model/)
- [Keras `Model.fit` API (`validation_split` 동작)](https://keras.io/api/models/model_training_apis/)
- [Keras MNIST 데이터셋](https://keras.io/api/datasets/mnist/)
- [Keras CIFAR-10 데이터셋 (알려진 라벨 노이즈 안내)](https://keras.io/api/datasets/cifar10/)
- [CIFAR-10 / CIFAR-100 공식 페이지(클래스 목록·출처)](https://www.cs.toronto.edu/~kriz/cifar.html)
- [NumPy 선형대수 (`numpy.linalg`)](https://numpy.org/doc/stable/reference/routines.linalg.html)
