# 🧠 딥러닝 완전 입문 가이드 — DAY5. 딥러닝 모델 훈련 및 평가

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: DAY1(개념·설치) · DAY2(신경망 알고리즘·계층구조) · DAY3(데이터 처리·데이터셋) · DAY4(네트워크 모델 설계)
> **프레임워크 안내**: 이번 글의 **완성형 MNIST 예제만 Keras(`tf.keras`)** 입니다. 이번 DAY의 강의 PDF가 TensorFlow/Keras를 소개하기 때문입니다. (DAY2·DAY4는 PyTorch였습니다.) 합성곱·경사하강법 **직관 실습은 NumPy**로 작성했고, 그 출력값은 **직접 실행해 확인한 값**입니다.

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 내용
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 사전지식·실무 주의점
> - **NumPy 코드(경사하강법·합성곱)** 는 실제 실행해 확인한 값입니다. **Keras MNIST 코드는 이 글을 작성한 환경에 TensorFlow가 설치되지 않아 실행하지 못했습니다.** 정확도 수치를 임의로 적지 않았으니 직접 실행해 확인하세요.

---

## 1. 이번 DAY에서 배우는 것

지금까지는 신경망을 "어떻게 설계하는가"를 봤습니다. 이번 DAY는 설계한 모델을 **실제로 학습시키고(훈련), 잘 됐는지 확인하는(평가)** 전체 과정을 다룹니다.

- 딥러닝이 학습하는 한 사이클: **순전파 → 손실 계산 → 역전파 → 가중치 갱신**
- 학습의 수학적 심장인 **경사하강법(Gradient Descent)** 과 **학습률(Learning Rate)** — NumPy로 직접 확인
- 모델을 평가하는 법: **Train/Validation/Test 분할**, 분류 지표(정확도·정밀도·재현율·F1), 회귀 지표(MAE·RMSE·R²), **과적합/과소적합**
- 딥러닝 3대 프레임워크(PDF 기준)와 **Keras MNIST 첫 코드**
- **CNN(합성곱 신경망)**: 합성곱이 무엇이고(이동평균에서 출발), 영상에서 어떻게 동작하며, 출력 크기·Padding·Stride·Pooling·Bias·배치 정규화까지

> 🟩 이번 PDF는 범위가 넓습니다. **"학습 = 손실을 줄이는 반복"** 이라는 큰 줄기를 먼저 잡고, CNN은 "이미지를 위한 특별한 층"으로 이해하는 데 집중하세요.

---

## 2. 딥러닝 학습의 큰 그림 🟦

딥러닝 모델 훈련(Training)이란 **신경망이 데이터로부터 규칙과 특징을 스스로 학습하는 과정**입니다. 예를 들어 고양이·강아지 사진을 구분하려면, 수많은 이미지와 정답(라벨)을 모델에 주고 각 동물의 특징을 학습시킵니다.

한 번의 학습 사이클은 다음 순서로 진행됩니다.

```text
①데이터 입력 → ②순전파(예측) → ③손실 계산 → ④역전파 → ⑤가중치 갱신 → (다시 ①)
```

### 2.1 순전파(Forward Propagation) — 예측 만들기

입력이 신경망의 각 층을 앞쪽으로 차례로 통과합니다. 각 뉴런은 입력에 가중치를 곱하고 편향을 더한 값 $z$ 를 계산한 뒤, **활성화 함수**를 통과시켜 다음 층으로 보냅니다. 🟦

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

대표적인 활성화 함수는 **Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax** 입니다(자세한 비교는 6장에서 다시 봅니다).

모든 층을 통과하면 최종 출력이 나옵니다. 예를 들어 손글씨 숫자(0~9) 분류 모델은 다음처럼 각 숫자일 확률을 출력합니다. 🟦

| 숫자 | 0 | 1 | 2 | 3 | … |
|---|---|---|---|---|---|
| 확률 | 0.01 | 0.02 | 0.03 | **0.91** | … |

이 경우 확률이 가장 높은 **숫자 3** 으로 예측합니다.

### 2.2 손실(Loss) — 얼마나 틀렸는가

예측과 실제 정답의 차이를 수치로 표현한 것이 **손실 함수(Loss Function)** 입니다. 같은 데이터와 같은 손실함수를 기준으로 볼 때 손실이 작을수록 해당 학습 목표에 더 잘 맞춘 것입니다. 다만 **훈련 손실만 작다고 새로운 데이터에서도 좋은 모델은 아닙니다.** 과적합 여부를 확인하려면 validation loss와 실제 평가지표를 함께 봐야 합니다. 🟦

- **회귀 문제 → 평균제곱오차(MSE)**: 실제 값과 예측 값의 차이를 제곱해 평균
  $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
- **분류 문제 → 교차 엔트로피(Cross Entropy)**: 예측 확률과 실제 정답 확률의 차이를 측정

> 🟩 "어떤 문제에 어떤 출력층·라벨·손실을 짝지어야 하는지"는 DAY4에서 자세히 다뤘습니다. 핵심만 다시 적으면: **회귀=MSE/MAE**, **이진분류=이진 교차 엔트로피**, **다중분류=교차 엔트로피**.

### 2.3 역전파(Backpropagation)와 가중치 갱신

손실이 나오면, 모델은 **"각 가중치가 이 오차에 얼마나 기여했는지"** 를 출력층에서 입력층 방향으로(=뒤로) 계산합니다. 이때 미분과 **연쇄법칙(Chain Rule)** 이 쓰입니다. 이렇게 구한 값이 **기울기(Gradient)** 이고, 이 기울기를 이용해 가중치를 조금씩 수정합니다(경사하강법, 3장에서 자세히). 🟦

$$
w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}
$$

여기서 $\eta$(에타)가 **학습률**입니다. 너무 크면 학습이 불안정하고, 너무 작으면 매우 느려집니다.

### 2.4 Epoch, Batch, 그리고 옵티마이저

- **Epoch(에폭)**: 전체 학습 데이터를 한 번 모두 사용한 것. 데이터 10,000개를 모두 쓰면 1 Epoch. 🟦
- **Batch(배치)**: 데이터를 한 번에 다 처리하기 어려우므로 작은 묶음으로 나눠 학습. 데이터 10,000개를 Batch Size 100으로 나누면, **100번 반복하면 1 Epoch** 가 끝납니다. 🟦
- **최적화 알고리즘(Optimizer)**: 가중치를 갱신하는 방법. **SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW** 등이 있고, PDF는 *"현재 실무에서는 Adam·AdamW가 가장 널리 쓰인다"* 고 소개합니다. 🟦

> 🟩 **Epoch / Batch / 갱신 1회(step)** 는 다릅니다. 한 step은 배치 하나로 가중치를 한 번 갱신하는 것이고, 여러 step이 모여 1 Epoch가 됩니다. "배치가 크면 무조건 빠르고 좋다"는 말은 사실이 아닙니다 — 메모리·일반화 성능과 균형을 봐야 합니다.

---

## 3. 경사하강법과 기울기 🟦

> 학습의 수학적 핵심입니다. PDF 2장 전체(p9~16)가 이 한 가지를 설명합니다.

### 3.1 왜 "한 번에" 정답을 못 구할까

딥러닝의 목표는 **예측 오차(Loss)를 최소화하는 가중치**를 찾는 것입니다. 수학적으로는 손실을 가중치로 미분했을 때 0이 되는 지점을 찾으면 됩니다. 하지만 실제 모델은 가중치가 **수백만~수천억 개** 입니다(예: 이미지 분류 CNN, 자연어 Transformer, GPT 같은 대규모 언어모델). 이걸 한 번에 정확히 푸는 것은 사실상 불가능합니다.

그래서 딥러닝은 정답을 한 번에 구하는 대신 **현재 위치에서 조금씩 더 나은 방향으로 반복해서 이동**합니다.

### 3.2 직관: 안개 낀 산에서 내려오기

경사하강법은 *"산에서 가장 낮은 계곡으로 내려가는 것"* 에 비유됩니다. 🟦

> ① 현재 위치 확인 → ② 주변 경사 확인 → ③ 가장 낮아지는 방향 선택 → ④ 조금 이동 → ⑤ 다시 경사 계산 → (반복)

아래 설명은 가중치가 **하나인 단순한 1차원 손실 곡선**의 직관입니다. 실제 신경망에서는 기울기가 0인 지점이 항상 전역 최적점은 아니며, 지역 최솟값·극댓값·안장점일 수도 있습니다. 여러 파라미터가 있을 때는 **gradient vector의 반대 방향**으로 이동합니다. 🟩

1차원 직관에서 **기울기(Gradient)** 의 부호는 이동 방향을 알려줍니다.

- 기울기가 **양수** → 손실이 줄어드는 쪽(값을 낮추는 쪽)으로 이동
- 기울기가 **음수** → 반대 방향으로 이동
- 기울기가 **0 근처** → 해당 축에서는 최적점에 가까운 상태

### 3.3 Python 실습 — 손실 곡선을 직접 내려가 보기 🟩

가장 단순한 손실 $f(w) = (w-3)^2$ 을 생각해 봅시다. 정답은 누가 봐도 $w=3$ 이지만, 모델은 그걸 모른 채 **기울기만으로** 찾아가야 합니다. 미분하면 $f'(w) = 2(w-3)$ 입니다.

```python
# 경사하강법 직관 실습 (NumPy / 프레임워크 무관) — 실제 실행 확인됨
import numpy as np

def f(w):    return (w - 3) ** 2     # 손실 함수
def grad(w): return 2 * (w - 3)      # 기울기(미분)

w  = 0.0      # 시작 가중치
lr = 0.1      # 학습률

print("step    w       loss     grad")
for step in range(6):
    print(f"{step:>3}  {w:7.4f}  {f(w):7.4f}  {grad(w):7.4f}")
    w = w - lr * grad(w)             # 핵심: 기울기 반대 방향으로 이동
```

**실행 결과(이 환경에서 확인):**

```text
step    w       loss     grad
  0   0.0000   9.0000  -6.0000
  1   0.6000   5.7600  -4.8000
  2   1.0800   3.6864  -3.8400
  3   1.4640   2.3593  -3.0720
  4   1.7712   1.5099  -2.4576
  5   2.0170   0.9664  -1.9661
```

**해석**: $w$ 가 0 → 0.6 → 1.08 …처럼 **정답 3을 향해** 이동하고, 그때마다 손실(9 → 5.76 → 3.69 …)이 줄어듭니다. 50번 반복하면 $w \approx 3.0$, 손실 $\approx 0$ 에 도달합니다. 이것이 경사 기반 학습의 핵심 원리입니다 — **기울기를 보고, 손실이 줄어드는 방향으로 조금씩 이동.**

### 3.4 학습률(Learning Rate)이 왜 그렇게 중요한가 🟦

같은 문제를 학습률만 바꿔 50번 반복한 결과입니다.

```python
# 독립 실행 가능 (NumPy 실행 확인됨)
import numpy as np

def f(w):    return (w - 3) ** 2
def grad(w): return 2 * (w - 3)

def run(lr, steps=50):
    w = 0.0
    for _ in range(steps):
        w = w - lr * grad(w)
    return w, f(w)

for lr in [0.01, 0.1, 1.01]:
    w, loss = run(lr)
    print(f"lr={lr:<5}  ->  w={w:8.4f},  loss={loss:.4f}")
```

**실행 결과(이 환경에서 확인):**

```text
lr=0.01   ->  w=  1.9075,  loss=1.1936
lr=0.1    ->  w=  3.0000,  loss=0.0000
lr=1.01   ->  w= -5.0748,  loss=65.2018
```

| 학습률 | 결과 | 의미 |
|---|---|---|
| 0.01 (너무 작음) | $w=1.91$, 아직 3에 못 감 | 학습이 너무 **느림** |
| 0.1 (적절) | $w=3.00$, 손실 0 | 잘 **수렴** |
| 1.01 (너무 큼) | $w=-5.07$, 손실 폭증 | 최적점을 지나쳐 **발산** |

> 🟩 PDF의 *"학습률이 크면 최적점을 지나치고, 작으면 오래 걸린다"* 는 말이 숫자로 그대로 나타났습니다. Adam에서 `0.001`은 자주 쓰이는 출발점 중 하나지만, validation 곡선을 보며 조정해야 합니다.

이 반복 과정(① 입력 → ② 예측 → ③ 손실 → ④ 미분 → ⑤ 기울기 → ⑥ 가중치 수정 → ⑦ 다시 예측)을 수천·수만 번 반복하는 것이 바로 학습입니다. 🟦

---

## 4. 모델 평가 🟦

학습이 끝나면 **학습에 쓰지 않은 데이터**로 실제 성능을 측정해야 합니다.

### 4.1 데이터 분할 — Train / Validation / Test

| 데이터 | 용도 |
|---|---|
| **Train Set** | 모델 학습 |
| **Validation Set** | 하이퍼파라미터 조정 |
| **Test Set** | 최종 성능 평가 |

PDF 예시는 **Train 70% / Validation 15% / Test 15%** 입니다. 🟦

> 🟩 가장 중요한 원칙: **Test Set으로 하이퍼파라미터를 조정하면 안 됩니다.** 조정은 Validation으로 하고, Test는 "마지막에 딱 한 번" 보는 시험지처럼 다뤄야 점수를 신뢰할 수 있습니다. 클래스 비율이 치우쳐 있으면 분할 시 **계층적 분할(stratify)** 을 고려하세요.

### 4.2 분류(Classification) 평가 지표

TP·FP·FN·TN 네 칸에서 출발합니다. (TP: 양성을 양성으로, FP: 음성을 양성으로, FN: 양성을 음성으로, TN: 음성을 음성으로 예측)

| 지표 | 정의(말로) | 식 |
|---|---|---|
| 정확도(Accuracy) | 전체 예측 중 정답 비율 | $\dfrac{TP+TN}{TP+TN+FP+FN}$ |
| 정밀도(Precision) | 양성이라 **예측한 것 중** 실제 양성 | $\dfrac{TP}{TP+FP}$ |
| 재현율(Recall) | **실제 양성 중** 모델이 찾아낸 비율 | $\dfrac{TP}{TP+FN}$ |
| F1 Score | 정밀도·재현율의 조화평균 | $2\cdot\dfrac{P\cdot R}{P+R}$ |

PDF의 핵심 경고: **정확도는 데이터가 불균형하면 신뢰하기 어렵습니다.** 🟦

> 🟩 위 공식은 먼저 **이진 분류** 기준으로 이해하면 쉽습니다. MNIST처럼 클래스가 10개인 다중 분류에서는 클래스마다 지표를 계산한 뒤 macro·micro·weighted 방식 등으로 평균합니다.
>
> - 놓치면 위험한 양성을 최대한 찾는 것이 중요하면 → **Recall**
> - 거짓 경보를 줄이는 것이 중요하면 → **Precision**
> - 두 기준의 균형이 필요하면 → **F1**
>
> MNIST는 비교적 균형 잡힌 다중 분류이므로 첫 예제에서는 accuracy를 기본 지표로 사용해도 적절합니다.

### 4.3 회귀(Regression) 평가 지표 🟦

| 지표 | 의미 |
|---|---|
| **MAE** (평균절대오차) | 오차의 절댓값 평균. 단위가 직관적 |
| **RMSE** (평균제곱근오차) | 오차를 제곱→평균→제곱근. 큰 오차에 더 민감 |
| **결정계수(R²)** | 모델 설명력. 1이 최선이며 단순 기준(평균 예측)보다 좋으면 양수, 못하면 음수도 가능 |

> 🟩 R²는 "1에 가까울수록 좋다"고만 외우면 안 됩니다. test 데이터에서 음수가 나올 수도 있으며, 단위가 있는 MAE·RMSE와 함께 해석해야 합니다.

### 4.4 과적합(Overfitting)과 과소적합(Underfitting) 🟦

- **과적합**: 훈련 성능은 높은데 **새로운 데이터에서 성능이 낮은** 현상. 학습 데이터를 "외워버린" 상태.
- **과소적합**: 모델이 **충분히 학습하지 못한** 상태.

PDF가 제시하는 과적합 완화 기법: **데이터 증강(Data Augmentation), Dropout, Batch Normalization, Early Stopping, 정규화(L1·L2), 더 많은 데이터 확보.** 🟦

> 🟩 어느 기법도 과적합을 *완전히* 없애지는 못합니다. 데이터·모델·설정에 따라 **줄여 줄 수 있다**고 이해하세요. 과적합/과소적합은 **훈련 곡선과 검증 곡선을 함께** 봐야 판단할 수 있습니다. MNIST 예제에서 이 곡선을 직접 확인하는 방법은 5절에서 다룹니다.

---

## 5. 프레임워크와 첫 코드 🟦

### 5.1 딥러닝 3대 프레임워크 (PDF 기준) 🟦

PDF는 **Keras · TensorFlow · PyTorch** 를 *"모두 파이썬 기반"* 으로 소개합니다.

| 프레임워크 | 한 줄 소개 (PDF 기준) |
|---|---|
| **TensorFlow** | 구글 브레인팀 제작. V1 2015 / V2 2019 공개, TPU 연동 |
| **Keras** | François Chollet가 2015 공개. TensorFlow를 쉽게 쓰도록 돕는 고수준 도구 |
| **PyTorch** | 연구·실험에서 인기. (DAY2·DAY4에서 사용) |

> 🟩 **버전 안내**: PDF는 Keras를 TensorFlow 전용 고수준 도구로 설명하는 당시 관점입니다. 현재 `tf.keras`는 TensorFlow에 통합된 API이고, 별도의 Keras 3은 TensorFlow·JAX·PyTorch 백엔드를 모두 지원합니다. 이 글의 코드는 **TensorFlow 백엔드의 `tf.keras`** 를 사용합니다.
>
> TensorFlow 설치 가능 Python 버전과 GPU 설치 방법은 시점에 따라 바뀝니다. 특히 Windows Native의 최신 공식 GPU 지원에는 제한이 있으므로 [TensorFlow 공식 설치 문서](https://www.tensorflow.org/install/pip)에서 현재 환경에 맞는 명령을 확인하세요.

### 5.2 Keras로 MNIST 손글씨 분류 🟦

PDF p20은 텐서플로 공식 튜토리얼의 MNIST 예제를 보여줍니다. MNIST는 0~9 손글씨 숫자 이미지(28×28, 흑백)를 모은 표준 데이터셋입니다.

아래 코드는 PDF의 예제를 현재 TensorFlow 공식 quickstart 권장 방식으로 보완한 것입니다. 🟩

- **변경점 1**: 출력층을 `softmax` 대신 **raw logits**로, 손실을 `SparseCategoricalCrossentropy(from_logits=True)` 로 — DAY4에서 배운 PyTorch의 `CrossEntropyLoss` 와 같은 논리입니다.
- **변경점 2**: `model.fit()`에 `validation_split=0.1` 을 추가해 훈련 중 과적합 여부를 확인합니다.
- **변경점 3**: test 평가는 학습·모델 선택이 끝난 뒤 **한 번만** 수행합니다.

> PDF p20의 원래 코드는 출력층에 `softmax`를 넣고 `sparse_categorical_crossentropy`(문자열)를 쓰는 방식입니다. 그 조합도 서로 호환되어 올바르게 동작하지만, 위 코드는 현재 공식 권장 방식을 따른 보충 예제입니다.

```python
# TensorFlow / Keras MNIST — 실제 실행하지 못했으므로 출력값을 임의로 적지 않음
# 설치: https://www.tensorflow.org/install/pip 에서 현재 명령 확인
import tensorflow as tf

tf.keras.utils.set_random_seed(42)   # 재현성 (하드웨어에 따라 완전히 동일하지 않을 수 있음)

# 1) 데이터 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 고정 상수(255)로 나누는 변환은 train 통계를 학습하지 않으므로 데이터 누수가 없습니다
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# 2) MLP 기준선 — Flatten + Dense (CNN이 아님)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),                       # (28, 28) -> (784,)
    tf.keras.layers.Dense(128, activation="relu"),   # 은닉층
    tf.keras.layers.Dropout(0.2),                    # 과적합 완화
    tf.keras.layers.Dense(10),                       # 출력: raw logits 10개
])

# 정수 라벨 + raw logits 조합
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# 3) 학습 — train의 10%를 validation으로 사용 (test는 여기서 보지 않음)
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=32,
)

# 4) 모든 선택이 끝난 뒤 test를 최종 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_acc:.4f}")
```

> ⚠️ **이 환경에 TensorFlow가 없어 위 코드를 실행하지 못했습니다.** 정확도 수치를 임의로 넣지 않았습니다. 직접 실행하면 각 Epoch마다 `loss`, `accuracy`, `val_loss`, `val_accuracy` 가 출력됩니다.

**shape · dtype · 손실 짝 맞추기** 🟩

| 항목 | shape | dtype / 설명 |
|---|---|---|
| `x_train` | `(60000, 28, 28)` | `float32`, 픽셀 범위 0~1 |
| `y_train` | `(60000,)` | 정수 클래스 인덱스 0~9 |
| `Flatten` 후 | `(batch, 784)` | 이미지 공간 구조를 한 줄로 펼침 |
| 모델 출력 | `(batch, 10)` | softmax 전 raw logits |
| 손실 | `SparseCategoricalCrossentropy(from_logits=True)` | 정수 라벨 + logits 조합 |

TensorFlow는 사용 가능한 CPU/GPU에 연산을 자동 배치합니다. GPU가 없어도 이 MNIST 예제는 CPU로 실행할 수 있습니다. GPU 여부를 확인하려면 `tf.config.list_physical_devices("GPU")` 를 사용하세요. 🟩

> 🟩 **이 예제는 이미지를 1차원으로 펼쳐 Dense 층에 넣는 MLP 기준선입니다.** 합성곱층(Conv2D)을 사용한 CNN은 아닙니다. CNN은 바로 다음 6장에서 설명합니다.

### 5.3 훈련·검증 곡선으로 과적합 확인하기 🟩

`history.history` 에는 Epoch마다의 손실과 정확도가 저장됩니다. 다음 코드로 곡선을 그려 과적합 여부를 확인하세요.

```python
# model.fit() 을 먼저 실행한 뒤 사용하세요
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"],     label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"],     label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```

곡선 해석 방법:

| 곡선 모양 | 판단 |
|---|---|
| train loss ↓, validation loss도 함께 ↓ | 학습이 잘 진행되는 중 |
| train loss ↓, validation loss ↑ (두 곡선이 벌어짐) | **과적합** 신호 |
| 두 곡선 모두 높고 거의 줄지 않음 | **과소적합** 또는 학습 설정 문제 |

> 🟩 Early Stopping은 validation loss가 더 이상 개선되지 않을 때 학습을 조기에 멈춰 과적합을 줄이는 기법입니다. PDF p8에서 소개한 기법이며, 다음처럼 `fit()`에 추가합니다.
>
> ```python
> early_stop = tf.keras.callbacks.EarlyStopping(
>     monitor="val_loss",
>     patience=2,
>     restore_best_weights=True,
> )
> history = model.fit(..., callbacks=[early_stop])
> ```
>
> `monitor`는 test loss가 아닌 **val_loss** 를 사용합니다. test 정보가 모델 선택에 새어 들어가지 않도록 하기 위해서입니다.

---

## 6. CNN — 합성곱 신경망 🟦

PDF 4장 전체(p21~58)가 CNN입니다. CNN은 **영상(이미지) 처리에 주로 쓰는** 신경망으로, 핵심은 **합성곱(Convolution) 연산** 입니다.

```text
영상 → [합성곱층 1] → [합성곱층 2] → … → [합성곱층 N] → [완전연결층(FC) 1~3개] → 출력
```

### 6.1 합성곱은 "이동 평균"에서 출발한다 🟦

PDF는 합성곱을 **이동 평균(Moving Average)** 으로 설명합니다. 데이터를 따라가며 이웃한 값들의 (가중)평균을 구해 급격한 변화를 평탄하게 만드는 것입니다.

```python
# 합성곱 직관 실습 (NumPy) — 실제 실행 확인됨
import numpy as np

x = np.array([10, 20, 50, 30, 10, 20, 60, 10], dtype=float)

# (1) 단순 이동 평균: 이웃 3개의 평균 (커널 = [1/3, 1/3, 1/3])
simple = np.convolve(x, np.ones(3) / 3, mode="valid")
print("단순 이동평균 :", np.round(simple, 2))

# (2) 가중 이동 평균: 가운데에 더 큰 가중치 (커널 = [1/4, 1/2, 1/4])
y = np.array([0.25*x[i-1] + 0.5*x[i] + 0.25*x[i+1] for i in range(1, len(x)-1)])
print("가중 이동평균 :", y)
```

**실행 결과(이 환경에서 확인):**

```text
단순 이동평균 : [26.67 33.33 30.   20.   30.   30.  ]
가중 이동평균 : [25.  37.5 30.  17.5 27.5 37.5]
```

가중 이동평균 결과 `[25, 37.5, 30, 17.5, 27.5, 37.5]` 는 **PDF p22의 값과 정확히 일치**합니다. 여기서 곱해지는 수들의 묶음 `[1/4, 1/2, 1/4]` 을 **커널(Kernel)**, 그 안의 값들을 **계수(Weight)** 라고 부릅니다. 🟦

1차원 합성곱의 일반식은 다음과 같습니다. 🟦

$$
y[n] = \sum_{k=-\frac{K-1}{2}}^{\frac{K-1}{2}} h[k]\, x[n+k]
\qquad (h: \text{커널})
$$

> 🟩 엄밀히 말하면 NumPy의 `np.convolve`는 커널을 뒤집어 계산하는 수학적 convolution입니다. 반면 딥러닝 프레임워크의 `Conv2D`는 커널을 뒤집지 않는 **cross-correlation** 방식으로 구현되지만, 학습 과정에서 필터 값 자체를 배우므로 관습적으로 convolution이라고 부릅니다. 이 예제의 커널 `[1/4, 1/2, 1/4]`과 `[-1, 2, -1]`은 대칭이라 두 방식의 결과가 같습니다.

### 6.2 필터링 — 저역통과(LPF)와 고역통과(HPF) 🟦

합성곱은 신호 처리에서 **필터링**에 많이 쓰입니다.

- **저역통과 필터(LPF)**: 천천히 변하는(낮은 주파수) 성분만 통과 → 영상이 **부드러워짐(blur)**. 위의 `[1/4, 1/2, 1/4]` 가 LPF입니다.
- **고역통과 필터(HPF)**: 급격히 변하는(높은 주파수) 성분만 통과 → **경계(엣지)** 만 남음. PDF 예시 커널은 `[-1, 2, -1]`. 🟦

```python
# HPF 직관 실습 (NumPy) — 실제 실행 확인됨
import numpy as np

# 평평하다가 계단처럼 올라갔다 내려오는 신호
x = np.array([0, 0, 10, 10, 10, 10, 10, 10, 0, 0], dtype=float)
y = np.array([-1*x[i-1] + 2*x[i] - 1*x[i+1] for i in range(1, len(x)-1)])
print("HPF [-1,2,-1] :", y)
```

**실행 결과(이 환경에서 확인):**

```text
HPF [-1,2,-1] : [-10.  10.   0.   0.   0.   0.  10. -10.]
```

**해석**: 값이 **변하는 경계** 에서만 ±10이 나오고, **평평한 구간은 모두 0** 입니다. PDF p25의 *"높은 주파수 성분이 없음 → 경계만 남음"* 이 숫자로 확인됩니다.

> 🟩 단순한 이미지 blur는 저역통과 필터의 대표적인 예입니다. Prewitt·Sobel 같은 엣지 검출 커널도 같은 원리의 고역통과 필터입니다. CNN이 학습을 통해 찾아내는 필터가 바로 이런 구조입니다.

### 6.3 영상에서의 합성곱 🟦

- **영상은 픽셀(pixel)** 이라는 아주 작은 점들로 이루어집니다. 흑백 영상은 픽셀마다 밝기 숫자를 가집니다. 🟦
- **컬러 영상**은 한 픽셀을 **R·G·B 세 값**으로 표현합니다. 🟦
- **해상도(Resolution)** 는 사진의 픽셀 개수입니다(예: Full HD = 1920×1080). 🟦

영상에서의 합성곱은 1차원이 2차원으로 확장된 것입니다. **$K \times K$ 크기의 커널을 이미지 위에서 한 칸씩 미끄러뜨리며** 곱하고 더해 새 값을 만듭니다. 평균 필터는 흐림(blur), Prewitt·Sobel 커널은 경계 검출(edge)에 쓰입니다. 🟦

### 6.4 출력 크기, Padding, Stride 🟦

합성곱을 하면 **출력이 입력보다 작아집니다.** N개 데이터에 크기 K인 커널을 적용하면 출력은 **N − K + 1개**. (예: 7개 → 5개) 🟦

아래 식은 **dilation=1** 인 경우의 한 축 출력 크기이며, 높이와 너비는 각각 따로 계산합니다. 🟩

$$
R = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1
$$

| 설정 | 조건 | 효과 |
|---|---|---|
| valid (`P=0, S=1`) | — | 가장자리 $K-1$ 만큼 줄어듦 |
| same (`P=(K−1)/2`) | stride 1, dilation 1, 홀수 커널 | 입력과 **같은 크기** 유지 |
| stride S | S칸씩 건너뜀 | 대략 $1/S$ 크기로 줄어듦. 정확한 값은 floor와 입력·커널·패딩 조합에 따라 달라짐 |

PDF가 소개하는 Padding 방식: **Padding(0으로 채움) · Copying(가장자리 값 복사) · Warping(반대쪽 값 순환)**. 🟦  
마지막 방식의 일반적인 용어는 **Wrapping 또는 circular padding** 입니다. 🟩

```python
# 출력 크기 공식 확인 (NumPy 실행 확인됨, dilation=1 가정)
def out_size(H, K, P=0, S=1):
    return (H - K + 2*P) // S + 1

print("valid  N=7,  K=3        :", out_size(7, 3))        # 5
print("same   H=10, K=3, P=1   :", out_size(10, 3, P=1))  # 10
print("stride H=10, K=2, S=2   :", out_size(10, 2, S=2))  # 5
```

**실행 결과(이 환경에서 확인):**

```text
valid  N=7,  K=3        : 5
same   H=10, K=3, P=1   : 10
stride H=10, K=2, S=2   : 5
```

### 6.5 CNN의 구조 — 합성곱층, 채널, 연산량 🟦

- **첫 번째 합성곱층**: 입력(예: RGB 3채널)에 여러 개의 커널을 적용합니다. **커널 1개 = 한 종류의 특징 검출기**. 커널이 너무 적으면 특징을 충분히 잡지 못합니다. 🟦
- **N개의 커널 → N개의 출력 Feature Map(=채널 N개).** 🟦
- **다음 층의 커널**은 깊이가 입력 채널 수와 같습니다. 3×3 커널이라도 입력 채널이 N이면 실제로는 **3×3×N** 짜리 3차원 커널입니다. 🟦

#### 곱셈 횟수 🟦

PDF는 한 합성곱층의 곱셈 횟수를 다음과 같이 제시합니다.

$$
\text{곱셈 횟수} = M \times R \times C \times N \times K \times K
$$

($M$: 출력 채널, $R \times C$: 출력 크기, $N$: 입력 채널, $K \times K$: 커널 크기)

```python
# PDF p44 예시 — 실제 실행 확인됨
M = N = 100; R = C = 200; K = 3
print(f"{M*R*C*N*K*K:,}")   # 3,600,000,000
```

**실행 결과**: `3,600,000,000` (= **3.6 × 10⁹**, PDF와 일치). 이 값은 **샘플 하나가 한 합성곱층을 통과할 때** 필요한 곱셈을 단순화한 수치입니다(배치 크기, bias 덧셈, 활성화 비용 등은 미포함). CPU로도 계산할 수 있지만 연산량이 커질수록 **GPU·TPU 같은 가속기**가 실용적인 학습 시간을 만드는 데 유리합니다. 🟩

- **완전 연결층(Fully-Connected Layer, FC)**: 합성곱층들을 지난 뒤 **3차원 Feature Map을 1차원 벡터로 펴서** 일반 신경망층(보통 1~3개)에 넣어 최종 출력을 만듭니다. 🟦

> 🟩 5절의 MNIST 예제는 바로 이 구조(Flatten + Dense)만 사용한 **MLP 기준선**입니다. 실제 이미지 과제에서는 앞에 합성곱층을 쌓은 CNN이 공간 구조를 더 잘 활용합니다.

### 6.6 활성화 함수 — 과거 vs 현재 🟦

활성화 함수는 합성곱층·FC층 사이에 들어가 비선형성을 줍니다.

| 시기 | 함수 | 특징 |
|---|---|---|
| 과거 | **Sigmoid** $\dfrac{1}{1+e^{-x}}$, **Tanh** | 양 끝에서 미분값이 매우 작아 **기울기 소실** 발생 |
| 현재 | **ReLU** $\max(0,x)$ | 음수는 0, 양수는 그대로. 간단하고 빠름 |
| 현재 | **Leaky ReLU** $\max(0.1x,\,x)$ | 음수도 작은 값으로 통과(ReLU의 dead 뉴런 위험 완화) |

> 🟩 ReLU가 기울기 소실 문제를 *완전히* 없애는 것은 아닙니다. 음수 입력이 계속되면 뉴런이 죽는 dead ReLU 위험이 있습니다. **출력층** 의 활성화는 문제 유형에 맞춥니다(다중분류=softmax, 또는 logits 출력 후 손실에서 처리).

### 6.7 Pooling 층, Bias, 배치 정규화 🟦

**Pooling 층**: 각 Feature Map(채널)마다 **MAX 또는 AVG** 연산을 적용해 크기를 줄입니다. 합성곱이 채널 방향으로 합산하는 것과 달리, Pooling은 **채널마다 따로** 수행합니다. 🟦

**Bias**: 한 출력 Feature Map의 모든 위치에 **같은 편향 $b$** 를 더합니다(채널마다 하나). 🟦

**배치 정규화(Batch Normalization)** 🟦
- **문제(공변량 쉬프트)**: 학습이 진행되면 각 층의 입력 분포가 계속 바뀝니다. 층이 깊을수록 심해집니다.
- **해결**: 미니배치 단위로 평균 $\mu_B$, 표준편차 $\sigma_B$ 를 구해 **평균 0, 표준편차 1** 로 정규화한 뒤, 학습 가능한 **Scale($\gamma$)·Shift($\beta$)** 를 되돌립니다: $y_i = \gamma \hat{x}_i + \beta$. 보통 **활성화 함수 직전**에 넣습니다.
- **PDF가 제시하는 효과**: 학습률을 크게 해도 학습이 안정적으로 진행되고, 과적합이 줄며, 계수 초기화에 덜 민감해집니다. 🟦

> 🟩 이 효과들은 PDF가 소개하는 동기이며, 실제 효과는 모델 구조·배치 크기·데이터에 따라 달라집니다. Dropout을 항상 대체하거나 큰 학습률을 무조건 허용하는 것은 아닙니다. 프레임워크에서는 `BatchNormalization()`(Keras) / `BatchNorm1d·BatchNorm2d`(PyTorch) 한 줄로 추가합니다.

---

## 7. 자주 하는 실수 🟩

- **훈련 손실만 보고 좋아하기**: 새로운 데이터에서의 성능은 validation/test 지표로 확인해야 합니다.
- **Test Set으로 튜닝하기**: 하이퍼파라미터는 validation을 기준으로 조정하고, test는 마지막 한 번만.
- **학습률을 아무렇게나 설정**: 너무 크면 발산(위 실습의 `lr=1.01`), 너무 작으면 학습이 진행되지 않습니다.
- **출력층·라벨·손실 짝 안 맞추기**: softmax 출력엔 `from_logits=False`, raw logits엔 `from_logits=True`. 정수 라벨엔 `sparse_`, 원-핫엔 `Categorical`. (DAY4 참고)
- **합성곱 출력 크기 착각**: padding이 없으면 매 층 크기가 $K-1$ 만큼 줄어듭니다. same padding은 stride 1, 홀수 커널일 때 크기를 유지합니다.
- **MNIST Flatten 예제를 CNN으로 오해**: 이미지를 1차원으로 펼쳐 Dense에 넣는 것은 MLP입니다. CNN은 `Conv2D` 층을 씁니다.
- **프레임워크 섞기**: 한 예제 안에서 Keras와 PyTorch API를 섞지 마세요.

---

## 8. DAY5 핵심 정리

```text
딥러닝 학습의 핵심
  ├─ 한 사이클: 순전파 → 손실 → 역전파 → 가중치 갱신
  ├─ 경사하강법: gradient 반대 방향으로 조금씩 이동 (학습률이 핵심)
  │    1D 볼록 함수의 직관; 실제 신경망에선 지역 최솟값·안장점도 존재
  └─ 단위: step(배치 1회) → Epoch(전체 1회)

평가
  ├─ 분할: Train(학습) / Validation(튜닝) / Test(최종, 1회)
  ├─ 분류: Accuracy·Precision·Recall·F1 (불균형이면 Accuracy만 믿지 말 것)
  ├─ 회귀: MAE·RMSE·R² (R²는 음수도 가능, 단위 있는 지표와 함께 해석)
  └─ 과적합 확인: 훈련·검증 곡선을 함께 보기

Keras MNIST (MLP 기준선, CNN 아님)
  ├─ logits 출력 + SparseCategoricalCrossentropy(from_logits=True)
  ├─ validation_split으로 훈련 중 과적합 모니터링
  └─ test는 모든 선택이 끝난 뒤 1회 평가

CNN (이미지를 위한 신경망)
  ├─ 합성곱: 커널을 미끄러뜨리며 곱·합 (이동평균·필터에서 출발)
  ├─ 출력 크기: R = ⌊(H+2P−K)/S⌋+1  (dilation=1 가정)
  ├─ 채널: 커널 N개 → 출력 채널 N개, 다음 층 커널 깊이 = 입력 채널
  ├─ FC층: 3D feature map → 1D vector → 최종 출력
  ├─ 활성화: 은닉층 ReLU/Leaky ReLU (Sigmoid·Tanh는 기울기 소실 위험)
  └─ 안정화: Pooling(채널별 MAX/AVG), Bias(채널당 1개), 배치 정규화
```

---

## 참고 자료

- 강의 PDF: **DAY5. 딥러닝 모델 훈련 및 평가** (`sources/DAY5_딥러닝_모델훈련및평가.pdf`), 전 58페이지
- TensorFlow beginner quickstart (MNIST): <https://www.tensorflow.org/tutorials/quickstart/beginner>
- TensorFlow 설치: <https://www.tensorflow.org/install/pip>
- Keras 3 소개: <https://keras.io/keras_3/>
- 이전 편: DAY4 딥러닝 네트워크 모델 설계(출력층·라벨·손실 짝 맞추기), DAY2 신경망 알고리즘·계층 구조
