# 🧠 딥러닝 완전 입문 가이드 — DAY1. 딥러닝 개념과 프레임워크 설치

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편(머신러닝)**: DAY1~7 — 핵심 개념 · 회귀/분류 · SVM/KNN · 결정트리 · 앙상블/신경망 · 차원 축소 · 비지도학습
> **딥러닝 DAY1**: 딥러닝의 개념과 프레임워크 설치 ← 딥러닝 편의 첫 글

> 📝 이 글부터 **딥러닝 편을 DAY1로 새로 시작**합니다.
> 다만 내용은 **머신러닝 편(DAY1~7)에서 다룬 지도학습 · 정규화 · 평가지표를 그대로 활용**하므로, 머신러닝 편을 먼저 보고 오시길 권합니다.

---

## 1. 이번 DAY에서 배우는 것

이번 글은 **딥러닝을 처음 시작할 때 꼭 알아야 할 개념과 환경 준비**를 다룹니다. 모델을 깊게 파고들기보다는 "딥러닝이 무엇이고, 왜 지금 가능해졌으며, 무엇을 깔아야 시작할 수 있는가"에 초점을 맞춥니다.

- 인공지능 · 머신러닝 · 딥러닝의 관계
- 신경망의 역사: 퍼셉트론 → XOR 한계 → 은닉층 → 딥러닝
- 딥러닝이 가능해진 이유와 대표 활용 분야
- 프레임워크(TensorFlow · Keras · PyTorch)와 GPU · CUDA · cuDNN
- DNN으로 회귀(집값)와 분류(심장병)를 푸는 **전체 흐름**

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 내용
> - 🟩 **(보충)** : 입문자를 위해 글쓴이가 덧붙인 사전지식 · 실무 주의점
> 강의 자료에는 **개념 그림과 데이터 설명**은 있지만 **실행 코드 전체는 실려 있지 않습니다.** 아래 실습 코드는 강의 흐름을 따라 **재구성한 실행 가능 예제**이며, 그 점을 분명히 표시했습니다.

---

## 2. 인공지능 · 머신러닝 · 딥러닝 🟦

세 단어는 자주 섞여 쓰이지만 **포함 관계**입니다.

```
┌─────────────────────────────────────────────┐
│ 인공지능 (AI)                                  │
│  사람의 인지 기능(학습·추론·문제해결·의사결정·     │
│  언어이해·시각인지)을 컴퓨터로 흉내내는 분야 전체   │
│  ┌───────────────────────────────────────┐   │
│  │ 머신러닝 (ML)                            │   │
│  │  규칙을 사람이 일일이 짜는 대신,           │   │
│  │  데이터에서 패턴을 '학습'해 새 값을 예측    │   │
│  │  ┌─────────────────────────────────┐   │   │
│  │  │ 딥러닝 (DL)                        │   │   │
│  │  │  인공신경망(은닉층 多)으로 학습      │   │   │
│  │  └─────────────────────────────────┘   │   │
│  └───────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### 2.1 머신러닝의 핵심 — 특징(Feature)과 분류기

강의에서는 "고양이 vs 개 사진 구분"을 예로 듭니다. 사람이 직접 규칙을 짜는 대신, 머신러닝은 **특징(feature)** — 귀 모양, 코 모양, 털 길이 같은 단서 — 을 보고 학습합니다. 학습이 끝나면 **분류기(classifier)** 가 새 사진에 대해 "고양이일 확률 95%" 같은 답을 냅니다.

머신러닝은 학습 방식에 따라 크게 나뉩니다(DAY1~7에서 다룬 내용).

| 종류 | 정답(Label) | 한 줄 설명 |
|---|---|---|
| 지도학습 | 있음 | 입력 + 정답으로 학습 (회귀·분류) |
| 비지도학습 | 없음 | 데이터 구조 스스로 발견 (군집·차원축소) |
| 강화학습 | 보상 | 시행착오로 행동 정책 학습 |

### 2.2 딥러닝은 무엇이 다른가

딥러닝(Deep Learning)은 **인공신경망(Artificial Neural Network)** 을 사용하는 머신러닝의 한 갈래입니다. **은닉층(Hidden Layer)** 을 여러 겹 쌓아 "깊게(Deep)" 만든다고 해서 붙은 이름입니다.

- 구조: **입력층 → 은닉층(여러 개) → 출력층**
- 가장 큰 차이: 전통 머신러닝은 사람이 특징을 골라 줘야 하지만, 딥러닝은 **특징을 스스로 추출**합니다.

> 🟩 **(보충)** "신경망이 사람 뇌를 그대로 구현한다"는 표현은 비유일 뿐입니다. 인공신경망은 뇌에서 **영감을 받은 단순화된 수학 모델**이지, 생물학적 뇌의 복제가 아닙니다.

---

## 3. 신경망의 역사 — 퍼셉트론에서 딥러닝까지 🟦

딥러닝은 갑자기 등장한 기술이 아니라 **70여 년에 걸친 발전**의 결과입니다. 강의는 세 단계로 정리합니다.

### 3.1 1단계 — 인공 뉴런 (McCulloch & Pitts, 1943)

신경세포(뉴런)가 신호를 받아 임계값을 넘으면 "발화(ON=1)", 못 넘으면 "정지(OFF=0)" 하는 동작을 **수학으로 옮긴** 최초의 모델입니다.

$$ y = f(w_1x_1 + w_2x_2 + \cdots + b) $$

| 기호 | 의미 |
|---|---|
| $x$ | 입력값 |
| $w$ | 가중치(weight) — 입력의 중요도 |
| $b$ | 편향(bias) — 발화 기준선을 위아래로 옮김 |
| $f$ | 활성화 함수 — 발화 여부 결정 |
| $y$ | 출력값 |

> 🟩 **(보충)** 가중치 $w$는 "학습으로 정해지는 계수"이지, **곧바로 "이 특징이 몇 배 중요하다"로 읽으면 안 됩니다.** 입력의 크기·정규화 여부에 따라 값의 의미가 달라지기 때문입니다.

### 3.2 2단계 — 퍼셉트론과 XOR의 벽 (Rosenblatt, 1958)

Rosenblatt는 **가중치를 데이터로 학습**할 수 있는 퍼셉트론(Perceptron)을 제안했습니다. 하지만 1969년 Minsky와 Papert가 **단층 퍼셉트론은 XOR조차 풀 수 없다**는 것을 보였습니다.

```
XOR 진리표              하나의 직선으로
0 XOR 0 = 0            (0,0),(1,1) ↔ (0,1),(1,0) 을
0 XOR 1 = 1            동시에 가를 수 없음  → 단층 퍼셉트론 한계
1 XOR 0 = 1
1 XOR 1 = 0
```

이 한계로 신경망 연구가 크게 위축된 시기를 **AI 겨울(AI Winter)** 이라고 부릅니다.

### 3.3 3단계 — 은닉층과 딥러닝의 부활 (Hinton, 1995~)

해결책은 **입력층과 출력층 사이에 은닉층(Hidden Layer)을 넣은 다층 구조** 였습니다. 이렇게 층을 깊게 쌓은 망을 **심층 신경망(Deep Neural Network, DNN)** 이라고 합니다.

- "Deep"은 **층이 깊다(많다)** 는 뜻입니다.
- 층이 깊어지면 낮은 층은 **선 → 모서리 → 윤곽** 처럼 점점 복잡한 특징을 단계적으로 학습합니다.
- 2012년 **AlexNet** 이 이미지 인식 대회에서 압도적 성능을 보이며 딥러닝 시대가 본격화되었습니다.

```
1943            1958~1969        1969~          1995~          2012        2020~
McCulloch&Pitts  Rosenblatt       Minsky&Papert  Hinton         AlexNet     Transformer
인공 뉴런         퍼셉트론          XOR 한계        은닉층/DNN      이미지 인식  ChatGPT 등
```

> 🟩 **(보충)** 다층 신경망이나 딥러닝을 **한 사람이 발명했다고 말하기는 어렵습니다.** 여러 연구자의 누적된 기여로 보는 것이 정확합니다. 위 연표의 연도·인물은 강의 자료 기준이며, 개략적 흐름으로 받아들이세요.

### 3.4 활성화 함수 한눈에 보기

은닉층을 쌓을 때 **활성화 함수**가 꼭 필요합니다. 활성화 함수가 없으면 층을 아무리 쌓아도 **하나의 1차식으로 합쳐져** 깊은 망의 의미가 사라지기 때문입니다.

| 함수 | 출력 범위 | 특징 |
|---|---|---|
| Sigmoid | 0 ~ 1 | 확률처럼 해석, 깊은 층에선 기울기 소실 |
| Tanh | -1 ~ 1 | sigmoid보다 기울기 소실 덜함 |
| ReLU | 0 ~ ∞ | `max(0, x)`, 계산 단순, 현재 은닉층 기본 |
| Leaky ReLU | -∞ ~ ∞ | ReLU의 "죽은 뉴런" 완화 |
| Softmax | 합=1 | 다중 분류 출력층(클래스 확률) |

강의는 은닉층에서 **주로 ReLU**를 쓴다고 정리합니다. (자세한 함수별 비교는 6장에서 다시 다룹니다.)

---

## 4. 딥러닝은 왜 지금 가능해졌나 & 어디에 쓰이나 🟦

### 4.1 가능해진 3가지 이유

1. **빅데이터** — 인터넷·SNS·센서 등에서 학습할 데이터가 폭발적으로 늘었습니다.
2. **GPU 발전** — GPU(Graphics Processing Unit)는 **수많은 단순 계산을 동시에(병렬)** 처리해, 신경망 학습에 필요한 방대한 행렬 연산을 빠르게 해냅니다.
3. **다양한 모델/알고리즘** — CNN·RNN·LSTM·Transformer·GAN 등 문제 유형별 구조가 발전했습니다.

### 4.2 대표 활용 분야와 모델

| 모델 | 잘하는 일 |
|---|---|
| CNN | 이미지 인식 · 분류 |
| RNN | 순차(시계열·텍스트) 데이터 |
| LSTM | 긴 순서 의존성(장기 기억) |
| Transformer | 현대 생성형 AI의 핵심 (ChatGPT·Gemini·Claude) |
| GAN | 이미지 등 생성 |
| AutoEncoder | 차원 축소 · 특징 추출 |

활용 분야는 **이미지(분류·객체 탐지), 자연어 처리(번역·챗봇), 음성, 생성형 AI(ChatGPT·Claude·Gemini·Stable Diffusion·Midjourney)** 등으로 넓어지고 있습니다.

> 🟩 **(보충)** "무엇이 가장 많이 쓰인다"류의 순위는 **시간에 따라 바뀝니다.** 위 표는 각 모델이 잘하는 영역을 직관적으로 묶은 것으로 이해하세요.

---

## 5. 프레임워크와 환경 설치 🟦

### 5.1 대표 프레임워크

| 프레임워크 | 개발 | 특징 |
|---|---|---|
| **TensorFlow** | Google | 산업·배포에 강함 |
| **Keras** | — | TensorFlow 위에서 동작하는 **고수준 API** (간결한 모델 작성) |
| **PyTorch** | Meta(구 Facebook) | 연구에서 널리 사용, 직관적 |

> 🟩 **(보충)** Keras는 TensorFlow와 별개의 경쟁 프레임워크가 아니라, **TensorFlow를 쉽게 쓰게 해 주는 상위 인터페이스**입니다(현재 `tf.keras`). 6장 실습 코드는 이 Keras 스타일로 작성합니다.

### 5.2 CPU vs GPU, 그리고 CUDA · cuDNN

- **CPU** 로도 학습은 됩니다. 다만 큰 신경망은 느립니다.
- **GPU** 는 병렬 연산이 강해 학습을 크게 가속합니다(NVIDIA GPU 기준).
- **CUDA** : NVIDIA가 만든 **병렬 연산 플랫폼/런타임**. GPU를 일반 계산에 쓰게 해 줍니다.
- **cuDNN** : CUDA 위에서 도는 **딥러닝 전용 가속 라이브러리**(CNN·RNN 등 연산 최적화).

> ⚠️ **셋은 같은 종류의 "프로그램"이 아닙니다.**
> CUDA(플랫폼) · cuDNN(라이브러리) · 프레임워크(PyTorch/TensorFlow)는 **역할이 다른 계층**입니다. 또한 **모든 컴퓨터에 NVIDIA GPU(=CUDA)가 있는 것은 아닙니다.** GPU가 없으면 CPU로 동작합니다.

### 5.3 설치 순서 (강의 안내)

```
1) Python  →  2) Anaconda  →  3) CUDA  →  4) cuDNN  →  5) PyTorch
                      ↘  실습 환경:  Jupyter Notebook  또는  Google Colab
```

> 🟩 **(보충)** CUDA·cuDNN 설치는 **GPU가 있을 때만** 필요합니다. 환경 설정이 부담된다면 **Google Colab**을 추천합니다. 설치 없이 브라우저에서 무료 GPU를 쓸 수 있습니다. 설치 명령과 지원 버전은 **시간이 지나면 바뀌므로**, 실제 설치 전 PyTorch·TensorFlow 공식 사이트의 최신 안내를 확인하세요.

### 5.4 PyTorch 동작 확인 코드

강의 자료에는 GPU 확인·텐서 생성 코드가 나옵니다. 다만 **GPU가 없는 컴퓨터에서 그대로 실행하면 오류**가 나므로, 아래처럼 **GPU 사용 가능 여부를 먼저 확인**하는 방식을 권합니다.

```python
# 프레임워크: PyTorch
import torch

# 1) 사용할 장치를 안전하게 선택 (GPU 없으면 자동으로 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# 2) 텐서 생성 — 텐서는 '딥러닝용 다차원 배열'
x = torch.tensor([1, 2, 3])
print(x, x.shape)          # tensor([1, 2, 3]) torch.Size([3])

# 3) 선택한 장치로 이동
x = x.to(device)
print(x.device)            # cpu 또는 cuda:0
```

> ⚠️ **강의 자료의 `torch.cuda.get_device_name(0)` 이나 `.to("cuda")` 는 GPU가 있을 때만 동작합니다.** CPU만 있는 환경에서는 위처럼 `torch.cuda.is_available()` 로 먼저 확인한 뒤 분기해야 안전합니다.

#### 텐서 vs NumPy 배열

강의는 PyTorch 텐서와 NumPy 배열을 비교합니다. 둘 다 **n차원 배열**이지만,

| | PyTorch 텐서 | NumPy 배열 |
|---|---|---|
| GPU 지원 | ⭕ | ❌ |
| 자동 미분 | ⭕ (학습에 필수) | ❌ |
| 강점 | 속도, 딥러닝 연동 | 안정성, 풍부한 생태계 |

핵심 차이는 **텐서는 GPU 가속과 자동 미분(역전파)을 지원**한다는 점입니다. 그래서 신경망 학습에는 텐서를 씁니다.

---

## 6. DNN 실습의 전체 흐름 🟦 + 🟩

강의 6장은 **두 가지 DNN 예제**를 다룹니다.

1. **회귀** — 보스턴 집값 예측 (출력: 연속값)
2. **분류** — 심장병 여부 판별 (출력: 0/1)

> ⚠️ **프레임워크 안내**: 5장 설치는 **PyTorch 중심**이지만, 6장 실습의 흐름(`evaluate`·`predict` 함수, `MY_EPOCH`·`MY_BATCH` 같은 하이퍼파라미터)은 **Keras(TensorFlow) API**에 해당합니다. 그래서 아래 **실습 예제는 Keras로 통일**해 작성합니다. 같은 글 안에서 프레임워크가 바뀌는 지점을 분명히 표시합니다.

### 6.1 공통 준비 단계 (두 예제 모두 동일)

어떤 DNN이든 학습 전에 거치는 공통 단계가 있습니다.

**(1) 데이터 살펴보기 — 상자그림(Box Plot)**
데이터의 다섯 숫자(최솟값·Q1·중앙값·Q3·최댓값)와 **특이점(Outlier)** 을 한눈에 봅니다. 특이점은 $Q1 - 1.5 \times IQR$ 보다 작거나 $Q3 + 1.5 \times IQR$ 보다 큰 값입니다($IQR = Q3 - Q1$). 특이점이 많은 특성은 학습에 방해가 될 수 있습니다.

**(2) Z-점수 정규화(표준화)**
특성마다 값의 범위가 다르면, **범위가 큰 특성이 학습을 지배**합니다(예: 안정혈압 94~200 vs oldpeak 0~6.2). 이를 막기 위해 각 값을 아래처럼 변환해 **평균 0, 표준편차 1** 로 맞춥니다.

$$ Z = \frac{X - \text{평균}}{\text{표준편차}} $$

> 🟩 **(데이터 누수 주의)** 정규화의 **평균·표준편차는 반드시 "학습 데이터"에서만** 계산해야 합니다. 전체 데이터로 먼저 정규화한 뒤 나누면, 평가 데이터 정보가 학습에 새어 들어가 성능이 부풀려집니다. → 코드에서는 `fit_transform`(학습)과 `transform`(평가)을 구분합니다.

**(3) 학습용 / 평가용 분할**
지도학습은 데이터를 입력값/정답, 학습용/평가용으로 나눕니다.

```
X_train / Y_train  (학습용 입력 / 정답)
X_test  / Y_test   (평가용 입력 / 정답)
```

### 6.2 예제 ① 회귀 — 보스턴 집값 예측

강의 설계도(개념):

```
입력층 12개  →  은닉1 (200, ReLU)  →  은닉2 (1000, ReLU)  →  출력 1개(활성화 없음 → 집값)
```

- 출력층에 **활성화 함수가 없습니다.** 회귀는 **연속값(집값)** 을 그대로 내보내야 하기 때문입니다.
- 손실 함수는 **MSE(평균 제곱 오차)** 를 씁니다: 완벽하면 0, 오차가 크면 커지고 **큰 오차에 더 민감**합니다.

$$ MSE = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2 $$

학습은 **경사하강법(SGD)** — 손실 함수의 기울기를 따라 손실이 낮아지는 쪽으로 가중치를 조금씩 보정 — 으로 진행합니다.

> 🟩 **(은닉층 뉴런 수)** 위 200·1000 같은 뉴런 수는 정해진 공식이 있는 게 아니라 **실험적으로 고른 값**입니다. (강의 자료에서도 "실험적으로 정해짐"이라고 적고 있습니다.)

> ⚠️ **보스턴 집값 데이터셋 주의 (중요)**
> 이 데이터셋은 **인종 관련 변수(`B`)** 등 윤리적 문제가 제기되어, **scikit-learn 1.2 버전부터 `load_boston()` 이 제거**되었습니다. 학습·개념 이해용으로는 등장하지만, **새 프로젝트의 기본 데이터셋으로 권하지 않습니다.** 실행 가능한 회귀 예제가 필요하면 `fetch_california_housing()`(캘리포니아 집값) 같은 **유지·관리되는 대체 데이터**를 사용하세요.

회귀용 모델 정의(부분 코드, Keras):

```python
# [부분 코드] 회귀용 출력층/손실 — 분류와의 차이만 강조
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(12,)),          # 입력 특성 수
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dense(1000, activation="relu"),
    tf.keras.layers.Dense(1),                     # 출력 1개, 활성화 없음(연속값)
])
model.compile(optimizer="sgd", loss="mse", metrics=["mae"])  # 회귀 → MSE
```

### 6.3 예제 ② 분류 — 심장병 판별 (전체 실행 예제)

**데이터**: UCI Heart Disease — 환자 303명, 특성 13개(나이·성별·가슴통증 유형·안정혈압·콜레스테롤·공복혈당·안정심전도·최대심박수·운동유발협심증·oldpeak·slope·주요혈관수·thal), 정답은 **심장병 여부(0=정상, 1=심장병)**.

강의 설계도(개념):

```
입력층 13개  →  은닉1 (1000, Tanh)  →  은닉2 (1000, Tanh)  →  출력 1개(Sigmoid → 심장병 확률)
```

- 출력층 **Sigmoid** 는 결과를 **0~1 확률**로 바꿔, 이진 분류(심장병/정상)에 맞습니다.
- 평가는 **혼동행렬(Confusion Matrix)** 과 **F1 점수**로 합니다.

| 지표 | 식 | 의미 |
|---|---|---|
| 정밀도(Precision) | TP / (TP + FP) | "양성"이라 한 것 중 진짜 양성 비율 |
| 재현율(Recall) | TP / (TP + FN) | 실제 양성 중 잡아낸 비율 |
| 정확도(Accuracy) | (TP + TN) / 전체 | 전체 중 맞힌 비율 |
| F1 | 2·P·R / (P + R) | 정밀도·재현율의 조화평균 |

> 🟩 **(보충)** 질병처럼 **놓치면 위험한 문제**에서는 정확도만 봐서는 안 됩니다. 실제 환자를 놓치지 않는 **재현율**과, 정밀도까지 균형 있게 보는 **F1**이 더 중요합니다.

> 🟩 **(Sigmoid에 대한 정정)** 강의 자료는 sigmoid를 "딥러닝에서 가장 많이 쓰이는 활성화 함수"라고 소개하지만, 이는 다소 과한 표현입니다. **오늘날 은닉층의 기본은 ReLU 계열**이고, **sigmoid는 주로 이진 분류의 출력층**에서 쓰입니다. 또한 sigmoid·tanh는 깊은 층에서 **기울기 소실(gradient vanishing)** 이 생길 수 있어, ReLU가 이를 **완화**합니다(완전히 없애지는 못합니다).

아래는 강의 흐름(설계도·정규화·evaluate·predict·혼동행렬)을 따라 **재구성한 전체 실행 예제**입니다. **강의 PDF에 이 코드 전체가 실려 있지는 않습니다.**

```python
# === PDF 강의 흐름을 따라 재구성한 실행 예제 (프레임워크: TensorFlow / Keras) ===
# 설치: pip install tensorflow scikit-learn ucimlrepo
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 0) 재현성을 위한 시드 고정
#    (GPU·병렬 연산에서는 결과가 완벽히 같지 않을 수 있습니다)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1) 데이터 로드 (UCI Heart Disease, id=45) ─ 공식 채널
from ucimlrepo import fetch_ucirepo
heart = fetch_ucirepo(id=45)
X_df = heart.data.features                 # (303, 13)
y_raw = heart.data.targets.iloc[:, 0]      # 0~4 (0=정상, 1~4=심장병 단계)

df = pd.concat([X_df, y_raw], axis=1).dropna()   # 결측치 행 제거
X = df.iloc[:, :-1].values                  # (n, 13)
y = (df.iloc[:, -1].values > 0).astype(int) # 0=정상, 1=심장병으로 이진화

# 2) 학습/평가 분할 ─ 분류이므로 stratify로 클래스 비율 유지
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 3) Z-점수 정규화 ─ 통계량은 '학습 데이터'에서만 계산 (데이터 누수 방지)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # 학습: 평균·표준편차 계산 + 변환
X_test  = scaler.transform(X_test)          # 평가: 학습 통계로 '변환만'

# 4) DNN 설계 (입력13 → 1000 tanh → 1000 tanh → 1 sigmoid)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(13,)),
    tf.keras.layers.Dense(1000, activation="tanh"),
    tf.keras.layers.Dense(1000, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"),   # 이진 분류 → 0~1 확률
])

# 5) 컴파일
#    출력이 sigmoid 1개인 이진 분류에는 binary_crossentropy가 표준입니다.
#    (강의의 손실 설명은 회귀용 MSE였지만, 분류에는 크로스엔트로피가 더 적합 ─ 보충)
MY_EPOCH = 50      # 전체 데이터를 몇 번 반복 학습할지
MY_BATCH = 32      # 한 번의 가중치 업데이트에 쓰는 표본 수
model.compile(optimizer="sgd",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# 6) 학습 ─ 학습/검증 지표를 따로 추적
history = model.fit(
    X_train, y_train,
    epochs=MY_EPOCH, batch_size=MY_BATCH,
    validation_split=0.2, verbose=0
)

# 7) 평가 (강의의 evaluate 함수에 해당) ─ 모델 선택이 끝난 뒤 test에 한 번만
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 정확도: {acc:.3f}")

# 8) 예측 (강의의 predict 함수에 해당) ─ 확률 → 0/1 라벨
proba = model.predict(X_test, verbose=0).ravel()   # (n,) 0~1 확률
pred  = (proba >= 0.5).astype(int)

# 9) 혼동행렬 / 정밀도·재현율·F1
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=3))
```

**중요 텐서 모양(shape) 정리**

```
입력 X            : (배치, 13)
은닉1 출력         : (배치, 1000)
은닉2 출력         : (배치, 1000)
출력(sigmoid)     : (배치, 1)     → 0~1 확률
정답 y            : (배치,)       → 0 또는 1
```

> ⚠️ **이 코드는 현재 작성 환경에 TensorFlow가 없어 실행 검증을 하지 못했습니다.** 구조·손실·출력·정규화 흐름은 점검했으나, 실제 정확도 수치는 환경에서 직접 실행해 확인하세요. (강의는 회귀에 SGD+MSE를 썼지만, 분류 출력층이 sigmoid이므로 손실만 `binary_crossentropy`로 바꿨습니다.)

### 6.4 직접 바꿔 보며 배우기 (강의 실습 문제)

강의는 코드를 한 줄씩 바꿔 보며 **결과가 어떻게 변하는지 관찰**하라고 권합니다. 입문자에게 아주 좋은 학습법입니다.

- `MY_EPOCH` 를 0으로 → 학습이 거의 안 된 상태 확인
- `MY_BATCH` 를 16/32로 → 학습 안정성·속도 비교
- 은닉층 추가/뉴런 수 변경 → 표현력 변화
- Z-점수 정규화 생략 → 정규화의 효과 체감
- 활성화 함수 Tanh ↔ ReLU 교체
- 최적화 함수 SGD ↔ RMSprop 교체
- 특정 특성(예: `fbs`, `chol`) 제거 → 특성 중요도 감 잡기
- 데이터 일부만 사용(예: 303개 중 100개) → 데이터 양의 영향

> 🟩 **(보충)** 바꿀 때는 **한 번에 하나씩** 바꾸세요. 여러 개를 동시에 바꾸면 어떤 변화가 결과에 영향을 줬는지 알 수 없습니다.

---

## 7. 자주 하는 실수

```
❌ GPU가 당연히 있다고 가정      → torch.cuda.is_available()로 먼저 확인
❌ 전체 데이터로 정규화 후 분할   → 학습 데이터로만 fit, 평가는 transform만
❌ 분류 출력에 MSE 사용          → sigmoid 출력은 binary_crossentropy
❌ 활성화 함수 없이 층만 쌓기     → 비선형성 사라져 1층과 동일
❌ sigmoid가 항상 최선이라 믿기  → 은닉층 기본은 ReLU, sigmoid는 출력층용
❌ test로 하이퍼파라미터 튜닝     → test는 마지막 한 번만, 검증은 validation으로
❌ 보스턴 집값을 기본 데이터로    → sklearn에서 제거됨, 대체 데이터 사용
```

---

## 8. 딥러닝 DAY1 핵심 정리

```
✅ AI ⊃ 머신러닝 ⊃ 딥러닝
   딥러닝 = 은닉층을 깊게 쌓은 인공신경망, 특징을 스스로 추출

✅ 신경망의 역사
   인공뉴런(1943) → 퍼셉트론(1958) → XOR 한계/AI겨울(1969)
   → 은닉층·DNN(1995~) → AlexNet(2012) → Transformer/ChatGPT(2020~)

✅ 딥러닝이 가능해진 이유 = 빅데이터 + GPU + 다양한 모델
   모델: CNN(이미지)·RNN/LSTM(순차)·Transformer(생성형)·GAN(생성)·AutoEncoder

✅ 환경 = 프레임워크(PyTorch/TensorFlow·Keras) + (선택)GPU·CUDA·cuDNN
   설치 부담되면 Google Colab → 텐서는 GPU·자동미분을 지원하는 배열

✅ DNN 실습 흐름
   상자그림으로 데이터 확인 → Z-점수 정규화(학습 데이터 기준)
   → 학습/평가 분할 → 모델 설계 → 학습 → evaluate/predict → 평가

✅ 과제 유형별 출력층·손실 (중요!)
   회귀  : 출력 활성화 없음  +  MSE
   이진분류: 출력 sigmoid    +  binary_crossentropy
   평가는 회귀=MSE/MAE, 분류=혼동행렬·F1

✅ 주의
   GPU 가정 금지 / 정규화는 학습 데이터로만 / sigmoid는 출력층용
   보스턴 집값 데이터는 윤리 문제로 sklearn에서 제거됨
```

---

## 🔗 참고 자료

- [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)
- [TensorFlow 공식 설치 안내](https://www.tensorflow.org/install)
- [Keras 공식 문서](https://keras.io/)
- [UCI Heart Disease 데이터셋](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [scikit-learn `fetch_california_housing` (보스턴 대체)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [scikit-learn에서 보스턴 데이터셋이 제거된 이유](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)
- [Google Colab](https://colab.research.google.com/)

---

> 📌 **작성 메모(초안 v1)**
> - 강의 자료(딥러닝 DAY1, 52쪽)에는 **개념 그림·데이터 설명**만 있고 **실행 코드 전체는 없어**, 6장 실습 코드는 강의 흐름에 맞춰 **재구성**했습니다(현재 환경에 TF 미설치로 미실행).
> - 보스턴 집값(회귀)은 데이터셋 윤리 문제로 **개념 설명 위주**로 두고, **실행 예제는 심장병 분류**로 통일했습니다.
> - 설치는 PyTorch 중심(강의), 실습은 Keras(강의의 evaluate/predict 흐름) — **프레임워크 전환 지점을 명시**했습니다.
> - DAY 번호는 **딥러닝 편 DAY1**로 확정(머신러닝 DAY1~7과 별도 카운터, 머리말에 연결 링크 표기).
> - 검토(Stage 2)에서 확인 요청: ① 회귀(보스턴) 예제도 실행 코드로 넣을지 ② 설계도의 뉴런 수(슬라이드 그림 200/1000 vs 주석 2000/1000 불일치) 표기 방식.
