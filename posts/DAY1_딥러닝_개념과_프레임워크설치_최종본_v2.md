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
- 신경망의 역사: 퍼셉트론 → XOR 한계 → 다층 신경망 → 딥러닝
- 신경망이 학습하는 원리: 순전파 · 손실 · 역전파
- 딥러닝이 가능해진 이유와 대표 활용 분야
- 프레임워크(TensorFlow · Keras · PyTorch)와 GPU · CUDA · cuDNN
- DNN으로 회귀(집값)와 분류(심장병)를 푸는 **전체 흐름**

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 내용
> - 🟩 **(보충)** : 입문자를 위해 글쓴이가 덧붙인 사전지식 · 실무 주의점
>
> 강의 자료에는 **개념 그림과 데이터 설명**은 있지만 **실행 코드 전체는 실려 있지 않습니다.** 7장의 실습 코드는 강의 흐름을 따라 **재구성한 실행 가능 예제**이며, 이 점은 7장에서 한 번만 명확히 표시합니다.

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

머신러닝은 학습 방식에 따라 크게 나뉩니다(머신러닝 편 DAY1~7에서 다룬 내용).

| 종류 | 정답(Label) | 한 줄 설명 |
|---|---|---|
| 지도학습 | 있음 | 입력 + 정답으로 학습 (회귀·분류) |
| 비지도학습 | 없음 | 데이터 구조 스스로 발견 (군집·차원축소) |
| 강화학습 | 보상 | 시행착오로 행동 정책 학습 |

### 2.2 딥러닝은 무엇이 다른가

딥러닝(Deep Learning)은 **인공신경망(Artificial Neural Network)** 을 사용하는 머신러닝의 한 갈래입니다. **은닉층(Hidden Layer)** 을 여러 겹 쌓아 "깊게(Deep)" 만든다고 해서 붙은 이름입니다.

- 구조: **입력층 → 은닉층(여러 개) → 출력층**
- 큰 차이: 전통 머신러닝은 사람이 특징을 골라 주는 경우가 많지만, 딥러닝은 **유용한 특징을 비교적 자동으로 학습**하는 경향이 있습니다.

> 🟩 **(보충)** "딥러닝이 특징을 100% 알아서 다 만든다"는 절대적 표현은 피하는 게 좋습니다. 데이터 전처리, 입력 설계, 도메인 지식은 여전히 중요합니다. 또한 "신경망이 사람 뇌를 그대로 구현한다"는 표현도 비유일 뿐입니다. 인공신경망은 뇌에서 **영감을 받은 단순화된 수학 모델**이지, 생물학적 뇌의 복제가 아닙니다.

---

## 3. 신경망의 역사 — 퍼셉트론에서 딥러닝까지 🟦

딥러닝은 갑자기 등장한 기술이 아니라 **오랜 기간 누적된 발전**의 결과입니다. 강의는 흐름을 쉽게 보여 주기 위해 세 단계로 정리합니다.

### 3.1 1단계 — 인공 뉴런 (McCulloch & Pitts, 1943)

신경세포(뉴런)가 신호를 받아 임계값을 넘으면 "발화(ON=1)", 못 넘으면 "정지(OFF=0)" 하는 동작을 **수학으로 옮긴** 초기 모델입니다.

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

이런 한계와 더불어 여러 요인이 겹쳐 신경망 연구가 위축된 시기를 **AI 겨울(AI Winter)** 이라고 부릅니다.

### 3.3 3단계 — 다층 신경망과 딥러닝의 부상

핵심 해결책은 **입력층과 출력층 사이에 은닉층(Hidden Layer)을 넣은 다층 구조** 와, 이를 학습시키는 **역전파(backpropagation)** 였습니다. 이렇게 층을 깊게 쌓은 망을 **심층 신경망(Deep Neural Network, DNN)** 이라고 합니다.

- "Deep"은 **층이 깊다(많다)** 는 뜻입니다.
- 층이 깊어지면 낮은 층은 **선 → 모서리 → 윤곽** 처럼 점점 복잡한 특징을 단계적으로 학습하는 경향이 있습니다.
- 2012년 **AlexNet** 이 이미지 인식 대회에서 큰 성능 향상을 보이며 딥러닝이 본격적으로 주목받았습니다.

```
1943            1958~1969     1969~          1980년대~        2012        2017         2022
McCulloch&Pitts  Rosenblatt    Minsky&Papert  역전파 확산·     AlexNet     Transformer  ChatGPT
인공 뉴런         퍼셉트론       XOR 한계        심층학습 연구     이미지 인식  논문 발표     공개
```

> 🟩 **역사 설명 주의 (정정)**
> 강의 PDF는 입문 흐름을 위해 `퍼셉트론 → XOR 한계 → Hinton → AlexNet`으로 단순화하고, "1995년에 은닉층을 넣어 DNN을 부활시켰다"고 설명합니다. 하지만 **은닉층은 1995년에 처음 나온 개념이 아니며**, 딥러닝의 발전은 한 사건·한 연구자만으로 설명하기 어렵습니다. 다층 신경망, **1980년대 역전파의 확산**, 2000년대 후반 심층 학습 연구, 계산 자원(GPU), 대규모 데이터, 학습 기법이 **누적된 결과**입니다. AI 겨울 역시 기술적 한계뿐 아니라 과도한 기대와 투자 축소가 함께 작용했습니다. (참고: Transformer 논문 *Attention Is All You Need* 는 **2017년**, ChatGPT 공개는 **2022년**입니다.)

---

## 4. 신경망은 어떻게 학습하는가 🟦

PDF 39~40쪽의 핵심 내용입니다. 신경망 학습은 네 단계가 반복되는 과정입니다.

1. **순전파(Forward propagation)**: 입력값이 은닉층과 출력층을 차례로 지나 **예측값**이 됩니다. (입력 → 왼쪽에서 오른쪽으로 계산)
2. **손실 계산(Loss)**: 예측값과 정답의 차이를 **손실함수**로 수치화합니다. 예를 들어 정답이 3인데 예측이 3.5이면 오차 0.5가 생깁니다.
3. **역전파(Backpropagation)**: 각 가중치가 손실에 **얼마나 영향을 줬는지** 연쇄법칙(chain rule)으로 계산해 기울기(gradient)를 구합니다. (출력 → 입력 방향, 즉 역방향)
4. **가중치 갱신(Optimizer step)**: SGD 같은 옵티마이저가 기울기를 이용해 **손실이 작아지는 방향**으로 가중치를 조금씩 이동시킵니다.

이 1~4를 데이터에 대해 여러 번 반복하면서 신경망은 점점 정답에 가까워집니다.

### 4.1 활성화 함수와 기울기 소실

은닉층을 쌓을 때 **활성화 함수**가 꼭 필요합니다. 활성화 함수가 없으면 층을 아무리 쌓아도 **하나의 1차식(선형 변환)으로 합쳐져** 깊은 망의 의미가 사라지기 때문입니다.

| 함수 | 출력 범위 | 특징 |
|---|---|---|
| Sigmoid | 0 ~ 1 | 확률처럼 해석, 깊은 층에선 기울기 소실 |
| Tanh | -1 ~ 1 | sigmoid보다 기울기 소실이 덜한 편 |
| ReLU | 0 ~ ∞ | `max(0, x)`, 계산 단순, 현재 은닉층의 흔한 기본값 |
| Leaky ReLU | -∞ ~ ∞ | ReLU의 "죽은 뉴런" 문제 완화 |
| Softmax | 합=1 | 다중 분류 출력층(클래스 확률) |

sigmoid와 tanh를 깊게 쌓으면 역전파에서 기울기가 점점 작아져 학습이 잘 안 되는 **기울기 소실(Gradient Vanishing)** 이 생길 수 있습니다. ReLU 계열은 이를 **완화하는 데 도움**이 되지만, **모든 경우에 문제를 완전히 없애는 것은 아닙니다.**

---

## 5. 딥러닝은 왜 지금 가능해졌나 & 어디에 쓰이나 🟦

### 5.1 가능해진 3가지 이유

1. **빅데이터** — 인터넷·SNS·센서 등에서 학습할 데이터가 폭발적으로 늘었습니다.
2. **GPU 발전** — GPU(Graphics Processing Unit)는 **수많은 단순 계산을 동시에(병렬)** 처리해, 신경망 학습에 필요한 방대한 행렬 연산을 빠르게 해냅니다.
3. **다양한 모델/알고리즘** — CNN·RNN·LSTM·Transformer·GAN 등 문제 유형별 구조가 발전했습니다.

### 5.2 대표적으로 잘 알려진 활용

| 모델 | 대표적으로 잘 알려진 활용 |
|---|---|
| CNN | 이미지 인식 · 분류 |
| RNN | 순차(시계열·텍스트) 데이터 |
| LSTM | 긴 순서 의존성(장기 기억) |
| Transformer | 현대 생성형 AI의 핵심 (ChatGPT·Gemini·Claude) |
| GAN | 이미지 등 생성 |
| AutoEncoder | 차원 축소 · 특징 추출 |

> 🟩 **(보충)** 위 표는 **1:1로 고정된 규칙이 아닙니다.** 예컨대 이미지에도 Transformer(ViT)가 쓰이고, 시계열에도 CNN이 쓰입니다. "대표적으로 잘 알려진 조합" 정도로 이해하세요. "무엇이 가장 많이 쓰인다"류의 순위도 시간에 따라 바뀝니다.

활용 분야는 이미지(분류·객체 탐지), 자연어 처리(번역·챗봇), 음성, 생성형 AI(ChatGPT·Claude·Gemini·Stable Diffusion·Midjourney) 등으로 넓어지고 있습니다.

---

## 6. 프레임워크와 환경 설치 🟦

### 6.1 대표 프레임워크

| 프레임워크 | 개발·생태계 | 특징 |
|---|---|---|
| **TensorFlow** | Google | 산업·배포에 강함 |
| **Keras** | Google(François Chollet) | 간결한 고수준 API, 현재는 멀티 백엔드 |
| **PyTorch** | Meta(구 Facebook) | 연구에서 널리 사용, 직관적 |

> 🟩 **Keras에 대한 정정 (중요)**
> 강의 PDF는 Keras를 "TensorFlow 위에서 동작하는 고수준 API"로 소개합니다. 과거 `tf.keras`를 설명할 때는 맞는 말이었지만, **현재의 Keras 3는 TensorFlow뿐 아니라 JAX·PyTorch도 백엔드로 쓸 수 있는 멀티 백엔드 API**입니다. 그래서 이 글에서는 "Keras 전체 = TensorFlow 전용"이라고 일반화하지 않고, **"아래 실습 예제는 TensorFlow 백엔드의 Keras(`tf.keras`)를 사용한다"** 고만 한정하겠습니다. ([Keras 3 소개](https://keras.io/keras_3/))

### 6.2 CPU vs GPU, 그리고 CUDA · cuDNN

- **CPU** 로도 학습은 됩니다. 다만 큰 신경망은 느립니다.
- **GPU** 는 병렬 연산이 강해 학습을 크게 가속합니다(NVIDIA GPU 기준).
- **CUDA** : NVIDIA가 만든 **병렬 연산 플랫폼/런타임**. GPU를 일반 계산에 쓰게 해 줍니다.
- **cuDNN** : CUDA 위에서 도는 **딥러닝 전용 가속 라이브러리**(CNN·RNN 등 연산 최적화).

> ⚠️ **셋은 같은 종류의 "프로그램"이 아닙니다.**
> CUDA(플랫폼) · cuDNN(라이브러리) · 프레임워크(PyTorch/TensorFlow)는 **역할이 다른 계층**입니다. 또한 **모든 컴퓨터에 NVIDIA GPU(=CUDA)가 있는 것은 아닙니다.** GPU가 없으면 CPU로 동작합니다.

### 6.3 설치 — "강의 안내" vs "현재 권장 방식"

강의 PDF는 설치 순서를 이렇게 안내합니다.

```
1) Python  →  2) Anaconda  →  3) CUDA  →  4) cuDNN  →  5) PyTorch
                      ↘  실습 환경:  Jupyter Notebook  또는  Google Colab
```

> 🟩 **설치에 대한 정정 (중요)**
> 위 순서가 **모든 사용자에게 항상 필요한 필수 절차는 아닙니다.**
> - **CPU만 쓰는 경우 CUDA·cuDNN은 필요 없습니다.** 이 둘은 NVIDIA GPU를 쓸 때만 설치합니다.
> - 요즘은 CUDA·cuDNN을 손으로 먼저 깔기보다, **OS·GPU에 맞춰 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)가 알려 주는 명령**을 그대로 쓰는 것이 우선입니다.
> - **Windows에서 최신 TensorFlow GPU**를 쓰려면 주의가 필요합니다. TensorFlow 공식 문서상 **네이티브 Windows GPU 지원은 2.10이 마지막**이며, 2.11 이상은 **WSL2 경로**를 안내합니다. ([TensorFlow 설치](https://www.tensorflow.org/install/pip))
> - 환경 설정이 부담된다면 **Google Colab**을 추천합니다. 설치 없이 브라우저에서 무료 GPU를 쓸 수 있습니다.
> 설치 명령과 지원 버전은 **시간이 지나면 바뀌므로**, 실제 설치 전에는 항상 공식 문서를 확인하세요.

### 6.4 PyTorch 동작 확인 코드

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

강의는 PyTorch 텐서와 NumPy 배열을 비교합니다. 둘 다 **n차원 배열**이지만, 텐서는 **GPU 가속**과 **자동 미분(autograd)** 을 쓸 수 있다는 점이 다릅니다.

| | PyTorch 텐서 | NumPy 배열 |
|---|---|---|
| GPU 지원 | ⭕ | ❌ |
| 자동 미분 | 가능 (조건 필요) | ❌ |
| 강점 | 속도, 딥러닝 연동 | 안정성, 풍부한 생태계 |

> 🟩 **자동 미분 정정** — **모든 텐서가 자동으로 기울기를 추적하는 것은 아닙니다.** `requires_grad=True`로 만든 **실수형(float)** 텐서만 추적됩니다. 위의 `torch.tensor([1, 2, 3])`은 정수형(`int64`)이라 미분 대상이 될 수 없습니다.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # 실수형 + 추적 켜기
y = (x ** 2).sum()
y.backward()

print(x.dtype)  # torch.float32
print(x.grad)   # tensor([2., 4., 6.])  ← d(x²)/dx = 2x
```

---

## 7. DNN 실습의 전체 흐름 🟦 + 🟩

> ⚠️ **이 장에 대한 안내 (한 번만 정리)**
> - 강의 PDF에는 **개념 그림·데이터 설명만 있고 실행 코드 전체는 없습니다.** 아래 코드는 강의 흐름(설계도·정규화·`evaluate`/`predict`·혼동행렬)을 따라 **재구성한 실행 예제**입니다.
> - 강의 실습 흐름은 **Keras(TensorFlow 백엔드) API** 에 해당하므로, 실습 코드는 모두 `tf.keras`로 작성합니다. (설치 장의 PyTorch 텐서 예제와 프레임워크가 다른 지점입니다.)
> - 작성 환경에 TensorFlow가 없어 **학습 실행 수치는 검증하지 못했습니다.** 구조·손실·출력·전처리 흐름은 점검했으니, 정확도 등 실제 수치는 직접 실행해 확인하세요.

강의 6장은 **두 가지 DNN 예제**를 다룹니다.

1. **회귀** — 보스턴 집값 예측 (출력: 연속값)
2. **분류** — 심장병 여부 판별 (출력: 0/1)

### 7.1 공통 준비 단계

어떤 DNN이든 학습 전에 거치는 공통 단계가 있습니다. 표 형태의 데이터는 보통 **Pandas DataFrame**(행=표본, 열=특성)으로 다룹니다.

**(1) 데이터 살펴보기 — 상자그림(Box Plot)과 산점도**

상자그림은 Q1·중앙값·Q3와 데이터의 퍼짐을 요약합니다. 일반적인 Tukey 방식에서 **whisker(수염)** 는 `Q1 − 1.5×IQR`과 `Q3 + 1.5×IQR` 안쪽의 **가장 먼 관측값**까지 이어지므로, **항상 실제 최솟값·최댓값을 뜻하지는 않습니다**($IQR = Q3 − Q1$). whisker 밖의 점도 **자동 삭제 대상이 아니라**, 측정 오류인지 실제 희귀 사례인지 먼저 확인해야 합니다.

산점도(scatter)는 한 특성과 목표값 사이의 관계를 봅니다. `seaborn.regplot`처럼 **회귀선이 그려져도 인과관계가 증명되는 것은 아니며**, 휘어진 관계나 집단별 차이는 한 직선으로 충분히 설명되지 않을 수 있습니다.

**(2) Z-점수 정규화(표준화)**

특성마다 값의 범위가 다르면, **범위가 큰 특성이 학습을 지배**합니다(예: 안정혈압 94~200 vs oldpeak 0~6.2). 이를 막기 위해 각 값을 아래처럼 변환해 **평균 0, 표준편차 1** 로 맞춥니다.

$$ Z = \frac{X - \text{평균}}{\text{표준편차}} $$

**(3) 학습 / 검증 / 평가 분할**

지도학습은 데이터를 입력값/정답으로, 그리고 학습용/검증용/평가용으로 나눕니다.

```
X_train / y_train  (학습용)   → 가중치 학습 + 전처리 통계 계산
X_val   / y_val    (검증용)   → 모델·하이퍼파라미터 선택
X_test  / y_test   (평가용)   → 마지막에 딱 한 번만 성능 측정
```

> 🟩 **데이터 누수 방지 (가장 중요)**
> 정규화의 **평균·표준편차(그리고 결측치 대체·인코딩 같은 모든 "학습되는" 전처리)는 반드시 학습 데이터에서만 `fit`** 해야 합니다. 검증·평가 데이터에는 `transform`만 적용합니다.
> 흔한 실수: 전체 데이터를 한 번에 표준화한 뒤 나누거나, `X_train`을 표준화한 다음 그 안에서 `validation_split`으로 검증을 떼는 것 — 이러면 **검증 데이터 정보가 전처리 통계에 새어 들어갑니다.** 그래서 아래 예제는 **나누기를 먼저** 하고 전처리를 나중에 합니다.

### 7.2 예제 ① 회귀 — 보스턴 집값 예측 (설계도 중심)

강의 설계도(개념):

```
입력층 12개  →  은닉1 (200, ReLU)  →  은닉2 (1000, ReLU)  →  출력 1개(활성화 없음 → 집값)
```

- 출력층에 **활성화 함수가 없습니다.** 회귀는 **연속값(집값)** 을 그대로 내보내야 하기 때문입니다.
- 손실 함수는 **MSE(평균 제곱 오차)** 를 씁니다: 완벽하면 0, 오차가 크면 커지고 **큰 오차에 더 민감**합니다.

$$ MSE = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2 $$

학습은 **경사하강법(SGD)** — 손실의 기울기를 따라 손실이 낮아지는 쪽으로 가중치를 조금씩 보정 — 으로 진행합니다.

> 🟩 **(은닉층 뉴런 수)** 위 200·1000 같은 뉴런 수는 정해진 공식이 있는 게 아니라 **실험적으로 고른 값**입니다(강의 자료에도 "실험적으로 정해짐"이라고 적혀 있습니다). 참고로 강의 슬라이드의 그림(200·1000)과 옆 주석(2,000·1,000)이 서로 달라, 여기서는 **설계도 그림 기준**으로 적었습니다.

> ⚠️ **보스턴 집값 데이터셋 주의 (중요)**
> 이 데이터셋은 **인종 관련 변수(`B`)** 등 윤리적 문제가 제기되어, **scikit-learn 1.2부터 `load_boston()` 이 제거**되었습니다(직접 확인함). 학습·개념 이해용으로는 등장하지만 **새 프로젝트의 기본 데이터셋으로 권하지 않습니다.** 실행 가능한 회귀 예제가 필요하면 `fetch_california_housing()`(캘리포니아 집값)이나 Ames 주택 데이터 같은 **유지·관리되는 대체 데이터**를 사용하세요.

회귀와 분류의 **출력층·손실 차이만** 보이기 위한 부분 코드입니다(전체 실행 예제는 7.3에서 분류로 제공합니다).

```python
# [부분 코드] 회귀용 출력층/손실 — 분류와 비교하기 위한 발췌
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(12,)),          # 입력 특성 수
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dense(1000, activation="relu"),
    tf.keras.layers.Dense(1),                     # 출력 1개, 활성화 없음(연속값)
])
model.compile(optimizer="sgd", loss="mse", metrics=["mae"])  # 회귀 → MSE
```

### 7.3 예제 ② 분류 — 심장병 판별 (전체 실행 예제)

**데이터**: UCI Heart Disease — 환자 303명, **입력 특성 13개 + 정답 1개**(강의의 "14가지 항목"은 이 둘을 합한 수입니다). 특성은 나이·성별·가슴통증 유형(cp)·안정혈압(trestbps)·콜레스테롤(chol)·공복혈당(fbs)·안정심전도(restecg)·최대심박수(thalach)·운동유발협심증(exang)·oldpeak·slope·주요혈관수(ca)·thal 입니다.

**강의 설계도(개념):**

```
입력층 13개  →  은닉1 (1000, Tanh)  →  은닉2 (1000, Tanh)  →  출력 1개(Sigmoid → 심장병 확률)
```

- 출력층 **Sigmoid** 는 결과를 **0~1 확률**로 바꿔, 이진 분류(심장병/정상)에 맞습니다.

> ⚠️ **이 설계도를 그대로 실전 모델로 쓰지 마세요.** 표본이 약 300개인데 `Dense(1000) → Dense(1000)`이면 **파라미터가 100만 개를 넘어** 과적합 위험이 매우 큽니다. 설계도는 **개념 설명용**으로 두고, 아래 실행 예제는 **작은 기준 모델(64–32) + 조기 종료(EarlyStopping)** 로 더 안전하게 작성합니다.

> 🟩 **두 가지 데이터 처리 주의**
> 1. **정답 이진화**: UCI 원본의 표적 `num`은 0/1이 아니라 원래 **0~4**(0=정상, 1~4=심장병 단계)입니다. 아래 코드는 `> 0` 기준으로 **0=정상 / 1=심장병** 으로 묶습니다.
> 2. **범주형 코드**: `sex, cp, fbs, restecg, exang, slope, ca, thal` 은 숫자로 적혀 있지만 **범주(카테고리) 코드**입니다. 이를 전부 연속형처럼 표준화하면 코드 사이에 없는 거리·순서 의미를 부여하게 됩니다. 그래서 아래 예제는 **연속형만 표준화하고 범주형은 원-핫 인코딩**합니다. (강의처럼 13개를 모두 수치로 넣는 것은 흐름을 단순 재현하기 위한 교육용 단순화입니다.)

> 🟩 **(Sigmoid에 대한 정정)** 강의 자료는 sigmoid를 "딥러닝에서 가장 많이 쓰이는 활성화 함수"라고 소개하지만 다소 과한 표현입니다. **오늘날 은닉층의 흔한 기본은 ReLU 계열**이고, **sigmoid는 주로 이진 분류의 출력층**에서 쓰입니다.

```python
# === 강의 흐름을 따라 재구성한 실행 예제 (TensorFlow 백엔드 Keras) ===
# 설치: pip install tensorflow scikit-learn ucimlrepo matplotlib
#   (TensorFlow 설치 가능 여부는 Python 버전·운영체제에 따라 다르므로,
#    설치 전 공식 지원 버전을 확인하세요: https://www.tensorflow.org/install/pip)
import numpy as np
import tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 0) 재현성을 위한 시드 고정
#    (GPU·병렬 연산에서는 결과가 완벽히 같지 않을 수 있습니다)
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# 1) 데이터 로드 (UCI Heart Disease, id=45)
heart = fetch_ucirepo(id=45)
X = heart.data.features.copy()                       # 13개 특성
y = (heart.data.targets.iloc[:, 0] > 0).astype("float32")  # 0~4 → 0/1 이진화

# 2) train / validation / test 를 '전처리보다 먼저' 분리 (누수 방지)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

# 3) 연속형은 표준화, 범주형은 원-핫 (학습 데이터에만 fit)
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

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

X_train = preprocessor.fit_transform(X_train).astype("float32")  # fit은 학습 데이터만
X_val   = preprocessor.transform(X_val).astype("float32")        # 변환만
X_test  = preprocessor.transform(X_test).astype("float32")       # 변환만

# 정답 shape·dtype을 출력과 맞춰 (batch, 1) float32 로
y_train = y_train.to_numpy(dtype="float32").reshape(-1, 1)
y_val   = y_val.to_numpy(dtype="float32").reshape(-1, 1)
y_test  = y_test.to_numpy(dtype="float32").reshape(-1, 1)

# 4) 작은 기준 모델 (설계도의 1000-1000 대신 과적합을 줄인 64-32)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),   # 이진 분류 → 0~1 확률
])

# 5) 컴파일 — sigmoid 출력 1개인 이진 분류에는 binary_crossentropy가 표준
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

# 6) 조기 종료 — 검증 손실이 더 나아지지 않으면 멈추고 가장 좋은 가중치 복원
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# 7) 학습 — 검증 세트로 학습/검증 지표를 따로 추적
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[early_stopping], verbose=0,
)

# 8) 평가(evaluate) — 모델 선택이 끝난 뒤 test를 '한 번만'
results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
print(results)

# 9) 예측(predict) — 확률 → 0/1 라벨
proba = model.predict(X_test, verbose=0).ravel()   # 0~1 확률
pred  = (proba >= 0.5).astype(int)
true  = y_test.ravel().astype(int)

# 10) 혼동행렬 / 정밀도·재현율·F1
print(confusion_matrix(true, pred))
print(classification_report(true, pred, digits=3, zero_division=0))
```

**중요 텐서 모양(shape)·자료형(dtype)**

```
입력 X      : (batch_size, 전처리 후 특성 수), float32   # 원-핫으로 13개보다 늘어남
모델 출력   : (batch_size, 1), float32                  # sigmoid 확률
정답 y      : (batch_size, 1), float32
```

> 🟩 **`evaluate` vs `predict`**
> - `evaluate(X_test, y_test)` : 정답과 비교해 **loss·지표를 계산**(성능 측정).
> - `predict(X_new)` : 정답 없이 **새 입력의 예측(확률)을 생성**(실제 서비스 추론).
> 위 코드의 `predict(X_test)`는 혼동행렬을 만들기 위한 **평가용 예측**이라는 점에서, 정답이 없는 실제 추론과는 목적이 다릅니다.

#### 학습 곡선으로 과적합 확인

`history`에는 에폭별 손실이 담깁니다. 학습/검증 손실을 함께 그려야 과적합 여부를 해석할 수 있습니다.

```python
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.xlabel("epoch"); plt.ylabel("binary cross-entropy")
plt.legend(); plt.title("학습/검증 손실 곡선")
plt.show()
```

- 두 손실이 **모두 높음** → 과소적합(underfitting) 가능성
- train 손실은 내려가는데 **validation 손실이 다시 올라감** → 과적합(overfitting) 가능성

> ⚠️ **의료 데이터 주의** — 이 예제는 **학습용**입니다. 정확도나 F1이 높아도 **실제 진단 도구의 임상적 유효성을 뜻하지 않습니다.** 표본이 작고 오래됐으며, 실제 의료 적용에는 외부 검증, 민감도·특이도, 확률 보정(calibration), 임계값과 오류 비용 검토가 필요합니다.

### 7.4 직접 바꿔 보며 배우기 (강의 실습 문제)

강의는 코드를 조금씩 바꿔 보며 **결과가 어떻게 변하는지 관찰**하라고 권합니다. 입문자에게 아주 좋은 학습법입니다. (강의는 에폭·배치 크기를 `MY_EPOCH`·`MY_BATCH`라는 이름으로 부릅니다.)

- **에폭(epochs)을 0으로** → "학습이 덜 된 상태"가 아니라 **가중치를 한 번도 갱신하지 않은 초기 모델**의 평가 결과를 봅니다.
- **배치 크기(batch_size) 변경(16/32 등)** → 학습 안정성·속도 비교.
- **은닉층 추가 / 뉴런 수 변경** → 표현력 변화.
- **Z-점수 정규화 생략** → 정규화의 효과 체감.
- **활성화 함수 Tanh ↔ ReLU**, **옵티마이저 SGD ↔ RMSprop** 교체.
- **특정 특성(예: `fbs`, `chol`) 제거** → 단, **한 번의 성능 변화만으로 특성 중요도를 단정하지 마세요.**
- **데이터 일부만 사용(예: 303개 중 앞 100개)** → 앞부분만 자르면 클래스 분포가 치우칠 수 있으니, 비교 실험에선 `stratify`로 비율을 유지하세요.

> 🟩 **공정한 비교의 원칙**
> ① **한 번에 하나씩만** 바꾸세요(무엇이 영향을 줬는지 알 수 있게). ② 설정 비교는 **test가 아니라 validation 결과**로 하세요. test는 마지막에 한 번만 봅니다. ③ 임의 시드를 제거하면 결과가 매번 달라질 수 있는데, **모델끼리 공정하게 비교할 때는 같은 시드를 유지**합니다.

---

## 8. 자주 하는 실수

```
❌ GPU가 당연히 있다고 가정       → torch.cuda.is_available()로 먼저 확인
❌ CUDA·cuDNN을 항상 먼저 설치     → CPU만 쓰면 불필요, 공식 설치 선택기 우선
❌ 전체(또는 train 전체)로 정규화  → train에만 fit, val·test는 transform만
❌ 분류 출력에 MSE 사용           → sigmoid 출력은 binary_crossentropy
❌ 범주형 코드를 그냥 표준화        → 연속형만 표준화, 범주형은 원-핫
❌ 활성화 함수 없이 층만 쌓기       → 비선형성 사라져 1층과 동일
❌ 작은 데이터에 초대형 모델        → 과적합, EarlyStopping·작은 모델로 시작
❌ test로 하이퍼파라미터 튜닝       → 비교는 validation, test는 마지막 1회
❌ sigmoid가 항상 최선이라 믿기    → 은닉층 기본은 ReLU, sigmoid는 출력층용
❌ 보스턴 집값을 기본 데이터로      → sklearn에서 제거됨, 대체 데이터 사용
❌ 교실 정확도를 임상 성능으로 오해  → 의료 적용엔 별도 검증 필요
```

---

## 9. 딥러닝 DAY1 핵심 정리

```
✅ AI ⊃ 머신러닝 ⊃ 딥러닝
   딥러닝 = 은닉층을 깊게 쌓은 인공신경망, 유용한 특징을 비교적 자동으로 학습

✅ 신경망 학습 = 순전파 → 손실 계산 → 역전파 → 가중치 갱신 (반복)
   활성화 함수 없으면 깊이 쌓아도 1차식 / 깊은 sigmoid·tanh는 기울기 소실↑
   ReLU는 이를 '완화'(완전 제거 아님)

✅ 역사(강의 단순화 + 정정)
   인공뉴런(1943) → 퍼셉트론(1958) → XOR 한계/AI겨울(1969)
   → 역전파 확산(1980s~)·심층학습 → AlexNet(2012) → Transformer(2017) → ChatGPT(2022)
   한 사람·한 사건의 발명이 아니라 누적된 발전

✅ 환경 = 프레임워크(PyTorch / TensorFlow·Keras) + (선택)GPU·CUDA·cuDNN
   CPU만 쓰면 CUDA·cuDNN 불필요 / Windows 최신 TF-GPU는 WSL2
   Keras 3는 멀티 백엔드(TF·JAX·PyTorch) / 부담되면 Colab

✅ DNN 실습 흐름
   데이터 확인(상자그림·산점도) → train/val/test 먼저 분할
   → 전처리는 train에만 fit(연속형 표준화·범주형 원-핫)
   → 모델 → 학습(EarlyStopping) → 학습곡선 해석 → test 1회 평가

✅ 과제 유형별 출력층·손실 (중요!)
   회귀   : 출력 활성화 없음  +  MSE          / 평가 MSE·MAE
   이진분류: 출력 sigmoid     +  binary_crossentropy / 평가 혼동행렬·F1

✅ 주의
   GPU 가정 금지 / 전처리는 train에만 fit / 범주형은 원-핫
   sigmoid는 출력층용 / 작은 데이터엔 작은 모델+조기종료
   보스턴 집값은 윤리 문제로 제거됨 / 의료 예제는 임상 성능 아님
```

---

## 🔗 참고 자료

- [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)
- [TensorFlow 설치와 Windows GPU 제한](https://www.tensorflow.org/install/pip)
- [Keras 3 소개 (멀티 백엔드)](https://keras.io/keras_3/)
- [UCI Heart Disease 데이터셋](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [scikit-learn `fetch_california_housing` (보스턴 대체)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [scikit-learn 1.1 `load_boston` 문서 (보스턴 제거 이유·윤리 문제)](https://scikit-learn.org/1.1/modules/generated/sklearn.datasets.load_boston.html)
- [Google Colab](https://colab.research.google.com/)
