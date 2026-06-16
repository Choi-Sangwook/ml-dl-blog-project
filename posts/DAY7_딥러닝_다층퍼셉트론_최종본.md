# 🧠 딥러닝 완전 입문 가이드 — DAY7. 다층 퍼셉트론(MLP)·옵티마이저·RNN·GAN

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: DAY1(개념·설치) · DAY2(신경망·계층구조) · DAY3(데이터 처리·데이터셋) · DAY4(네트워크 모델 설계) · DAY5(훈련·평가) · DAY6(모델 저장·활용)

> 💡 **이 글의 표기 약속 (꼭 읽어 주세요)**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 개념·수식·코드
> - 🟩 **(보충 / PyTorch 재구성 코드)** : 입문자를 위한 설명, 또는 PDF에 **설계도만 있고 코드가 없는** 부분(항공 LSTM·GAN)을 PyTorch로 다시 구현한 코드. **PDF 원문 코드가 아닙니다.**
> - **프레임워크는 전부 PyTorch로 통일**했습니다. PDF의 유일한 실제 코드(`nn.RNN`)가 PyTorch라서입니다. PDF는 항공 LSTM·GAN을 **Keras 용어**(`predict`, `train_on_batch`)로 설명하지만, 이 글에서는 코드 일관성을 위해 **PyTorch로 옮겼고 그 사실을 코드마다 표시**했습니다.
> - ⚠️ 작성 환경에 PyTorch가 없어 **PyTorch 코드는 실행하지 못했습니다.** 손실·정확도·예측값을 임의로 적지 않았습니다. PDF에 출력이 적힌 `nn.RNN` 예제만 그 값을 인용하고, **전처리(NumPy·scikit-learn)·shape은 직접 실행해 확인**했습니다. 나머지는 직접 실행해 확인하세요.

> 🟨 **설치 참고** — 이 글의 코드는 **PyTorch / torchvision**을 씁니다. 설치 명령은 OS(Windows/macOS/Linux), Python 버전, CPU/CUDA 여부에 따라 달라지므로 **게시 시점의 공식 설치 페이지**(<https://pytorch.org/get-started/locally/>)에서 본인 환경을 선택해 확인하세요. MNIST 예제는 `torchvision.datasets.MNIST(download=True)`로 **처음 실행 시 인터넷 연결**이 필요합니다.

---

## 1. 이번 DAY에서 배우는 것

이번 강의는 분량이 큽니다(79쪽). 딥러닝의 기본기와 두 가지 대표 응용을 한 번에 다룹니다.

1. **다층 퍼셉트론(MLP)** — 모든 딥러닝의 기본 구조, XOR로 보는 "은닉층이 필요한 이유"
2. **최적화 함수(Optimizer)** — 가중치를 어떻게 수정할지 정하는 알고리즘(SGD → Adam → AdamW)
3. **RNN** — 순서가 있는 데이터를 다루는 신경망(은닉 상태·BPTT·기울기 소실), *항공 여행자 수 예측*
4. **GAN** — 가짜를 만들어 내는 신경망, *손글씨 숫자 생성*

> 🟩 **(보충) 진행 방식** — PDF 순서를 따르되 각 주제를 **왜 필요한가 → 동작 원리 → 코드 → 해석 → 주의** 흐름으로 정리했습니다. RNN의 기본 예제는 **PDF에 실린 실제 PyTorch 코드**이고, 항공 LSTM·GAN은 **설계도를 옮긴 PyTorch 보충 코드**입니다.

---

## 2. 다층 퍼셉트론(MLP)

### 2.1 퍼셉트론과 "단층의 한계" 🟦 (강의 PDF)

**퍼셉트론(Perceptron)** 은 뉴런을 수학으로 흉내 낸 가장 단순한 신경망입니다. 입력에 가중치를 곱해 더하고(가중합), 활성화 함수를 적용해 출력합니다.

```text
입력(x1,x2,x3) → 가중치(w1,w2,w3) → 가중합 Σ → 활성화 함수 → 출력 y
```

가중합 `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`, 출력 `y = f(z)` 입니다(`b`=편향).

단층 퍼셉트론은 **직선 하나로 나눌 수 있는 문제만** 풉니다. AND·OR은 되지만 **XOR은 못 풉니다.**

| A | B | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

> 🟩 **(보충)** XOR에서 정답 `1`인 점 (0,1)·(1,0)과 정답 `0`인 점 (0,0)·(1,1)은 **직선 하나로 분리할 수 없습니다.** 이 비선형 문제를 풀려고 **은닉층을 추가한 것이 MLP**입니다.

### 2.2 MLP의 구조 🟦 (강의 PDF)

입력층과 출력층 사이에 **은닉층(Hidden Layer)** 을 하나 이상 둡니다.

```text
입력층 → 은닉층1 → 은닉층2 → ... → 출력층
(예) 입력 4 → 은닉 8 → 은닉 4 → 출력 2
```

- **입력층**: 데이터를 받음(붓꽃이면 꽃받침·꽃잎 길이/너비 → 입력 4개).
- **은닉층**: 입력의 특징을 추출. 많을수록 복잡한 패턴 학습.
- **출력층**: **이진 분류=노드 1개**, **다중 분류=클래스 수만큼**(setosa/versicolor/virginica → 3개).

한 층의 계산은 *가중합 → 활성화 함수* 입니다. PDF 표기로는 은닉층 `h = f(W·x + b)`, 출력층 `y = g(W·h + b)` 형태로 이어집니다.

### 2.3 활성화 함수 🟦 (강의 PDF)

> **MLP의 핵심.** 활성화 함수가 없으면 층을 쌓아도 하나의 선형식과 같아져 은닉층의 의미가 사라집니다.

| 함수 | 수식 | 특징 |
|---|---|---|
| **Sigmoid** | `1/(1+e⁻ˣ)` | 0~1, 확률 해석 가능, **기울기 소실** 발생 |
| **Tanh** | `(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)` | −1~1, Sigmoid보다 학습이 잘 되는 편 |
| **ReLU** | `max(0,x)` | 단순·빠름, 은닉층에서 널리 사용 |
| **Leaky ReLU** | `x>0:x, x≤0:0.01x` | "죽은 ReLU" 완화 |
| **Softmax** | `eᶻⁱ/Σeᶻʲ` | **다중 분류 출력층**, 합=1 |

> 🟩 **(보충)** 보통 **은닉층=ReLU 계열, 출력층=문제 유형에 맞게**(이진=Sigmoid, 다중=Softmax, 회귀=활성화 없음)를 씁니다. "어떤 함수가 항상 최선"은 아니며 데이터·문제에 따라 달라집니다.

### 2.4 학습은 한 사이클의 반복 🟦 (강의 PDF)

```text
순전파(예측) → 손실(MSE=회귀 / Cross Entropy=분류)
            → 역전파(기울기 계산) → 경사하강법(가중치 수정) → 반복
```

(순전파·역전파·경사하강의 자세한 수식은 DAY5·DAY6에서 다뤘습니다.)

### 2.5 Python 실습: PyTorch로 XOR 풀기 🟩 (보충 코드, PDF 원문 코드 아님)

> 🟩 PDF는 XOR을 "단층이 못 푸는 문제"로 소개만 합니다. 아래는 그 XOR을 MLP로 직접 푸는 보충 예제입니다. (프레임워크: **PyTorch**)

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

torch.manual_seed(42)   # 재현성: 모델을 만들기 "전"에 시드 고정

# 입력 (4,2), 정답 (4,1) — 이진 분류라 0/1 실수 라벨
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
y = torch.tensor([[0], [1], [1], [0]],             dtype=torch.float32).to(device)

model = nn.Sequential(
    nn.Linear(2, 8),   # 은닉층(입력 2 → 8)
    nn.ReLU(),         # 비선형성 부여
    nn.Linear(8, 1),   # 출력층(8 → 1), 활성화 없이 "로짓"을 그대로 출력
).to(device)

# 출력이 로짓이므로 손실은 BCEWithLogitsLoss (시그모이드+BCE 결합)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(2000):
    pred = model(X)                 # (4,1) 로짓
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    prob = torch.sigmoid(model(X))  # 추론할 때만 시그모이드로 확률화
    print(prob)
```

**무엇을 봐야 하나** — 학습이 잘 되면 `(0,1),(1,0)`의 예측 확률은 1에 가깝고 `(0,0),(1,1)`은 0에 가깝습니다. **은닉층 + 활성화 함수** 덕분에 단층이 못 풀던 XOR이 풀립니다. (네 점만으로 원리를 보는 예제라, 일반화 성능을 말하는 용도는 아닙니다.)

> ⚠️ **(실행 안내)** 작성 환경에 PyTorch가 없어 실행하지 못했습니다. 예측 숫자는 직접 실행해 확인하세요.

> 🟩 **PyTorch 출력층·라벨·손실 조합 (입문자가 가장 많이 헷갈리는 부분)**
> | 문제 | 모델 출력 | 라벨 | 손실 |
> |---|---|---|---|
> | 이진 분류 | raw logit `(B,1)` | 0/1 `float` | `nn.BCEWithLogitsLoss()` |
> | 다중 클래스 | class별 raw logits `(B, C)` | 정수 class index `torch.long` | `nn.CrossEntropyLoss()` |
> | 회귀 | 연속값 `(B,1)` | `float` | `nn.MSELoss()` / `nn.L1Loss()` |
>
> - `BCEWithLogitsLoss`는 내부에 시그모이드가 있으니 **모델 마지막에 `nn.Sigmoid()`를 넣지 않습니다.**
> - `CrossEntropyLoss`는 **raw logits를 받아 내부에서 log-softmax를 처리**합니다. 학습용 모델 마지막에 `Softmax`를 붙이지 마세요. 사람이 확률을 보고 싶을 때만 추론 단계에서 `torch.softmax(logits, dim=1)`을 적용합니다.

### 2.6 MLP의 장단점·활용 🟦 (강의 PDF)

- **장점**: 비선형·XOR 해결, 다양한 분류/회귀, 딥러닝의 기본 구조.
- **단점**: 데이터가 많아야 하고 학습이 길며 과적합 위험. **이미지엔 CNN, 시계열엔 RNN/LSTM**이 더 효율적.

| 분야 | 활용 예 |
|---|---|
| 이미지 | 손글씨 숫자 분류, 품질 검사 |
| 금융 | 대출 심사, 신용 평가, 사기 탐지 |
| 의료 | 질병 진단, 환자 분류 |
| 제조 | 불량품 판정, 예측 유지보수 |

---

## 3. 최적화 함수(Optimizer)

### 3.1 왜 필요한가 🟦 (강의 PDF)

신경망은 처음에 가중치를 **무작위**로 둡니다(실제값 100인데 예측 20처럼 오차가 큼). 학습을 반복하며 `20 → 45 → 70 → 90 → 99`처럼 정답에 다가가도록 **가중치를 조금씩 수정**하는 일이 옵티마이저입니다.

기본은 **경사하강법**: 손실이 줄어드는 방향으로 이동합니다.

```text
w_new = w_old − η × (∂L/∂w)      η = 학습률(Learning Rate)
```

**학습률**은 한 걸음의 크기입니다. 너무 작으면(0.0000001) 매우 느리고, 너무 크면(10) 최적점을 지나쳐 **발산**, 적절하면(0.001~0.0001) 안정적으로 수렴합니다. 🟦

### 3.2 데이터를 얼마나 보고 갱신할까 🟦 (강의 PDF)

| 방식 | 한 번 갱신에 쓰는 데이터 | 특징 |
|---|---|---|
| **Batch GD** | 전체 | 정확·안정적이나 느리고 메모리 큼 |
| **SGD** | 1개씩 | 빠르고 지역최적 탈출, 진동·불안정 |
| **Mini-Batch GD** | 32개 등 일부 | 속도·안정성 균형, GPU 활용 — 가장 흔히 사용 |

> 🟩 **(보충)** 데이터 10,000개를 배치 32로 나누면 한 바퀴(1 **epoch**)에 약 313번 갱신합니다. "epoch=전체 몇 바퀴, batch=한 번에 보는 묶음, step=한 번의 갱신".

### 3.3 발전된 옵티마이저들 🟦 (강의 PDF)

| Optimizer | 핵심 아이디어 |
|---|---|
| **Momentum** | 이전 이동 방향 기억(관성) → 진동↓·수렴↑ |
| **NAG** | 이동할 위치를 미리 예측해 기울기 계산 |
| **AdaGrad** | 파라미터별 학습률. **누적 제곱 기울기** 때문에 학습률이 계속 줄어 나중엔 거의 멈춤 |
| **RMSProp** | AdaGrad의 단점을 완화 — 누적 대신 **최근 기울기 평균**만 반영. RNN에 효과적 |
| **Adam** | **Momentum + RMSProp 결합.** 기본값 `lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8` |
| **AdamW** | Adam의 weight decay 개선. Transformer 계열에서 많이 쓰임 |
| **Lion** | Google 제안, Adam보다 메모리 적게 사용 |

> 🟩 **(보충) 실무 선택** — 보통 **Adam/AdamW로 시작**하고, 이미지 분류처럼 일반화가 중요하면 **SGD+Momentum**도 많이 씁니다. "표준/최고/최신" 같은 표현은 시점에 따라 달라지니 절대적 기준으로 받아들이지 않는 게 좋습니다.

### 3.4 PyTorch에서 옵티마이저 바꾸기 🟩 (보충)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)               # Adam
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)          # RMSProp
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # SGD+Momentum
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)            # AdamW
```

---

## 4. RNN — 순서가 있는 데이터

> 🟦 PDF 실습 주제: **"RNN으로 항공 여행자 수 예측하기"**

### 4.1 RNN이 필요한 이유 🟦 (강의 PDF)

**RNN(Recurrent Neural Network)** 은 **순서(Sequence)** 가 있는 데이터를 위한 신경망입니다. 일반 신경망(DNN)은 입력들이 서로 독립이라고 보지만, RNN은 **이전 입력의 정보를 기억**해 다음 처리에 씁니다.

> 🟦 예) "나는 → 학교에 → 갔다"는 순서가 바뀌면 의미가 달라집니다. 일반 DNN은 단어 순서를 기억하지 못하지만, RNN은 *이전 단어 정보 + 현재 단어 정보*로 다음을 계산합니다.

활용은 시계열뿐이 아닙니다. PDF는 **자동 이미지 캡셔닝(CNN으로 인식→RNN으로 문장 생성), 기계 번역, 문장 생성, 검색, 감성 분석, 음성 인식·합성, 주가·환율 예측, 센서/이상 탐지** 등을 예로 듭니다.

### 4.2 RNN의 구조와 은닉 상태 🟦 (강의 PDF)

RNN은 **은닉층의 출력이 다시 자기 자신에게 입력**됩니다. 이 되먹임을 **순환(loop)** 이라 합니다. 시간축으로 펼쳐(unroll) 보면 이렇게 이어집니다.

```text
h0 → (x1→h1→y1) → (x2→h2→y2) → (x3→h3→y3) → ...   (h = 은닉 상태가 다음으로 전달)
```

RNN의 핵심은 **은닉 상태(Hidden State)** `h_t` 입니다.

```text
h_t = f(W_xh · x_t + W_hh · h_{t-1} + b)      ← 현재 입력 + 이전 기억
y_t = g(W_hy · h_t + b_y)                      ← 그 시점의 출력
```

| 기호 | 의미 | 기호 | 의미 |
|---|---|---|---|
| `x_t` | 현재 입력 | `W_xh` | 입력 가중치 |
| `h_{t-1}` | 이전 상태 | `W_hh` | **순환 가중치** |
| `b` | 편향 | `f` | 활성화 함수 |

### 4.3 BPTT와 기울기 소실·폭주 🟦 (강의 PDF)

RNN은 시간 방향으로 역전파하므로 **BPTT(Backpropagation Through Time)** 라고 합니다. 그런데 역전파에서 `∂h_t/∂h_{t-1}` 이 **계속 곱해집니다.**

- **기울기 소실(Vanishing)**: `0.5 × 0.5 × 0.5 × ... → 0`에 가까워짐 → **오래된 정보가 사라져 장기 기억 불가.**
- **기울기 폭주(Exploding)**: `2 × 2 × 2 × ... → 1024, 4096, 16384 ...`로 커짐 → **학습 불안정.**

> 🟩 **(보충, 직접 계산)** `0.5¹⁰ ≈ 0.00098`, `2¹⁴ = 16384` — PDF 예시와 같은 방향임을 직접 확인했습니다. 곱이 1보다 작으면 0으로, 1보다 크면 폭발적으로 커집니다.

### 4.4 RNN 유형과 LSTM·GRU 🟦 (강의 PDF)

기울기 소실(장기 의존성) 문제를 해결하려고 등장한 것이 **LSTM(Long Short-Term Memory)** (기억할/버릴/출력할 정보를 **게이트**로 제어)과 이를 단순화한 **GRU(Gated Recurrent Unit)** 입니다.

| 유형 | 입력→출력 | 예시 |
|---|---|---|
| One-to-One | 1 → 1 | 일반 신경망 |
| One-to-Many | 1 → 여러 개 | 이미지 캡셔닝 |
| Many-to-One | 여러 개 → 1 | 감성 분석, **항공 여행자 예측** |
| Many-to-Many | 여러 개 → 여러 개 | 기계 번역 |

### 4.5 Python 실습: PyTorch `nn.RNN` 🟦 (강의 PDF 실제 코드)

> 🟦 이 코드는 **PDF에 실제로 실린 PyTorch 예제**입니다(p47~48). — 이 글에서 PDF 원문 코드는 이 블록 하나뿐입니다.

```python
import torch
import torch.nn as nn

input_size = 10     # 한 시점 입력 특성 수
hidden_size = 20    # 은닉 상태 크기

rnn = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_first=True,        # 입력 형태를 (배치, 시퀀스, 특성)으로
)

# 배치 4, 시퀀스 길이 5, 특성 10
x = torch.randn(4, 5, 10)

output, hidden = rnn(x)
print(output.shape)   # torch.Size([4, 5, 20])
print(hidden.shape)   # torch.Size([1, 4, 20])
```

**출력 (PDF에 적힌 값):**

```text
torch.Size([4, 5, 20])     # output: 모든 시점의 은닉 상태
torch.Size([1, 4, 20])     # hidden: 마지막 시점의 은닉 상태
```

**형태 해석** 🟩 (직접 검증)
- `output (4, 5, 20)` = `(배치 4, 시퀀스 5, 은닉 20)` → **각 시점마다** 은닉 상태가 하나씩.
- `hidden (1, 4, 20)` = `(층×방향 1, 배치 4, 은닉 20)` → **마지막 시점**의 은닉 상태.
- `batch_first=True`라서 배치가 맨 앞으로 옵니다.

> 🟩 **(보충)** PDF도 지적하듯, 기본 RNN은 기울기 소실·폭주 때문에 실제로는 **LSTM·GRU**, 최근에는 **Transformer** 계열이 더 널리 쓰입니다.

### 4.6 실전: 항공 여행자 수 예측 (PyTorch LSTM) 🟩 (보충 코드, PDF 원문 코드 아님)

> 🟩 **PDF 설계 / PyTorch 보충 코드 구분** — PDF는 이 항공 여행자 예측을 **Keras 용어**(`predict`)로 *설명*하지만 **완전한 실행 코드를 싣지는 않습니다.** 아래 코드는 PDF의 설계도를 **PyTorch로 다시 구현한 보충 예제**이며, PDF 원문 코드가 아닙니다.

**데이터 배경** 🟦 — *AirPassengers*는 1949~1960년 **월별 항공 승객 수**로 총 **144개 시점**입니다. 해가 갈수록 증가하는 **추세**와 매년 반복되는 **계절성**이 함께 보이는 전형적인 시계열입니다. 목표는 **과거 12개월을 보고 13번째 달을 맞히는 지도학습(supervised) 문제**로 바꾸는 것입니다.

**설계(PDF)**: MinMax 정규화 → 12개월 윈도우 → LSTM → 출력 → 예측 후 `inverse_transform`으로 원래 단위 복원 → seaborn으로 시각화. PDF는 입력 정규화 범위 `[0,1]`에 맞춰 출력에 **sigmoid**를 쓰는 설계를 보여줍니다.

> 🟩 **(보충) 두 가지 중요한 수정 이유**
> 1. **데이터 누수 방지** — 시계열은 **먼저 시간순으로 train/val/test를 나누고, `MinMaxScaler`는 train 구간에만 `fit`** 해야 합니다. 전체 시계열에 `fit_transform`을 하면 미래(test)의 최소·최댓값이 학습 전처리에 새어 들어갑니다.
> 2. **출력은 sigmoid가 아니라 선형** — train 구간 기준으로만 스케일링하면 **미래 값이 1을 넘을 수 있습니다**(이 예시 데이터에서 test 정규화 값이 약 `1.44`까지 커지는 것을 직접 확인했습니다). sigmoid는 출력을 `[0,1]`로 막아 버려 이런 값을 예측하지 못합니다. 그래서 **회귀 예제에서는 `nn.Linear(hidden, 1)` 선형 출력 + `MSELoss`** 가 안전합니다. (PDF의 `sigmoid + MSE`는 "전체 정규화 [0,1]" 설계에서의 재현입니다.)

먼저 **전처리**는 프레임워크가 필요 없어 **직접 실행해 확인**했습니다.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 실제로는 Kaggle 'Air Passengers'의 AirPassengers.csv를 읽습니다.
#   import pandas as pd
#   df = pd.read_csv("AirPassengers.csv")          # 컬럼: Month, #Passengers
#   series = df["#Passengers"].values.astype("float32").reshape(-1, 1)
# 여기서는 shape 확인용 144개 예시 시계열을 사용합니다.
series = np.arange(1, 145, dtype="float32").reshape(-1, 1)   # (144, 1)
W = 12
n = len(series)

# 1) 시간순 분할 (셔플 금지) — 미래 정보가 train 전처리에 들어가지 않게 먼저 나눈다
train_end, val_end = int(n * 0.7), int(n * 0.85)
train_raw = series[:train_end]
val_raw   = series[train_end - W:val_end]   # 경계에서 윈도우를 만들 수 있게 W만큼 겹침
test_raw  = series[val_end - W:]

# 2) 스케일러는 train에만 fit, val/test는 transform만
scaler = MinMaxScaler()
train_s = scaler.fit_transform(train_raw)
val_s   = scaler.transform(val_raw)
test_s  = scaler.transform(test_raw)

def make_windows(values, window=12):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window, 0])   # 과거 12개월
        y.append(values[i + window, 0])     # 13번째 달
    X = np.array(X, dtype="float32").reshape(-1, window, 1)
    y = np.array(y, dtype="float32").reshape(-1, 1)
    return X, y

X_train, y_train = make_windows(train_s, W)
X_val,   y_val   = make_windows(val_s, W)
X_test,  y_test  = make_windows(test_s, W)
print(X_train.shape, X_val.shape, X_test.shape)
```

**실행 결과(확인됨):**

```text
(88, 12, 1) (22, 12, 1) (22, 12, 1)
```

이제 **LSTM 모델**입니다(PyTorch 보충 코드).

```python
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def to_t(a):
    return torch.tensor(a, dtype=torch.float32, device=device)

Xtr, ytr = to_t(X_train), to_t(y_train)
Xva, yva = to_t(X_val),   to_t(y_val)
Xte      = to_t(X_test)

class LSTMRegressor(nn.Module):
    def __init__(self, hidden=300):                 # PDF 설계: 셀 차원 300
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)        # out: (n, 12, 300)
        last = out[:, -1, :]         # 마지막 시점만 사용 → (n, 300)
        return self.fc(last)         # 회귀는 선형 출력 (sigmoid 아님)

model = LSTMRegressor().to(device)
criterion = nn.MSELoss()                                   # 연속값 예측 → MSE
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)   # PDF: RMSProp

for epoch in range(300):
    model.train()
    loss = criterion(model(Xtr), ytr)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (epoch + 1) % 50 == 0:                # 검증 손실로 과적합 여부 확인
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xva), yva)
        print(f"epoch {epoch+1}: train={loss.item():.4f}, val={val_loss.item():.4f}")

# 예측 → 역정규화로 "원래 여행자 수" 단위 복원 → 회귀 지표
model.eval()
with torch.no_grad():
    pred = scaler.inverse_transform(model(Xte).cpu().numpy())
truth = scaler.inverse_transform(y_test)

mae = mean_absolute_error(truth, pred)
rmse = mean_squared_error(truth, pred) ** 0.5
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# 시각화: 예측 vs 실제 (seaborn lineplot은 1차원 입력 → flatten)
import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x=range(len(pred)),  y=pred.flatten(),  label="pred")
sns.lineplot(x=range(len(truth)), y=truth.flatten(), label="truth")
plt.title("Air Passengers: pred vs truth")
plt.show()
```

**무엇을 봐야 하나** — 손실 그래프가 그럴듯해 보여도 그것만으로 판단하면 안 됩니다. **검증 손실(val)** 이 함께 줄어드는지로 과적합을 점검하고, 원래 단위로 되돌린 **MAE/RMSE**(예: "평균 ±몇 명 오차")로 성능을 해석합니다. 마지막으로 그래프로 추세·계절성을 따라가는지 봅니다.

> ⚠️ **(실행 안내)** PyTorch 미설치로 모델 학습·예측은 실행하지 못했습니다(전처리·shape `88/22/22`만 실행 확인). 손실·MAE·RMSE를 임의로 적지 않았으니 직접 실행해 확인하세요. 데이터는 Kaggle `AirPassengers.csv`가 필요하며, 컬럼명(`Month`, `#Passengers`)은 환경에 따라 다를 수 있으니 확인하세요.

> 🟩 **(보충) Keras ↔ PyTorch 대응** — PDF의 Keras `model.predict()`에 해당하는 PyTorch 추론은 **`model.eval()` + `torch.no_grad()`** 조합입니다. 또 **MinMax는 값 범위를 [0,1]로 맞추지만 이상치(outlier)에 약합니다** — 극단값이 있으면 나머지 값이 좁은 구간에 몰립니다. 이상치가 많으면 다른 스케일러(예: 표준화·RobustScaler) 검토가 필요합니다.

---

## 5. GAN — 손글씨 숫자 생성

> 🟦 PDF 실습 주제: **"DNN-GAN으로 손글씨 모방하기"**

### 5.1 GAN의 아이디어 🟦 (강의 PDF)

**GAN(Generative Adversarial Network)** (Ian Goodfellow, 2014)은 **두 신경망의 경쟁**으로 학습합니다.

- **생성자(Generator) = 위조범**: 노이즈에서 **가짜 이미지**를 만들어 감별자를 속입니다.
- **감별자(Discriminator) = 탐정**: 이미지가 **진짜인지 가짜인지** 판별합니다.

minimax 목표를 풀어 보면, `D(x)`는 **진짜 이미지를 진짜로 판단할 확률**, `D(G(z))`는 **생성 이미지를 진짜로 착각할 확률**입니다. 감별자는 `D(x)`는 1, `D(G(z))`는 0에 가깝게(잘 맞히도록) 학습하고, 생성자는 `D(G(z))`를 1에 가깝게(속이도록) 학습합니다.

> 🟦 GAN의 응용으로 PDF는 스케치→실물 제품, 얼굴 속성(나이) 변환, 도메인 변환(사진↔그림), 이미지 캡션, video-to-video 합성 등을 소개합니다.

### 5.2 DNN-GAN 설계도 🟦 (강의 PDF)

- **데이터**: MNIST 손글씨(28×28 흑백, 7만 장).
- **은닉층 활성화**: **Leaky ReLU**.
- **생성자**: `노이즈(100)` → `Dense(128)`→LeakyReLU → `Dense(128)`→LeakyReLU → `Dense(784)` → `Reshape(28,28,1)`
- **감별자**: `이미지(28,28,1)` → `Flatten(784)` → `Dense(128)`→LeakyReLU → `Dense(1, sigmoid)`(진짜일 확률)

### 5.3 학습 방식 🟦 (강의 PDF)

- **라벨**: `ones`(=1, 진짜) / `zeros`(=0, 가짜).
- 매 배치마다 **감별자 학습(진짜+가짜) → 생성자 학습**을 번갈아, 여러 epoch 반복.
- 생성자를 학습할 땐 **감별자를 고정**합니다.

> 🟩 **(보충) `train_on_batch` ↔ PyTorch 루프** — PDF의 Keras `train_on_batch`는 "한 배치로 한 번 가중치를 갱신"한다는 뜻입니다. PyTorch에서는 같은 일을 **배치 루프에서 직접 `loss.backward()` → `optimizer.step()`** 으로 작성합니다.

### 5.4 Python 실습: PyTorch DNN-GAN 🟩 (보충 코드, PDF 원문 코드 아님)

> 🟩 PDF는 GAN을 **Keras**로 설명하지만 코드는 없습니다. 아래는 **PyTorch 보충 구현**이며 PDF 원문 코드가 아닙니다. GAN은 이번 글에서 가장 고급 주제이니 **전체 구조 이해**에 집중하세요(완전한 실전 튜토리얼로 확장하지 않습니다).

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
NOISE = 100

# 생성자: 노이즈(100) → 가짜 이미지(1,28,28)
generator = nn.Sequential(
    nn.Linear(NOISE, 128), nn.LeakyReLU(0.01),
    nn.Linear(128, 128),   nn.LeakyReLU(0.01),
    nn.Linear(128, 784),   nn.Sigmoid(),       # 픽셀값 [0,1]
).to(device)

# 감별자: 이미지(1,28,28) → 진짜일 확률(sigmoid)
discriminator = nn.Sequential(
    nn.Flatten(),                              # (B,1,28,28) → (B,784)
    nn.Linear(784, 128), nn.LeakyReLU(0.01),
    nn.Linear(128, 1),   nn.Sigmoid(),
).to(device)

criterion = nn.BCELoss()                       # 라벨 0/1 ↔ 확률 (PDF 설계에 맞춘 버전)
opt_g = torch.optim.Adam(generator.parameters(),     lr=2e-4)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

# MNIST [0,1] — 처음 실행 시 인터넷 다운로드 필요
loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)

for epoch in range(20):                        # 데모용(실제로는 더 많이)
    generator.train(); discriminator.train()
    for real, _ in loader:
        real = real.to(device)                 # (B,1,28,28), [0,1]
        b = real.size(0)
        ones  = torch.ones(b, 1, device=device)    # 진짜=1
        zeros = torch.zeros(b, 1, device=device)   # 가짜=0

        # (a) 감별자 학습 — 가짜는 detach()로 생성자까지 기울기가 가지 않게
        noise = torch.randn(b, NOISE, device=device)
        fake = generator(noise).view(-1, 1, 28, 28)
        d_loss = criterion(discriminator(real), ones) \
               + criterion(discriminator(fake.detach()), zeros)
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # (b) 생성자 학습 — 가짜를 "진짜(1)"로 속이도록
        noise = torch.randn(b, NOISE, device=device)
        fake = generator(noise).view(-1, 1, 28, 28)
        g_loss = criterion(discriminator(fake), ones)
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()

# 샘플 확인 (matplotlib)
import matplotlib.pyplot as plt
generator.eval()
with torch.no_grad():
    samples = generator(torch.randn(16, NOISE, device=device)).view(-1, 28, 28).cpu()
fig, axes = plt.subplots(4, 4, figsize=(4, 4))
for img, ax in zip(samples, axes.flatten()):
    ax.imshow(img, cmap="gray"); ax.axis("off")
plt.show()
```

**무엇을 봐야 하나** — 학습 초반에는 노이즈 같던 그림이, 진행될수록 **숫자 비슷한 형태**로 변하는지 봅니다. 입문 단계에서는 **샘플 이미지 시각 확인**이면 충분합니다(FID 같은 정량 평가는 선택 보충). GAN은 두 손실의 균형이 까다로워 결과가 들쭉날쭉할 수 있습니다.

> ⚠️ **(실행 안내)** PyTorch·torchvision 미설치로 실행하지 못했습니다. 생성 품질·손실을 임의로 적지 않았으니 직접 실행해 확인하세요. GAN 학습은 CPU에서 느리니 **GPU(Colab 등)** 를 권장합니다.

> 🟩 **(보충) 두 가지 핵심 주의**
> - **`fake.detach()`** — 감별자를 학습할 때 가짜 이미지에서 생성자 쪽으로 기울기가 흘러가면 안 됩니다. `detach()`로 끊어야 "감별자만" 갱신됩니다(= PDF의 "생성자 학습 시 감별자 고정"을 PyTorch로 구현한 것).
> - 위는 PDF 설계대로 감별자 끝에 `Sigmoid` + `BCELoss`를 썼습니다. 수치적으로는 **마지막 시그모이드를 빼고 `BCEWithLogitsLoss`** 를 쓰는 편이 더 안정적입니다(실전 버전).

---

## 6. 자주 하는 실수

- **MLP에 활성화 함수 누락** — 층을 쌓아도 선형 모델과 같아져 XOR을 못 풉니다.
- **다중분류에서 `Softmax`를 모델에 붙이고 `CrossEntropyLoss` 사용** — `CrossEntropyLoss`는 **raw logits**를 받아 내부에서 log-softmax를 처리합니다. 학습용 모델엔 softmax를 넣지 말고, 추론 확률이 필요할 때만 `torch.softmax`를 적용하세요.
- **시그모이드 + `BCEWithLogitsLoss` 중복** — 시그모이드가 두 번 적용됩니다.
- **시계열을 전체 정규화하거나 셔플** — 미래 정보가 새는 **데이터 누수**. 시간순 분할 + 학습 구간에만 `fit`.
- **회귀 출력에 무조건 sigmoid** — 정규화를 train 기준으로 하면 값이 1을 넘을 수 있어 sigmoid가 예측을 막습니다. 회귀에서는 **선형 출력**을 두는 경우가 많습니다.
- **RNN/LSTM 입력 차원 실수** — `(배치, 시퀀스, 특성)` 3차원, `batch_first` 설정 확인.
- **GAN에서 `detach()` 누락** — 감별자 학습 때 생성자까지 갱신되어 학습이 망가집니다.
- **모델·입력의 device 불일치** — `model.to(device)`와 입력 `.to(device)`를 함께. 실행하지 않은 손실·정확도를 단정적으로 적지 않기.

---

## 7. DAY7 핵심 정리

```text
MLP
  - 은닉층 + 활성화 함수로 비선형(XOR) 해결
  - 출력층/라벨/손실: 이진=로짓+BCEWithLogitsLoss / 다중=로짓+CrossEntropyLoss(softmax는 추론용) / 회귀=MSE

Optimizer
  - 경사하강법 + 학습률이 기본, Mini-Batch가 흔함
  - SGD→Momentum→AdaGrad→RMSProp→Adam→AdamW, 보통 Adam/AdamW로 시작

RNN
  - 은닉 상태 h_t = f(W_xh x_t + W_hh h_{t-1} + b)로 이전 정보를 기억
  - BPTT에서 기울기 소실(→0)·폭주(→∞) → LSTM/GRU로 보완
  - nn.RNN 출력: output(배치,시퀀스,은닉) / hidden(층, 배치, 은닉)
  - 항공 예측: 12개월→13번째(지도학습), 시간순 분할 + train만 정규화 fit,
              회귀는 선형 출력 + MSE, 예측 후 inverse_transform, MAE/RMSE로 평가

GAN
  - 생성자(위조범) vs 감별자(탐정)의 경쟁 학습 (D(x)=1, D(G(z))=0 쪽으로)
  - 생성자: 노이즈(100)→...→이미지(28,28) / 감별자: 이미지→확률(sigmoid)
  - ones/zeros 라벨, 번갈아 학습, 생성자 학습 땐 감별자 고정(detach)
```

> 다음 DAY 주제는 이번 강의 PDF에서 확인할 수 없어 따로 예고하지 않습니다.

---

## 참고 자료

- 강의 자료: `DAY7_딥러닝_다층퍼셉트론.pdf` (교과목 2 — 데이터 분석과 머신러닝/딥러닝, 단원 3)
- ※ 본문 중 **`nn.RNN` 예제만 PDF에 실린 실제 코드**이며, XOR·항공 LSTM·GAN 코드는 **PDF 설계를 PyTorch로 옮긴 보충 예제**입니다(PDF는 항공·GAN을 Keras로 설명).
- PyTorch 설치(환경별 선택) — <https://pytorch.org/get-started/locally/>
- PyTorch 공식 문서 — `torch.nn.RNN`, `torch.nn.LSTM`, `BCEWithLogitsLoss`, `CrossEntropyLoss`
- torchvision — `datasets.MNIST`
- scikit-learn — `MinMaxScaler`, `mean_absolute_error`, `mean_squared_error`
- 데이터셋 — Kaggle *Air Passengers* (`AirPassengers.csv`), MNIST
