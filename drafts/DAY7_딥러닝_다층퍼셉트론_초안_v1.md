<!--
============================================================
기획 문서 (드래프트 메타) — 게시 전 이 주석 블록은 제거 예정
============================================================

[작업 모드] Generate (Stage 1: 초안)
[1차 출처] sources/DAY7_딥러닝_다층퍼셉트론.pdf (총 79페이지)
[분야] 딥러닝 / [독자] 파이썬 기본은 알지만 딥러닝은 처음
[이전 편] DAY1~DAY6

[같은 DAY7의 다른 버전 주의]
  - 이 프로젝트에는 비슷한 파일이 있었음(DAY7_딥러닝_다중퍼셉트론.pdf, 72p) → 별도 초안(_다중_초안_v1.md) 존재.
  - 이번 파일은 "다층퍼셉트론(79p)"로 더 확장된 버전. 두 파일을 혼동하지 말 것. 이 초안은 79p 파일 기준.
  - 차이: 이 버전은 3장 RNN에 "텍스트 이론 + 실제 PyTorch nn.RNN 코드(p47~48)"가 추가됨.

[전 페이지 확인 — 이미지 전용 블록 전수 확인]
  - 텍스트 페이지: 1장 MLP(p3~12), 2장 Optimizer(p13~22), 3장 RNN 이론(p38~48).
  - 이미지 전용 블록: RNN 개념도(p23~37), 항공여행자 LSTM 설계 슬라이드(p49~59), GAN(p60~79). → 모두 렌더링해 확인함.
  - p49~59, p60~79 슬라이드는 "다중" 버전과 동일한 그림(항공 LSTM 설계·GAN 설계).

[중요: PDF의 코드/프레임워크 상황]
  - "실제 코드 텍스트"는 단 하나: 3장의 PyTorch nn.RNN 예제(p47~48). 출력값도 PDF에 명시됨(torch.Size([4,5,20]), [1,4,20]).
  - 항공여행자 LSTM(p49~59)·GAN(p60~79)은 "설계도/산문"만 있고 코드 없음. 산문에는 Keras predict / train_on_batch 등 Keras 용어가 등장.
  => 본 글은 "코드 일관성(항목6)"을 위해 실행 코드를 전부 PyTorch로 통일:
       · MLP XOR  : PyTorch 재구성(⚙️)
       · RNN 기본 : PDF 실제 PyTorch 코드(🟦) — 그대로
       · 항공 LSTM: PyTorch 재구성(⚙️) — "PDF는 Keras로 설명하나 일관성 위해 PyTorch로 옮김" 명시
       · GAN      : PyTorch 재구성(⚙️) — 동일 명시
  => 재구성 코드는 PDF 원문 코드라고 표기하지 않음(설계도를 옮긴 것).

[실행 환경 한계 — 임의 결과값 금지(항목8)]
  - 이 환경에 torch/tensorflow 미설치 → XOR·LSTM·GAN 코드는 "직접 실행" 안내, 손실/정확도/예측 수치 임의 기입 금지.
  - PDF nn.RNN 예제의 출력(shape)은 "PDF에 적힌 값"이며, 형태 논리도 직접 검증함(아래).
  - 직접 실행해 확인한 것(프레임워크 무관):
      · nn.RNN 출력 형태 논리: batch_first, x(4,5,10), hidden=20 → output(4,5,20), h_n(1,4,20)  [논리 검증]
      · 윈도잉: 144길이, W=12 → X(132,12,1), y(132,1)  [실행 확인]
      · 기울기 소실/폭주 수치: 0.5^10≈0.00098(→점점 0), 2^14=16384  [실행 확인, PDF 예시와 방향 일치]
  - sklearn 설치됨 / torch·tensorflow 미설치.

------------------------------------------------------------
PDF 페이지 → 핵심 주제 지도 (전 79페이지 확인)
------------------------------------------------------------
[1장 MLP] p3~12 (텍스트, "다중" 버전과 동일 내용)
  p3 퍼셉트론 구조·식 z=Σwx+b / p4 단층 한계·XOR 진리표 / p5 MLP 구조·입력층 / p6 은닉·출력층(이진1·다중3)
  p7 계산과정·활성화 도입 / p8 Sigmoid·Tanh·ReLU 식 / p9 LeakyReLU·Softmax·순전파 / p10 손실(MSE/CE)·역전파
  p11 경사하강·장점 / p12 단점·활용·요약
[2장 Optimizer] p13~22 (텍스트)
  p13 정의·필요성 / p14 학습률(0.0000001/10) / p15 적절(0.001)·Batch GD / p16 SGD·Mini-Batch / p17 Mini-Batch장점·Momentum
  p18 NAG·AdaGrad / p19 RMSProp·Adam / p20 Adam설정값·단점 / p21 AdamW·Lion / p22 비교표·요약
[3장 RNN] p23~59
  p23 표지 "RNN으로 항공 여행자 수 예측"(자동 이미지 캡션 대회) / p24~37 (이미지)RNN·LSTM 개념도(unfold, LSTM셀/게이트 등)
  p38 RNN 정의·활용(NLP/번역/음성/주가/시계열) / p39 왜 필요(단어 순서)·RNN 구조
  p40 순환 루프·펼친 구조 / p41 은닉상태 식 h_t=f(W_xh x_t + W_hh h_{t-1}+b), 출력 y_t=g(W_hy h_t + b_y)
  p42 동작 예시 "I love deep learning" / p43 BPTT / p44 Vanishing(0.5×..→0.0001)·Exploding(2×..→16384)
  p45 LSTM·GRU·RNN유형(One-to-One/One-to-Many) / p46 Many-to-One/Many-to-Many·활용분야 / p47 [PyTorch nn.RNN 실제 코드]
  p48 출력 torch.Size([4,5,20])·[1,4,20] 해석·요약 / p49~59 (이미지)항공 LSTM 설계: MinMax·윈도잉·RMSprop·inverse_transform·seaborn
[4장 GAN] p60~79 (이미지, "다중" 버전과 동일)
  p60 표지 "DNN-GAN으로 손글씨 모방" / p61~72 GAN 응용·원리(생성자/감별자)·minimax·MNIST·LeakyReLU
  p73 생성자 설계도(100→Dense128→leakyR→Dense128→leakyR→Dense784→Reshape28×28×1)
  p74 감별자 설계도(28×28×1→Flatten784→Dense128→leakyR→Dense1 sigmoid)
  p75 DNN-GAN 연결 / p76~78 역전파·train_on_batch·라벨(ones/zeros) / p79 샘플 생성(matplotlib)·학습 요약

[검토 메모]
  - 분량 큼(4주제). 입문 유지: MLP·Optimizer는 개념+표+소형 실습, RNN은 이론+실제코드+항공LSTM 재구성, GAN은 개념+압축 재구성.
  - 출력층/라벨/손실(항목7): XOR=로짓+BCEWithLogitsLoss(추론시 sigmoid), LSTM=sigmoid[0,1]+MSE, GAN 감별자=sigmoid+BCE+ones/zeros.
  - device 처리: 모든 PyTorch 예제에 cuda/cpu 분기 포함.
  - 데이터 누수: 시계열 셔플 금지·시간순 분할, MinMax는 PDF대로 전체 fit하되 학습구간 fit 권장 주의.
  - 다음 DAY 주제 PDF에서 확인 불가 → 예고 안 함(항목9).
============================================================
-->

# 🧠 딥러닝 완전 입문 가이드 — DAY7. 다층 퍼셉트론(MLP)·옵티마이저·RNN·GAN

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: DAY1(개념·설치) · DAY2(신경망·계층구조) · DAY3(데이터 처리·데이터셋) · DAY4(네트워크 모델 설계) · DAY5(훈련·평가) · DAY6(모델 저장·활용)

> 💡 **이 글의 표기 약속 (꼭 읽어 주세요)**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 개념·수식·코드
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 사전지식·실무 주의점
> - ⚙️ **(설계도 재구성 코드)** : PDF에 **설계도만 있고 코드가 없는** 부분(항공 LSTM·GAN)을 옮겨 만든 코드. PDF 원문 코드가 아닙니다.
> - **프레임워크는 전부 PyTorch로 통일**했습니다. PDF의 유일한 실제 코드(RNN 예제)가 PyTorch라서입니다. (PDF는 항공 LSTM·GAN을 Keras 용어로 설명하지만, 이 글에서는 코드 일관성을 위해 **PyTorch로 다시 구현**하고 그 사실을 표시했습니다.)
> - ⚠️ 작성 환경에 PyTorch가 없어 **코드를 실행하지 못했습니다.** 손실·정확도·예측값을 임의로 적지 않았습니다. PDF에 출력이 적혀 있는 RNN 예제만 그 값을 인용하고, 형태(shape) 논리는 직접 검증했습니다. 나머지는 직접 실행해 확인하세요.

---

## 1. 이번 DAY에서 배우는 것

이번 강의는 분량이 큽니다(79쪽). 딥러닝의 기본기와 두 가지 대표 응용을 한 번에 다룹니다.

1. **다층 퍼셉트론(MLP)** — 모든 딥러닝의 기본 구조, XOR로 보는 "은닉층이 필요한 이유"
2. **최적화 함수(Optimizer)** — 가중치를 어떻게 수정할지 정하는 알고리즘(SGD → Adam → AdamW)
3. **RNN** — 순서가 있는 데이터를 다루는 신경망(은닉 상태·BPTT·기울기 소실), *항공 여행자 수 예측*
4. **GAN** — 가짜를 만들어 내는 신경망, *손글씨 숫자 생성*

> 🟩 **(보충) 진행 방식** — PDF 순서를 따르되 각 주제를 **왜 필요한가 → 동작 원리 → 코드 → 해석 → 주의** 흐름으로 정리했습니다. RNN의 기본 예제는 **PDF에 실린 실제 PyTorch 코드**이고, 항공 LSTM·GAN은 **설계도를 옮긴 PyTorch 재구성 코드**입니다.

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

### 2.3 활성화 함수 🟦 (강의 PDF)

> **MLP의 핵심.** 활성화 함수가 없으면 층을 쌓아도 하나의 선형식과 같아져 은닉층의 의미가 사라집니다.

| 함수 | 수식 | 특징 |
|---|---|---|
| **Sigmoid** | `1/(1+e⁻ˣ)` | 0~1, 확률 해석 가능, **기울기 소실** 발생 |
| **Tanh** | `(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)` | −1~1, Sigmoid보다 학습이 잘 되는 편 |
| **ReLU** | `max(0,x)` | 단순·빠름, 현재 널리 사용 |
| **Leaky ReLU** | `x>0:x, x≤0:0.01x` | "죽은 ReLU" 완화 |
| **Softmax** | `eᶻⁱ/Σeᶻʲ` | **다중 분류 출력층**, 합=1 |

> 🟩 **(보충)** "ReLU가 가장 많이 쓰인다"는 시점에 따라 달라질 수 있는 경향입니다. 보통 **은닉층=ReLU 계열, 출력층=문제 유형에 맞게**(이진=Sigmoid, 다중=Softmax, 회귀=활성화 없음).

### 2.4 학습은 한 사이클의 반복 🟦 (강의 PDF)

```text
순전파(예측) → 손실(MSE=회귀 / Cross Entropy=분류)
            → 역전파(기울기 계산) → 경사하강법(가중치 수정) → 반복
```

(순전파·역전파·경사하강의 자세한 수식은 DAY5·DAY6에서 다뤘습니다.)

### 2.5 Python 실습: PyTorch로 XOR 풀기 ⚙️ (재구성 코드)

> 🟩 PDF는 XOR을 "단층이 못 푸는 문제"로 소개만 합니다. 아래는 그 XOR을 MLP로 직접 푸는 보충 예제입니다. (프레임워크: **PyTorch**)

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 입력 (4,2), 정답 (4,1) — 이진 분류라 0/1 실수 라벨
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
y = torch.tensor([[0], [1], [1], [0]],             dtype=torch.float32).to(device)

torch.manual_seed(42)
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

**무엇을 봐야 하나** — 학습이 잘 되면 `(0,1),(1,0)`의 예측 확률은 1에 가깝고 `(0,0),(1,1)`은 0에 가깝습니다. **은닉층 + 활성화 함수** 덕분에 단층이 못 풀던 XOR이 풀립니다.

> ⚠️ **(실행 안내)** 작성 환경에 PyTorch가 없어 실행하지 못했습니다. 예측 숫자는 직접 실행해 확인하세요.

> 🟩 **출력층·라벨·손실 조합 점검(중요)** — `BCEWithLogitsLoss`는 내부에 시그모이드가 들어 있으므로 **모델 마지막에 `nn.Sigmoid()`를 넣지 않습니다.** 둘 다 넣으면 시그모이드가 두 번 적용됩니다. 확률이 필요한 추론에서만 `torch.sigmoid()`를 적용합니다.

### 2.6 MLP의 장단점·활용 🟦 (강의 PDF)

- **장점**: 비선형·XOR 해결, 다양한 분류/회귀, 딥러닝의 기본 구조.
- **단점**: 데이터가 많아야 하고 학습이 길며 과적합 위험. **이미지엔 CNN, 시계열엔 RNN/LSTM**이 더 효율적.
- **활용**: 손글씨 분류, 대출 심사·사기 탐지, 질병 진단, 불량품 판정 등.

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
| **Mini-Batch GD** | 32개 등 일부 | **가장 많이 사용**, 속도·안정성 균형, GPU 활용 |

> 🟩 **(보충)** 데이터 10,000개를 배치 32로 나누면 한 바퀴(1 **epoch**)에 약 313번 갱신합니다. "epoch=전체 몇 바퀴, batch=한 번에 보는 묶음, step=한 번의 갱신".

### 3.3 발전된 옵티마이저들 🟦 (강의 PDF)

| Optimizer | 핵심 아이디어 |
|---|---|
| **Momentum** | 이전 이동 방향 기억(관성) → 진동↓·수렴↑ |
| **NAG** | 이동할 위치를 미리 예측해 기울기 계산 |
| **AdaGrad** | 파라미터별 학습률(희소·NLP에 강함). 학습률이 계속 줄어 나중엔 거의 멈춤 |
| **RMSProp** | AdaGrad 개선 — 최근 기울기만 반영. RNN에 효과적 |
| **Adam** | **Momentum+RMSProp.** 기본값 `lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8` |
| **AdamW** | weight decay 개선. Transformer·BERT·GPT·ViT에서 표준 |
| **Lion** | Google 제안, Adam보다 메모리 적게 사용 |

> 🟩 **(보충) 실무 선택** — 보통 **Adam/AdamW로 시작**하고, 이미지 분류처럼 일반화가 중요하면 **SGD+Momentum**도 많이 씁니다. "어느 하나가 항상 최고"는 아닙니다.

### 3.4 PyTorch에서 옵티마이저 바꾸기 ⚙️ (보충)

```python
# 같은 모델에서 옵티마이저만 교체
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)          # Adam
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)     # RMSProp
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD+Momentum
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)       # AdamW
```

---

## 4. RNN — 순서가 있는 데이터

> 🟦 PDF 실습 주제: **"RNN으로 항공 여행자 수 예측하기"**

### 4.1 RNN이 필요한 이유 🟦 (강의 PDF)

**RNN(Recurrent Neural Network)** 은 **순서(Sequence)** 가 있는 데이터를 위한 신경망입니다. 일반 신경망(DNN)은 입력들이 서로 독립이라고 보지만, RNN은 **이전 입력의 정보를 기억**해 다음 처리에 씁니다.

> 🟦 예) "나는 → 학교에 → 갔다"는 순서가 바뀌면 의미가 달라집니다. 일반 DNN은 단어 순서를 기억하지 못하지만, RNN은 *이전 단어 정보 + 현재 단어 정보*로 다음을 계산합니다.

활용: 자연어 처리(번역·챗봇·문장 생성·감성 분석), 음성 인식·합성, 주가·환율 예측, 센서·시계열 분석.

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

- **기울기 소실(Vanishing)**: `0.5 × 0.5 × 0.5 × ... → 0`에 가까워짐(예: `0.5¹⁰ ≈ 0.001`). → **오래된 정보가 사라져 장기 기억 불가.**
- **기울기 폭주(Exploding)**: `2 × 2 × 2 × ... → 1024, 4096, 16384 ...`로 커짐 → **학습 불안정.**

> 🟩 **(보충, 직접 계산)** `0.5¹⁰ ≈ 0.00098`, `2¹⁴ = 16384` — PDF의 예시와 같은 방향임을 직접 확인했습니다. 즉 곱이 1보다 작으면 0으로, 1보다 크면 폭발적으로 커집니다.

### 4.4 RNN 유형과 LSTM·GRU 🟦 (강의 PDF)

기울기 소실(장기 의존성) 문제를 해결하려고 등장한 것이 **LSTM(Long Short-Term Memory)** (기억할/버릴/출력할 정보를 **게이트**로 제어)과 이를 단순화한 **GRU(Gated Recurrent Unit)** 입니다.

| 유형 | 입력→출력 | 예시 |
|---|---|---|
| One-to-One | 1 → 1 | 일반 신경망 |
| One-to-Many | 1 → 여러 개 | 이미지 캡셔닝 |
| Many-to-One | 여러 개 → 1 | 감성 분석, **항공 여행자 예측** |
| Many-to-Many | 여러 개 → 여러 개 | 기계 번역 |

### 4.5 Python 실습: PyTorch `nn.RNN` 🟦 (강의 PDF 실제 코드)

> 🟦 이 코드는 **PDF에 실제로 실린 PyTorch 예제**입니다(p47~48).

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
- `batch_first=True`라서 배치가 맨 앞으로 옵니다. (이 형태 논리는 직접 검증했습니다.)

> 🟩 **(보충)** PDF도 지적하듯, 기본 RNN은 기울기 소실·폭주 때문에 실제로는 **LSTM·GRU**, 그리고 최근에는 **Transformer** 계열이 더 널리 쓰입니다.

### 4.6 실전: 항공 여행자 수 예측 (PyTorch LSTM) ⚙️ (재구성 코드)

> 🟩 **프레임워크 안내** — PDF는 이 항공 여행자 예측을 **Keras 워크플로(Keras predict, MinMax, seaborn)** 로 *설명*하지만 **코드는 싣지 않습니다.** 이 글에서는 위 RNN 예제와 **코드 일관성을 맞추기 위해 PyTorch(LSTM)로 다시 구현**했습니다. 정규화(scikit-learn)·시각화(seaborn)는 프레임워크와 무관하게 그대로 씁니다.

PDF의 설계는 이렇습니다: **과거 12개월 → 13번째 달 예측**, **MinMax [0,1] 정규화**, **LSTM 셀 차원 300**, **출력 Sigmoid**(입력이 [0,1]이라서), 옵티마이저 **RMSProp**, 손실 **MSE**, 예측 후 **inverse_transform**으로 원래 단위 복원.

먼저 **데이터 준비**는 프레임워크가 필요 없어 **직접 실행해 확인**했습니다.

```python
# (직접 실행해 확인) 정규화 + 윈도우 묶기
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 실제로는 Kaggle 'Air Passengers'의 AirPassengers.csv(144개월)를 읽습니다:
#   import pandas as pd
#   series = pd.read_csv("AirPassengers.csv")["#Passengers"].values.astype("float32").reshape(-1,1)
series = np.arange(1, 145, dtype="float32").reshape(-1, 1)   # 형태 확인용 144개 예시

scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)          # [0,1], shape (144,1)

W = 12
X = np.array([scaled[i:i+W, 0] for i in range(len(scaled)-W)]).reshape(-1, W, 1)  # (132,12,1)
y = np.array([scaled[i+W, 0]   for i in range(len(scaled)-W)]).reshape(-1, 1)     # (132,1)
print("X:", X.shape, "y:", y.shape)
print("역정규화 복원 일치:", np.allclose(scaler.inverse_transform(scaled), series))
```

**실행 결과(확인됨):**

```text
X: (132, 12, 1) y: (132, 1)
역정규화 복원 일치: True
```

이제 **LSTM 모델**입니다(PyTorch 재구성).

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시계열은 순서가 중요 → 셔플하지 않고 시간 순서대로 분할
split = int(len(X) * 0.8)
X_train = torch.tensor(X[:split], dtype=torch.float32).to(device)   # (n,12,1)
y_train = torch.tensor(y[:split], dtype=torch.float32).to(device)   # (n,1)
X_test  = torch.tensor(X[split:], dtype=torch.float32).to(device)

class LSTMRegressor(nn.Module):
    def __init__(self, hidden=300):                 # PDF 설계: 셀 차원 300
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)        # out: (n, 12, hidden)
        last = out[:, -1, :]         # 마지막 시점만 사용 → (n, hidden)
        return torch.sigmoid(self.fc(last))   # 출력 [0,1] (입력 정규화 범위와 일치)

torch.manual_seed(42)
model = LSTMRegressor().to(device)
criterion = nn.MSELoss()                                  # 연속값 예측 → MSE
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)   # PDF: RMSProp

for epoch in range(300):
    model.train()
    loss = criterion(model(X_train), y_train)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# 예측 → 역정규화로 "원래 여행자 수" 단위 복원
model.eval()
with torch.no_grad():
    pred_scaled = model(X_test).cpu().numpy()
pred  = scaler.inverse_transform(pred_scaled)
truth = scaler.inverse_transform(y[split:])

# 시각화: 예측 vs 실제 (seaborn lineplot은 1차원 입력 → flatten)
import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x=range(len(pred)),  y=pred.flatten(),  label="pred")
sns.lineplot(x=range(len(truth)), y=truth.flatten(), label="truth")
plt.title("Air Passengers: pred vs truth")
plt.show()
```

**무엇을 봐야 하나** — 예측선이 실제선의 **증가 추세와 계절적 출렁임**을 얼마나 따라가는지 봅니다. 손실(MSE) 숫자 하나보다 **그래프로 비교**하는 편이 직관적입니다.

> ⚠️ **(실행 안내)** PyTorch 미설치로 모델 학습·예측은 실행하지 못했습니다(데이터 준비만 실행 확인). 손실·예측을 임의로 적지 않았으니 직접 실행해 확인하세요. 데이터는 Kaggle에서 `AirPassengers.csv`가 필요합니다.

> 🟩 **(보충) 데이터 누수·출력 조합 주의**
> - 위는 PDF처럼 **전체 시계열에 MinMax를 fit** 했지만, 더 엄밀히는 **학습 구간에만 `fit`** 하고 평가 구간엔 `transform`만 적용해야 미래 정보가 새지 않습니다. 시계열은 **셔플 금지**.
> - 입력을 [0,1]로 정규화했기에 출력도 `sigmoid`([0,1])로 맞췄습니다. **정규화 범위와 출력 활성화는 한 세트**입니다(표준화로 바꾸면 sigmoid는 부적절).

---

## 5. GAN — 손글씨 숫자 생성

> 🟦 PDF 실습 주제: **"DNN-GAN으로 손글씨 모방하기"**

### 5.1 GAN의 아이디어 🟦 (강의 PDF)

**GAN(Generative Adversarial Network)** (Ian Goodfellow, 2014)은 **두 신경망의 경쟁**으로 학습합니다.

- **생성자(Generator) = 위조범**: 노이즈에서 **가짜 이미지**를 만들어 감별자를 속입니다.
- **감별자(Discriminator) = 탐정**: 이미지가 **진짜인지 가짜인지** 판별합니다.

생성자는 더 그럴듯하게, 감별자는 더 날카롭게 — 서로를 밀어 올립니다(minimax: 감별자는 맞힐 확률 **최대화**, 생성자는 들킬 확률 **최소화**).

### 5.2 DNN-GAN 설계도 🟦 (강의 PDF)

- **데이터**: MNIST 손글씨(28×28 흑백, 7만 장).
- **은닉층 활성화**: **Leaky ReLU**.
- **생성자(p73)**: `노이즈(100)` → `Dense(128)`→LeakyReLU → `Dense(128)`→LeakyReLU → `Dense(784)` → `Reshape(28,28,1)`
- **감별자(p74)**: `이미지(28,28,1)` → `Flatten(784)` → `Dense(128)`→LeakyReLU → `Dense(1, sigmoid)`(진짜일 확률)

### 5.3 학습 방식 🟦 (강의 PDF)

- **라벨**: `ones`(=1, 진짜) / `zeros`(=0, 가짜).
- 매 배치마다 **감별자 학습(진짜+가짜) → 생성자 학습**을 번갈아, 여러 epoch 반복.
- 생성자를 학습할 땐 **감별자를 고정**합니다.

### 5.4 Python 실습: PyTorch DNN-GAN ⚙️ (재구성 코드)

> 🟩 **프레임워크 안내** — PDF는 GAN을 **Keras(train_on_batch)** 로 설명하지만 코드는 없습니다. 여기서는 **PyTorch로 재구성**했습니다. GAN은 이번 글에서 가장 고급 주제이므로 **전체 구조 파악**에 집중하세요.

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

criterion = nn.BCELoss()                       # 라벨 0/1 ↔ 확률
opt_g = torch.optim.Adam(generator.parameters(),     lr=2e-4)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

# MNIST [0,1]
loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)

torch.manual_seed(42)
for epoch in range(20):                        # 데모용(실제로는 더 많이)
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

**무엇을 봐야 하나** — 학습 초반에는 노이즈 같던 그림이, 진행될수록 **숫자 비슷한 형태**로 변하는지 봅니다. GAN은 두 손실의 균형이 까다로워 결과가 들쭉날쭉할 수 있습니다.

> ⚠️ **(실행 안내)** PyTorch 미설치로 실행하지 못했습니다(`torchvision`도 필요). 생성 품질·손실을 임의로 적지 않았으니 직접 실행해 확인하세요. GAN 학습은 CPU에서 느리니 **GPU(Colab 등)** 를 권장합니다.

> 🟩 **(보충) 두 가지 핵심 주의**
> - **`fake.detach()`** — 감별자를 학습할 때 가짜 이미지에서 생성자 쪽으로 기울기가 흘러가면 안 됩니다. `detach()`로 끊어야 "감별자만" 갱신됩니다(= PDF의 "생성자 학습 시 감별자 고정"을 PyTorch로 구현한 것).
> - 위는 설계도대로 감별자 끝에 `Sigmoid` + `BCELoss`를 썼습니다. 수치적으로는 **마지막 시그모이드를 빼고 `BCEWithLogitsLoss`** 를 쓰는 편이 더 안정적입니다.

---

## 6. 자주 하는 실수

- **MLP에 활성화 함수 누락** — 층을 쌓아도 선형 모델과 같아져 XOR을 못 풉니다.
- **출력층·라벨·손실 불일치** — 이진=로짓+`BCEWithLogitsLoss`(추론 시 sigmoid), 다중=Softmax+CrossEntropy, 회귀=MSE.
- **시그모이드 + `BCEWithLogitsLoss` 중복** — 둘 다 넣으면 시그모이드가 두 번 적용됩니다.
- **RNN/LSTM 입력 차원 실수** — `(배치, 시퀀스, 특성)` 3차원이어야 합니다. `batch_first` 설정도 확인.
- **시계열 셔플·전체 정규화 fit** — 미래 정보가 새는 데이터 누수. 시간순 분할 + 학습 구간에만 `fit`.
- **정규화 범위와 출력 활성화 불일치** — [0,1]엔 sigmoid, [-1,1]엔 tanh.
- **GAN에서 `detach()` 누락** — 감별자 학습 때 생성자까지 갱신되어 학습이 망가집니다.
- **모델·입력의 device 불일치** — `model.to(device)`와 입력 `.to(device)`를 함께. 실행하지 않은 손실·정확도를 단정적으로 적지 않기.

---

## 7. DAY7 핵심 정리

```text
MLP
  - 은닉층 + 활성화 함수로 비선형(XOR) 해결
  - 출력층/라벨/손실은 한 세트(이진=로짓+BCEWithLogits, 다중=Softmax+CE, 회귀=MSE)

Optimizer
  - 경사하강법 + 학습률이 기본, Mini-Batch가 표준
  - SGD→Momentum→AdaGrad→RMSProp→Adam→AdamW, 보통 Adam/AdamW로 시작

RNN
  - 은닉 상태 h_t = f(W_xh x_t + W_hh h_{t-1} + b)로 이전 정보를 기억
  - BPTT에서 기울기 소실(→0)·폭주(→∞) → LSTM/GRU로 보완
  - nn.RNN 출력: output(배치,시퀀스,은닉) / hidden(층, 배치, 은닉)
  - 항공 예측: 12개월→13번째, MinMax[0,1]+LSTM+sigmoid+MSE, 예측 후 inverse_transform, 시계열 셔플 금지

GAN
  - 생성자(위조범) vs 감별자(탐정)의 경쟁 학습
  - 생성자: 노이즈(100)→...→이미지(28,28) / 감별자: 이미지→확률(sigmoid)
  - ones/zeros 라벨, 감별자·생성자 번갈아 학습, 생성자 학습 땐 감별자 고정(detach)
```

> 다음 DAY 주제는 이번 강의 PDF에서 확인할 수 없어 따로 예고하지 않습니다.

---

## 참고 자료

- 강의 자료: `DAY7_딥러닝_다층퍼셉트론.pdf` (교과목 2 — 데이터 분석과 머신러닝/딥러닝, 단원 3)
- ※ 본문 중 **`nn.RNN` 예제만 PDF에 실린 실제 코드**이며, XOR·항공 LSTM·GAN 코드는 **PDF 설계를 옮겨 재구성한 PyTorch 예제**입니다(PDF는 항공·GAN을 Keras로 설명).
- PyTorch 공식 문서 — `torch.nn.RNN`, `torch.nn.LSTM`, `BCEWithLogitsLoss`
- scikit-learn 공식 문서 — `MinMaxScaler`, `inverse_transform`
- 데이터셋 — Kaggle *Air Passengers* (`AirPassengers.csv`), MNIST(`torchvision.datasets.MNIST`)
