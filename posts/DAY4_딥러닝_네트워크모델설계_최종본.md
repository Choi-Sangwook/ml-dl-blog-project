# 🧠 딥러닝 완전 입문 가이드 — DAY4. 딥러닝 네트워크 모델 설계

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: DAY1(개념·설치) · DAY2(신경망 알고리즘·계층구조) · DAY3(데이터 처리·데이터셋)
> **프레임워크**: 이 글의 코드는 모두 **PyTorch** 기준입니다. (PDF가 `nn.MSELoss`·`nn.CrossEntropyLoss` 등 PyTorch API를 사용)

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 내용
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 사전지식·실무 주의점
> - 손실 공식의 **숫자 예시(MSE·MAE·BCE·CE·ReLU)** 는 **NumPy로 실제 실행해 확인**한 값입니다. **PyTorch `nn.*` 코드는 직접 실행해 결과를 확인**하세요(자세한 실행 안내는 5장에). 임의의 결과는 넣지 않았습니다.

---

## 1. 이번 DAY에서 배우는 것

- 딥러닝 **네트워크 모델 설계**가 "층을 쌓는 것"을 넘어 무엇을 결정하는 과정인지
- **입력층 → 은닉층 → 출력층** 을 문제에 맞게 설계하는 법
- 이 글의 핵심: **문제 유형별 "출력층 · 라벨 형식 · 손실함수" 짝 맞추기** (정확한 shape·dtype 포함)
- 회귀(MSE·MAE·Huber), 이진(BCE), 다중(Cross Entropy), 다중 라벨, 그리고 NLL·KL 손실
- **최적화(Adam)** 와 **과적합 방지(Dropout·BatchNorm·EarlyStopping·증강)**, 손실 vs 평가지표
- 대표 딥러닝 모델(퍼셉트론 → MLP/DNN → CNN/RNN → LSTM/GRU → AutoEncoder/GAN → Transformer/BERT/GPT → ViT/Diffusion) **한눈에 보기**

---

## 2. 네트워크 모델 설계란 무엇인가 🟦

딥러닝 모델 설계는 단순히 층(Layer)을 여러 개 쌓는 작업이 아니라, **풀려는 문제와 데이터 특성을 분석해 문제에 적절한 신경망 구조를 정하는 과정**입니다. 설계에서 정해야 할 것은 계층 구조, 활성화 함수, 손실 함수, 최적화 알고리즘 등입니다.

### 2.1 첫 단계 — 문제 정의

무슨 문제냐에 따라 구조가 달라집니다.

| 문제 유형 | 예시 | 주로 쓰는 구조 |
|---|---|---|
| 분류(Classification) | 고양이/개 구분 | MLP·CNN |
| 회귀(Regression) | 집값·온도 예측 | MLP |
| 객체 탐지(Object Detection) | 사진 속 물체 위치 | CNN 기반 |
| 자연어 처리(NLP) | 번역·챗봇 | RNN·LSTM·Transformer |
| 이미지 생성 | 그림 생성 | GAN·Diffusion |

### 2.2 입력 데이터 분석과 입력층

입력 데이터의 종류에 맞는 구조를 고릅니다 — **표/수치 데이터 → MLP**, **이미지 → CNN**, **문장·시계열 → RNN·LSTM·GRU·Transformer**.

**표 형태 데이터를 MLP에 넣는 경우**, 입력층의 `in_features`는 **샘플당 특성(feature) 수**와 같습니다. 예를 들어 학생 한 명을 `(국어, 영어, 수학)` 3개 특성으로 표현하면 `in_features=3`입니다.

> 🟩 **(보충)** 이미지·문장·시계열은 다릅니다. 이미지는 보통 `(N, C, H, W)`, 시퀀스는 `(N, L, D)` 처럼 여러 축을 유지하므로 "입력 노드 수 = 특성 수"라는 규칙을 그대로 적용하지 않습니다. 이때는 입력 채널 수, 시퀀스 길이, 임베딩 차원 등 **구조에 맞는 입력 크기**를 설계합니다.

### 2.3 은닉층 — 특징을 추출하는 곳

은닉층(Hidden Layer)은 입력에서 **중요한 특징을 추출**합니다. 각 뉴런은 입력에 가중치를 곱하고 편향을 더한 값을 계산한 뒤, **활성화 함수**를 통과시켜 다음 층으로 보냅니다.

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

활성화 함수는 신경망에 **비선형성**을 줘서 복잡한 패턴을 학습하게 합니다. 대표적으로 Sigmoid·Tanh·ReLU·Leaky ReLU·ELU·Swish가 있고, 현재는 계산이 간단하고 기울기 소실이 적은 **ReLU**를 은닉층 기본으로 많이 씁니다.

$$
\text{ReLU}(x) = \max(0,\, x)
$$

> 🟩 **(보충)** 은닉층의 **개수·너비**는 문제 복잡도뿐 아니라 데이터 규모, 계산 자원, 과적합 정도, 사용할 아키텍처에 따라 달라집니다. 처음에는 **작은 기준 모델에서 시작해 검증(validation) 성능을 보며 늘리는** 편이 안전합니다. 또 "ReLU가 항상 최고"는 아닙니다 — 음수 입력에서 기울기가 0이라 일부 뉴런이 죽는 **Dead ReLU**가 생기면 Leaky ReLU 등을 시도합니다. (활성화 함수 자세한 비교는 DAY2 참고.)

### 2.4 설계의 전체 흐름 (12단계)

PDF는 설계 과정을 다음 순서로 정리합니다.

```
1. 문제 정의              7. 손실 함수 선택
2. 데이터 수집·분석        8. 최적화 알고리즘 선택
3. 입력층 설계            9. 학습률·하이퍼파라미터 설정
4. 은닉층 개수·노드 수     10. 과적합 방지 기법 적용
5. 활성화 함수 선택        11. 모델 학습·평가
6. 출력층 설계            12. 성능 개선·재설계
```

즉, **입력층에서 시작 → 은닉층으로 특징 학습 → 출력층에서 결과 생성 → 손실함수·역전파로 가중치를 반복 수정**하며 최적의 예측 모델을 만드는 전체 과정이 "네트워크 모델 설계"입니다.

---

## 3. 출력층과 손실함수는 항상 함께 설계한다 🟦 (이 글의 핵심)

출력층은 모델이 **어떤 형태의 답**을 내보낼지 정하고, 손실함수는 그 답이 **정답과 얼마나 다른지** 재는 기준입니다. 둘은 따로 생각하면 안 되고 **항상 짝으로** 설계합니다.

### 3.1 한눈에 보는 짝 맞추기 표

아래 shape에서 `N`은 배치 크기, `C`는 클래스 개수입니다.

| 문제 유형 | 모델의 raw 출력 | 라벨 형식 | PyTorch 손실함수 |
|---|---|---|---|
| **회귀(값 1개)** | `(N, 1)` float, 출력 활성화 없음 | `(N, 1)` float | `nn.MSELoss` / `nn.L1Loss` / `nn.HuberLoss` |
| **이진 분류** | logits `(N, 1)` float | `(N, 1)` float, 값 0~1 | `nn.BCEWithLogitsLoss` |
| **다중 분류** | logits `(N, C)` float | `(N,)` `torch.long`, 값 0~C-1 | `nn.CrossEntropyLoss` |
| **다중 라벨 분류** | logits `(N, C)` float | `(N, C)` float, 각 원소 0~1 | `nn.BCEWithLogitsLoss` |

`CrossEntropyLoss`에는 Softmax 전 logits를, `BCEWithLogitsLoss`에는 Sigmoid 전 logits를 전달합니다. 확률이 필요할 때만 평가·추론 단계에서 각각 `softmax` 또는 `sigmoid`를 적용합니다.

> 🟩 **(아주 중요) PyTorch에서 활성화는 "손실함수가 대신 계산"합니다**
> - `nn.CrossEntropyLoss` 는 **내부에 LogSoftmax** 가 들어 있습니다. → 모델 출력층에 Softmax를 **넣지 말고 raw 점수(logit)** 를 그대로 내보냅니다.
> - `nn.BCEWithLogitsLoss` 는 **내부에 Sigmoid** 가 들어 있습니다. → 출력층에 Sigmoid를 **넣지 마세요.**
> - 위 표의 활성화에 괄호를 친 이유가 이것입니다. Sigmoid·Softmax는 **학습이 끝난 뒤 사람이 확률로 해석할 때** 적용합니다. (Softmax 수식과 확률 해석은 DAY3에서 다뤘습니다.)

### 3.2 다중 분류의 라벨 형식 — One-Hot vs 정수 인덱스 🟩

PDF는 다중 분류 정답을 **One-Hot**(예: 정답이 2번 → `[0,0,1,0,0]`)으로 설명합니다. 이는 손실 **수식**을 이해하기 위한 표현입니다. 일반적인 단일 정답 다중 분류에서 PyTorch `nn.CrossEntropyLoss` 는 정답을 **정수 클래스 인덱스(`torch.long`, shape `(N,)`)** 로 받습니다. 즉 `[0,0,1,0,0]` 대신 정수 `2` 를 넘깁니다.

> 🟩 PyTorch는 label smoothing·mixup 같은 경우를 위해 **클래스 확률(soft label) 타깃**도 지원하지만, 이 글의 입문 예제는 **클래스 인덱스 방식**을 사용합니다. ([CrossEntropyLoss 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html))

---

## 4. 손실함수 하나씩 — 왜·어떻게·예시 🟦 + 🟩

아래 숫자 예시는 모두 **NumPy로 실제 실행해 확인**한 값입니다. (각 코드 블록은 필요한 `import`를 포함해 단독 실행할 수 있습니다.)

### 4.1 회귀 — MSE · MAE · Huber

**MSE (평균제곱오차)** — 회귀에서 대표적으로 쓰는 손실. 오차를 제곱해 평균 내므로 **큰 오차에 강한 벌점**을 줍니다.

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

**MAE (평균절대오차)** — 오차에 절댓값을 씌워 평균. 제곱하지 않으므로 **이상치에 덜 민감**합니다.

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i\rvert
$$

```python
import numpy as np

y, yhat1, yhat2 = 100, 90, 50
print((y-yhat1)**2, (y-yhat2)**2)   # 100 2500   ← MSE는 오차 10→100, 50→2500 (급증)
print(abs(y-yhat1), abs(y-yhat2))   # 10 50      ← MAE는 오차 10→10, 50→50 (선형)
```

**Huber Loss** — MSE와 MAE의 장점을 합친 손실. 오차가 작으면(MSE처럼) 제곱, 크면(MAE처럼) 절댓값을 써서 **이상치에 과하게 흔들리지 않습니다**($a=y-\hat{y}$, $\delta$는 경계).

$$
L_\delta(a) = \begin{cases} \frac{1}{2}a^2 & \lvert a\rvert \le \delta \\ \delta\left(\lvert a\rvert - \frac{1}{2}\delta\right) & \lvert a\rvert > \delta \end{cases}
$$

PyTorch에서 (부분 코드 — 손실함수만 발췌):

```python
import torch.nn as nn

criterion = nn.MSELoss()     # 평균제곱오차
criterion = nn.L1Loss()      # 평균절대오차(MAE)
criterion = nn.HuberLoss()   # Huber (이상치가 섞인 회귀에 유용)
```

### 4.2 이진 분류 — BCE

**BCE (이진 교차 엔트로피)** — 정답이 0 또는 1인 문제(스팸/정상, 질병 있음/없음 등). 단순히 맞고 틀림만 보는 게 아니라 **얼마나 확신 있게 맞혔는지**까지 평가합니다.

$$
\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\left[\,y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\,\right]
$$

> 🟩 위 수식의 $\hat{y}$ 는 **Sigmoid를 통과한 확률**입니다. 하지만 `nn.BCEWithLogitsLoss` 에는 확률이 아니라 **Sigmoid 전 logits**를 전달하며, 손실함수가 내부에서 Sigmoid와 BCE를 **수치적으로 안정적으로** 함께 계산합니다.

```python
import numpy as np

# 정답 y=1일 때, 예측 확률에 따른 BCE (한 샘플)
for p in [0.95, 0.05]:
    print(p, round(-np.log(p), 4))   # 0.95 -> 0.0513 (정답에 가까움, 손실 작음)
                                     # 0.05 -> 2.9957 (정답과 멀어 손실 큼)
```

```python
import torch.nn as nn
criterion = nn.BCEWithLogitsLoss()   # Sigmoid + BCE를 함께 계산 → 모델에 Sigmoid 불필요
```

### 4.3 다중 분류 — Cross Entropy

**Cross Entropy** — 여러 클래스 중 하나를 고르는 문제(숫자 0~9, 동물 5종 등)에서 대표적으로 씁니다. **정답 클래스에 준 확률**이 높을수록 손실이 작습니다.

$$
\text{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i) \qquad (C=\text{클래스 개수})
$$

정답이 One-Hot이면 정답 위치만 1이라, 실제로는 **정답 클래스의 예측 확률**만 손실에 반영됩니다.

```python
import numpy as np

# 정답 = 고양이(0번). 모델이 각 클래스에 준 확률에 따른 CE
for name, probs in [("good", [0.90, 0.07, 0.03]), ("bad", [0.05, 0.80, 0.15])]:
    p = np.array(probs)
    print(name, round(-np.log(p[0]), 4))   # good -> 0.1054 (정답에 0.90, 손실 작음)
                                            # bad  -> 2.9957 (정답에 0.05, 손실 큼)
```

```python
import torch.nn as nn
criterion = nn.CrossEntropyLoss()   # 내부에 LogSoftmax 포함 → 모델은 raw logits, 라벨은 정수 인덱스
```

### 4.4 다중 라벨 분류

다중 라벨은 **여러 정답이 동시에** 참일 수 있는 문제입니다(한 사진에 사람·자동차·강아지가 모두 있음). "하나만 고르기"가 아니라 **클래스마다 0/1을 따로 판단**하므로, 각 클래스에 Sigmoid를 적용한 이진 분류를 여러 개 푸는 셈입니다.

```python
import torch.nn as nn
criterion = nn.BCEWithLogitsLoss()   # 라벨마다 0/1 → 다중 라벨에도 사용
```

### 4.5 기타 손실 — NLL · KL (참고)

- **NLLLoss (음의 로그 가능도)**: $\text{NLL} = -\log P(y)$, 정답 클래스 확률이 낮을수록 손실↑. PyTorch에서는 **LogSoftmax 출력**과 함께 씁니다. 보통은 이 둘을 합친 `CrossEntropyLoss`를 더 많이 씁니다.
- **KLDivLoss (KL 발산)**: 두 확률분포 $P$(기준)·$Q$(예측)가 **얼마나 다른지** 측정. 지식 증류, 생성 모델(VAE) 등에 쓰입니다.

$$
D_{KL}(P\,\Vert\,Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}
$$

> 🟩 **(중요) `nn.KLDivLoss`의 PyTorch 사용법** — 수학식 $D_{KL}(P\Vert Q)$ 에서 $P$ 는 기준 분포, $Q$ 는 모델 분포지만, PyTorch 호출 순서는 일반 손실함수처럼 `criterion(model_output, target)` 입니다. 따라서 **첫 번째 인자에는 모델 분포 $Q$ 의 log-확률**, **두 번째 인자에는 기준 분포 $P$ 의 확률**을 넣습니다. 또 기본값 `reduction="mean"` 은 수학적 KL 값과 일치하지 않으므로, 배치 단위의 수학적 정의에 맞추려면 **`reduction="batchmean"`** 을 씁니다. ([KLDivLoss 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html))

```python
# ⚠️ torch 미설치 환경이라 실행하지 못함 — 직접 실행해 확인하세요.
import torch
import torch.nn as nn
import torch.nn.functional as F

logits        = torch.randn(3, 5)        # 모델의 raw 출력 Q
target_logits = torch.randn(3, 5)        # 예시용 기준 분포 P

log_q = F.log_softmax(logits, dim=1)     # input  : log-확률
p     = F.softmax(target_logits, dim=1)  # target : 확률

criterion = nn.KLDivLoss(reduction="batchmean")
loss = criterion(log_q, p)               # 인자 순서: (모델 log-확률, 기준 확률)
print(loss.item())
```

### 4.6 손실함수 선택 기준 (요약) 🟦

| 문제 유형 | 출력 형태 | 대표 손실함수 |
|---|---|---|
| 회귀 | 연속 숫자 | `MSELoss`, `L1Loss`, `HuberLoss` |
| 이진 분류 | 0 또는 1 | `BCEWithLogitsLoss` |
| 다중 분류 | 여러 클래스 중 하나 | `CrossEntropyLoss` |
| 다중 라벨 분류 | 여러 클래스 동시 | `BCEWithLogitsLoss` |
| 확률분포 비교 | 분포 vs 분포 | `KLDivLoss` |
| 로그 확률 기반 분류 | 로그 확률 | `NLLLoss` |

---

## 5. 하나로 합치기 — PyTorch 한 스텝 학습 구조 확인

아래 코드는 다중 분류에서 **입력 → 모델 → logits → 손실 → 역전파 → 가중치 갱신**이 어떻게 연결되는지 **한 스텝**으로 확인하는 예제입니다. 무작위로 만든 8개 샘플만 사용하므로 **실제 성능을 평가하는 완전한 학습 예제는 아닙니다.**

> 🟩 **실제 프로젝트에서는** 데이터를 먼저 **train/validation/test** 로 나눕니다. 모델은 train으로 학습하고, 구조·학습률·epoch 수와 Early Stopping 시점은 **validation**으로 선택하며, **test는 모든 선택이 끝난 뒤 최종 성능 확인에 한 번만** 사용합니다. 스케일러·결측치 대치처럼 데이터에서 값을 학습하는 전처리는 **train에만 `fit`** 하고 validation/test에는 `transform`만, **Data Augmentation도 보통 train에만** 적용해야 데이터 누수를 막습니다.

> **PyTorch 설치** — 설치 명령은 운영체제와 CPU/CUDA 환경에 따라 달라집니다. [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 OS·패키지 관리자·연산 플랫폼을 선택한 뒤 표시되는 명령을 사용하세요.

> ⚠️ **이 코드는 작성 환경에 torch가 없어 실행하지 못했습니다.** 손실 값 등 실제 수치는 직접 실행해 확인하세요. (텐서 shape·dtype처럼 확정적인 부분만 주석으로 표기했습니다.)

```python
# 프레임워크: PyTorch
# 설치 명령은 https://pytorch.org/get-started/locally/ 에서 환경별로 확인
import torch
import torch.nn as nn

# 1) 장치 안전 선택 (GPU 없으면 자동 CPU) — DAY1에서 다룬 패턴
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)                         # 재현성(완전한 일치는 아닐 수 있음)

# 2) 설정 가능한 MLP: 입력 → 은닉(여러 층, ReLU) → 출력(활성화 없음 = raw logits)
def build_mlp(in_features, hidden_sizes, out_features):
    layers, prev = [], in_features
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_features))   # 출력층: Softmax를 넣지 않음
    return nn.Sequential(*layers)

# 3) 다중 분류 예: 특성 20개 → 은닉 [64, 32] → 클래스 5개
model = build_mlp(20, [64, 32], out_features=5).to(device)

# 4) 가짜 데이터 (배치 8개)
X = torch.randn(8, 20, device=device)              # 입력  (8, 20) float32
y = torch.randint(0, 5, (8,), device=device)       # 정답  (8,)   int64(long) = 클래스 인덱스

# 5) 손실·옵티마이저
criterion = nn.CrossEntropyLoss()                  # 내부 LogSoftmax 포함 → 모델은 logits
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6) 한 스텝 학습 (순전파 → 손실 → 역전파 → 갱신)
model.train()
logits = model(X)                                  # 출력 (8, 5) raw logits
loss = criterion(logits, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(logits.shape, logits.dtype)   # torch.Size([8, 5]) torch.float32
print(y.shape, y.dtype)             # torch.Size([8]) torch.int64
print(loss.item())                  # 손실 값(스칼라) — 실행해서 확인
```

**텐서 모양(shape)·자료형(dtype) 점검**

```
입력 X    : (8, 20),  float32
출력 logits: (8, 5),   float32   # 클래스 5개 → raw 점수 5개 (Softmax 미적용)
정답 y    : (8,),     int64(long) # 클래스 '인덱스' (One-Hot 아님)
손실      : CrossEntropyLoss  ← (N,C) logits + (N,) long 라벨 짝
장치      : model과 X·y가 같은 device에 있어야 함
```

회귀·이진 분류로 바꾸려면 **출력층 크기와 손실함수, 라벨 형식만** 바꾸면 됩니다(부분 코드).

```python
import torch.nn as nn
# 회귀 head (부분): 출력 1, 활성화 없음 / 라벨은 실수 (N,1)
reg_head, reg_loss = nn.Linear(32, 1), nn.MSELoss()        # 또는 nn.L1Loss(), nn.HuberLoss()

# 이진 분류 head (부분): 출력 1 logit / 라벨은 0,1 float (N,1) / 모델에 Sigmoid 넣지 않음
bin_head, bin_loss = nn.Linear(32, 1), nn.BCEWithLogitsLoss()
```

### 5.1 평가·추론할 때 — 모드 전환과 확률 변환

학습이 끝나면 **평가 모드**로 바꿔 검증·추론합니다. (위 `model`·`criterion`을 이어서 사용한다고 가정한 부분 코드입니다.)

```python
# 검증·평가 단계: Dropout·BatchNorm을 평가 모드로 전환
model.eval()
with torch.no_grad():                       # gradient 계산 끄기
    logits = model(X)                        # raw logits
    probabilities = torch.softmax(logits, dim=1)   # 추론 시에만 확률로 변환
    predicted_class = probabilities.argmax(dim=1)  # 가장 큰 확률의 클래스
```

> 🟩 **(보충)** `model.eval()` 은 Dropout·BatchNorm 등의 **동작 모드만** 바꾸고 gradient 계산까지 끄지는 않습니다. 그래서 평가 때는 `torch.no_grad()`(또는 `torch.inference_mode()`)를 **함께** 써야 메모리·속도에 유리합니다. 또 손실 계산에는 Softmax를 넣지 않지만, 위처럼 **추론 단계에서만** `softmax`로 확률을 만들어 해석합니다.
>
> 재현성 주석의 `torch.manual_seed(42)` 도 **완전한 일치를 보장하지는 않습니다.** GPU 연산·라이브러리 버전·비결정적 연산 때문에 결과가 조금 달라질 수 있습니다.

---

## 6. 최적화와 과적합 방지 🟦 + 🟩

### 6.1 최적화 알고리즘(Optimizer)

손실을 줄이는 방향으로 가중치를 갱신하는 규칙입니다. 가장 기본 식은 다음과 같습니다($\eta$는 학습률).

$$
w_{\text{new}} = w_{\text{old}} - \eta\,\frac{\partial L}{\partial w}
$$

대표적으로 **SGD · Momentum · RMSProp · Adam** 이 있고, **Adam**(Momentum+RMSProp의 장점 결합)이 빠르고 안정적이라 널리 쓰입니다.

> 🟩 **(보충)** 학습률 $\eta$ 는 너무 크면 학습이 불안정(발산)하고 너무 작으면 느립니다. 또 "Adam이 항상 최선"은 아닙니다 — 문제에 따라 SGD+Momentum이 더 잘 일반화하기도 합니다. **무난한 기본 선택** 정도로 이해하세요.

### 6.2 과적합 방지 기법

훈련 데이터에만 지나치게 맞춰지는 **과적합**을 줄이는 대표 기법입니다.

| 기법 | 한 줄 설명 |
|---|---|
| **Dropout** | 학습 중 일부 뉴런을 무작위로 꺼서 특정 뉴런 의존을 줄임 |
| **Batch Normalization** | 미니배치의 중간 활성값을 정규화하고 학습 가능한 scale·shift를 적용 |
| **Early Stopping** | 검증 성능이 더 나아지지 않으면 학습을 멈춤 |
| **Data Augmentation** | 데이터를 변형해 양·다양성을 늘림(주로 이미지, 보통 train에만) |

> 🟩 **(보충)** 어떤 기법도 과적합을 **항상 없애 주지는 않습니다.** Batch Normalization은 많은 모델에서 학습을 안정시키는 데 도움이 되지만 **효과는 구조와 배치 크기에 따라 달라집니다.** 또 Dropout·BatchNorm은 학습과 평가에서 동작이 다르므로, 평가 때는 5.1처럼 `model.eval()` 로 전환해야 합니다. Early Stopping의 기준은 **test가 아니라 validation** 지표입니다.

---

## 7. 손실함수와 평가지표는 역할이 다르다 🟩

입문자는 손실과 평가지표를 같은 것으로 오해하기 쉽습니다. **손실함수**는 역전파로 가중치를 업데이트하기 위한 **학습 목표**이고, **평가지표**는 실제 문제에서 모델 성능을 해석하는 **기준**입니다.

| 문제 | 학습 손실 예 | 평가 지표 예 |
|---|---|---|
| 회귀 | MSE, MAE, Huber | MAE, RMSE, R² |
| 균형 잡힌 분류 | Cross Entropy | Accuracy, F1 |
| 불균형 이진 분류 | BCEWithLogitsLoss | Precision, Recall, F1, PR-AUC |
| 다중 라벨 | BCEWithLogitsLoss | micro/macro F1, 라벨별 precision/recall |

지표는 데이터와 업무 비용에 맞춰 고릅니다. 예를 들어 **양성 사례가 매우 드물면 정확도(accuracy)가 높아도 좋은 모델이 아닐 수 있으므로** precision·recall을 함께 확인해야 합니다.

> 🟩 **(보충)** 클래스 불균형이 심하면 손실 단계에서 `nn.CrossEntropyLoss(weight=...)` 나 `nn.BCEWithLogitsLoss(pos_weight=...)` 로 가중치를 줄 수 있습니다. 가중치 설정은 이 글의 범위를 넘으니, 우선 **"불균형에서는 정확도만으로 판단하지 않는다"** 만 기억하세요.

---

## 8. 딥러닝 모델 종류 한눈에 🟦

이번 DAY의 중심은 설계와 손실함수이므로, 대표 모델은 **한 줄 개요**로 정리합니다.

| 모델 | 한 줄 개요 |
|---|---|
| **Perceptron** | 가장 기본 인공뉴런(Rosenblatt, 1957~1958). 단층은 XOR을 못 풂 |
| **MLP** | 은닉층을 추가해 비선형 문제 해결(예: 784→128→64→10) |
| **DNN** | 은닉층을 여러 개 쌓은 심층 신경망. 표현력↑(보통 더 많은 데이터·정규화·사전학습의 도움이 필요할 수 있음) |
| **CNN** | 합성곱으로 이미지의 공간 특징 추출(Conv→ReLU→Pool→FC). LeNet·AlexNet·VGG·ResNet |
| **RNN** | 순서가 있는 데이터 처리(이전 상태 기억). 장기 기억·기울기 소실 문제 |
| **LSTM** | 게이트(Forget·Input·Output)와 Cell State로 RNN의 장기 기억 개선 |
| **GRU** | LSTM을 단순화. 파라미터가 적어 더 빠를 수 있고, 성능은 데이터에 따라 다름 |
| **AutoEncoder** | 입력을 압축(Encoder)했다 복원(Decoder). 차원 축소·노이즈 제거·이상 탐지 |
| **VAE** | AutoEncoder를 발전시킨 **생성 모델**(잠재공간 학습) |
| **GAN** | 생성자 vs 판별자가 경쟁하며 학습. 얼굴·이미지 생성 |
| **Transformer** | RNN 없이 **Self-Attention**으로 장거리 관계를 직접 모델링(병렬 처리에 유리) |
| **BERT** | Transformer **Encoder** 기반, 양방향 문맥 이해(문서 분류·검색) |
| **GPT** | Transformer **Decoder** 기반 생성형 모델(ChatGPT 등) |
| **ViT** | 이미지를 패치로 나눠 Transformer로 처리(CNN 없이 이미지 분석) |
| **Diffusion** | 노이즈를 더했다 제거하며 학습해 이미지 생성(Stable Diffusion 등) |

**발전 순서(PDF)**: Perceptron → MLP → DNN → CNN/RNN → LSTM/GRU → AutoEncoder → GAN → Transformer → BERT/GPT → ViT → Diffusion

**데이터 유형별 대표 모델(PDF)**

| 데이터 유형 | 대표 모델 |
|---|---|
| 표(Tabular) | DNN, MLP |
| 이미지 분류 | CNN, ViT |
| 객체 탐지 | CNN 기반 YOLO |
| 시계열 | LSTM, GRU |
| 자연어 처리 | Transformer, BERT |
| 생성형 AI | GPT |
| 이미지 생성 | GAN, Diffusion |
| 차원 축소 | AutoEncoder |
| 이상 탐지 | AutoEncoder, Transformer |

> 🟩 **(보충)** 위 "유형 → 모델"은 **고정 규칙이 아니라 자주 쓰이는 조합**입니다. 예컨대 이미지에도 Transformer(ViT)가, 시계열에도 CNN·Transformer가 쓰입니다. "무엇이 가장 많이/최고"라는 순위는 시간이 지나면 바뀝니다. AutoEncoder를 "PCA 대체"라고도 하지만, 정확히는 **선형 AutoEncoder가 PCA와 비슷한 역할**을 하는 것이고 일반 AutoEncoder는 비선형 압축까지 학습합니다.

---

## 9. 자주 하는 실수

```text
❌ 출력층·손실함수를 따로 설계           → 문제 유형에 맞춰 '짝'으로 설계(shape·dtype까지)
❌ CrossEntropyLoss인데 출력에 Softmax    → 내부에 포함됨, 모델은 raw logits 출력
❌ BCEWithLogitsLoss인데 출력에 Sigmoid   → 내부에 포함됨, Sigmoid 넣지 않기
❌ CrossEntropyLoss 라벨을 One-Hot으로     → 단일 정답은 정수 인덱스(long), shape (N,)
❌ 이진 분류 라벨 dtype을 long으로         → BCE 계열 타깃은 float, 출력과 같은 shape (N,1)
❌ 회귀 출력층에 Sigmoid/Softmax          → 회귀는 활성화 없음(선형)
❌ 다중 분류 vs 다중 라벨 혼동             → 하나만=Softmax/CE, 동시 여럿=Sigmoid/BCE
❌ KLDivLoss를 기본 reduction으로          → batchmean 사용, 인자는 (log-확률 Q, 확률 P)
❌ 평가 때 model.eval()·no_grad() 누락     → Dropout·BatchNorm 모드 전환 + gradient 끄기
❌ 모델과 데이터 device 불일치            → model·X·y를 같은 device로 .to(device)
❌ 분류 성능을 정확도만으로 판단           → 불균형이면 precision·recall·F1 등 함께
❌ Adam·ReLU가 항상 정답이라 가정          → 무난한 기본일 뿐, 문제에 따라 다름
```

---

## 10. DAY4 핵심 정리

```text
네트워크 설계
  - 문제 정의 → 입력층(표 데이터면 특성 수) → 은닉층(특징 추출) → 출력층 → 손실 → 최적화 → 과적합 방지
  - 은닉층 z = Σ w·x + b 후 활성화(ReLU=max(0,x))로 비선형성 부여

출력층·라벨·손실 짝 (가장 중요)
  - 회귀    : 출력 (N,1) 선형     + 실수 라벨 (N,1)        + MSELoss/L1Loss/HuberLoss
  - 이진분류 : 출력 (N,1) logit    + 0/1 float 라벨 (N,1)    + BCEWithLogitsLoss (Sigmoid 내장)
  - 다중분류 : 출력 (N,C) logits   + 정수 인덱스 (N,) long    + CrossEntropyLoss (Softmax 내장)
  - 다중라벨 : 출력 (N,C) logits   + 0/1 float 라벨 (N,C)     + BCEWithLogitsLoss

손실 직관 / 평가지표
  - MSE 큰 오차 강벌점 / MAE 이상치 둔감 / Huber 절충 / BCE·CE 정답 확률 높을수록 손실↓
  - 손실(학습 목표) ≠ 평가지표(성능 해석): 불균형이면 정확도 말고 precision·recall·F1

최적화·과적합·평가
  - 갱신 w ← w - η·∂L/∂w / Adam 무난한 기본(항상 최선은 아님)
  - 과적합 방지: Dropout·BatchNorm·EarlyStopping·DataAugmentation(보통 train에만)
  - 실제 데이터: train/validation/test 분리, 전처리는 train에만 fit, eval()+no_grad()

모델 발전: Perceptron→MLP→DNN→CNN/RNN→LSTM/GRU→AutoEncoder→GAN→Transformer→BERT/GPT→ViT→Diffusion
```

---

## 🔗 참고 자료

- 강의 PDF: `DAY4_딥러닝_네트워크모델설계.pdf` (본문의 1차 출처)
- [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/) (OS·연산 플랫폼별 설치 명령)
- [PyTorch — Loss Functions 목록](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [`nn.CrossEntropyLoss` (라벨 형식·내부 LogSoftmax)](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [`nn.BCEWithLogitsLoss` (Sigmoid 결합)](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [`nn.KLDivLoss` (입력=log-확률, `batchmean`)](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
- [PyTorch 옵티마이저 (`torch.optim`)](https://pytorch.org/docs/stable/optim.html)
