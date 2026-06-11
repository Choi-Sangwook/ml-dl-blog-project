# 🧠 딥러닝 완전 입문 가이드 — DAY2. 신경망 알고리즘과 계층구조

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: 딥러닝 DAY1 — 딥러닝 개념과 프레임워크 설치
> **프레임워크**: 이 글의 모든 코드는 **PyTorch** 기준입니다. (검증 환경: `torch 2.12.0+cpu`)

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 내용
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 사전지식·실무 주의점
> - 본문 코드의 출력은 모두 **실제 실행으로 확인**한 값입니다(무작위 초기화에 따라 환경마다 소수점은 조금 다를 수 있습니다).

---

## 1. 이번 DAY에서 배우는 것

DAY1에서 딥러닝이 무엇이고 PyTorch를 어떻게 설치하는지 살펴봤습니다.
이번에는 딥러닝의 가장 기본 부품인 **뉴런(퍼셉트론)** 에서 시작해서, 이 부품을 여러 층으로 쌓아 만든 **신경망(Neural Network)** 이 어떻게 학습하는지까지 한 번에 따라갑니다.

- 퍼셉트론이 무엇이고, `nn.Linear`가 그중 **어느 부분**을 담당하는가
- 퍼셉트론으로 AND·OR·NAND 같은 논리회로를 만드는 방법
- 단층 퍼셉트론이 **XOR 문제**를 못 푸는 이유와, 층을 쌓아 해결하는 방법
- 입력층 → 은닉층 → 출력층으로 이어지는 신경망의 **계층구조**
- **순전파 → 손실 → 역전파 → 가중치 업데이트**로 이어지는 학습의 전체 흐름
- 신경망에 비선형성을 더하는 **활성화 함수**의 종류와 선택 기준

---

## 2. 퍼셉트론: 인공뉴런의 출발점 🟦

### 2.1 왜 필요한가 — 뇌의 뉴런을 흉내 내다

연구자들은 인간의 뇌를 관찰하면서 "컴퓨터도 인간처럼 학습할 수 없을까?"라는 질문을 던졌고, 여기서 **인공신경망(Artificial Neural Network)** 이 탄생했습니다. 인공신경망은 인간의 뉴런을 **수학적으로 단순화한 모델**입니다.

생물학적 뉴런은 다음 세 부분으로 동작합니다.

| 뉴런 구조 | 역할 | PyTorch에 비유하면 |
|---|---|---|
| 수상돌기(Dendrite) | 다른 뉴런에서 신호를 **입력** 받음 | `x = torch.tensor([1.0, 2.0])` (입력 데이터) |
| 세포체(Cell Body) | 입력 신호를 **계산** | `y = model(x)` (계산 과정) |
| 축색돌기(Axon) | 계산 결과를 다음 뉴런으로 **전달** | `output` (출력 신호) |

> 🟩 **(보충)** 인공신경망은 뇌를 "본떠 만든 영감"일 뿐, 실제 뇌의 복제가 아닙니다. 어디까지나 단순화된 수학 모델이라는 점을 기억하세요.

### 2.2 동작 원리 — 가중치와 편향

**퍼셉트론(Perceptron)** 은 이 인공뉴런을 수학적으로 구현한 모델입니다. 입력값 $x_1, \cdots, x_n$ 과 가중치 $w_1, \cdots, w_n$ 를 각각 곱해 더한 뒤, 함수 $f$ 를 통과시킵니다.

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

- **가중치(Weight, $w$)**: 각 입력에 곱해지는 **학습되는 계수**.
- **편향(Bias, $b$)**: 기본값(출발선). 모든 입력이 0이어도 출력이 $b$만큼 이동합니다.
- $f$: **활성화 함수**(8장에서 자세히 다룹니다).

> 🟩 **(보충) "가중치 = 중요도"는 어디까지나 거친 직관입니다**
> PDF에는 `weight = [[0.8, 0.1]]` 이면 "첫 입력이 두 번째보다 8배 중요"라는 예시가 나옵니다. 방향을 잡는 직관으로는 좋지만, **문자 그대로 "정확히 8배"** 라고 단정할 수는 없습니다. 가중치 크기는 **입력의 단위·스케일, 다른 가중치, 활성화 함수**에 따라 의미가 달라지기 때문입니다. (예: 면적을 m²로 쓰느냐 km²로 쓰느냐에 따라 같은 정보라도 가중치 크기가 달라집니다.) "학습으로 정해지는 계수" 정도로 이해하는 것이 안전합니다.

### 2.3 PyTorch의 `nn.Linear`는 무엇을 계산할까?

PyTorch에서 가장 기본이 되는 계층은 `nn.Linear`입니다.

```python
import torch
import torch.nn as nn

# 입력 2개 -> 출력 1개
model = nn.Linear(2, 1)

print(model.weight)   # 가중치, shape: [1, 2]
print(model.bias)     # 편향,  shape: [1]
```

> 🟩 **(보충) `nn.Linear`는 "퍼셉트론 전체"가 아니라 그중 선형 변환 부분입니다**
> 고전적 퍼셉트론은 보통 `가중합 + 편향 + 계단 함수`까지 포함하지만, `nn.Linear`는 공식 문서 기준으로 **선형 변환 + 편향**($z = Wx + b$)만 계산합니다. 비선형성은 뒤에서 활성화 함수로 따로 붙입니다.
>
> ```text
> nn.Linear       : z = Wx + b 를 계산하는 선형 계층
> 퍼셉트론        : z를 계산한 뒤 계단 함수로 0/1을 결정하는 모델
> 일반 신경망 뉴런 : z를 계산한 뒤 ReLU·Sigmoid 같은 활성화 함수를 적용
> ```

배치(여러 샘플) 입력을 넣을 때의 shape 약속은 다음과 같습니다. PyTorch가 내부에 저장하는 가중치는 `(out_features, in_features)` 모양이라, 개념적으로 `weight.T`를 곱합니다.

```text
입력 X       : (batch_size, in_features)
weight       : (out_features, in_features)
bias         : (out_features,)
출력 Z       : (batch_size, out_features)
개념적 계산  : Z = X @ weight.T + bias
```

예를 들어 `nn.Linear(10, 64)`는 `(8, 10)` 입력을 `(8, 64)`로 바꿉니다. 맨 앞 **배치 크기 8은 그대로 유지**되고, 마지막 특징 차원만 10에서 64로 바뀝니다.

> 🟩 **(보충)** 수학 교재는 종종 샘플 하나를 열벡터로 보고 $Wx$ 로 적지만, PyTorch는 배치를 행으로 쌓아 `X @ weight.T` 로 계산합니다. 결과는 같은 선형 변환이며, 입문 단계에서는 **"앞 계층 출력 수 = 다음 계층 입력 수"** 만 맞으면 됩니다.

### 2.4 결과 해석과 주의사항

- `nn.Linear`는 활성화 함수가 **포함되지 않은** 순수 선형 계산입니다.
- 가중치/편향 초깃값은 무작위라서 실행마다 출력 숫자가 다릅니다. 재현이 필요하면 `torch.manual_seed(42)`로 시드를 고정하세요.

---

## 3. 논리회로로 퍼셉트론 이해하기 🟦

퍼셉트론은 간단한 **논리회로(Logic Gate)** 를 구현할 수 있습니다. 대표적으로 **AND, OR, NAND** 게이트가 있고, 모두 입력 2개·출력 1개(0 또는 1)입니다.

| $x_1$ | $x_2$ | AND | NAND | OR |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 0 |
| 1 | 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 1 | 1 | 0 | 1 |

핵심 아이디어는 **적절한 $w_1, w_2, b$를 고르면** `x1*w1 + x2*w2 + b > 0` 여부로 게이트를 흉내 낼 수 있다는 것입니다.

### 3.1 AND 게이트 — 둘 다 1일 때만 1

```python
# AND 게이트 : 입력이 모두 1일 때만 1 출력
import torch

def AND(x1, x2):
    x = torch.tensor([x1, x2], dtype=torch.float32)  # 입력을 텐서로
    w = torch.tensor([0.5, 0.5])                      # AND를 만족하는 가중치
    b = -0.7                                          # 편향
    tmp = torch.sum(x * w) + b                        # 가중합 + 편향
    if tmp <= 0:                                      # 임계값(0) 기준 판단
        return 0
    else:
        return 1

print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))
```

**실행 결과**

```text
0 0 0 1
```

> 🟩 **(보충)** `(1,1)`만 보면 `1*0.5 + 1*0.5 - 0.7 = 0.3 > 0` 이라 1, 나머지는 합이 0 이하라 0입니다. 여기서 `if tmp <= 0` 부분이 곧 **계단(Step) 활성화 함수** 역할을 합니다.

### 3.2 NAND 게이트 — AND의 반대

NAND는 AND 결과를 뒤집은 게이트로, **음수 가중치**를 쓰는 것이 포인트입니다.

```python
import torch

def NAND(x1, x2):
    x = torch.tensor([x1, x2], dtype=torch.float32)
    w = torch.tensor([-0.2, -0.2])   # 음수 가중치
    b = 0.3
    tmp = torch.sum(x * w) + b
    return 0 if tmp <= 0 else 1

print(NAND(0, 0), NAND(0, 1), NAND(1, 0), NAND(1, 1))
```

**실행 결과**

```text
1 1 1 0
```

### 3.3 OR 게이트 — 하나라도 1이면 1

```python
import torch

def OR(x1, x2):
    x = torch.tensor([x1, x2], dtype=torch.float32)
    w = torch.tensor([0.7, 0.7])
    b = -0.4
    tmp = torch.sum(x * w) + b
    return 0 if tmp <= 0 else 1

print(OR(0, 0), OR(0, 1), OR(1, 0), OR(1, 1))
```

**실행 결과**

```text
0 1 1 1
```

### 3.4 가중치를 "직접 정하지 않고" 찾을 수는 없을까?

위에서는 가중치를 사람이 직접 골랐습니다. PDF에는 **무작위로 가중치를 뽑아보며** 조건을 만족하는 값을 찾는 단순 탐색 예제가 나옵니다. 다만 원본의 `while True`는 멈추지 않을 수 있으므로, 아래처럼 **시도 횟수에 상한(`max_trials`)을 두고** 못 찾으면 안내 메시지를 출력하도록 했습니다.

```python
# 학습 없이 랜덤 Weight 탐색 : 논리회로를 만족하는 Weight 찾기
import torch

def check(x1, x2, y, max_trials=100000):
    for trial in range(max_trials):     # while True 대신 시도 횟수 제한
        cnt = 0
        w1 = torch.randn(1).item()      # 정규분포에서 무작위 Weight
        w2 = torch.randn(1).item()
        b  = torch.randn(1).item()

        for i in range(len(x1)):
            x = torch.tensor([x1[i], x2[i]], dtype=torch.float32)
            w = torch.tensor([w1, w2])
            tmp = torch.sum(x * w) + b
            result = 0 if tmp <= 0 else 1
            if y[i] == result:
                cnt += 1

        if cnt == 4:                    # 4개 데이터를 모두 맞히면 종료
            print(f"{trial + 1}번째 시도 성공 → w1 = {w1}, w2 = {w2}, b = {b}")
            return (w1, w2, b)

    print(f"{max_trials}번 안에 조건을 만족하는 가중치를 찾지 못했습니다.")
    return None

# 정답(y)만 바꾸면 AND/NAND/OR 모두 같은 방식으로 탐색
check([0,1,0,1], [0,0,1,1], [0,0,0,1])   # AND
```

> ⚠️ **주의 — 이 코드는 "학습"이 아니라 운에 기대는 무작위 탐색입니다**
> 조건을 만족하는 가중치를 우연히 뽑을 때까지 반복하므로, 운이 나쁘면 오래 걸릴 수 있습니다. 그래서 위 코드는 `while True` 대신 **`max_trials`로 시도 횟수에 상한**을 두고, 못 찾으면 안내 메시지를 출력하도록 바꿨습니다. 입력·가중치가 조금만 늘어도 무작위 탐색은 급격히 비효율적이 되므로, 실제로는 다음 절(3.5)의 **경사하강법 학습**을 씁니다.

### 3.5 진짜 "학습": 경사하강법으로 가중치를 배우게 하기

무작위 탐색 대신, 현대 딥러닝은 **오차를 보고 가중치를 조금씩 고쳐 나갑니다**. PDF가 소개하는 AND 게이트 학습 코드입니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)                              # 🟩 재현을 위해 시드 고정

model = nn.Linear(2, 1)                            # 이진 분류용 선형 계층(logit 출력)
criterion = nn.BCEWithLogitsLoss()                 # 이진 분류용 손실
optimizer = optim.Adam(model.parameters(), lr=0.01)

X = torch.tensor([[0., 0.], [0., 1.],
                  [1., 0.], [1., 1.]])             # 입력 shape: (4, 2)
y = torch.tensor([[0.], [0.],
                  [0.], [1.]])                     # AND 정답, shape: (4, 1)

for epoch in range(1000):
    pred = model(X)                                # 순전파, pred shape: (4, 1) = logit
    loss = criterion(pred, y)                      # 손실 계산
    optimizer.zero_grad()                          # 이전 기울기 초기화
    loss.backward()                                # 역전파
    optimizer.step()                               # 가중치 업데이트
    if epoch % 200 == 0:
        print(epoch, round(loss.item(), 4))
```

**실행 결과(시드 42 기준)**

```text
0 0.6903
200 0.3782
400 0.2523
600 0.1823
800 0.1376
```

손실이 꾸준히 줄어듭니다. 그런데 **손실이 줄었다고 곧바로 "학습 성공"은 아닙니다.** 실제로 네 입력을 모두 맞히는지 확인해야 합니다. 여기서 출력의 **세 단계 — logit → 확률 → 최종 라벨** 을 구분하는 것이 중요합니다.

```python
model.eval()                          # 평가 모드
with torch.no_grad():                 # 기울기 계산 끄기
    logits = model(X)                 # (4,1) 가공 전 점수(logit)
    prob   = torch.sigmoid(logits)    # 0~1 확률로 변환
    pred   = (prob >= 0.5).float()    # 0.5 기준으로 0/1 라벨
    acc    = (pred == y).float().mean()

print(prob.flatten())     # tensor([0.0030, 0.1290, 0.1290, 0.8620])
print(pred.flatten())     # tensor([0., 0., 0., 1.])
print(f"진리표 정확도: {acc.item():.2%}")   # 진리표 정확도: 100.00%
```

> 🟩 **(보충) 이 정확도 100%의 의미**
> 이 예제의 네 행은 **가능한 AND 입력 전체**입니다. 따라서 정확도 100%는 "네 행의 진리표를 재현했다"는 뜻이지, **새로운 현실 데이터에 대한 일반화 성능**을 검증했다는 뜻은 아닙니다. 진짜 데이터에서의 평가 방법은 6.5에서 짧게 정리합니다.

> 🟩 **(보충) `BCEWithLogitsLoss`를 쓸 때 Sigmoid를 모델에 넣지 않는 이유**
> 위 `model`에는 Sigmoid가 없습니다. `BCEWithLogitsLoss`가 **내부에서 Sigmoid + 이진 교차엔트로피를 함께 계산**하기 때문입니다. 모델 끝에 `nn.Sigmoid()`를 또 붙이면 **이중 적용**이 되어 잘못됩니다. Sigmoid는 위처럼 **학습이 끝난 뒤 확률로 보고 싶을 때만** 적용합니다. (출력층-손실함수 짝은 6.2 표로 정리합니다.)

---

## 4. 단층 퍼셉트론의 한계와 XOR 문제 🟦

### 4.1 XOR은 직선 하나로 나눌 수 없다

XOR(배타적 논리합)은 "두 입력이 다를 때만 1"인 게이트입니다.

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

단층 퍼셉트론의 결정 경계는 $y = w_1 x_1 + w_2 x_2 + b$ 형태, 즉 **직선**입니다. AND·OR은 직선 하나로 0과 1을 가를 수 있지만, **XOR은 어떤 직선으로도 0/1을 깔끔히 나눌 수 없습니다.**

> 🟩 **(보충)** "복잡한 데이터는 무조건 곡선 경계가 필요하다"고 일반화할 수는 없습니다. 정확히는 **XOR처럼 하나의 직선으로 분리되지 않는 문제**에 비선형 결정 경계가 필요합니다.

### 4.2 해결책: 층을 쌓아 만든 다층 퍼셉트론(MLP)

XOR은 **기존 게이트(NAND, OR, AND)를 조합**하면 만들 수 있습니다.

- **0층(입력)**: $x_1, x_2$
- **1층(은닉)**: $s_1 = \text{NAND}(x_1, x_2)$, $s_2 = \text{OR}(x_1, x_2)$
- **2층(출력)**: $y = \text{AND}(s_1, s_2)$

중간 계산까지 펼친 진리표는 다음과 같습니다.

| $x_1$ | $x_2$ | $s_1$(NAND) | $s_2$(OR) | $y$(AND) |
|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 0 |
| 1 | 0 | 1 | 1 | 1 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 1 | 0 | 1 | 0 |

마지막 $y$ 열이 정확히 XOR 진리표와 같습니다.

> 🟩 **(보충)** 아래 코드는 PDF의 **스크린샷(이미지) 페이지**에 있던 것을 복원한 것입니다. 앞서 정의한 `AND`, `OR`, `NAND` 함수(3장)가 먼저 정의되어 있어야 동작합니다.

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)   # 1층
    s2 = OR(x1, x2)     # 1층
    y  = AND(s1, s2)    # 2층
    return y

input_data = [(0, 0), (1, 0), (0, 1), (1, 1)]
print('XOR')
for i in input_data:
    print(XOR(i[0], i[1]))
```

**실행 결과**

```text
XOR
0
1
1
0
```

은닉층(Hidden Layer)을 하나 추가하니, 직선 하나로는 못 풀던 문제가 풀립니다. 이것이 **다층 퍼셉트론(Multi-Layer Perceptron)** 의 핵심 아이디어입니다.

### 4.3 더 깊게: 심층 신경망(DNN)

현실 문제는 XOR보다 훨씬 복잡합니다(얼굴 인식, 음성 인식, 자연어 처리, 자율주행 등). 이런 문제를 풀기 위해 은닉층을 **여러 개** 쌓은 구조가 등장했고, 이를 **심층 신경망(Deep Neural Network, DNN)** 이라고 부릅니다. 여기서 "딥(Deep)"이 바로 **층이 깊다**는 뜻입니다.

---

## 5. 딥러닝 신경망의 계층구조 🟦

딥러닝 신경망은 여러 인공뉴런을 **층(layer)** 으로 연결한 학습 구조입니다. 각 뉴런은 입력값에 가중치 $W$를 곱하고 편향 $b$를 더한 뒤, 활성화 함수를 통과시켜 다음 층으로 전달합니다.

$$
z = Wx + b \qquad a = f(z)
$$

- $z$: 뉴런 내부의 계산값(**pre-activation**)
- $a$: 활성화 함수를 통과한 출력값(**activation**)

> 🟩 **(보충)** 실제 PyTorch 코드에서는 2.3에서 본 것처럼 배치를 행으로 쌓아 `Z = X @ weight.T + bias` 로 계산합니다. 그림의 한 줄 수식 $z = Wx + b$ 는 **샘플 하나**에 대한 표현이라고 생각하면 됩니다.

```text
입력층(Input Layer)
      ↓
은닉층(Hidden Layer 1)
      ↓
은닉층(Hidden Layer 2)
      ↓
은닉층(Hidden Layer 3)
      ↓
출력층(Output Layer)
```

**입력층(Input Layer)** — 데이터를 모델에 넣는 부분입니다.
- 집값 예측이라면: 방 개수, 면적, 위치, 건축연도, 역세권 여부
- 이미지 분류라면: 픽셀값

**은닉층(Hidden Layer)** — 실제로 **특징을 학습**하는 부분입니다.

> 🟩 **(보충) "얕은 층=선, 깊은 층=얼굴"은 경향이지 규칙이 아닙니다**
> PDF는 "첫 층은 선·점, 다음 층은 눈·코·입, 깊은 층은 얼굴"이라고 설명합니다. 이는 주로 **이미지 모델(CNN) 시각화**에서 자주 관찰되는 직관입니다. 실제로는 얕은 층이 비교적 단순한 패턴에, 깊은 층이 더 복합적인 패턴에 반응하는 **경향**이 있지만, 각 층·각 뉴런이 정확히 그런 의미를 학습한다는 보장은 없습니다. 표현은 모델 구조와 데이터에 따라 달라집니다.

**출력층(Output Layer)** — 최종 결과를 내보내는 부분으로, 문제 유형에 따라 구조가 달라집니다.

| 문제 유형 | 출력층 예시 |
|---|---|
| 집값 예측(회귀) | 숫자 1개 |
| 이진 분류 | 점수 1개 → (확률화 후) 0/1 |
| 다중 분류 | 클래스 개수만큼 출력 |
| 이미지 생성 | 이미지 픽셀값 |
| 문장 생성 | 다음 단어에 대한 점수(→확률) |

> 🟩 **(보충)** 학습 중 모델이 내보내는 것은 보통 가공 전 점수인 **logit**입니다. "0 또는 1", "다음 단어 확률"은 logit에 Sigmoid/Softmax를 적용하고 임계값·argmax를 거친 **최종 해석값**입니다. (자세한 짝은 6.2 표)

---

## 6. 딥러닝 학습의 전체 흐름 🟦

딥러닝 학습은 다음을 **반복**합니다.

1. 입력 데이터 준비
2. **순전파(Forward Propagation)** — 예측값 계산
3. **손실값(Loss) 계산**
4. **역전파(Backpropagation)** — 각 파라미터의 기울기 계산
5. **가중치·편향 업데이트**
6. 반복

> 한 줄 요약: **예측한다 → 틀린 정도를 계산한다 → 각 파라미터가 손실에 얼마나 영향을 주는지 계산한다 → 조금 고친다**

### 6.1 순전파 (Forward Propagation)

입력 데이터가 앞에서 뒤로 이동하며 예측값을 만드는 과정입니다. 입력이 $x_1$=방 개수, $x_2$=면적, $x_3$=건축연도라면:

$$
z = w_1 x_1 + w_2 x_2 + w_3 x_3 + b \qquad a = \text{ReLU}(z)
$$

이렇게 계산된 값이 다음 층으로 전달됩니다.

### 6.2 손실 함수 (Loss Function)와 출력층의 짝

손실 함수는 **예측이 실제와 얼마나 틀렸는지**를 숫자로 나타냅니다. 비슷하면 작고, 많이 틀리면 큽니다. 회귀에 흔히 쓰는 **MSE(평균 제곱 오차)** 는 다음과 같습니다($\hat{y}$=예측, $y$=정답, $n$=표본 수).

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
$$

**모델의 학습 출력, 라벨 형식, 손실 함수, 출력층 활성화는 서로 짝이 맞아야 합니다.** 이 표 하나만 기억하면 됩니다.

| 문제 | 모델의 학습 출력 | 라벨 형식 | PyTorch 손실 | 사람이 해석할 때 |
|---|---|---|---|---|
| 회귀 | `(N, 1)` 실수 | `float32`, `(N, 1)` | `nn.MSELoss()` | 그대로 예측값 |
| 이진 분류 | `(N, 1)` **logit** | 0/1 `float32`, 같은 shape | `nn.BCEWithLogitsLoss()` | `sigmoid` 후 임계값 |
| 다중 분류 | `(N, C)` **logits** | 클래스 인덱스 `torch.long`, `(N,)` | `nn.CrossEntropyLoss()` | `softmax` 후 확률, `argmax` |

> ⚠️ **주의 — 출력층 활성화 중복에 주의**
> - `nn.BCEWithLogitsLoss()` = Sigmoid + 이진 교차엔트로피. 모델 끝에 Sigmoid를 **붙이지 않습니다.**
> - `nn.CrossEntropyLoss()` = log-softmax + NLL. 출력층에 Softmax를 **붙이지 않습니다.** 모델은 logits를 그대로 내보냅니다.
> - 굳이 Sigmoid를 모델에 넣고 싶다면 손실을 `nn.BCELoss()`로 바꿔야 합니다. 즉 **`Sigmoid 출력 + BCELoss`** 또는 **`raw logit + BCEWithLogitsLoss`** 중 하나로 맞춰야 하며, 입문 단계에서는 수치적으로 더 안정적인 **후자**를 권장합니다.

### 6.3 역전파 (Backpropagation)와 Gradient

순전파가 예측을 만드는 과정이라면, 역전파는 **손실에 대한 각 파라미터의 기울기를 출력층에서 입력층 방향으로 계산**하는 과정입니다. 즉 손실값을 각 가중치에 대해 **미분**하며, 이 값이 **Gradient(기울기)** 입니다.

Gradient는 **지금 이 파라미터 값에서 손실이 얼마나 민감하게 변하는지**를 나타냅니다.

- **절댓값**: 현재 지점에서의 국소적 민감도(크면 작은 변화에도 손실이 크게 출렁임)
- **부호**: 파라미터를 어느 방향으로 바꾸면 손실이 **증가**하는지

> 🟩 **(보충)** Gradient가 크다고 해서 그 입력이 **전역적으로 가장 중요**하거나 오차에 **인과적으로 가장 책임**이 있다는 뜻은 아닙니다. 어디까지나 현재 미니배치·현재 파라미터 지점에서의 국소 정보입니다.

### 6.4 가중치 업데이트와 Optimizer

Gradient를 계산한 뒤, **Optimizer(최적화 알고리즘)** 가 가중치와 편향을 수정합니다. 가장 기본적인 업데이트 식은 다음과 같습니다.

$$
w \leftarrow w - \eta \,\frac{\partial L}{\partial w}
$$

- $L$: 손실, $\dfrac{\partial L}{\partial w}$: 가중치에 대한 gradient, $\eta$: **학습률(learning rate)**

> 🟩 **(보충) 학습률($\eta$)** 은 "한 번에 얼마나 크게 고칠지"를 정합니다. 너무 크면 최솟값을 지나치거나 손실이 발산할 수 있고, 너무 작으면 학습이 매우 느려집니다.

PyTorch에서는 이 세 줄이 한 묶음입니다.

```python
optimizer.zero_grad()   # 이전 단계에 남은 기울기를 0으로 초기화
loss.backward()         # 모든 가중치·편향의 gradient 계산(역전파)
optimizer.step()        # 계산된 gradient로 w, b 업데이트
```

> ⚠️ **주의**: `zero_grad()`를 빼먹으면 이전 단계의 기울기가 누적되어 학습이 이상해집니다. 매 반복마다 호출하세요.

대표적인 Optimizer로 SGD, Momentum, RMSProp, **Adam**, **AdamW**가 있습니다. 실무에서는 보통 Adam 또는 AdamW를 많이 씁니다.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### 6.5 학습 코드의 기본 골격

기본적인 PyTorch 학습 루프의 골격입니다(미니배치·스케줄러 등을 쓰면 세부는 달라집니다).

```python
for epoch in range(100):
    pred = model(X_train)              # 1. 예측 (순전파)
    loss = criterion(pred, y_train)    # 2. 손실 계산
    optimizer.zero_grad()              # 3. 이전 gradient 초기화
    loss.backward()                    # 4. 역전파
    optimizer.step()                   # 5. 가중치·편향 업데이트
    print(epoch, loss.item())          # 6. 학습 상태 출력
```

> 🟩 **(보충) 실제 데이터에서는 데이터의 역할을 셋으로 나눕니다**
> 위 골격은 **훈련 손실만** 출력합니다. 진짜 데이터에서는 다음을 지켜야 모델 선택과 최종 평가가 공정해집니다.
> - **train**: 가중치를 학습합니다.
> - **validation**: 모델 구조·학습률·epoch를 선택하고 과적합을 확인합니다.
> - **test**: 모든 선택이 끝난 뒤 **한 번만** 최종 성능을 확인합니다.
>
> 또한 학습할 때는 `model.train()`, 검증·예측할 때는 `model.eval()` + `torch.no_grad()`를 사용합니다(불필요한 gradient 저장을 막습니다). 표준화·결측치 대체처럼 **데이터에서 규칙을 배우는 전처리는 train에만 `fit`** 하고, validation·test에는 같은 규칙을 적용만 합니다.
>
> (3.5의 AND 예제는 가능한 입력 네 개를 모두 학습하는 **진리표 재현 예제**라 따로 나눌 필요가 없었습니다.)

> 🟩 **(보충) 용어 구분**: **epoch**은 전체 데이터를 한 번 다 학습하는 단위, **batch**는 한 번에 처리하는 데이터 묶음, **step**은 가중치를 한 번 고치는 동작입니다.

---

## 7. PyTorch 신경망 예제 — SimpleDNN 🟦

지금까지 배운 계층구조를 코드로 만들어 봅니다. 가장 기본적인 다층 퍼셉트론입니다.

```python
import torch
import torch.nn as nn

class SimpleDNN(nn.Module):
    def __init__(self):
        super().__init__()                  # nn.Module 초기화 (필수)
        self.fc1 = nn.Linear(10, 64)         # 입력층 -> 은닉층1
        self.fc2 = nn.Linear(64, 32)         # 은닉층1 -> 은닉층2
        self.fc3 = nn.Linear(32, 1)          # 은닉층2 -> 출력층
        self.relu = nn.ReLU()                # 활성화 함수

    def forward(self, x):
        x = self.relu(self.fc1(x))           # 선형 변환 후 ReLU
        x = self.relu(self.fc2(x))           # 선형 변환 후 ReLU
        x = self.fc3(x)                      # 출력층 (활성화 없음)
        return x

model = SimpleDNN()
print(model)
```

**실행 결과**

```text
SimpleDNN(
  (fc1): Linear(in_features=10, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=32, bias=True)
  (fc3): Linear(in_features=32, out_features=1, bias=True)
  (relu): ReLU()
)
```

이 모델은 입력값 10개를 받아 최종적으로 숫자 1개를 출력합니다. 집값·매출·점수 예측 같은 **회귀 문제**에 쓸 수 있습니다(출력층에 활성화가 없으므로 연속값을 그대로 내보냅니다).

> 🟩 **(보충) 텐서 shape으로 따라가기**
> 입력 `(batch, 10)` → fc1 → `(batch, 64)` → ReLU → fc2 → `(batch, 32)` → ReLU → fc3 → `(batch, 1)`.
> `nn.Linear(a, b)`에서 **앞 계층의 출력 수와 다음 계층의 입력 수가 일치**해야 합니다(64→64, 32→32). 배치 차원(맨 앞)은 계산 내내 유지됩니다.
>
> ```python
> x = torch.randn(8, 10)     # 샘플 8개, 특징 10개
> print(model(x).shape)      # torch.Size([8, 1])
>
> num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
> print(num_params)          # 2817
> ```
>
> 입력이 10개뿐이어도 은닉층을 쌓으면 학습할 파라미터가 **2,817개**로 금세 늘어납니다.

**계층별 역할 요약**

```text
입력층     -> 원본 데이터를 받음
은닉층 1   -> 단순 패턴 학습
은닉층 2   -> 조합된 패턴 학습
은닉층 3+  -> 더 복합적인 특징 학습
출력층     -> 최종 예측값 생성
```

---

## 8. 활성화 함수 자세히 보기 🟦

### 8.1 왜 비선형성이 필요한가

활성화 함수는 신경망에 **비선형성(Non-linearity)** 을 더합니다. **활성화 함수가 없으면, 아무리 층을 많이 쌓아도 결국 하나의 선형식과 같아집니다**(선형 변환을 여러 번 해도 하나의 선형 변환으로 합쳐지기 때문). 그러면 4장의 XOR처럼 직선으로 분리되지 않는 문제를 풀 수 없습니다. 즉 활성화 함수는 **딥러닝이 복잡한 문제를 풀 수 있게 하는 핵심**입니다.

### 8.2 자주 쓰는 활성화 함수

아래 결과는 모두 입력 `x = torch.tensor([-2., -1., 0., 1., 2.])` 에 대한 값이며 **실제 실행으로 확인**했습니다. 아래 코드 블록들은 **맨 위부터 이어서 실행**하는 연속 예제로, 첫 블록에서 만든 `x`와 `import`(`torch`, `torch.nn as nn`)를 그대로 재사용합니다.

**Step Function (계단 함수)** — 0을 기준으로 0 또는 1.

```python
import torch
x = torch.tensor([-2., -1., 0., 1., 2.])
print((x > 0).float())   # tensor([0., 0., 0., 1., 1.])
```
거의 모든 구간에서 기울기가 0이라 **경사 기반 학습(역전파)에 부적합** → 현재는 거의 쓰지 않습니다.

**Sigmoid** — 출력을 0~1로 누르는 S자 함수.

```python
import torch.nn as nn
print(nn.Sigmoid()(x))   # tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
```
확률처럼 해석할 수 있어 **이진 분류 출력층**에 쓰지만, 입력 절댓값이 크면 기울기가 거의 0이 되는 **포화(기울기 소실)** 가 있습니다.

**Tanh** — 출력 범위 -1~1, 중심이 0.

```python
print(nn.Tanh()(x))   # tensor([-0.9640, -0.7616,  0.0000,  0.7616,  0.9640])
```
0 중심이라는 장점이 있지만, **Sigmoid처럼 큰 절댓값에서 포화**합니다(항상 더 빠르다고 단정할 수는 없습니다).

**ReLU** — 음수는 0, 양수는 그대로. 현재 은닉층의 흔한 기본값.

```python
print(nn.ReLU()(x))   # tensor([0., 0., 0., 1., 2.])
```
계산이 단순하고 양수 구간 기울기가 1이라 Sigmoid/Tanh의 포화 문제를 줄여 줍니다. 다만 **음수 구간 기울기는 0** 이라 일부 뉴런이 죽는 **Dead ReLU**가 생길 수 있고, 깊은 망 전체의 기울기 소실·폭주를 **완전히 없애지는 않습니다.**

**Leaky ReLU** — 음수도 조금(기본 0.01배) 통과.

```python
print(nn.LeakyReLU(negative_slope=0.01)(x))   # tensor([-0.0200, -0.0100, 0.0000, 1.0000, 2.0000])
```
Dead ReLU 위험을 **완화**하지만 해결을 보장하지는 않습니다.

### 8.3 더 발전된 함수들 (이름과 용도 위주)

> 🟩 **(보충)** PDF는 PReLU·ELU·SELU·GELU·SiLU(Swish)·Mish도 소개합니다. 입문 단계에서는 이름과 "언제 쓰는지"만 알아두면 충분합니다.
> - **PReLU**: 음수쪽 기울기까지 학습 / **ELU·SELU**: 평균을 0 근처로 맞춰 학습 안정화(SELU의 자기 정규화는 특정 초기화·구조 조건에서 기대) / **GELU**: 부드러운 곡선형, BERT 계열 등 일부 Transformer에서 사용 / **SiLU(Swish)** = `x * sigmoid(x)`, 여러 현대 모델에서 사용 / **Mish**: 부드러운 곡선형.

### 8.4 비교와 실무 선택

PDF의 비교표는 "속도 등급"·"기울기 소실 없음"·"가장 많이 사용" 같은 단정이 많은데, 이는 하드웨어·구현·시점에 따라 바뀝니다. 핵심 성질 중심으로 정리합니다.

| 함수 | 핵심 성질 | 주의점·대표 용도 |
|---|---|---|
| Step | 0 또는 1 출력 | 기울기가 대부분 0 → 역전파 학습에 부적합 |
| Sigmoid | 0~1 출력 | 이진 확률 해석에 사용, 큰 절댓값에서 포화 |
| Tanh | -1~1, 0 중심 | 역시 큰 절댓값에서 포화 |
| ReLU | 음수 0, 양수 그대로 | 단순·효율적, 음수 구간 기울기 0(Dead ReLU) |
| Leaky ReLU | 음수에도 작은 기울기 | Dead ReLU 위험 완화(해결 보장은 아님) |
| GELU | 입력을 부드럽게 조절 | BERT 계열 등 일부 Transformer |
| SiLU(Swish) | `x * sigmoid(x)` | 여러 현대 모델에서 사용 |

> ⚠️ **주의**: "ReLU가 항상 최고", "GELU면 무조건 좋다", "GELU는 LLM 표준" 같은 단정은 피하세요. 어떤 함수가 좋은지는 **데이터와 모델 구조에 따라 달라집니다.** GELU·SiLU·Mish는 음수쪽에 **유한한 최솟값**이 있어 출력 범위가 $-\infty$까지 내려가지 않습니다.

**실무 출발점(가이드라인)**

```text
은닉층 기본            : nn.ReLU()
Dead ReLU가 의심되면  : nn.LeakyReLU()
이진 분류 출력층      : nn.Sigmoid()  (단, 손실이 BCEWithLogitsLoss면 생략)
다중 분류 출력층      : 보통 생략 (CrossEntropyLoss가 내부 처리)
```

---

## 9. 대표적인 딥러닝 신경망 한눈에 🟦

이번 DAY의 중심은 계층구조와 학습 흐름이므로, 대표 모델은 한 줄 개요만 정리합니다.

| 알고리즘 | 한 줄 개요 |
|---|---|
| **DNN** | 가장 기본적인 완전연결 신경망. 표·수치 데이터, 기본 분류·회귀 |
| **CNN** | 합성곱으로 이미지의 공간적 특징을 학습 |
| **RNN** | 순서가 있는 데이터(시계열·문장·음성) 처리 |
| **LSTM / GRU** | RNN의 장기 기억 문제를 개선 |
| **Transformer** | Self-Attention으로 문장 전체 관계를 파악, 생성형 AI의 핵심 구조 |

> CNN·RNN·Transformer는 이후 단계에서 더 깊이 다룹니다.

---

## 10. 자주 하는 실수

```text
❌ nn.Linear가 곧 퍼셉트론 전체    → nn.Linear는 선형 변환(z=Wx+b)만, 활성화는 별도
❌ 출력층에 활성화 중복            → BCEWithLogitsLoss/CrossEntropyLoss에 Sigmoid/Softmax 또 붙이지 않기
❌ 손실 감소만 보고 학습 성공 판단  → 실제 예측 라벨/정확도까지 확인
❌ logit을 곧바로 확률·라벨로 해석  → logit → sigmoid/softmax → 임계값/argmax 순서
❌ optimizer.zero_grad() 누락      → 기울기 누적으로 학습 망가짐
❌ 계층 간 shape 불일치            → 앞 출력 수 = 다음 입력 수
❌ 활성화 함수 없이 층만 쌓기       → 깊이 쌓아도 1차식과 동일
❌ Gradient·가중치를 절대적 중요도로 → 국소 민감도일 뿐, 전역 중요도 아님
❌ while True 무한 탐색            → 시도 횟수 상한 두기
❌ test로 하이퍼파라미터 튜닝       → 비교는 validation, test는 마지막 1회
```

---

## 11. DAY2 핵심 정리

```text
퍼셉트론
  - 입력 × 가중치 + 편향 → 활성화 함수로 통과시키는 인공뉴런
  - nn.Linear는 그중 선형 변환(z=Wx+b)만 담당, 활성화는 별도

논리회로와 XOR
  - AND/OR/NAND는 단층 퍼셉트론(직선 경계)으로 구현 가능
  - XOR은 직선으로 못 나눔 → 은닉층을 추가한 다층 퍼셉트론으로 해결

계층구조
  - 입력층 → 은닉층(특징 학습) → 출력층
  - z = Wx + b (선형) 후 a = f(z) (비선형)로 다음 층에 전달
  - 배치 계산: Z = X @ weight.T + bias, (batch, in) → (batch, out)

학습 흐름
  - 순전파(예측) → 손실 → 역전파(gradient) → Optimizer 업데이트 → 반복
  - 핵심 3줄: zero_grad() → backward() → step()
  - 업데이트: w ← w - lr × gradient
  - 실제 데이터: train/validation/test 분리, eval()·no_grad()로 검증

출력층·손실 짝 (가장 중요)
  - 회귀    : 활성화 없음 + MSELoss
  - 이진분류: logit + BCEWithLogitsLoss (Sigmoid는 해석할 때만)
  - 다중분류: logits + CrossEntropyLoss (Softmax는 내부 처리)

활성화 함수
  - 비선형성을 더해 복잡한 문제를 풀 수 있게 함
  - 은닉층 기본은 ReLU / 어떤 함수가 좋은지는 데이터·구조에 따라 다름
```

---

## 🔗 참고 자료

- 강의 PDF: `DAY2_딥러닝_신경망알고리즘_계층구조.pdf` (본문의 1차 출처)
- [PyTorch 공식 문서 — `torch.nn`](https://pytorch.org/docs/stable/nn.html) (계층·활성화 함수 동작)
- [PyTorch 공식 문서 — Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions) (`BCEWithLogitsLoss`, `CrossEntropyLoss` 내부 동작)
- [PyTorch 공식 문서 — `nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (선형 변환과 가중치 shape)
