<!--
============================================================
기획 문서 (드래프트 메타) — 게시 전 이 주석 블록은 제거 예정
============================================================

[작업 모드] Generate (Stage 1: 초안)
[원본 PDF] sources/DAY4_딥러닝_네트워크모델설계.pdf (총 28쪽)
[분야] 딥러닝 / [독자] 파이썬 기본은 알지만 딥러닝은 처음
[프레임워크] PyTorch (PDF가 nn.MSELoss·nn.L1Loss·nn.HuberLoss·nn.BCEWithLogitsLoss·
            nn.CrossEntropyLoss·nn.NLLLoss·nn.KLDivLoss를 사용 → PyTorch로 일관)

------------------------------------------------------------
1) 페이지 → 핵심 주제 지도 (전 28쪽 확인. 수식·다이어그램은 이미지로 박혀 있어 p4,5,10~13,15 렌더링하여 공식 육안 확인)
------------------------------------------------------------
■ 1장. 딥러닝 네트워크 모델 설계 (p3~6)
  p3  설계의 의미 / 문제정의(분류·회귀·탐지·NLP·생성) / 입력데이터 분석(MLP·CNN·RNN…) / 입력층·은닉층 설계
  p4  뉴런 계산 z=Σw_i x_i+b, 활성화함수(ReLU 등), ReLU=max(0,x)[이미지], 출력층(이진 Sigmoid/다중 Softmax/회귀 선형), 손실함수 개요
  p5  MSE 공식[이미지], 역전파·경사하강, 가중치 갱신 w_new=w_old-η∂L/∂w[이미지], 옵티마이저(SGD·Momentum·RMSProp·Adam), 과적합방지(Dropout·BatchNorm·EarlyStopping·DataAug)
  p6  설계 12단계 흐름 정리
■ 2장. 출력층 설계와 손실함수 (p7~16)
  p7  출력층 의미 / 회귀 출력층(노드1, 선형)
  p8  이진분류(노드1, Sigmoid) / 다중분류(노드=클래스수, Softmax, argmax)
  p9  출력층 활성화 표(회귀/이진/다중/다중라벨), 다중라벨≠다중분류, 손실함수 정의
  p10 손실함수 기본개념(y, ŷ, L), MSE 공식[이미지]
  p11 MSE 예시(2500)[이미지], MAE 공식[이미지]·예시, nn.MSELoss()/nn.L1Loss()
  p12 Huber Loss 공식[이미지, a=y-ŷ], nn.HuberLoss() / BCE 시작, BCE 공식[이미지]
  p13 BCE 해석, nn.BCEWithLogitsLoss()(Sigmoid 포함) / CrossEntropy 공식 CE=-Σ y log(ŷ)[이미지], One-Hot
  p14 CE 예시(고양이), nn.CrossEntropyLoss()(내부 Softmax 포함) / 다중라벨 손실
  p15 다중라벨 BCEWithLogitsLoss / NLLLoss=-log P(y)[이미지](LogSoftmax) / KLDivLoss 공식[이미지]
  p16 KL 사용처, nn.KLDivLoss() / 손실함수 선택 기준 표
■ 3장. 딥러닝 모델 종류 (p17~28)
  p17 Perceptron(1957 Rosenblatt, XOR 불가) / 수식 y=f(Σw x+b)
  p18 MLP(은닉층 추가, 784→128→64→10) / DNN(은닉층 다수)
  p19 DNN 특징·활용 / CNN(Conv-ReLU-Pool-FC)
  p20 CNN Convolution 필터 예, 대표모델(LeNet·AlexNet·VGG16·GoogLeNet·ResNet) / RNN(이전상태 기억)
  p21 RNN 문제(장기기억·Gradient Vanishing) / LSTM(Forget·Input·Output Gate, Cell State)
  p22 GRU(LSTM 단순화) / AutoEncoder(Encoder-잠재공간-Decoder)
  p23 VAE(생성) / GAN(Generator vs Discriminator)
  p24 대표 GAN(DCGAN·CycleGAN·StyleGAN) / Transformer(Self-Attention, RNN 없이)
  p25 Transformer 장점(병렬·긴문장) / BERT(Encoder, 양방향)
  p26 GPT(Decoder, 생성) / ViT(이미지 패치) / Diffusion(노이즈 추가·제거)
  p27 Diffusion 특징 / 모델 발전 순서(Perceptron→…→Diffusion)
  p28 데이터유형→대표모델 표 / 현재 Transformer·Diffusion 주류

------------------------------------------------------------
2) 재구성(집필) 계획 — 입문자 순서
------------------------------------------------------------
- PDF 1장(설계 프로세스)을 "입력→은닉→출력→손실→최적화→과적합" 흐름으로 재배열.
- PDF 2장(출력층+손실)을 글의 중심으로: '문제 유형별 출력층·라벨·손실 짝' 표를 핵심으로 두고,
  손실함수는 회귀(MSE/MAE/Huber)·이진(BCE)·다중(CE)·다중라벨(BCE)·기타(NLL/KL)로 묶음.
- 코드: PyTorch 하나로 일관. '설정 가능한 MLP' 하나를 완전 실행 예제(다중분류 end-to-end)로,
  회귀·이진 head는 부분 코드로. tensor shape/dtype/device/출력층/라벨/손실 점검 포함.
- PDF 3장(모델 종류)은 카탈로그라 표 중심으로 간결히(각 모델 한 줄). 발전 순서·데이터유형 표 포함.

[실행/검증 메모]
- 본 환경: NumPy만 설치(torch 없음). 손실 공식의 숫자 예시(MSE/MAE/softmax/BCE/CE/ReLU)는 NumPy로 실제 실행해 확인.
- PyTorch nn.* 코드는 실행하지 못함 → 손실 '값'은 비워 두고 shape/dtype만 명시(확정값). 임의 결과 X.

[보충/정정 포인트(🟩)]
- nn.CrossEntropyLoss 라벨은 PyTorch에서 '정수 클래스 인덱스(long), shape (N,)' — PDF의 One-Hot은 수학 개념. (현대 버전은 확률 타깃도 받지만 입문은 정수 인덱스 기준)
- CrossEntropyLoss=내부 Softmax 포함 → 모델 출력층에 Softmax 넣지 않음(raw logits). BCEWithLogitsLoss=내부 Sigmoid 포함 → Sigmoid 넣지 않음.
- NLLLoss는 LogSoftmax 출력을 입력으로 받음. KLDivLoss는 입력=log-확률, 타깃=확률(주의).
- "Adam이 가장 널리" → 무난한 기본 선택이나 항상 최선은 아님. ReLU도 마찬가지(+Dead ReLU).
- Perceptron 연도는 1957~1958(Rosenblatt)로 표기(자료별 상이). AutoEncoder=PCA '대체'는 선형 AE 한정 비유. Diffusion "GAN보다 안정적"은 알려진 경향.
- 다음 DAY 주제는 PDF에 없음 → 예고하지 않음.
============================================================
-->

# 🧠 딥러닝 완전 입문 가이드 — DAY4. 딥러닝 네트워크 모델 설계

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: DAY1(개념·설치) · DAY2(신경망 알고리즘·계층구조) · DAY3(데이터 처리·데이터셋)
> **프레임워크**: 이 글의 코드는 모두 **PyTorch** 기준입니다. (PDF가 `nn.MSELoss`·`nn.CrossEntropyLoss` 등 PyTorch API를 사용)

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 내용
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 사전지식·실무 주의점
> - 손실 공식의 **숫자 예시(MSE·MAE·Softmax·BCE·CE·ReLU)** 는 **NumPy로 실제 실행해 확인**한 값입니다. 반면 **PyTorch `nn.*` 학습 코드는 작성 환경에 torch가 없어 실행하지 못했습니다.** 손실 '값'은 비워 두었고(직접 실행해 확인), 텐서 **shape·dtype**처럼 확정적인 부분만 표기했습니다. (임의의 결과를 지어내지 않았습니다.)

---

## 1. 이번 DAY에서 배우는 것

- 딥러닝 **네트워크 모델 설계**가 "층을 쌓는 것"을 넘어 무엇을 결정하는 과정인지
- **입력층 → 은닉층 → 출력층** 을 문제에 맞게 설계하는 법
- 이 글의 핵심: **문제 유형별 "출력층 · 라벨 형식 · 손실함수" 짝 맞추기**
- 회귀(MSE·MAE·Huber), 이진(BCE), 다중(Cross Entropy), 다중 라벨, 그리고 NLL·KL 손실
- **최적화(Adam)** 와 **과적합 방지(Dropout·BatchNorm·EarlyStopping·증강)**
- 대표 딥러닝 모델(퍼셉트론 → MLP/DNN → CNN/RNN → LSTM/GRU → AutoEncoder/GAN → Transformer/BERT/GPT → ViT/Diffusion) **한눈에 보기**

---

## 2. 네트워크 모델 설계란 무엇인가 🟦

딥러닝 모델 설계는 단순히 층(Layer)을 여러 개 쌓는 작업이 아니라, **풀려는 문제와 데이터 특성을 분석해 최적의 신경망 구조를 정하는 과정**입니다. 설계에서 정해야 할 것은 계층 구조, 활성화 함수, 손실 함수, 최적화 알고리즘 등입니다.

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

**입력층(Input Layer)의 노드 수 = 특성(feature) 개수**입니다. 예를 들어 학생 성적이 (국어, 영어, 수학) 3개 특성이면 입력층 노드는 3개입니다.

### 2.3 은닉층 — 특징을 추출하는 곳

은닉층(Hidden Layer)은 입력에서 **중요한 특징을 추출**합니다. 은닉층 개수·노드 수는 문제 복잡도에 따라 정합니다(단순한 문제는 1~2개, 복잡한 이미지·자연어는 수십~수백 층).

각 뉴런은 입력에 가중치를 곱하고 편향을 더한 값을 계산한 뒤, **활성화 함수**를 통과시켜 다음 층으로 보냅니다.

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

활성화 함수는 신경망에 **비선형성**을 줘서 복잡한 패턴을 학습하게 합니다. 대표적으로 Sigmoid·Tanh·ReLU·Leaky ReLU·ELU·Swish가 있고, 현재는 계산이 간단하고 기울기 소실이 적은 **ReLU**를 은닉층 기본으로 많이 씁니다.

$$
\text{ReLU}(x) = \max(0,\, x)
$$

> 🟩 **(보충)** "ReLU가 항상 최고"는 아닙니다. 음수 입력에서 기울기가 0이라 일부 뉴런이 죽는 **Dead ReLU**가 생길 수 있고, 그럴 땐 Leaky ReLU 등을 시도합니다. (활성화 함수 자세한 비교는 DAY2 참고.)

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

| 문제 유형 | 출력 노드 수 | 출력층 활성화 | 라벨(정답) 형식 | PyTorch 손실함수 |
|---|---|---|---|---|
| **회귀** | 예측값 개수(보통 1) | 없음(선형) | 실수 | `nn.MSELoss` / `nn.L1Loss` / `nn.HuberLoss` |
| **이진 분류** | 1 | (Sigmoid) | 0/1 | `nn.BCEWithLogitsLoss` |
| **다중 분류** | 클래스 개수 C | (Softmax) | 정수 클래스 인덱스 | `nn.CrossEntropyLoss` |
| **다중 라벨 분류** | 클래스 개수 C | (Sigmoid, 라벨마다) | 0/1 벡터 | `nn.BCEWithLogitsLoss` |

> 🟩 **(아주 중요) PyTorch에서 활성화는 "손실함수가 대신 계산"합니다**
> - `nn.CrossEntropyLoss` 는 **내부에 Softmax(정확히는 LogSoftmax)** 가 들어 있습니다. → 모델 출력층에 Softmax를 **넣지 말고 raw 점수(logit)** 를 그대로 내보냅니다.
> - `nn.BCEWithLogitsLoss` 는 **내부에 Sigmoid** 가 들어 있습니다. → 출력층에 Sigmoid를 **넣지 마세요.**
> - 위 표의 활성화에 괄호를 친 이유가 이것입니다. Sigmoid·Softmax는 **학습이 끝난 뒤 사람이 확률로 해석할 때** 적용합니다.

### 3.2 다중 분류의 라벨 형식 — One-Hot vs 정수 인덱스 🟩

PDF는 다중 분류 정답을 **One-Hot**(예: 정답이 2번 → `[0,0,1,0,0]`)으로 설명합니다. 이는 손실 **수식**을 이해하기 위한 표현입니다. 하지만 **PyTorch `nn.CrossEntropyLoss` 는 정답을 정수 클래스 인덱스(`torch.long`, shape `(N,)`)** 로 받습니다. 즉 `[0,0,1,0,0]` 대신 정수 `2` 를 넘깁니다. (One-Hot을 직접 곱해 계산할 필요가 없습니다.)

---

## 4. 손실함수 하나씩 — 왜·어떻게·예시 🟦 + 🟩

아래 숫자 예시는 모두 **NumPy로 실제 실행해 확인**한 값입니다.

```python
import numpy as np
```

### 4.1 회귀 — MSE · MAE · Huber

**MSE (평균제곱오차)** — 회귀에서 가장 많이 쓰는 손실. 오차를 제곱해 평균 내므로 **큰 오차에 강한 벌점**을 줍니다.

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

**MAE (평균절대오차)** — 오차에 절댓값을 씌워 평균. 제곱하지 않으므로 **이상치에 덜 민감**합니다.

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i\rvert
$$

```python
y, yhat1, yhat2 = 100, 90, 50
print((y-yhat1)**2, (y-yhat2)**2)   # 100 2500   ← MSE는 오차 10→100, 50→2500 (급증)
print(abs(y-yhat1), abs(y-yhat2))   # 10 50      ← MAE는 오차 10→10, 50→50 (선형)
```

**Huber Loss** — MSE와 MAE의 장점을 합친 손실. 오차가 작으면(MSE처럼) 제곱, 크면(MAE처럼) 절댓값을 써서 **이상치에 과하게 흔들리지 않습니다**($a=y-\hat{y}$, $\delta$는 경계).

$$
L_\delta(a) = \begin{cases} \frac{1}{2}a^2 & \lvert a\rvert \le \delta \\ \delta\left(\lvert a\rvert - \frac{1}{2}\delta\right) & \lvert a\rvert > \delta \end{cases}
$$

PyTorch에서:

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

```python
# 정답 y=1일 때, 예측 확률에 따른 BCE (한 샘플)
for p in [0.95, 0.05]:
    print(p, -np.log(p))     # 0.95 -> 0.0513 (정답에 가까움, 손실 작음)
                             # 0.05 -> 2.9957 (정답과 멈, 손실 큼)
```

```python
criterion = nn.BCEWithLogitsLoss()   # Sigmoid + BCE를 함께 계산 → 모델에 Sigmoid 불필요
```

### 4.3 다중 분류 — Cross Entropy

**Cross Entropy** — 여러 클래스 중 하나를 고르는 문제(숫자 0~9, 동물 5종 등)에서 가장 많이 씁니다. **정답 클래스에 준 확률**이 높을수록 손실이 작습니다.

$$
\text{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i) \qquad (C=\text{클래스 개수})
$$

정답이 One-Hot이면 정답 위치만 1이라, 실제로는 **정답 클래스의 예측 확률**만 손실에 반영됩니다.

```python
# 정답 = 고양이(0번). 모델이 각 클래스에 준 확률에 따른 CE
for name, probs in [("good", [0.90, 0.07, 0.03]), ("bad", [0.05, 0.80, 0.15])]:
    p = np.array(probs)
    print(name, round(-np.log(p[0]), 4))   # good -> 0.1054 (정답에 0.90, 손실 작음)
                                            # bad  -> 2.9957 (정답에 0.05, 손실 큼)
```

```python
criterion = nn.CrossEntropyLoss()   # 내부에 (Log)Softmax 포함 → 모델은 raw logits 출력, 라벨은 정수 인덱스
```

### 4.4 다중 라벨 분류

다중 라벨은 **여러 정답이 동시에** 참일 수 있는 문제입니다(한 사진에 사람·자동차·강아지가 모두 있음). "하나만 고르기"가 아니라 **클래스마다 0/1을 따로 판단**하므로, 각 클래스에 Sigmoid를 적용한 이진 분류를 여러 개 푸는 셈입니다.

```python
criterion = nn.BCEWithLogitsLoss()   # 라벨마다 0/1 → 다중 라벨에도 사용
```

### 4.5 기타 손실 — NLL · KL (참고)

- **NLLLoss (음의 로그 가능도)**: $\text{NLL} = -\log P(y)$, 정답 클래스 확률이 낮을수록 손실↑. PyTorch에서는 **LogSoftmax 출력**과 함께 씁니다. 보통은 이 둘을 합친 `CrossEntropyLoss`를 더 많이 씁니다.
- **KLDivLoss (KL 발산)**: 두 확률분포 $P$(실제)·$Q$(예측)가 **얼마나 다른지** 측정. 지식 증류, 생성 모델(VAE) 등에 쓰입니다.

$$
D_{KL}(P\,\Vert\,Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}
$$

> 🟩 **(보충) PyTorch 사용 시 주의** — `nn.NLLLoss`는 입력으로 **log-확률**(`LogSoftmax` 출력)을 기대합니다. `nn.KLDivLoss`도 입력을 **log-확률**, 타깃을 **확률**로 받습니다. 형식을 안 맞추면 값이 엉뚱해지니, 입문 단계에서는 분류엔 `CrossEntropyLoss`를 기본으로 쓰는 편이 안전합니다.

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

## 5. 하나로 합치기 — PyTorch 설계 예제

지금까지의 "입력 → 은닉 → 출력 → 손실"을 **하나의 PyTorch 코드**로 묶어 봅니다. 아래는 **다중 분류**를 끝까지 연결한 예제입니다.

> ⚠️ **이 코드는 작성 환경에 torch가 없어 실행하지 못했습니다.** 구조·shape·dtype·출력층·손실 짝은 점검했으니, **손실 값 등 실제 수치는 직접 실행**해 확인하세요.

```python
# 프레임워크: PyTorch  (설치: pip install torch)
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
criterion = nn.CrossEntropyLoss()                  # 내부 Softmax 포함 → 모델은 logits
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6) 한 스텝 학습 (순전파 → 손실 → 역전파 → 갱신)
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
# 회귀 head (부분): 출력 1, 활성화 없음 / 라벨은 실수 (N,1)
reg_head, reg_loss = nn.Linear(32, 1), nn.MSELoss()        # 또는 nn.L1Loss(), nn.HuberLoss()

# 이진 분류 head (부분): 출력 1 logit / 라벨은 0,1 float (N,1) / 모델에 Sigmoid 넣지 않음
bin_head, bin_loss = nn.Linear(32, 1), nn.BCEWithLogitsLoss()
```

> 🟩 **(보충) device 한 줄 점검** — 모델과 입력·정답 텐서가 **같은 장치**에 있어야 합니다(`.to(device)`). GPU 모델에 CPU 텐서를 넣으면 오류가 납니다. GPU 인식 여부는 `torch.cuda.is_available()` 로 확인하세요.

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
| **Batch Normalization** | 층의 입력 분포를 정규화해 학습을 안정·가속 |
| **Early Stopping** | 검증 성능이 더 나아지지 않으면 학습을 멈춤 |
| **Data Augmentation** | 데이터를 변형해 양·다양성을 늘림(주로 이미지) |

> 🟩 **(보충)** 어떤 기법도 과적합을 **항상 없애 주지는 않습니다.** 또 Dropout은 학습 때만 켜고 평가 때는 꺼야 하므로, PyTorch에서는 `model.train()` / `model.eval()` 전환을 잊지 마세요(DAY2 참고).

---

## 7. 딥러닝 모델 종류 한눈에 🟦

이번 DAY의 중심은 설계와 손실함수이므로, 대표 모델은 **한 줄 개요**로 정리합니다.

| 모델 | 한 줄 개요 |
|---|---|
| **Perceptron** | 가장 기본 인공뉴런(Rosenblatt, 1957~1958). 단층은 XOR을 못 풂 |
| **MLP** | 은닉층을 추가해 비선형 문제 해결(예: 784→128→64→10) |
| **DNN** | 은닉층을 여러 개 쌓은 심층 신경망. 표현력↑, 대량 데이터 필요 |
| **CNN** | 합성곱으로 이미지의 공간 특징 추출(Conv→ReLU→Pool→FC). LeNet·AlexNet·VGG·ResNet |
| **RNN** | 순서가 있는 데이터 처리(이전 상태 기억). 장기 기억·기울기 소실 문제 |
| **LSTM** | 게이트(Forget·Input·Output)와 Cell State로 RNN의 장기 기억 개선 |
| **GRU** | LSTM을 단순화. 구조 간단·학습 빠름, 성능은 LSTM과 유사 |
| **AutoEncoder** | 입력을 압축(Encoder)했다 복원(Decoder). 차원 축소·노이즈 제거·이상 탐지 |
| **VAE** | AutoEncoder를 발전시킨 **생성 모델**(잠재공간 학습) |
| **GAN** | 생성자 vs 판별자가 경쟁하며 학습. 얼굴·이미지 생성 |
| **Transformer** | RNN 없이 **Self-Attention**으로 관계 파악. 병렬 처리·긴 문장에 강함 |
| **BERT** | Transformer **Encoder** 기반, 양방향 문맥 이해(문서 분류·검색) |
| **GPT** | Transformer **Decoder** 기반 생성형 모델(ChatGPT 등) |
| **ViT** | 이미지를 패치로 나눠 Transformer로 처리(CNN 없이 이미지 분석) |
| **Diffusion** | 노이즈를 더했다 제거하며 학습해 이미지 생성(Stable Diffusion·DALL·E) |

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

## 8. 자주 하는 실수

```text
❌ 출력층·손실함수를 따로 설계           → 문제 유형에 맞춰 '짝'으로 설계
❌ CrossEntropyLoss인데 출력에 Softmax    → 내부에 포함됨, 모델은 raw logits 출력
❌ BCEWithLogitsLoss인데 출력에 Sigmoid   → 내부에 포함됨, Sigmoid 넣지 않기
❌ CrossEntropyLoss 라벨을 One-Hot으로     → PyTorch는 정수 인덱스(long), shape (N,)
❌ 회귀 출력층에 Sigmoid/Softmax          → 회귀는 활성화 없음(선형), 숫자 범위 제한 X
❌ 다중 분류 vs 다중 라벨 혼동             → 하나만=Softmax/CE, 동시 여럿=Sigmoid/BCE
❌ NLL·KL에 일반 점수 입력                → NLL·KL은 log-확률 입력 필요
❌ 모델과 데이터 device 불일치            → model·X·y를 같은 device로 .to(device)
❌ Adam·ReLU가 항상 정답이라 가정          → 무난한 기본일 뿐, 문제에 따라 다름
❌ 과적합 방지 기법이 항상 해결한다고 믿기  → 완화일 뿐, eval()에서 Dropout 꺼짐 주의
```

---

## 9. DAY4 핵심 정리

```text
네트워크 설계
  - 문제 정의 → 입력층(특성 수) → 은닉층(특징 추출) → 출력층 → 손실 → 최적화 → 과적합 방지
  - 은닉층 z = Σ w·x + b 후 활성화(ReLU=max(0,x))로 비선형성 부여

출력층·라벨·손실 짝 (가장 중요)
  - 회귀    : 출력 선형        + 실수 라벨        + MSELoss/L1Loss/HuberLoss
  - 이진분류 : 출력 1 logit     + 0/1 라벨         + BCEWithLogitsLoss (Sigmoid 내장)
  - 다중분류 : 출력 C logits    + 정수 인덱스 라벨   + CrossEntropyLoss (Softmax 내장)
  - 다중라벨 : 출력 C logits    + 0/1 벡터 라벨      + BCEWithLogitsLoss

손실함수 직관
  - MSE 큰 오차에 강한 벌점 / MAE 이상치에 둔감 / Huber 둘의 절충
  - BCE·CE 정답에 준 확률이 높을수록 손실 작음

최적화·과적합
  - 갱신 w ← w - η·∂L/∂w / Adam 무난한 기본(항상 최선은 아님)
  - 과적합 방지: Dropout·BatchNorm·EarlyStopping·DataAugmentation

모델 발전: Perceptron→MLP→DNN→CNN/RNN→LSTM/GRU→AutoEncoder→GAN→Transformer→BERT/GPT→ViT→Diffusion
```

---

## 🔗 참고 자료

- 강의 PDF: `DAY4_딥러닝_네트워크모델설계.pdf` (본문의 1차 출처)
- [PyTorch 공식 문서 — Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions) (`MSELoss`·`BCEWithLogitsLoss`·`CrossEntropyLoss` 등)
- [PyTorch `nn.CrossEntropyLoss` (라벨 형식·내부 LogSoftmax)](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch `nn.BCEWithLogitsLoss` (Sigmoid 결합)](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch 옵티마이저 (`torch.optim`)](https://pytorch.org/docs/stable/optim.html)
