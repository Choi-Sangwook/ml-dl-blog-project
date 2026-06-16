<!--
============================================================
기획 문서 (드래프트 메타) — 게시 전 이 주석 블록은 제거 예정
============================================================

[작업 모드] Generate (Stage 1: 초안)
[1차 출처] sources/DAY7_딥러닝_다중퍼셉트론.pdf (총 72페이지)
[분야] 딥러닝 / [독자] 파이썬 기본은 알지만 딥러닝은 처음
[이전 편] DAY1~DAY6 (개념·설치 / 신경망·계층 / 데이터셋 / 모델설계 / 훈련·평가 / 모델저장·활용)

[중요 발견 — 전 페이지(이미지 전용 26~72 포함) 확인 결과]
  - PDF에는 "실행 가능한 파이썬 코드 텍스트가 한 곳도 없음".
  - 3장 RNN(p26~52), 4장 GAN(p53~72)은 전부 개념도 + "설계도(층 차원·활성화·옵티마이저·라벨 방식)" + 산문 설명.
    예) RNN: "Keras Sequential, LSTM 셀 차원 300, 출력 sigmoid, MinMax[0,1], RMSprop, MSE, inverse_transform, numpy flatten, seaborn lineplot"
        GAN: 생성자 Dense128→leakyReLU→Dense128→leakyReLU→Dense784→Reshape(28,28,1) / 감별자 Flatten→Dense128→leakyReLU→Dense1(sigmoid) / train_on_batch / numpy ones·zeros 라벨
  => 따라서 RNN·GAN 코드는 "PDF 설계도를 충실히 옮긴 재구성 코드(🟩)"로 명시하고, PDF 원문 코드라고 표기하지 않음.

[프레임워크 결정]
  - 실행 코드는 전부 "Keras(tf.keras)" 하나로 통일(항목6). PDF가 RNN을 명시적으로 Keras로 기술하고 GAN도 Keras(train_on_batch/compile) 기반이기 때문.
  - MLP 실습도 같은 프레임워크 유지를 위해 Keras로 XOR 예제 작성(PDF가 XOR을 단층 한계의 핵심 예로 강조).

[실행 환경 한계 — 임의 결과값 금지(항목8)]
  - 이 환경에 tensorflow/keras 미설치 → 모든 Keras 코드(XOR·LSTM·GAN)는 "직접 실행" 안내, 손실/정확도/예측 수치 임의 기입 금지.
  - 실제 실행으로 검증한 것(프레임워크 무관):
      · 윈도잉: 144길이 시계열, window=12 → X (132,12,1), y (132,1)  [실행 확인]
      · MinMax 정규화 ↔ inverse 역정규화 왕복 일치 (np.allclose True)  [실행 확인]
      · XOR 입력 X (4,2) float  [실행 확인]
  - sklearn 1.7.2 설치됨 / tensorflow·torch 미설치.

------------------------------------------------------------
PDF 페이지 → 핵심 주제 지도 (전 72페이지 확인, 이미지 전용 26~72 전수 확인)
------------------------------------------------------------
p1-2    표지/목차 (4장: ①MLP ②최적화 함수 ③RNN ④GAN)
[1장 MLP] (텍스트)
p3      퍼셉트론 정의, 구조(입력→가중치→가중합Σ→활성화→출력), 식 z=Σwx+b, y=f(z)
p4      단층 퍼셉트론 한계: AND/OR 가능, XOR 불가(선형분리), XOR 진리표
p5      MLP 구조(입력-은닉-출력), 예: 입력4-은닉8-은닉4-출력2; 입력층/은닉층 설명(Iris 4입력)
p6-7    (이미지)빈 + 출력층(이진=1노드, 다중=3노드), MLP 계산 1단계 입력 x1=1,x2=2
p8      (이미지)은닉층 계산 h1=f(w11x1+w12x2+b1),h2=..., 출력 y=f(v1h1+v2h2+b3); 활성화 함수 Sigmoid/Tanh 식
p9      (이미지)ReLU max(0,x), Leaky ReLU(0.01x), Softmax 식
p10     순전파 4과정, 손실(MSE 회귀 / Cross Entropy 분류)
p11     역전파 과정, 경사하강법, MLP 장점
p12     MLP 단점(CNN/RNN 대비), 활용 분야, 1장 요약
[2장 최적화 함수] (텍스트)
p13     (빈)
p14     Optimizer 정의(손실 최소화 위해 W,b 갱신), 학습 흐름, 왜 필요(랜덤 초기화)
p15     (이미지)예측 20→45→70→90→99, 경사하강법 w_new=w_old-η∂L/∂w, 학습률(너무작음0.0000001/너무큼10)
p16-17  (이미지)적절 0.001~0.0001; 1.Batch GD 2.SGD(1개마다)
p18     Mini-Batch GD(가장 많이 사용, 배치32 예)
p19     (이미지)4.Momentum(이전 방향 기억), 5.NAG
p20-21  (이미지)6.AdaGrad(파라미터별 학습률, 희소데이터/NLP, 학습률 계속 감소 단점)
p22     (이미지)7.RMSProp(최근 기울기만, RNN 효과적) 8.Adam(Momentum+RMSProp, lr0.001 b1 0.9 b2 0.999 eps1e-8)
p23-24  (이미지)Adam 단점, 9.AdamW(weight decay 개선, Transformer/BERT/GPT/ViT) 10.Lion(Google, 메모리 적음)
p25     Optimizer 비교표(속도/안정성/메모리/활용도), 2장 요약(Adam·AdamW 표준, 이미지엔 SGD+Momentum)
[3장 RNN] (전부 이미지/개념도, 코드 없음)
p26     3장 표지 "RNN으로 항공 여행자 수 예측하기"
p27     (이미지)이미지 검색/dense caption 예시
p28     (이미지)순환신경망 RNN(Rumelhart 1986), 시계열, unfold 풀어헤침
p29     (이미지)RNN 종류(one-to-many/many-to-one[이번]/many-to-many), CNN-RNN 캡셔닝
p30-36  (이미지)LSTM: 기울기소실(vanishing), Hochreiter&Schmidhuber 1997, cell state, 4개 게이트(망각/입력/cell출력/현재단출력)
p37     (이미지)1950년대 비행기 여행(도입 사진)
p38     (이미지)Air Passengers Dataset(Kaggle, AirPassengers.csv, 1949.1~1960.12, 144개, 1000명 단위)
p39     (이미지)시계열 데이터 그래프(AP vs Time)
p40-41  (이미지)RNN·LSTM·Dataset 요약
p42-44  (이미지)MinMax 정규화 공식·종류(Standard/MinMax/Robust/Normalizer), 역전환 공식 x=scaled(max-min)+min, inverse_transform
p45-46  (이미지)시계열 분할 묶음(12입력+13번째 출력, sliding window), 파이썬 리스트로 분할
p47     (이미지)RNN 설계도: 12달 입력→13번째 예측, Keras Sequential LSTM, 셀 차원 300, 출력 sigmoid, [0,1] MinMax
p48     (이미지)Numpy Flatten: X_test 3D, Y_test 2D, Pred 2D, seaborn lineplot은 1D 필요
p49     (이미지)Sigmoid 활성화함수(미분 s'(x)=s(x)(1-s(x)), 기울기소실)
p50     (이미지)RMSprop(Hinton), 식 ν=ρν+(1-ρ)g², Δω=-η/√(ν+ε)·g
p51     (이미지)Keras RMSprop+MSE, Seaborn lineplot(pred vs truth 비교 그래프)
p52     (이미지)inverse_transform로 원래 단위 복귀, lineplot 시각화 요약
[4장 GAN] (전부 이미지/개념도, 코드 없음)
p53     4장 표지 "DNN-GAN으로 손글씨 모방하기"
p54-57  (이미지)GAN 응용 예시(스케치→실물, 나이변환, Monet↔photo, video synthesis)
p58     (이미지)GAN 소개(Ian Goodfellow 2014), 생성자/감별자
p59     (이미지)GAN 원리(Generator=위조범, Discriminator=탐정, Real/Fake)
p60     (이미지)생성자 학습방식/감별자 학습방식(가중치 고정·가변 주의점)
p61     (이미지)원논문 Algorithm 1(minibatch SGD)
p62     (이미지)생성자 손실(최소화)/감별자 손실(최대화) minimax 식
p63-64  (이미지)손글씨 판별, MNIST(NIST, 28×28 흑백 7만개=학습6만+평가1만)
p65     (이미지)Leaky ReLU(죽는 뉴런 방지, y=0.01x, Parametric ReLU)
p66     (이미지)생성자 설계도: 100→Dense128→leakyR→Dense128→leakyR→Dense784→Reshape(28×28×1)
p67     (이미지)감별자 설계도: 28×28×1→Flatten784→Dense128→leakyR→Dense1(sigmoid)
p68     (이미지)DNN-GAN 설계도(생성자+감별자 연결, 노이즈100→가짜(28,28,1)→판별1 sigmoid)
p69     (이미지)DNN-GAN 역전파(생성자 학습=감별자 고정+GAN전체 compile / 감별자 학습=감별자만)
p70     (이미지)train_on_batch vs fit 비교표
p71     (이미지)numpy zeros(가짜 라벨0)/ones(진짜 라벨1)로 학습 유도
p72     (이미지)샘플 이미지 생성(matplotlib subplots/imshow), DNN-GAN 학습(매 batch 감별자1회+생성자1회, epoch 반복)

[검토 메모]
  - 분량이 큼(4개 주제). 입문 범위 유지를 위해: MLP·Optimizer는 개념+핵심표+소형 실습, RNN·GAN은 개념 압축 + "설계도를 옮긴" 재구성 코드 1개씩.
  - 데이터 누수: LSTM은 시계열이므로 셔플 금지·시간순 분할. MinMax는 PDF가 전체에 적용하나, 엄밀히는 학습분할에만 fit 권장 → 🟩 주의로 안내.
  - 출력층/라벨/손실 조합 점검(항목7): XOR=sigmoid+BCE+0/1, LSTM=sigmoid([0,1])+MSE, GAN 감별자=sigmoid+BCE+ones/zeros.
  - 다음 DAY 주제: PDF/파일에서 확인 불가 → 예고하지 않음(항목9).
============================================================
-->

# 🧠 딥러닝 완전 입문 가이드 — DAY7. 다층 퍼셉트론(MLP)·옵티마이저·RNN·GAN

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: DAY1(개념·설치) · DAY2(신경망·계층구조) · DAY3(데이터 처리·데이터셋) · DAY4(네트워크 모델 설계) · DAY5(훈련·평가) · DAY6(모델 저장·활용)

> 💡 **이 글의 표기 약속 (꼭 읽어 주세요)**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 개념·설계도
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 사전지식·실무 주의점
> - ⚙️ **(설계도 재구성 코드)** : **이 강의 PDF에는 실행 코드가 없습니다.** 3장(RNN)·4장(GAN)은 "설계도(층 차원·활성화 함수·옵티마이저·라벨 방식)"만 그림으로 제시합니다. 그 설계도를 충실히 옮겨 **재구성한 Keras 코드**이며, PDF 원문 코드가 아닙니다.
> - 실행 코드는 모두 **하나의 프레임워크(Keras / `tf.keras`)** 로 통일했습니다.
> - ⚠️ 이 글을 쓴 환경에는 **TensorFlow가 설치되어 있지 않아 Keras 코드를 실행하지 못했습니다.** 손실·정확도·예측값을 임의로 적지 않았으니 직접 실행해 확인하세요. 데이터 윈도잉·정규화처럼 **직접 실행해 확인한 부분은 따로 표시**했습니다.

---

## 1. 이번 DAY에서 배우는 것

이번 강의는 분량이 큽니다(72쪽). 네 가지 주제를 한 번에 다룹니다.

1. **다층 퍼셉트론(MLP)** — 모든 딥러닝의 기본 구조, XOR 문제로 보는 "은닉층이 필요한 이유"
2. **최적화 함수(Optimizer)** — 가중치를 어떻게 수정할지 정하는 알고리즘(SGD → Adam → AdamW)
3. **RNN / LSTM** — 시계열을 다루는 신경망, *항공 여행자 수 예측*
4. **GAN** — 가짜를 만들어 내는 신경망, *손글씨 숫자 생성*

> 🟩 **(보충) 순서·범위 안내** — PDF 순서를 따르되, 각 주제는 **왜 필요한가 → 동작 원리 → (있다면) 코드 → 해석 → 주의** 흐름으로 재구성했습니다. RNN·GAN은 개념을 압축하고, **PDF 설계도를 그대로 옮긴 실행용 코드**를 한 개씩 붙였습니다(위 ⚙️ 표기 참고).

---

## 2. 다층 퍼셉트론(MLP)

### 2.1 퍼셉트론과 "단층의 한계" 🟦 (강의 PDF)

**퍼셉트론(Perceptron)** 은 사람의 뉴런을 수학으로 흉내 낸 가장 단순한 신경망입니다. 입력값에 가중치를 곱해 모두 더한 뒤(가중합), 활성화 함수를 적용해 출력합니다.

```text
입력(x1,x2,x3) → 가중치(w1,w2,w3) → 가중합 Σ → 활성화 함수 → 출력 y
```

수식으로는 가중합 `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`, 출력 `y = f(z)` 입니다. (`b`는 편향 Bias)

**단층 퍼셉트론**은 직선 하나로 나눌 수 있는 문제만 풉니다. AND·OR은 되지만 **XOR은 못 풉니다.**

| A | B | XOR 출력 |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

> 🟩 **(보충) 왜 XOR이 안 될까?** — XOR의 정답 `1`인 점(0,1),(1,0)과 정답 `0`인 점(0,0),(1,1)은 **직선 하나로 분리할 수 없습니다.** 이 "비선형" 문제를 풀려고 **은닉층을 추가한 것이 MLP**입니다.

### 2.2 MLP의 구조 🟦 (강의 PDF)

MLP는 입력층과 출력층 사이에 **은닉층(Hidden Layer)** 을 하나 이상 둡니다.

```text
입력층 → 은닉층1 → 은닉층2 → ... → 출력층
(예) 입력 4개 → 은닉 8개 → 은닉 4개 → 출력 2개
```

- **입력층**: 데이터를 받습니다. (예: 붓꽃 데이터면 꽃받침 길이/너비, 꽃잎 길이/너비 → 입력 노드 4개)
- **은닉층**: 입력의 특징을 추출합니다. 층이 많을수록 복잡한 패턴을 배웁니다.
- **출력층**: 최종 결과. **이진 분류는 노드 1개**(고양이/개), **다중 분류는 클래스 수만큼**(setosa/versicolor/virginica → 3개).

한 뉴런의 계산은 가중합 뒤 활성화 함수를 적용하는 식입니다(PDF p8 예).

```text
은닉층:  h₁ = f(w₁₁x₁ + w₁₂x₂ + b₁),   h₂ = f(w₂₁x₁ + w₂₂x₂ + b₂)
출력층:  y  = f(v₁h₁ + v₂h₂ + b₃)
```

### 2.3 활성화 함수(Activation Function) 🟦 (강의 PDF)

> **MLP의 핵심.** 활성화 함수가 없으면 층을 아무리 쌓아도 결국 하나의 선형식과 같아져, 은닉층을 둔 의미가 사라집니다.

| 함수 | 수식 | 출력 범위 | 특징 |
|---|---|---|---|
| **Sigmoid** | `1 / (1 + e⁻ˣ)` | 0~1 | 확률처럼 해석 가능, **기울기 소실** 발생 |
| **Tanh** | `(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)` | −1~1 | Sigmoid보다 학습이 잘 되는 편 |
| **ReLU** | `max(0, x)` | 0~∞ | 계산 단순·빠름, 현재 가장 널리 사용 |
| **Leaky ReLU** | `x>0:x, x≤0:0.01x` | −∞~∞ | "죽은 ReLU" 문제 완화 |
| **Softmax** | `eᶻⁱ / Σ eᶻʲ` | 0~1(합=1) | **다중 분류 출력층**에서 사용 |

> 🟩 **(보충)** "ReLU가 현재 가장 많이 쓰인다"는 PDF 표현은 시점에 따라 달라질 수 있는 경향입니다. 또 **하나의 함수가 항상 최고는 아니며**, 은닉층에는 보통 ReLU 계열, 출력층에는 문제 유형에 맞는 함수(이진=Sigmoid, 다중=Softmax, 회귀=활성화 없음)를 씁니다.

### 2.4 학습은 한 사이클의 반복 🟦 (강의 PDF)

MLP는 아래 한 사이클을 반복하며 가중치를 고쳐 갑니다. (자세한 수식은 DAY5·DAY6에서 다뤘습니다.)

```text
순전파(예측) → 손실 계산(MSE=회귀, Cross Entropy=분류)
            → 역전파(각 가중치의 기울기 계산) → 경사하강법(가중치 수정) → 반복
```

### 2.5 Python 실습: Keras로 XOR 풀기 ⚙️ (설계도 재구성 코드)

> 🟩 PDF는 XOR을 "단층이 못 푸는 문제"로 소개만 합니다. 아래는 그 XOR을 **MLP로 직접 풀어 보는 보충 예제**입니다. (프레임워크: **Keras**)

```python
import numpy as np
import tensorflow as tf

# XOR 데이터: 입력 (4,2), 정답 (4,1) — 이진 분류라 0/1 실수 라벨
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)  # shape (4, 2)
y = np.array([[0], [1], [1], [0]],            dtype=np.float32)    # shape (4, 1)

tf.random.set_seed(42)   # 재현성
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="relu"),      # 은닉층(비선형성 부여)
    tf.keras.layers.Dense(1, activation="sigmoid"),   # 출력층: 이진분류 → 노드 1개
])
# 출력 sigmoid + 라벨 0/1 → 손실은 binary_crossentropy 조합
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X, y, epochs=500, verbose=0)

print(model.predict(X))   # 4개 입력에 대한 예측 확률(0~1)
```

**무엇을 봐야 하나** — 학습이 잘 되면 `(0,1),(1,0)` 입력의 예측은 1에 가깝고, `(0,0),(1,1)` 입력은 0에 가깝게 나옵니다. **은닉층 덕분에** 단층이 못 풀던 XOR이 풀립니다.

> ⚠️ **(실행 안내)** 이 환경에 TensorFlow가 없어 실행하지 못했습니다. 예측 숫자는 임의로 적지 않았으니 직접 실행해 확인하세요. 무작위 초기화 때문에 실행마다 값이 조금 다를 수 있습니다.

> 🟩 **출력층·라벨·손실 조합 점검(중요)** — 이진 분류는 *출력 노드 1개 + Sigmoid + `binary_crossentropy` + 0/1 실수 라벨* 이 한 세트입니다. 다중 분류라면 *출력 노드 = 클래스 수 + Softmax + `categorical`/`sparse_categorical_crossentropy`* 로 바뀝니다.

### 2.6 MLP의 장단점과 활용 🟦 (강의 PDF)

- **장점**: 비선형·XOR 문제 해결, 다양한 분류/회귀에 적용, 딥러닝의 기본 구조.
- **단점**: 데이터가 많아야 성능이 나고, 학습이 오래 걸리며 과적합 위험. **이미지엔 CNN, 시계열엔 RNN/LSTM이 더 효율적.**
- **활용**: 손글씨 숫자 분류, 대출 심사·사기 탐지(금융), 질병 진단(의료), 불량품 판정(제조) 등.

---

## 3. 최적화 함수(Optimizer)

### 3.1 왜 필요한가 🟦 (강의 PDF)

신경망은 처음에 가중치를 **무작위**로 둡니다. 그래서 실제값 100인데 예측 20처럼 오차가 큽니다. 학습을 반복하며 `20 → 45 → 70 → 90 → 99`처럼 정답에 가까워지도록 **가중치를 조금씩 수정하는 일**, 그걸 하는 것이 **옵티마이저**입니다.

기본은 **경사하강법(Gradient Descent)**: 손실이 줄어드는 방향(기울기 반대)으로 이동합니다.

```text
w_new = w_old − η × (∂L/∂w)      η = 학습률(Learning Rate)
```

**학습률**은 한 걸음의 크기입니다. 🟦

- 너무 작으면(예 0.0000001): 학습이 매우 느림
- 너무 크면(예 10): 최적점을 지나쳐 **발산**
- 적절하면(예 0.001~0.0001): 안정적으로 수렴

### 3.2 데이터를 얼마나 보고 갱신할까 — 배치 방식 🟦 (강의 PDF)

| 방식 | 한 번 갱신에 쓰는 데이터 | 특징 |
|---|---|---|
| **Batch GD** | 전체 데이터 | 방향이 정확·안정적이나 느리고 메모리 큼 |
| **SGD** | 1개씩 | 빠르고 지역최적 탈출 가능하나 진동·불안정 |
| **Mini-Batch GD** | 32개 등 일부씩 | **현재 가장 많이 사용**, 속도·안정성 균형, GPU 활용 |

> 🟩 **(보충) 용어 정리** — 데이터 10,000개를 배치 크기 32로 나누면 한 번 다 도는 데(1 **epoch**) 약 313번 가중치를 갱신합니다. "epoch=전체를 몇 바퀴", "batch=한 번에 보는 묶음", "step=한 번의 갱신"입니다.

### 3.3 발전된 옵티마이저들 🟦 (강의 PDF)

기본 경사하강법을 개선하며 여러 알고리즘이 나왔습니다.

| Optimizer | 핵심 아이디어 |
|---|---|
| **Momentum** | 이전 이동 방향을 기억(관성) → 진동↓·수렴↑ |
| **NAG** | 미리 이동할 위치를 예측해 기울기 계산 → Momentum보다 정확 |
| **AdaGrad** | 파라미터마다 다른 학습률(희소 데이터·NLP에 강함). 단, 학습률이 계속 줄어 나중엔 거의 학습 안 됨 |
| **RMSProp** | AdaGrad 개선 — 최근 기울기만 반영. RNN 학습에 효과적 |
| **Adam** | **Momentum + RMSProp 결합.** 기본값 `lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8` |
| **AdamW** | Adam의 weight decay 개선. Transformer·BERT·GPT·ViT에서 사실상 표준 |
| **Lion** | Google 제안, Adam보다 메모리 적게 사용(대규모 언어모델) |

> 🟩 **(보충) 실무 선택** — PDF 요약대로 **대개 Adam 또는 AdamW로 시작**하고, 이미지 분류처럼 일반화 성능이 중요하면 **SGD+Momentum**도 많이 씁니다. "어느 하나가 항상 최고"는 아니며 문제·데이터에 따라 다릅니다.

### 3.4 Keras에서 옵티마이저 바꾸기 ⚙️ (보충)

> 🟩 PDF엔 코드가 없지만, 실제로는 `compile`의 `optimizer` 한 줄만 바꾸면 됩니다.

```python
# 같은 모델에서 옵티마이저만 교체하는 예
model.compile(optimizer="adam",                 loss="mse")   # Adam
# model.compile(optimizer="rmsprop",            loss="mse")   # RMSProp
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
#               loss="mse")                                   # SGD+Momentum
```

---

## 4. RNN과 LSTM — 시계열 예측

> 🟦 PDF 실습 주제: **"RNN으로 항공 여행자 수 예측하기"**

### 4.1 RNN이 필요한 이유 🟦 (강의 PDF)

**순환신경망(RNN, Recurrent Neural Network)** 은 **시계열(시간 순서가 있는) 데이터**를 다루는 신경망입니다(Rumelhart, 1986). 이전 시점(t)의 정보를 다음 시점(t+1)으로 **연결**해 흘려보내는 것이 핵심이며, AI 번역·음성 인식·주가 예측 등에 쓰입니다.

RNN은 입력 길이만큼 같은 구조를 **풀어 헤쳐(unfold)** 계산합니다. RNN 종류 중 이번 예제는 **여러 입력 → 하나의 출력(many-to-one)** 형태입니다(과거 12개월 → 다음 1개월).

### 4.2 기울기 소실과 LSTM 🟦 (강의 PDF)

기본 RNN은 시점이 멀어질수록 **기울기 소실(vanishing gradient)** 로 옛 정보를 잊습니다. 이를 해결한 것이 **LSTM(Long-Short Term Memory)**(Hochreiter & Schmidhuber, 1997)입니다.

LSTM의 핵심은 **셀 상태(cell state)** 라는 "컨베이어 벨트"로, 정보를 거의 그대로 멀리까지 전달합니다(셀 상태 라인 자체엔 활성화 함수가 없어 기울기 소실에서 자유롭습니다). 정보를 더하거나 빼는 일은 **게이트(gate)** 가 맡습니다.

| 게이트 | 역할 |
|---|---|
| **망각 게이트** | 이전 정보를 얼마나 **잊을지** 결정 (0~1) |
| **입력(새 정보) 게이트** | 새 입력을 얼마나 **받아들일지** 결정 |
| **출력 게이트** | 다음 단계로 전할 셀 상태의 양을 결정 |

> 🟩 **(보충)** 게이트는 Sigmoid(0~1, "얼마나 통과시킬지")와 Tanh(−1~1, "어떤 값을 더할지")를 조합해 만듭니다. 자세한 게이트 수식은 입문 범위를 넘어가니, "셀 상태 + 게이트로 장기 기억을 조절한다"는 직관만 잡아도 충분합니다.

### 4.3 실습 설계: 항공 여행자 수 예측 🟦 (강의 PDF)

- **데이터**: Kaggle의 *Air Passengers*(`AirPassengers.csv`). 1949년 1월~1960년 12월 매월 여행자 수, **총 144개**(1,000명 단위).
- **정규화**: 값 범위가 크므로 **MinMax 정규화 [0,1]** → `x_scaled = (x − min) / (max − min)`. 예측 후에는 **역정규화** `x = x_scaled × (max − min) + min`(scikit-learn `inverse_transform`)로 원래 단위 복원.
- **윈도우 묶기**: **과거 12개월(입력) → 13번째 달(정답)**. 한 칸씩 밀며 여러 묶음을 만듭니다(sliding window).
- **설계도(PDF p47)**: `Keras Sequential` → `LSTM(셀 차원 300)` → `Dense(1, 출력 sigmoid)`. 입력을 [0,1]로 정규화했으므로 출력도 0~1인 Sigmoid를 씁니다. 옵티마이저 **RMSProp**, 손실 **MSE**.

먼저 **데이터 준비 로직**은 프레임워크가 필요 없어 **직접 실행해 확인**했습니다.

```python
# (직접 실행해 확인한 부분) 윈도우 묶기 + 정규화 왕복
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 실제로는 pandas로 CSV를 읽습니다:
#   import pandas as pd
#   series = pd.read_csv("AirPassengers.csv")["#Passengers"].values.astype("float32")
# 여기서는 형태 확인용으로 144개짜리 예시 시계열을 사용합니다.
series = np.arange(1, 145, dtype="float32").reshape(-1, 1)   # (144, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)        # [0,1]로 정규화, shape (144,1)

W = 12                                        # 과거 12개월
X, y = [], []
for i in range(len(scaled) - W):
    X.append(scaled[i:i+W, 0])                # 12개월 입력
    y.append(scaled[i+W, 0])                  # 13번째 달 정답
X = np.array(X).reshape(-1, W, 1)             # (132, 12, 1) ← LSTM 입력 형태
y = np.array(y).reshape(-1, 1)               # (132, 1)
print("X:", X.shape, "y:", y.shape)

# 역정규화 왕복 확인
restored = scaler.inverse_transform(scaled)
print("역정규화 복원 일치:", np.allclose(restored, series))
```

**실행 결과(확인됨):**

```text
X: (132, 12, 1) y: (132, 1)
역정규화 복원 일치: True
```

> 🟩 **(보충) LSTM 입력은 3차원** — `(샘플 수, 시점 길이, 특징 수)` = `(132, 12, 1)`. 시점마다 값이 1개(여행자 수)뿐이라 마지막 차원이 1입니다.

### 4.4 Python 실습: LSTM 모델 ⚙️ (설계도 재구성 코드)

> 위 설계도(p47~p52)를 그대로 옮긴 **Keras 재구성 코드**입니다. 데이터 준비(4.3)에 이어집니다.

```python
import tensorflow as tf

# 시계열은 순서가 중요 → 셔플하지 않고 "시간 순서대로" 분할
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(W, 1)),                      # (시점 12, 특징 1)
    tf.keras.layers.LSTM(300),                         # 셀 차원 300 (PDF 설계도)
    tf.keras.layers.Dense(1, activation="sigmoid"),    # 출력 [0,1]
])
model.compile(optimizer="rmsprop", loss="mse")         # RMSProp + MSE (PDF)
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

# 예측 → 역정규화로 "원래 여행자 수" 단위 복원
pred = scaler.inverse_transform(model.predict(X_test))   # (n,1)
truth = scaler.inverse_transform(y_test)

# 시각화: 예측 vs 실제 (seaborn lineplot은 1차원 입력 → flatten)
import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x=range(len(pred)),  y=pred.flatten(),  label="pred")
sns.lineplot(x=range(len(truth)), y=truth.flatten(), label="truth")
plt.title("Air Passengers: pred vs truth")
plt.show()
```

**무엇을 봐야 하나** — 예측선(pred)이 실제선(truth)의 **증가 추세와 계절적 출렁임**을 얼마나 따라가는지 봅니다. PDF는 "MSE 손실값 하나로는 부족하므로 그래프로 예측값과 정답을 함께 비교"하라고 안내합니다.

> ⚠️ **(실행 안내)** TensorFlow 미설치로 모델 학습·예측은 실행하지 못했습니다(데이터 준비 부분만 실행 확인). 손실·예측 수치를 임의로 적지 않았으니 직접 실행해 확인하세요. 데이터셋은 Kaggle에서 `AirPassengers.csv`를 받아야 합니다.

> 🟩 **(보충) 데이터 누수 주의** — 위는 PDF처럼 **전체 시계열에 MinMax를 fit** 했습니다. 더 엄밀히는 **학습 구간에만 `fit`** 하고 평가 구간엔 `transform`만 적용해야 미래 정보가 새지 않습니다. 또 시계열은 **절대 셔플하지 말고** 시간 순서대로 나눠야 합니다.

> 🟩 **출력층·라벨·손실 조합 점검** — 입력을 [0,1]로 정규화했기에 출력도 Sigmoid([0,1])로 맞췄고, 연속값 예측이라 손실은 MSE입니다. 만약 정규화를 표준화(평균0·분산1)로 바꾸면 출력 Sigmoid는 더 이상 맞지 않습니다(범위가 0~1을 벗어남). **정규화 방식과 출력 활성화는 한 세트로 함께 결정**해야 합니다.

---

## 5. GAN — 손글씨 숫자 생성

> 🟦 PDF 실습 주제: **"DNN-GAN으로 손글씨 모방하기"**

### 5.1 GAN의 아이디어 🟦 (강의 PDF)

**GAN(Generative Adversarial Network)**(Ian Goodfellow, 2014)은 **두 신경망이 경쟁하며 함께 성장**하는 방식입니다.

- **생성자(Generator) = 위조범**: 노이즈로부터 **가짜 이미지**를 만들어 감별자를 속이려 합니다.
- **감별자(Discriminator) = 탐정**: 주어진 이미지가 **진짜인지 가짜인지** 판별합니다.

생성자는 더 그럴듯한 가짜를, 감별자는 더 날카로운 판별을 학습하며 서로를 밀어 올립니다(minimax: 감별자는 맞힐 확률을 **최대화**, 생성자는 들킬 확률을 **최소화**).

### 5.2 DNN-GAN 설계도 🟦 (강의 PDF)

- **데이터**: MNIST 손글씨(28×28 흑백, 총 7만 장 = 학습 6만 + 평가 1만).
- **활성화**: 은닉층은 **Leaky ReLU**("죽는 뉴런" 방지).
- **생성자(p66)**: `노이즈(100)` → `Dense(128)` → LeakyReLU → `Dense(128)` → LeakyReLU → `Dense(784)` → `Reshape(28,28,1)`
- **감별자(p67)**: `이미지(28,28,1)` → `Flatten(784)` → `Dense(128)` → LeakyReLU → `Dense(1, sigmoid)`(진짜일 확률)
- **DNN-GAN(p68)**: 생성자의 출력(가짜 이미지)을 감별자의 입력으로 연결.

### 5.3 학습 방식 🟦 (강의 PDF)

- **라벨**: NumPy `ones`(=1, 진짜) / `zeros`(=0, 가짜)로 감별자에게 정답을 줍니다.
- **`train_on_batch`**: 한 배치로 한 번 가중치를 갱신하는 Keras 함수. 매 배치마다 **감별자 학습(진짜 1회 + 가짜 1회) → 생성자 학습**을 번갈아 진행합니다.
- **가중치 고정**: 생성자를 학습할 때는 **감별자 가중치를 고정**(DNN-GAN 전체를 묶어 학습), 감별자를 학습할 때는 감별자만 갱신합니다.

### 5.4 Python 실습: DNN-GAN ⚙️ (설계도 재구성 코드)

> 위 설계도(p66~p72)를 옮긴 **Keras 재구성 코드**입니다. GAN은 이번 글에서 가장 고급 주제이므로 **전체 구조를 잡는 것**에 집중하세요.

```python
import numpy as np
import tensorflow as tf

# 0) MNIST: 28x28 흑백 → [0,1] 정규화 → (N,28,28,1)
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28, 28, 1)

NOISE_DIM = 100

# 1) 생성자: 노이즈(100) → 가짜 이미지(28,28,1)
generator = tf.keras.Sequential([
    tf.keras.Input(shape=(NOISE_DIM,)),
    tf.keras.layers.Dense(128), tf.keras.layers.LeakyReLU(0.01),
    tf.keras.layers.Dense(128), tf.keras.layers.LeakyReLU(0.01),
    tf.keras.layers.Dense(784, activation="sigmoid"),   # 픽셀값 [0,1]
    tf.keras.layers.Reshape((28, 28, 1)),
], name="generator")

# 2) 감별자: 이미지(28,28,1) → 진짜일 확률(1, sigmoid)
discriminator = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),                          # 784
    tf.keras.layers.Dense(128), tf.keras.layers.LeakyReLU(0.01),
    tf.keras.layers.Dense(1, activation="sigmoid"),
], name="discriminator")
discriminator.compile(optimizer="adam", loss="binary_crossentropy")

# 3) DNN-GAN(결합): 생성자 → (감별자 고정) → 판별 결과
discriminator.trainable = False
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(optimizer="adam", loss="binary_crossentropy")

# 4) 학습: 매 배치마다 감별자(진짜+가짜) → 생성자
BATCH = 128
real_label = np.ones((BATCH, 1), dtype="float32")    # 진짜 = 1
fake_label = np.zeros((BATCH, 1), dtype="float32")   # 가짜 = 0

for step in range(2000):   # 데모용 횟수(실제로는 더 많이 반복)
    # (a) 감별자 학습
    idx = np.random.randint(0, x_train.shape[0], BATCH)
    real_imgs = x_train[idx]
    noise = np.random.normal(0, 1, (BATCH, NOISE_DIM)).astype("float32")
    fake_imgs = generator.predict(noise, verbose=0)

    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_imgs, real_label)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)

    # (b) 생성자 학습: 가짜를 "진짜(1)"로 속이도록 유도
    discriminator.trainable = False
    noise = np.random.normal(0, 1, (BATCH, NOISE_DIM)).astype("float32")
    g_loss = gan.train_on_batch(noise, real_label)

# 5) 샘플 이미지 확인 (matplotlib subplots + imshow)
import matplotlib.pyplot as plt
noise = np.random.normal(0, 1, (16, NOISE_DIM)).astype("float32")
samples = generator.predict(noise, verbose=0)        # (16,28,28,1)
fig, axes = plt.subplots(4, 4, figsize=(4, 4))
for img, ax in zip(samples, axes.flatten()):
    ax.imshow(img.squeeze(), cmap="gray")
    ax.axis("off")
plt.show()
```

**무엇을 봐야 하나** — 학습 초반에는 노이즈 같은 그림이, 학습이 진행될수록 점점 **숫자 비슷한 형태**가 나타나는지 봅니다. GAN은 학습이 까다로워(두 손실의 균형) 결과가 들쭉날쭉할 수 있습니다.

> ⚠️ **(실행 안내)** TensorFlow 미설치로 실행하지 못했습니다. 생성 품질·손실값을 임의로 적지 않았으니 직접 실행해 확인하세요. GAN 학습은 CPU에서 느리므로 **GPU 환경(예: Colab)** 을 권장합니다.

> 🟩 **(보충) 재구성에서 둔 가정** — PDF 설계도는 생성자 마지막 활성화 함수를 명시하지 않습니다. MNIST를 [0,1]로 정규화했으므로 여기서는 생성자 출력에 **Sigmoid**를 사용했습니다(데이터 범위와 일치). 만약 [-1,1]로 정규화한다면 보통 출력에 **Tanh**를 씁니다 — **정규화 범위와 출력 활성화는 함께 맞춰야 합니다.**

---

## 6. 자주 하는 실수

- **MLP에 활성화 함수를 빼먹음** — 층을 쌓아도 선형 모델과 같아져 XOR 같은 비선형 문제를 못 풉니다.
- **출력층·라벨·손실 불일치** — 이진=Sigmoid+BCE+0/1, 다중=Softmax+CrossEntropy, 회귀=활성화 없음+MSE. 짝을 맞춰야 합니다.
- **시계열을 셔플하거나 전체에 정규화 fit** — 미래 정보가 과거로 새는 **데이터 누수**가 생깁니다. 시간순 분할 + 학습 구간에만 `fit`.
- **정규화 범위와 출력 활성화 불일치** — [0,1] 정규화엔 Sigmoid, [-1,1]엔 Tanh.
- **LSTM 입력 차원 실수** — `(샘플, 시점, 특징)` 3차원이어야 합니다. 2차원으로 넣으면 에러.
- **GAN에서 생성자 학습 시 감별자를 같이 갱신** — 생성자 학습 단계에선 감별자를 **고정(`trainable=False`)** 해야 합니다.
- **실행하지 않은 손실·정확도를 단정적으로 적기** — 직접 실행해 확인한 값만 신뢰하세요.

---

## 7. DAY7 핵심 정리

```text
MLP (다층 퍼셉트론)
  - 은닉층 + 활성화 함수로 비선형 문제(XOR) 해결
  - 출력층/라벨/손실은 한 세트(이진=Sigmoid+BCE, 다중=Softmax+CE, 회귀=MSE)

Optimizer (최적화 함수)
  - 경사하강법 + 학습률이 기본, Mini-Batch가 표준
  - SGD→Momentum→AdaGrad→RMSProp→Adam→AdamW 로 발전, 보통 Adam/AdamW로 시작

RNN / LSTM (시계열)
  - RNN은 시점을 연결, 멀면 기울기 소실 → LSTM이 셀 상태+게이트로 장기 기억
  - 항공 여행자 예측: 12개월→13번째, MinMax[0,1]+LSTM+Sigmoid+MSE, 예측 후 inverse_transform
  - 시계열은 셔플 금지, 학습 구간에만 정규화 fit

GAN (생성)
  - 생성자(위조범) vs 감별자(탐정)의 경쟁 학습
  - DNN-GAN: 노이즈(100)→생성자→가짜(28,28,1)→감별자→진짜확률(sigmoid)
  - train_on_batch로 감별자·생성자 번갈아 학습, 생성자 학습 땐 감별자 고정
```

> 다음 DAY 주제는 이번 강의 PDF에서 확인할 수 없어 따로 예고하지 않습니다.

---

## 참고 자료

- 강의 자료: `DAY7_딥러닝_다중퍼셉트론.pdf` (교과목 2 — 데이터 분석과 머신러닝/딥러닝, 단원 3)
- ※ 이 PDF에는 실행 코드가 없으며, 본문의 RNN·GAN·XOR 코드는 PDF의 **설계도를 옮겨 재구성한 Keras 예제**입니다.
- Keras 공식 문서 — *The Sequential model*, *LSTM layer*, *train_on_batch*
- scikit-learn 공식 문서 — `MinMaxScaler`, `inverse_transform`
- 데이터셋 — Kaggle *Air Passengers* (`AirPassengers.csv`), MNIST(`tf.keras.datasets.mnist`)
