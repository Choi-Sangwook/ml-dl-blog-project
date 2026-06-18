<!--
============================================================
기획 문서 (드래프트 메타) — 게시 전 이 주석 블록은 제거 예정
============================================================

[작업 모드] Generate (Stage 1: 초안)
[1차 출처] sources/DAY8_딥러닝_오버피팅방지기법.pdf (총 25페이지)
[분야] 딥러닝 / [독자] 파이썬 기본은 알지만 딥러닝은 처음
[이전 편] DAY1~DAY7

[프레임워크 결정]
  - 이 PDF는 Keras와 PyTorch를 섞어 씀:
      · Keras: 데이터 증강(ImageDataGenerator, p4), 조기 종료(EarlyStopping, p5)
      · PyTorch: 드롭아웃·Weight Decay·BatchNorm·전이학습·라벨 스무딩·종합 예시(p5~12), 2장 전체(p14~25)
  - 유일한 "완성형 실행 예제"가 PyTorch(RegularizedModel, p11~12)이고 2장도 전부 PyTorch.
    → 실행 코드는 PyTorch로 통일(항목6). 데이터 증강·조기 종료는 PyTorch 대응으로 작성하고,
      PDF가 Keras로 보여준 부분은 🟦로 따로 표기(딥러닝 체크리스트1: 프레임워크 전환 명시).

[실행 환경 한계 — 임의 결과값 금지(항목8)]
  - torch 미설치 → PyTorch 코드(종합 모델 등)는 실행 못함. 손실/정확도 임의 기입 금지, "직접 실행" 안내.
  - 직접 실행해 확인한 것(프레임워크 무관):
      · 라벨 스무딩 [0,0,1,0] --(0.1)--> [0.025,0.025,0.925,0.025]  (PDF p9와 정확히 일치)
      · 정규화 x'=(x-min)/(max-min) → [0,1] / 표준화 z=(x-μ)/σ → 평균0·표준편차1  (실행 확인)
  - 종합 모델의 출력 shape (batch,10)은 계층 정의로 결정되는 "구조적 사실"이라 명시(임의 결과값 아님).

------------------------------------------------------------
PDF 페이지 → 핵심 주제 지도 (전 25페이지, 이미지/수식 페이지 시각 확인 완료)
------------------------------------------------------------
[1장 오버피팅 방지 기법]
p3   오버피팅 정의/발생 이유/확인(정상·오버피팅·언더피팅 표; 곡선: Epoch↑ Train↓ Val↓후↑)
p4   3.1 데이터 증강(표 + Keras ImageDataGenerator)
p5   3.2 Dropout(PyTorch nn.Dropout 0.5; 비율 0.2~0.5), 3.3 Early Stopping(Keras EarlyStopping, patience, restore_best_weights)
p6   3.4 Weight Decay/L2(Loss=예측오차+λ×가중치²의 합; AdamW weight_decay=0.01), 3.5 모델 단순화(표)
p7   3.5 복잡↔단순 모델 비교 코드(PyTorch), 3.6 BatchNorm(nn.BatchNorm1d)
p8   3.7 데이터 추가(표), 3.8 전이학습(torchvision resnet18 + fc 교체)
p9   3.9 교차검증(5-Fold), 3.10 라벨 스무딩([0,0,1,0]→[0.025,..,0.925,..]; CrossEntropyLoss label_smoothing=0.1)
p10  4. 기법 비교표(10종), 5. 실무 조합(이미지: 증강+전이+WD+ES / 일반: Dropout+BN+WD+ES / 데이터 적을때 6단계)
p11-12 6. 종합 PyTorch 예시(RegularizedModel: fc1→bn1→relu→dropout→fc2; CrossEntropyLoss(label_smoothing); AdamW)
p13  1장 요약(데이터 다양화/모델 단순화/과도학습 방지/가중치 억제/검증 기준 관리)
[2장 최적 모델 학습]
p14  최적 모델 = 일반화 최고, underfitting↔overfitting 중간
p15  조건(1) 낮은 손실(MSE 식), (2) 높은 정확도(Accuracy=정답/전체, 95/100=95%)
p16  (3) 일반화(Train99/Val98=좋음, 99/70=오버피팅); 절차 단계1 데이터수집(MNIST/CIFAR10/ImageNet/IMDB/Airline)
p17  단계2 EDA(df.info/describe/isnull), 단계3 전처리(정규화 0~1 / 표준화 평균0·표준편차1)
p18  단계4 분할(70/15/15 또는 80/10/10), 모델 설계(nn.Sequential 784-256-128-10)
p19  손실(회귀 MSE/MAE, 분류 BCE/CE), 최적화 SGD
p20  Momentum/RMSProp/Adam/AdamW(weight_decay)
p21  하이퍼파라미터(LR 시작 0.001, Batch Size, Epoch, Dropout, WD)
p22  오버피팅 방지 요약(Dropout/BN/WD/DataAug/ES), 평가(Accuracy/Precision)
p23  Recall/F1 식, 최적 모델 선택(①Val Loss최소 ②Val Acc최대 ③Test ④저장 torch.save state_dict)
p24  실무 조합(이미지 ResNet50+AdamW+CE+BN+Dropout+증강+ES / 텍스트 BERT+AdamW+CE+WD+ES / 시계열 LSTM+Adam+MSE+ES)
p25  정리 8단계 + 실무 조합(AdamW/CE/WD+Dropout/BN/ES/Lowest Val Loss)

[검토 메모]
  - 2장은 DAY5(훈련·평가)·DAY7(옵티마이저)과 크게 겹침 → 압축하고 "전체 워크플로 정리 + 모델 선택 기준"에 집중(반복 금지).
  - 10개 기법을 PDF의 평면 나열 대신 "데이터/모델/학습/평가" 4갈래로 묶어 재구성(입문자 이해용).
  - 출력층·라벨·손실 조합 점검(항목7): 종합 모델 = 로짓(batch,10) + 정수라벨 long + CrossEntropyLoss(softmax 안 붙임).
  - BatchNorm·Dropout → train()/eval() 구분 강조.
  - 정규화/표준화는 train에만 fit(데이터 누수) 주의. 교차검증은 고전 ML 기법이고 DL에선 비용 큼을 명시.
  - 다음 DAY 주제: PDF에서 확인 불가 → 예고 안 함(항목9).
============================================================
-->

# 🛡️ 딥러닝 완전 입문 가이드 — DAY8. 오버피팅 방지 기법과 최적 모델 학습

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: DAY1(개념·설치) · … · DAY5(훈련·평가) · DAY6(모델 저장·활용) · DAY7(MLP·옵티마이저·RNN·GAN)

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 개념·수식·코드
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 설명·실무 주의점
> - **프레임워크 안내**: PDF는 **Keras와 PyTorch를 섞어** 설명합니다(데이터 증강·조기 종료는 Keras, 나머지와 종합 예제는 PyTorch). 이 글은 **실행 코드를 PyTorch로 통일**하고, PDF가 Keras로 보여준 부분은 🟦로 따로 표시했습니다.
> - ⚠️ 작성 환경에 PyTorch가 없어 **PyTorch 코드는 실행하지 못했습니다.** 손실·정확도를 임의로 적지 않았습니다. **라벨 스무딩·정규화 수치는 직접 실행해 확인**했습니다.

---

## 1. 이번 DAY에서 배우는 것

- **오버피팅(과적합)이 무엇이고 왜 생기는지**, 그리고 **어떻게 알아채는지**
- 오버피팅을 막는 **대표 기법 10가지**를 "데이터 / 모델 / 학습 / 평가" 네 갈래로 정리
- 여러 기법을 합친 **종합 PyTorch 예제**
- 이 모든 것을 묶은 **최적 모델 학습 워크플로**와 **모델 선택 기준**

> 🟩 **(보충) 재구성 안내** — PDF는 기법을 3.1~3.10으로 평면 나열하지만, 이 글은 **"왜 효과가 있는가"** 기준으로 묶어 재배치했습니다. 2장(최적 모델 학습)은 DAY5·DAY7과 겹치는 부분이 많아 **반복을 줄이고 워크플로·모델 선택 기준**에 집중합니다.

---

## 2. 오버피팅이란? 🟦 (강의 PDF)

**오버피팅(Overfitting, 과적합)** 은 모델이 **학습 데이터는 매우 잘 맞추지만 새로운 데이터에는 성능이 떨어지는** 현상입니다. 일반적인 패턴이 아니라 학습 데이터의 **세부 특징·노이즈·우연한 패턴까지 외워버린** 상태입니다.

> 🟦 예) 고양이/강아지 분류 모델은 *귀 모양·얼굴 형태·털 패턴* 같은 일반 특징을 배워야 하는데, 특정 사진의 **배경·조명·촬영 각도까지 외우면** 새 사진에서 성능이 떨어집니다.

**왜 생기나** — 딥러닝 모델은 파라미터가 많아 표현력이 큽니다. 표현력이 크면 복잡한 문제를 잘 풀지만, **데이터가 부족하거나 모델이 너무 복잡하면** 데이터를 통째로 외워버릴 수 있습니다.

**어떻게 확인하나** — **학습 손실과 검증 손실의 차이**로 봅니다.

| 구분 | 학습 데이터 | 검증 데이터 | 의미 |
|---|---|---|---|
| 정상 학습 | 좋음 | 좋음 | 일반화 잘 됨 |
| **오버피팅** | 매우 좋음 | 나쁨 | 학습 데이터를 외움 |
| 언더피팅 | 나쁨 | 나쁨 | 충분히 학습 못 함 |

대표적인 신호는 이렇습니다. 🟦

```text
Epoch ↑
Train Loss      ↓ 계속 감소
Validation Loss ↓ 감소하다가 다시 증가  ← 이 지점부터 오버피팅
```

> 🟩 **(보충)** 검증 손실이 다시 오르기 시작하는 지점이 "외우기 시작하는" 순간입니다. 뒤에 나오는 **조기 종료**가 바로 이 지점을 노립니다.

---

## 3. 오버피팅을 막는 방법 — 4갈래로 정리

> 🟩 PDF의 10가지 기법을 **무엇에 손을 대는가**로 묶었습니다: ① 데이터 ② 모델 복잡도 ③ 학습 방식 ④ 평가.

### 3.1 데이터 측면 — 가장 근본적인 해결

**왜** — 오버피팅은 **데이터가 부족할 때** 자주 생깁니다. 데이터가 많고 다양하면 모델이 특정 샘플을 외우기보다 일반 패턴을 배웁니다. 🟦

**① 데이터 추가** 🟦 — 가장 근본적입니다. 실제 데이터 추가 수집(최선) · 공개 데이터셋 활용 · 데이터 증강 · 합성 데이터 생성.

**② 데이터 증강(Data Augmentation)** 🟦 — 기존 데이터를 변형해 새 학습 데이터처럼 씁니다. 이미지라면 회전·좌우 반전·확대/축소·밝기 조절·자르기·노이즈 추가. 고양이 사진 1장을 회전·밝기 변형하면 모델은 *다양한 고양이*를 학습하는 효과를 얻습니다.

> 🟦 **PDF는 이 부분을 Keras로 보여줍니다** — `ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)`.

🟩 이 글은 PyTorch로 통일하므로 **`torchvision.transforms`** 로 같은 일을 합니다.

```python
# 🟩 (PyTorch) 데이터 증강 — 학습 데이터에만 적용
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomRotation(20),                 # 최대 20도 회전
    transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),  # 확대/축소 + 자르기
    transforms.RandomHorizontalFlip(),             # 좌우 반전
    transforms.ToTensor(),
])
# 검증/테스트에는 증강을 적용하지 않습니다(ToTensor 등 최소 변환만).
```

> 🟩 **(보충) 주의 2가지** — ① 증강은 **학습 데이터에만** 적용합니다(검증·테스트에 적용하면 평가가 왜곡됩니다). ② **라벨이 바뀌는 변형은 피하세요**. 예를 들어 숫자 `6`을 180도 뒤집으면 `9`가 되므로, 손글씨 숫자에 상하 반전은 부적절합니다.

**③ 전이학습(Transfer Learning)** 🟦 — 대규모 데이터로 미리 학습된 모델을 가져와 내 문제에 맞게 다시 학습합니다. 데이터가 적을 때 처음부터 학습하면 오버피팅이 쉽지만, **이미 일반 특징을 아는 모델에서 시작**하면 적은 데이터로도 좋은 성능을 얻기 쉽습니다.

```python
# 🟦 (PyTorch) 전이학습 — ImageNet 사전학습 ResNet18 활용
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 사전학습 가중치
# 마지막 출력층을 "내 문제의 클래스 수"(예: 2)에 맞게 교체
model.fc = nn.Linear(model.fc.in_features, 2)
```

> 🟩 **(보충)** `weights=...DEFAULT`는 처음 실행 시 **가중치를 인터넷에서 내려받습니다**. 보통 앞쪽 층은 고정(freeze)하고 새로 바꾼 출력층 위주로 학습하지만, 데이터·상황에 따라 전체를 미세조정하기도 합니다(DAY6의 미세조정 참고).

### 3.2 모델 복잡도 측면 — 외울 여력을 줄이기

**① 모델 단순화** 🟦 — 데이터가 1,000개뿐인데 수백만 파라미터 모델을 쓰면 외우기 쉽습니다. 은닉층 수·뉴런 수·파라미터를 줄여 표현력을 제한합니다.

```python
# 🟦 (PyTorch) 같은 문제, 복잡한 모델 vs 단순한 모델
import torch.nn as nn

complex_model = nn.Sequential(           # 파라미터 많음 → 적은 데이터에선 과적합 위험↑
    nn.Linear(784, 1024), nn.ReLU(),
    nn.Linear(1024, 512), nn.ReLU(),
    nn.Linear(512, 10),
)
simple_model = nn.Sequential(            # 데이터가 적을수록 이런 단순 모델이 안전
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
```

**② 가중치 감쇠(Weight Decay = L2 정규화)** 🟦 — 가중치가 너무 크면 모델이 특정 입력에 과도하게 민감해집니다. 손실 함수에 **가중치 크기 벌점**을 더해 이를 억제합니다.

```text
일반 손실:  Loss = 예측 오차
L2 적용:    Loss = 예측 오차 + λ × (가중치²의 합)     (λ = 정규화 강도)
```

```python
# 🟦 (PyTorch) AdamW로 Weight Decay 적용
import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

> 🟩 **(보충)** `weight_decay` 값이 클수록 가중치를 더 강하게 억제합니다. 너무 크면 학습이 잘 안 되니 `0.01`처럼 작은 값에서 시작합니다.

**③ 드롭아웃(Dropout)** 🟦 — 학습 중 **일부 뉴런을 무작위로 끕니다.** 특정 뉴런에 지나치게 의존하는 것을 막아, 여러 뉴런이 고르게 일하도록 만듭니다.

```python
# 🟦 (PyTorch) 드롭아웃
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),      # 학습 중 뉴런의 50%를 무작위 비활성화
    nn.Linear(256, 10),
)
```

> 🟩 **(보충)** 드롭아웃 비율은 보통 **0.2~0.5**. 너무 높으면 학습이 잘 안 됩니다. 또 드롭아웃은 **학습할 때만** 작동하고 **예측할 때는 꺼져야** 합니다 → PyTorch에서는 `model.train()` / `model.eval()`로 전환합니다(아래 5장에서 강조).

**④ 라벨 스무딩(Label Smoothing)** 🟦 — 정답 라벨을 **100% 확신하지 않도록** 살짝 부드럽게 만듭니다. 모델이 정답을 너무 강하게 외우는 것을 줄입니다.

```text
원래 정답:      [0,    0,    1,     0   ]
라벨 스무딩 후: [0.025, 0.025, 0.925, 0.025]   (smoothing = 0.1, 4개 클래스)
```

> 🟩 **(보충, 직접 계산 확인)** 클래스가 `K`개, 스무딩 `α`일 때 *정답 클래스 = (1−α) + α/K*, *나머지 = α/K* 입니다. `α=0.1, K=4`면 정답 `0.9 + 0.1/4 = 0.925`, 나머지 `0.025` — **PDF 수치와 정확히 일치함을 직접 계산해 확인했습니다.**

```python
# 🟦 (PyTorch) 라벨 스무딩이 적용된 손실 함수
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 3.3 학습 방식 측면 — 안정적으로, 적당히

**① 배치 정규화(Batch Normalization)** 🟦 — 각 층으로 들어가는 값의 분포를 **평균·분산을 조정해 일정하게** 맞춥니다. 주목적은 **학습 안정화**지만, 약한 정규화 효과가 있어 오버피팅 완화에도 도움이 됩니다.

```python
# 🟦 (PyTorch) 배치 정규화
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),   # 256개 출력값을 정규화
    nn.ReLU(),
    nn.Linear(256, 10),
)
```

**② 조기 종료(Early Stopping)** 🟦 — **검증 성능이 더 이상 좋아지지 않으면 학습을 멈춥니다.** 오버피팅은 보통 *너무 오래 학습*할 때 생기므로, 검증 손실이 일정 기간 개선되지 않으면 중단하고 **가장 좋았던 시점의 가중치로 되돌립니다.**

> 🟦 **PDF는 이 부분을 Keras로 보여줍니다** — `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`. (예: 20번째 epoch에서 검증 손실이 가장 낮았고 이후 5번 개선이 없으면, 멈추고 20번째 상태로 복원.)

🟩 PyTorch에는 내장 콜백이 없어 **직접 구현**합니다(패턴만 보여주는 부분 코드).

```python
# 🟩 (PyTorch) 조기 종료 패턴 — 학습 루프 안에서 검증 손실을 추적
best_val = float("inf")
patience, bad = 5, 0
for epoch in range(100):
    # ... 학습 1 epoch 수행 ...
    val_loss = evaluate(model, val_loader)   # 검증 손실 계산(사용자 함수)
    if val_loss < best_val:
        best_val = val_loss
        bad = 0
        torch.save(model.state_dict(), "best_model.pth")  # 가장 좋은 가중치 저장
    else:
        bad += 1
        if bad >= patience:        # 5번 연속 개선 없으면 중단
            print(f"조기 종료: epoch {epoch}")
            break
# 학습 후 가장 좋았던 가중치를 다시 불러와 사용
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
```

### 3.4 평가 측면 — 성능을 안정적으로 가늠하기

**교차 검증(Cross Validation)** 🟦 — 데이터를 여러 조각으로 나눠 **여러 번 학습·검증을 반복**합니다. 5-Fold면 데이터를 5등분해, 매번 1조각은 검증·나머지 4조각은 학습으로 쓰고 5번 반복합니다. 데이터가 적을 때 성능을 **더 안정적으로 평가**하는 데 도움이 됩니다.

> 🟩 **(보충)** 교차 검증은 엄밀히는 *오버피팅을 막는* 기법이라기보다 **성능을 신뢰성 있게 평가**하는 기법입니다. 또 딥러닝에서는 한 번 학습이 오래 걸려 **k번 반복하는 비용이 커서**, 보통은 교차 검증 대신 **train/validation/test 한 번 분할**을 더 자주 씁니다(교차 검증은 데이터가 적은 경우에 한정). 그리고 정규화·증강 같은 전처리는 **각 fold의 학습 부분에서만 학습(fit)** 해야 정보가 새지 않습니다.

---

## 4. 기법 한눈에 보기 🟦 (강의 PDF)

| 기법 | 핵심 목적 | 적용 상황 |
|---|---|---|
| 데이터 증강 | 데이터 다양성 증가 | 이미지·음성·텍스트 |
| 드롭아웃 | 특정 뉴런 의존 방지 | 완전연결층, 일부 CNN |
| 조기 종료 | 과도한 학습 방지 | 거의 모든 학습 |
| Weight Decay | 가중치 크기 억제 | 대부분의 딥러닝 모델 |
| 모델 단순화 | 모델 복잡도 감소 | 데이터가 적을 때 |
| 배치 정규화 | 학습 안정화 | CNN, DNN |
| 데이터 추가 | 일반화 성능 향상 | 가장 근본적 해결 |
| 전이학습 | 사전 지식 활용 | 데이터가 적을 때 |
| 교차 검증 | 평가 안정화 | 데이터가 적을 때 |
| 라벨 스무딩 | 과도한 확신 방지 | 분류 문제 |

> 🟦 **실무 조합** — 하나만 쓰기보다 여러 개를 함께 씁니다.
> - 이미지 분류: **데이터 증강 + 전이학습 + Weight Decay + Early Stopping**
> - 일반 분류: **Dropout + BatchNorm + Weight Decay + Early Stopping**
> - 데이터가 적을 때 순서: **데이터 증강 → 전이학습 → 모델 축소 → Dropout → Weight Decay → Early Stopping**

---

## 5. 종합 예제: 여러 기법을 한 모델에 🟦 (강의 PDF, PyTorch)

PDF의 종합 예제입니다. **BatchNorm + Dropout + Weight Decay + Label Smoothing**을 한 번에 적용합니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RegularizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)      # 입력 784 → 은닉 256
        self.bn1 = nn.BatchNorm1d(256)      # 배치 정규화(학습 안정화)
        self.dropout = nn.Dropout(p=0.5)    # 드롭아웃(뉴런 의존 방지)
        self.fc2 = nn.Linear(256, 10)       # 출력 10개 클래스(로짓)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # (batch, 1, 28, 28) → (batch, 784)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)             # (batch, 10) 로짓 — softmax는 붙이지 않음
        return x

model = RegularizedModel().to(device)

# 출력이 로짓 → CrossEntropyLoss(내부에서 log-softmax 처리) + 라벨 스무딩
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# AdamW = Adam + Weight Decay(가중치 감쇠)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**텐서·출력·라벨·손실 점검** 🟩 (항목7)
- 입력: `(batch, 1, 28, 28)` → `view`로 `(batch, 784)`, `float32`.
- 출력: `(batch, 10)` **로짓**(softmax 없음). 예: 입력 `(32,1,28,28)` → 출력 `(32,10)` — 구조상 결정되는 형태입니다.
- 라벨: **정수 class index** `torch.long`, shape `(batch,)`.
- 손실: `nn.CrossEntropyLoss`는 **로짓을 받아 내부에서 log-softmax를 처리**합니다 → 모델 끝에 `Softmax`를 붙이지 않습니다(붙이면 이중 적용).

> ⚠️ **(실행 안내)** 작성 환경에 PyTorch가 없어 학습은 실행하지 못했습니다. 위 코드는 모델·손실·옵티마이저 **구성**을 보여주며, 손실·정확도 수치를 임의로 적지 않았습니다. 실제 학습에는 데이터 로더와 학습 루프가 더 필요합니다.

> 🟩 **(보충) 가장 중요한 주의 — `train()` / `eval()`** — **BatchNorm과 Dropout은 학습할 때와 예측할 때 동작이 다릅니다.** 학습 루프 전에는 `model.train()`, 검증·예측 전에는 `model.eval()` + `torch.no_grad()`로 전환해야 결과가 일관됩니다. 이걸 빠뜨리면 "학습은 되는데 예측이 이상한" 대표적 버그가 생깁니다.

---

## 6. 최적 모델 학습 워크플로 🟦 (강의 PDF)

> 🟩 이 장은 DAY5(훈련·평가)·DAY7(옵티마이저)과 겹치는 부분이 많아 **핵심 흐름과 새로운 부분(모델 선택 기준)** 위주로 정리합니다.

**최적 모델이란** — 학습 정확도만 높은 게 아니라 **새로운 데이터에서의 일반화 성능이 가장 높은** 모델입니다. 너무 단순하면 언더피팅, 너무 복잡하면 오버피팅 → 그 **중간 지점**이 최적입니다. 🟦

**최적 모델의 조건** 🟦
- **낮은 손실값** — 회귀는 `MSE = (1/N)Σ(yᵢ − ŷᵢ)²`. 낮을수록 좋음.
- **높은 정확도** — 분류는 `Accuracy = 정답 수 / 전체 수`(예: 100개 중 95개 → 95%).
- **높은 일반화 성능(가장 중요)** — Train 99% / **Val 98%** 면 좋은 모델, Train 99% / **Val 70%** 면 오버피팅.

**전체 절차** 🟦

| 단계 | 핵심 |
|---|---|
| 1. 데이터 수집 | 대표 데이터셋: 이미지(MNIST·CIFAR-10·ImageNet), 텍스트(IMDB), 시계열(Airline Passenger) |
| 2. 데이터 탐색(EDA) | `df.info()`, `df.describe()`, `df.isnull().sum()` — 결측치·이상치·분포·클래스 불균형 확인 |
| 3. 전처리 | 정규화 `x'=(x−min)/(max−min)`→[0,1] / 표준화 `z=(x−μ)/σ`→평균0·표준편차1 |
| 4. 데이터 분할 | Train/Val/Test = **70/15/15** 또는 80/10/10 |
| 5. 모델 설계 | 예: `nn.Sequential(nn.Linear(784,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU(), nn.Linear(128,10))` |
| 6. 손실/옵티마이저 | 회귀=MSE/MAE, 분류=BCE/CrossEntropy / 옵티마이저는 보통 **Adam·AdamW**로 시작 |
| 7. 하이퍼파라미터 | Learning Rate(시작 `0.001`), Batch Size, Epoch, Dropout, Weight Decay |
| 8. 오버피팅 방지 | 3장의 기법들 적용 |

> 🟩 **(보충) 정규화·표준화 직접 확인** — `x'=(x−min)/(max−min)`는 값을 `[0,1]`로, `z=(x−μ)/σ`는 평균 0·표준편차 1로 만듭니다(직접 실행해 확인). 둘 다 **통계량(min/max, μ/σ)은 학습 데이터에서만 계산**해 검증·테스트에 적용해야 미래 정보가 새지 않습니다(DAY7의 데이터 누수 참고).

> 🟩 **(보충) 옵티마이저 표현** — PDF는 Adam을 "가장 많이", AdamW를 "매우 많이 사용"으로 적지만, 이런 표현은 **시점에 따라 달라지는 경향**입니다. 보통 Adam/AdamW로 시작한다고 이해하면 충분합니다.

### 모델 선택 기준 🟦

학습이 끝난 뒤 **어떤 모델을 고를지**는 보통 이 순서로 판단합니다.

```text
① Validation Loss 최소   (가장 중요)
        ↓
② Validation Accuracy 최대 (분류 문제)
        ↓
③ Test 성능 확인          (최종 일반화 성능 — 모델 선택이 끝난 뒤 마지막에 1번만)
        ↓
④ 모델 저장
```

```python
# 🟦 (PyTorch) 가장 좋은 모델 저장 (DAY6 참고)
torch.save(model.state_dict(), "best_model.pth")
```

> 🟩 **(보충) 평가 지표** — 분류는 정확도 외에 **정밀도(Precision)·재현율(Recall)·F1**(= `2·P·R/(P+R)`)을 함께 봅니다. 클래스가 불균형하면 정확도만으로는 부족하기 때문입니다(DAY5의 평가 지표 참고). **Test 세트는 모델 선택이 끝난 뒤 마지막에 한 번만** 보고, 거기에 맞춰 다시 튜닝하지 않습니다.

> 🟦 **실무 조합 예시** — 이미지: `ResNet50 + AdamW + CrossEntropyLoss + BatchNorm + Dropout + 데이터 증강 + EarlyStopping` / 텍스트: `BERT + AdamW + CrossEntropyLoss + Weight Decay + EarlyStopping` / 시계열: `LSTM + Adam + MSELoss + EarlyStopping`.

---

## 7. 자주 하는 실수

- **증강을 검증·테스트에도 적용** — 평가가 왜곡됩니다. 증강은 **학습 데이터에만**.
- **라벨을 바꾸는 증강** — 숫자 `6`을 뒤집어 `9`로 만드는 식. 의미가 달라지는 변형은 금지.
- **예측 전에 `model.eval()` 누락** — BatchNorm·Dropout이 학습 모드로 동작해 결과가 흔들립니다.
- **정규화·표준화를 전체 데이터로 fit** — 미래 정보가 새는 데이터 누수. **학습 데이터에만 fit**.
- **`CrossEntropyLoss` 앞에 `Softmax` 추가** — 내부에서 처리하므로 이중 적용이 됩니다.
- **Dropout 비율을 너무 높게(예 0.8)** — 학습 자체가 안 될 수 있습니다(보통 0.2~0.5).
- **Test 세트로 튜닝** — Test는 최종 1회 확인용. 거기에 맞추면 또 다른 오버피팅입니다.
- **실행하지 않은 손실·정확도를 단정적으로 적기** — 직접 실행해 확인한 값만 신뢰.

---

## 8. DAY8 핵심 정리

```text
오버피팅 = 학습엔 강하고 새 데이터엔 약함 → Train Loss와 Validation Loss의 "차이"로 확인

막는 방법 (4갈래)
  데이터:  데이터 추가 · 데이터 증강 · 전이학습
  모델:    모델 단순화 · Weight Decay(L2) · Dropout · Label Smoothing
  학습:    BatchNorm · Early Stopping
  평가:    Cross Validation(데이터 적을 때)

종합 예제(PyTorch): BatchNorm + Dropout + AdamW(weight_decay) + CrossEntropyLoss(label_smoothing)
  - 출력은 로짓(softmax 안 붙임), 라벨은 정수 long
  - 학습 train() / 예측 eval()+no_grad() 전환 필수

최적 모델 워크플로
  데이터수집 → EDA → 전처리(정규화/표준화, train에만 fit) → 분할(70/15/15)
  → 설계 → 손실·옵티마이저 → 하이퍼파라미터 → 오버피팅 방지
  모델 선택: ① Val Loss 최소 ② Val Acc 최대 ③ Test 확인(마지막 1회) ④ 저장
```

> 다음 DAY 주제는 이번 강의 PDF에서 확인할 수 없어 따로 예고하지 않습니다.

---

## 참고 자료

- 강의 자료: `DAY8_딥러닝_오버피팅방지기법.pdf` (교과목 2 — 데이터 분석과 머신러닝/딥러닝, 단원 3)
- ※ PDF는 데이터 증강·조기 종료를 **Keras**로, 나머지와 종합 예제를 **PyTorch**로 설명합니다. 이 글은 실행 코드를 **PyTorch로 통일**하고 Keras 부분은 🟦로 표시했습니다.
- PyTorch 공식 문서 — `nn.Dropout`, `nn.BatchNorm1d`, `CrossEntropyLoss(label_smoothing)`, `optim.AdamW`
- torchvision 공식 문서 — `transforms`, `models.resnet18`
- 설치(환경별 선택) — <https://pytorch.org/get-started/locally/>
