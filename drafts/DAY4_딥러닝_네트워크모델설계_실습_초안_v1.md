<!--
============================================================
기획 문서 (드래프트 메타) — 게시 전 이 주석 블록은 제거 예정
============================================================

[작업 모드] Generate (Stage 1: 초안) — DAY4 "실습편"
[1차 출처] 제공된 Jupyter 노트북: fashion_pytorch_cnn.ipynb
            (sources/에 대응 PDF 없음 → 노트북이 supplied source material)
[개념편] posts/DAY4_딥러닝_네트워크모델설계_최종본(_v2).md 와 짝을 이루는 '실습'
[분야] 딥러닝 / [독자] 파이썬 기본은 알지만 딥러닝은 처음
[프레임워크] PyTorch (DAY2·DAY4 개념편과 일관)

------------------------------------------------------------
1) 노트북 셀 → 핵심 주제 지도 (전 셀 확인)
------------------------------------------------------------
- 도입 md: 핵심 흐름 6단계 + Keras(NHWC) vs PyTorch(NCHW) 차이
- 1 import + device(cpu) + seed(42)
- 2 하이퍼파라미터: M_EPOCH=20, M_BATCH=300, M_CLASS=10, M_LR=0.001, class_names 10개
- 3 데이터: FashionMNIST(train 60000/test 10000), ToTensor(0~1), 샘플 (1,28,28) label 9 Ankle boot
- 4 샘플 시각화(이미지+픽셀값)
- 5 DataLoader: train shuffle=True, test shuffle=False, 배치 (300,1,28,28) 라벨 (300,)
- 6~7 CNN 정의: conv1(1→32,k2,same)→pool→conv2(32→64,k2,same)→pool→flatten→fc1(3136→128)→fc2(128→10)
      더미 출력 (1,10), padding='same'+짝수커널 UserWarning
- 8 손실/옵티마이저: CrossEntropyLoss + Adam(lr=0.001) (출력층에 softmax 없음, 라벨=정수)
- 9 train_one_epoch: train(), zero_grad/backward/step, running_loss, acc
- 10 evaluate: eval()+no_grad(), 예측 수집, confusion_matrix, f1_score
- 11 학습 20epoch: E1 acc76.61% → E20 acc94.79%, 총 1391.6초(CPU)
- 12 학습곡선(loss↓, acc↑) 시각화
- 13 테스트: loss 0.2558, acc 91.08%, 혼동행렬, F1(micro) 0.911
- 14 혼동행렬 시각화
- 15 예측 샘플 10장 시각화

[실제 출력(노트북에서 실행 확인된 값만 사용 — 임의 생성 X)]
- device: cpu / train 60000, test 10000
- 샘플: (1,28,28), label 9 = Ankle boot
- 배치 입력 (300,1,28,28), 라벨 (300,), 앞10개 [5,7,4,7,3,8,9,5,3,1]
- model summary: conv1/pool1/conv2/pool2/relu/fc1(in=3136,out=128)/fc2(in=128,out=10), 더미출력 (1,10)
- UserWarning: padding='same' + 짝수 커널
- 학습 로그 20줄: E20 Loss 0.1442 Acc 94.79%, 총 1391.6초
- 테스트 Loss 0.2558, Acc 91.08%, 혼동행렬(10x10), F1 0.911

------------------------------------------------------------
2) 재구성(집필) 계획 — 입문자 순서
------------------------------------------------------------
- DAY4 개념(출력층·손실·설계)을 '실제로 돌려보는' 실습으로 연결.
- 순서: 왜 이미지엔 CNN인가 → 데이터/전처리(NCHW) → CNN 구조(conv·pool·flatten·fc, shape 추적)
        → 손실·옵티마이저(CrossEntropy=logits·정수라벨, Adam) → 학습 루프 → 평가(acc·혼동행렬·F1) → 결과 해석 → 주의.
- 코드: 노트북 기준 PyTorch 일관. 복붙 실행 가능하도록 import 포함. 주석은 입문용으로 다듬되 동작은 동일.
- DL 점검: shape/dtype/device/출력층/라벨/손실 명시.

[보충/주의(🟩)]
- 출력층에 Softmax 없음(=DAY4 핵심). 노트북 md의 Keras식 구조설명(Dense softmax)과 실제 PyTorch 코드 차이를 교육 포인트로.
- padding='same' + 짝수 커널(2) → PyTorch 경고. 보통 홀수 커널(3) 권장.
- validation 세트 없음(train/test만). train 94.79% vs test 91.08% 간극(가벼운 과적합 신호) → 모델 선택엔 validation 필요(DAY3·DAY4 원칙).
- 단일 라벨 다중분류에서 micro-F1 = accuracy (0.911 ≈ 0.9108).
- CPU 1391.6초 → GPU/에폭 축소 안내.
- 다음 DAY 주제는 추측하지 않음.
============================================================
-->

# 🧠 딥러닝 완전 입문 가이드 — DAY4 실습. PyTorch CNN으로 패션 이미지 분류 (Fashion-MNIST)

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **짝이 되는 개념편**: DAY4 — 딥러닝 네트워크 모델 설계(출력층·손실함수·모델 종류)
> **프레임워크**: PyTorch (DAY2·DAY4와 동일)

> 💡 **이 글의 표기 약속**
> - 🟦 **(실습 노트북)** : 제공된 실습 노트북 `fashion_pytorch_cnn.ipynb` 에 있는 코드·출력
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 설명·주의점
> - 본문에 적힌 **학습/평가 수치(정확도·혼동행렬·F1 등)는 노트북에서 실제 실행된 출력값**입니다. (임의로 지어내지 않았습니다. 무작위 초기화·환경에 따라 재실행하면 소수점은 달라질 수 있습니다.)
> - 이 글은 별도 PDF가 아니라 **실습 노트북을 1차 출처**로 정리한 DAY4의 실습편입니다.

---

## 1. 이번 실습에서 하는 것

DAY4(개념편)에서 배운 **"문제 유형 → 출력층 → 손실함수"** 설계를 **직접 돌려봅니다.** 손글씨 대신 **옷 사진(Fashion-MNIST)** 10종을 분류하는 **CNN(합성곱 신경망)** 을 PyTorch로 처음부터 끝까지 만듭니다.

```
① 데이터 불러오기·정규화  →  ② CNN 설계  →  ③ 손실·옵티마이저 설정
→  ④ 학습  →  ⑤ 평가(정확도·혼동행렬·F1)  →  ⑥ 결과 해석
```

DAY4에서 강조한 핵심이 코드로 어떻게 나타나는지 확인하는 게 목표입니다.

- 다중 분류 출력층 = **클래스 수(10)만큼 logit**, **Softmax는 모델에 넣지 않음**
- 손실 = `nn.CrossEntropyLoss` (내부에 LogSoftmax 포함), **라벨은 정수 인덱스**
- 옵티마이저 = `Adam`
- 평가 = 정확도 + **혼동행렬** + **F1**

---

## 2. 왜 이미지에는 CNN인가, 그리고 PyTorch의 이미지 모양 🟩 + 🟦

DAY3에서 MNIST를 **MLP**로 풀 때는 28×28 이미지를 **784개로 펼쳐서(Flatten)** 넣었습니다. 하지만 펼치면 **상하좌우 이웃 구조(공간 정보)** 가 입력에서 사라집니다. **CNN**은 작은 필터를 이미지 위에서 미끄러뜨리며 **국소적인 패턴(선·모서리·질감)** 을 직접 학습하도록 설계돼, 이미지에 더 잘 맞습니다.

> 🟦 **(노트북) 프레임워크마다 이미지 텐서 축 순서가 다릅니다.**
> - Keras: `(데이터수, 높이, 너비, 채널)` → 예 `(60000, 28, 28, 1)` (NHWC)
> - PyTorch: `(데이터수, 채널, 높이, 너비)` → 예 `(60000, 1, 28, 28)` (NCHW)
>
> 그래서 PyTorch CNN에는 **`1 × 28 × 28`(채널·높이·너비)** 형태의 이미지를 넣습니다.

> 🟩 **(보충)** Fashion-MNIST는 MNIST와 크기·형식이 똑같은(28×28 흑백, 10클래스, train 6만/test 1만) **대체 데이터셋**으로, 숫자 대신 **티셔츠·바지·신발 같은 옷 이미지**라 조금 더 어렵습니다. 입문 실습용으로 널리 쓰입니다.

---

## 3. 준비 — 라이브러리·장치·하이퍼파라미터 🟦

```python
import numpy as np
import matplotlib.pyplot as plt
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, f1_score

# 재현성을 위한 시드 고정(완전한 일치는 아닐 수 있음)
torch.manual_seed(42)
np.random.seed(42)

# GPU가 있으면 GPU, 없으면 CPU — 모델과 데이터는 같은 장치에 있어야 함
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)
```

**실행 결과**

```text
사용 장치: cpu
```

하이퍼파라미터(사람이 정하는 학습 설정값)는 다음과 같습니다.

```python
M_EPOCH = 20      # 전체 데이터를 20번 반복 학습
M_BATCH = 300     # 한 번에 300장씩 처리
M_CLASS = 10      # 클래스 10종
M_LR    = 0.001   # 학습률(Adam 기본값으로 흔히 사용)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

> 🟩 **(보충)** 모델은 정답을 **숫자(0~9)** 로 다루지만, 결과를 사람이 읽을 땐 `class_names` 로 이름을 붙이는 게 직관적입니다.

---

## 4. 데이터 불러오기와 전처리 🟦

`torchvision.datasets.FashionMNIST` 로 내려받고, **`transforms.ToTensor()`** 로 전처리합니다. `ToTensor()` 는 두 가지를 자동으로 합니다 — ① 이미지를 PyTorch 텐서로 변환, ② 픽셀값을 **0~255 → 0~1** 로 정규화.

```python
transform = transforms.Compose([
    transforms.ToTensor()           # (1,28,28) 텐서로, 픽셀 0~1
])

train_dataset = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

print("학습용 데이터 개수:", len(train_dataset))
print("평가용 데이터 개수:", len(test_dataset))

sample_image, sample_label = train_dataset[0]
print("샘플 이미지 텐서 모양:", sample_image.shape)
print("샘플 데이터 라벨 번호:", sample_label, "/ 이름:", class_names[sample_label])
```

**실행 결과**

```text
학습용 데이터 개수: 60000
평가용 데이터 개수: 10000
샘플 이미지 텐서 모양: torch.Size([1, 28, 28])
샘플 데이터 라벨 번호: 9 / 이름: Ankle boot
```

샘플 한 장은 **`(1, 28, 28)` = 채널 1 × 높이 28 × 너비 28** 입니다. 첫 번째 데이터의 정답은 9번(Ankle boot)이네요.

> 🟦 **(노트북) 데이터가 '숫자 배열'임을 눈으로 확인** — 이미지를 그려 보고 픽셀값 일부를 출력하면 0~1 범위로 정규화된 게 보입니다.
> ```python
> plt.imshow(sample_image.squeeze(), cmap="gray")   # (1,28,28)→(28,28)
> plt.title(f"Label: {class_names[sample_label]}"); plt.axis("off"); plt.show()
> ```

### 4.1 DataLoader — 미니배치로 공급

`DataLoader` 는 전체 데이터를 **배치 단위**로 잘라 모델에 넣어 줍니다. 학습 데이터는 매 epoch 순서를 섞고(`shuffle=True`), 테스트는 섞지 않습니다(`shuffle=False`).

```python
train_loader = DataLoader(train_dataset, batch_size=M_BATCH, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=M_BATCH, shuffle=False)

batch_images, batch_labels = next(iter(train_loader))
print("미니배치 입력 모양:", batch_images.shape)   # (배치, 채널, 높이, 너비)
print("미니배치 라벨 모양:", batch_labels.shape)
print("라벨 샘플:", batch_labels[:10])
```

**실행 결과**

```text
미니배치 입력 데이터 모양: torch.Size([300, 1, 28, 28])
미니배치 정답 라벨 모양: torch.Size([300])
미니배치 라벨 샘플: tensor([5, 7, 4, 7, 3, 8, 9, 5, 3, 1])
```

> 🟩 **(보충) 여기서 DAY4의 핵심이 보입니다** — 라벨이 `(300, 10)` One-Hot이 아니라 **`(300,)` 정수**입니다. `nn.CrossEntropyLoss` 가 **정수 클래스 인덱스**를 받기 때문입니다.

---

## 5. CNN 모델 설계 🟦

노트북의 모델 구조(개념)는 다음과 같습니다.

```text
입력 (1, 28, 28)
 → Conv2d(1→32, kernel 2, padding='same') → ReLU → MaxPool2d(2)
 → Conv2d(32→64, kernel 2, padding='same') → ReLU → MaxPool2d(2)
 → Flatten
 → Linear(64*7*7=3136 → 128) → ReLU
 → Linear(128 → 10)            # 출력층: logit 10개 (Softmax 없음)
```

> 🟩 **(보충, 아주 중요) 출력층에 Softmax가 없습니다** — DAY4에서 다룬 그대로입니다. PyTorch `nn.CrossEntropyLoss` 가 **내부에서 LogSoftmax를 처리**하므로, 모델 마지막은 확률이 아니라 **raw 점수(logits)** 를 내보냅니다. (노트북 설명글에는 Keras식으로 `Dense(10, activation='softmax')` 라고 적혀 있지만, **실제 PyTorch 코드에는 softmax를 넣지 않습니다.** 이 차이를 꼭 기억하세요.)

```python
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=2, padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(64 * 7 * 7, 128)   # 28→(pool)14→(pool)7 → 64채널×7×7
        self.fc2   = nn.Linear(128, M_CLASS)       # 출력 10 (logits)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))   # (N,1,28,28) → (N,32,14,14)
        x = self.pool2(self.relu(self.conv2(x)))   # → (N,64,7,7)
        x = x.view(x.size(0), -1)                  # Flatten → (N, 3136)
        x = self.relu(self.fc1(x))                 # → (N, 128)
        x = self.fc2(x)                            # → (N, 10) logits
        return x

model = FashionCNN().to(device)
print(model)

dummy = torch.randn(1, 1, 28, 28).to(device)       # 더미 입력 1장
print("더미 출력 모양:", model(dummy).shape)
```

**실행 결과**

```text
FashionCNN(
  (conv1): Conv2d(1, 32, kernel_size=(2, 2), stride=(1, 1), padding=same)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1), padding=same)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu): ReLU()
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
더미 출력 모양: torch.Size([1, 10])
```

**텐서 모양(shape) 추적** — CNN에서 모양이 어떻게 줄어드는지 보세요.

```
입력         (N, 1, 28, 28)
conv1+pool1  (N, 32, 14, 14)   # padding='same'으로 conv 후 28 유지 → pool로 절반(14)
conv2+pool2  (N, 64, 7, 7)     # 다시 절반(7)
flatten      (N, 3136)         # 64×7×7 = 3136  ← fc1의 in_features와 일치해야 함
fc1+relu     (N, 128)
fc2(출력)    (N, 10)           # 클래스 10개 logit
```

> ⚠️ **(주의) `padding="same"` + 짝수 커널(2) 경고** — 이 조합에서 PyTorch는 다음 경고를 띄웁니다(동작은 하지만 권장 형태는 아님).
> ```text
> UserWarning: Using padding='same' with even kernel lengths ... may require a zero-padded copy ...
> ```
> 🟩 입문 단계에서는 **홀수 커널(예: `kernel_size=3`)** 을 쓰면 이런 경고 없이 크기 계산도 깔끔합니다. 노트북은 원본 그대로 둔 것이니, 직접 바꿔 보며 비교해도 좋습니다.

---

## 6. 손실함수와 옵티마이저 🟦

DAY4의 "다중 분류 → Softmax(내장) + CrossEntropy" 짝을 그대로 씁니다.

```python
criterion = nn.CrossEntropyLoss()                  # 입력=logits, 정답=정수 라벨
optimizer = optim.Adam(model.parameters(), lr=M_LR)
```

| 항목 | 이 실습에서의 값 | DAY4 개념 연결 |
|---|---|---|
| 모델 출력 | `(N, 10)` float **logits** | 다중 분류 → 클래스 수만큼 출력, Softmax 없음 |
| 정답 라벨 | `(N,)` `torch.long` (0~9) | CrossEntropyLoss는 정수 인덱스 |
| 손실함수 | `nn.CrossEntropyLoss` | 내부 LogSoftmax 포함 |
| 옵티마이저 | `Adam(lr=0.001)` | 무난한 기본 선택 |

---

## 7. 학습 루프 🟦

PyTorch는 학습 루프를 직접 작성합니다. 핵심 3줄(`zero_grad → backward → step`)은 DAY2에서 다룬 그대로입니다.

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                                  # 학습 모드
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                      # ① 이전 기울기 초기화
        outputs = model(images)                    # ② 순전파 → logits (N,10)
        loss = criterion(outputs, labels)          # ③ 손실
        loss.backward()                            # ④ 역전파
        optimizer.step()                           # ⑤ 가중치 갱신

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)   # 가장 큰 logit의 클래스
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total   # 평균 손실, 정확도

train_losses, train_accuracies = [], []
begin = time()
for epoch in range(1, M_EPOCH + 1):
    loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(loss); train_accuracies.append(acc)
    print(f"Epoch [{epoch:02d}/{M_EPOCH}] Loss: {loss:.4f} Accuracy: {acc*100:.2f}%")
print("총 학습 시간: {:.1f}초".format(time() - begin))
```

**실행 결과 (일부)**

```text
CNN 학습 시작
Epoch [01/20] Loss: 0.6698 Accuracy: 76.61%
Epoch [05/20] Loss: 0.2874 Accuracy: 89.71%
Epoch [10/20] Loss: 0.2232 Accuracy: 91.80%
Epoch [15/20] Loss: 0.1802 Accuracy: 93.44%
Epoch [20/20] Loss: 0.1442 Accuracy: 94.79%
총 학습 시간: 1391.6초
```

손실은 꾸준히 내려가고 학습 정확도는 **76.61% → 94.79%** 로 올라갑니다.

> 🟩 **(보충) CPU에서 약 1392초(≈23분)** 걸렸습니다. 시간이 부담되면 **GPU 런타임**을 쓰거나, 처음엔 `M_EPOCH` 를 3~5로 줄여 동작만 확인한 뒤 늘리세요.

---

## 8. 평가 — 정확도·혼동행렬·F1 🟦

평가 때는 **`model.eval()` + `torch.no_grad()`** 로 전환합니다(DAY4에서 강조한 부분 — Dropout/BatchNorm 모드 전환 + 기울기 끄기).

```python
def evaluate(model, loader, criterion, device):
    model.eval()                                   # 평가 모드
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():                          # 기울기 계산 안 함
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss/total, correct/total, np.array(all_labels), np.array(all_preds)

test_loss, test_acc, truth, pred = evaluate(model, test_loader, criterion, device)
print("최종 테스트 손실값: {:.4f}".format(test_loss))
print("최종 정확도: {:.2f}%".format(test_acc * 100))
print(confusion_matrix(truth, pred))
print("F1 점수: {:.3f}".format(f1_score(truth, pred, average="micro")))
```

**실행 결과**

```text
최종 테스트 손실값: 0.2558
최종 정확도: 91.08%
혼동 행렬
[[891   0  22   7   3   1  74   0   2   0]
 [  3 978   1  11   2   0   4   0   1   0]
 [ 19   1 918   4  32   0  26   0   0   0]
 [ 30   5  18 892  30   0  21   0   4   0]
 [  3   0  93  22 836   0  46   0   0   0]
 [  0   0   0   0   0 989   0   9   1   1]
 [125   1  85  18  58   0 707   0   6   0]
 [  0   0   0   0   0  19   0 963   1  17]
 [  4   0   0   2   2   3   3   2 984   0]
 [  0   0   1   0   0  16   0  33   0 950]]
F1 점수: 0.911
```

### 8.1 결과 해석 🟩

- **테스트 정확도 91.08%** — 학습 정확도(94.79%)보다 약 3.7%p 낮습니다. 둘의 간극은 **가벼운 과적합** 신호입니다(훈련 데이터를 약간 더 잘 외움).
- **혼동행렬 읽기**: 행 = 실제 클래스, 열 = 예측 클래스. 대각선이 정답입니다.
  - 가장 헷갈리는 클래스는 **6번 Shirt(셔츠)** — 707개만 맞고, **T-shirt(0)로 125개, Pullover(2)로 85개, Coat(4)로 58개** 잘못 분류됐습니다. 모두 **상의 계열**이라 모양이 비슷해서 그렇습니다.
  - 반대로 **Trouser(978)·Bag(984)·Sandal(989)** 처럼 형태가 뚜렷한 클래스는 잘 맞힙니다.
- **F1(micro) 0.911 ≈ 정확도 0.9108** — 🟩 한 장당 정답이 하나뿐인 **단일 라벨 다중분류에서는 micro-F1이 정확도와 같습니다.** 그래서 이 경우 둘이 거의 일치합니다. (불균형이 심하거나 클래스별로 따로 보고 싶으면 `average="macro"` 를 쓰면 클래스별 F1의 평균을 봅니다.)

> 🟩 **(보충) 학습/검증 곡선으로 과적합 보기** — 노트북은 학습 손실·정확도 곡선을 그립니다. 손실이 내려가고 정확도가 올라가면 학습이 진행 중이라는 뜻입니다.
> ```python
> epochs = range(1, M_EPOCH + 1)
> plt.plot(epochs, train_losses, marker="o", label="Train Loss"); plt.legend(); plt.show()
> plt.plot(epochs, train_accuracies, marker="o", label="Train Accuracy"); plt.legend(); plt.show()
> ```

---

## 9. 예측 샘플 확인 🟦

테스트 이미지 몇 장을 골라 **실제(T) vs 예측(P)** 을 함께 표시하면 모델이 어디서 맞고 틀리는지 눈으로 볼 수 있습니다.

```python
images = torch.stack([test_dataset[i][0] for i in range(10)]).to(device)
labels = [test_dataset[i][1] for i in range(10)]

model.eval()
with torch.no_grad():
    _, predicted = torch.max(model(images), dim=1)
predicted = predicted.cpu().numpy()

plt.figure(figsize=(15, 4))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(images[i].cpu().squeeze(), cmap="gray")
    plt.title(f"T:{class_names[labels[i]]}\nP:{class_names[predicted[i]]}", fontsize=8)
    plt.axis("off")
plt.tight_layout(); plt.show()
```

> 🟩 **(보충) 추론에서 확률이 필요하면** logits에 softmax를 적용합니다(학습 손실 계산에는 넣지 않지만, **사람이 해석할 때만** 사용 — DAY4와 동일).
> ```python
> with torch.no_grad():
>     probs = torch.softmax(model(images), dim=1)   # (10, 10) 확률
>     pred  = probs.argmax(dim=1)                     # 가장 큰 확률의 클래스
> ```

---

## 10. 더 해보기 (실습 확장) 🟩

한 번에 **하나씩만** 바꿔 결과를 비교해 보세요(무엇이 영향을 줬는지 알 수 있게).

- **에폭/배치 크기** 바꾸기 → 학습 속도·정확도 변화
- **커널 크기 2 → 3(홀수)** → `padding="same"` 경고가 사라지는지 확인
- **은닉 채널 수(32·64)·`fc1` 노드(128)** 조절 → 표현력 변화
- **검증(validation) 세트 추가** → 🟩 이 노트북은 **train/test만** 씁니다. 원래는 train에서 일부를 떼어 **validation**으로 모델·에폭을 고르고(조기 종료 포함), **test는 마지막에 한 번만** 봐야 공정합니다(DAY3·DAY4 원칙).
- **MLP와 비교** → 같은 데이터를 DAY3식 MLP로 풀어 CNN과 정확도 비교

---

## 11. 자주 하는 실수

```text
❌ 출력층에 Softmax 추가              → CrossEntropyLoss가 내부 처리, 모델은 logits만
❌ 정답을 One-Hot으로 전달            → CrossEntropyLoss는 정수 라벨 (N,), torch.long
❌ 이미지 축 순서 혼동                → PyTorch는 NCHW (N,1,28,28), Keras는 NHWC
❌ fc1 입력 크기 잘못 계산            → 28→14→7, 64*7*7=3136 으로 맞추기
❌ padding='same' + 짝수 커널         → 경고 발생, 보통 홀수 커널(3) 권장
❌ 평가 때 eval()·no_grad() 누락      → 모드 전환 + 기울기 끄기
❌ 모델/데이터 device 불일치          → 둘 다 .to(device)
❌ validation 없이 test로 튜닝        → 비교·선택은 validation, test는 최종 1회
❌ 단일라벨인데 micro-F1을 정확도와 다른 지표로 오해 → 이 경우 둘은 동일
```

---

## 12. 핵심 정리

```text
CNN 이미지 분류 흐름 (PyTorch)
  데이터(NCHW, 0~1 정규화) → Conv+ReLU+Pool ×2 → Flatten → FC → 출력 10 logits
  손실: CrossEntropyLoss(정수 라벨, Softmax 내장) / 옵티마이저: Adam

shape 추적
  (N,1,28,28) → (N,32,14,14) → (N,64,7,7) → Flatten (N,3136) → (N,128) → (N,10)

학습/평가
  학습 3줄: zero_grad → backward → step / 평가: eval()+no_grad()
  평가지표: 정확도 + 혼동행렬 + F1 (단일라벨 다중분류는 micro-F1 = 정확도)

이번 실습 실제 결과(노트북 실행값)
  학습 20epoch: 정확도 76.61% → 94.79% (CPU 약 1392초)
  테스트: 손실 0.2558, 정확도 91.08%, F1(micro) 0.911
  가장 헷갈린 클래스: Shirt(상의 계열끼리 혼동) → 가벼운 과적합(94.79 vs 91.08)
```

---

## 🔗 참고 자료

- 실습 노트북: `fashion_pytorch_cnn.ipynb` (본문의 1차 출처)
- 짝이 되는 개념편: DAY4 — 딥러닝 네트워크 모델 설계
- [PyTorch `nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch `nn.CrossEntropyLoss` (logits·정수 라벨)](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [torchvision Fashion-MNIST 데이터셋](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)
- [Fashion-MNIST 원본 저장소(클래스·출처)](https://github.com/zalandoresearch/fashion-mnist)
