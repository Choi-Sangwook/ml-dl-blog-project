# 💾 딥러닝 완전 입문 가이드 — DAY6. 딥러닝 모델 저장 및 활용

> **시리즈**: 파이썬 기본만 있는 사람을 위한 딥러닝 입문
> **이전 편**: DAY1(개념·설치) · DAY2(신경망 알고리즘·계층구조) · DAY3(데이터 처리·데이터셋) · DAY4(네트워크 모델 설계) · DAY5(모델 훈련 및 평가)
> **프레임워크 안내**: 이번 글의 **모델 저장 예제는 PyTorch와 TensorFlow/Keras 두 가지를 모두** 보여줍니다. 강의 PDF가 두 프레임워크의 저장 방식을 **비교**하기 때문입니다. (DAY2·DAY4는 PyTorch, DAY5의 MNIST 예제는 Keras였습니다.) 코드 블록마다 어떤 프레임워크인지 표시했습니다.

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 내용
> - 🟩 **(보충)** : 입문자를 위해 덧붙인 사전지식·실무 주의점
> - **NumPy 예제**(이미지=숫자 배열)는 직접 실행해 확인한 값입니다.
> - **PyTorch·TensorFlow·OpenCV 코드는 이 글을 작성한 환경에 해당 라이브러리가 설치되지 않아 실행하지 못했습니다.** 예측값을 임의로 적지 않았으니, 표시된 부분은 **직접 실행해 확인**하세요.

---

## 1. 이번 DAY에서 배우는 것

DAY5까지 우리는 신경망을 **설계하고 → 학습시키고 → 평가**하는 과정을 봤습니다. 그런데 컴퓨터를 끄면 학습 결과는 어디로 갈까요? 이번 DAY의 핵심 질문입니다.

1. **오차 역전파법(Backpropagation)** — 신경망이 "무엇을" 학습해 저장하는지 빠르게 복습 (자세한 내용은 DAY5)
2. **모델 저장과 불러오기** — 학습 결과를 파일로 보관하고 다시 꺼내 쓰는 법 (PyTorch·TensorFlow) ← 이번 DAY의 제목
3. **컴퓨터비전 모델 활용** — 컴퓨터가 이미지를 어떻게 "숫자"로 보는지, 그리고 OpenCV 입문

> 🟩 **(보충) 순서를 바꾼 이유** — 강의 PDF는 "모델 저장 → 역전파 → 컴퓨터비전" 순서이지만, 이 글은 **"역전파(무엇을 학습하는가) → 저장(학습 결과 보관)"** 순으로 재배치했습니다. 저장하는 대상이 어떻게 만들어지는지 먼저 알면 왜 그것을 저장하는지가 자연스럽게 이해되기 때문입니다.

---

## 2. 신경망은 무엇을 학습하는가 — 오차 역전파법 (빠른 복습)

> DAY5에서 순전파·손실·경사하강법·학습률을 자세히 다뤘습니다. 여기서는 **저장과 이어지는 부분**만 짧게 정리합니다.

### 2.1 학습 한 사이클 🟦 (강의 PDF)

딥러닝 모델은 처음에 가중치가 **무작위 값**입니다. 그래서 고양이 사진을 넣어도 "강아지"라고 틀립니다. 이 오차를 줄이려고 아래 과정을 **반복**합니다.

```text
입력 → ① 순전파(예측 계산) → ② 손실 계산(정답과의 오차)
     → ③ 역전파(각 가중치가 오차에 준 영향 계산) → ④ 가중치 수정 → 반복
```

- **순전파 예시** 🟦 — 입력 `x=2`, 가중치 `w=3`이면 출력은 `y = x × w = 6`.
- **손실(MSE)** 🟦 — 실제값이 `10`이면 `Loss = (10 - 6)² = 16`. 손실이 클수록 더 많이 틀린 것입니다.
- **역전파** 🟦 — 출력층의 오차를 입력층 방향으로 **거꾸로(Backward)** 전달하며 각 가중치의 기울기(gradient)를 구합니다.

### 2.2 연쇄법칙과 자동미분 🟦 (강의 PDF)

신경망은 여러 층이 겹쳐 있어, 앞쪽 층까지 오차를 전달하려면 각 단계의 미분을 **연결해 곱합니다**. 이 규칙이 **연쇄법칙(Chain Rule)** 입니다.

```text
z = x·w,   a = sigmoid(z),   Loss = (y - a)²
dLoss/dw = dLoss/da × da/dz × dz/dw      ← 국소 미분을 곱해 연결
```

다행히 이 미분을 손으로 풀 필요는 없습니다. **PyTorch·TensorFlow가 자동미분으로 계산**해 줍니다.

```python
loss.backward()   # PyTorch: 이 한 줄이 모든 파라미터의 gradient를 자동 계산
```

> 🟩 **(보충) 흔한 오해 두 가지 바로잡기**
> - PDF에는 "출력 오차 0.8 × 영향 60% = 책임 0.48" 같은 예시가 나옵니다. 이는 *오차가 앞 단계로 전달된다*는 **직관용 비유**입니다. 실제 gradient는 퍼센트로 나뉘거나 합이 100%가 되지 않으며, 계산 그래프를 따라 국소 미분을 곱한 값입니다.
> - "Epoch가 늘면 손실이 계속 줄어든다"(예: 5.2→3.8→2.1→0.2)는 *학습이 잘 될 때*의 예시입니다. 실제로는 미니배치 순서·학습률 때문에 손실이 **오르내릴 수 있고**, train loss가 줄어도 validation loss는 과적합으로 커질 수 있습니다. 전체 추세와 검증 지표를 함께 봐야 합니다.

> 🟩 **(보충) 저장과의 연결고리** — 학습이 끝나면 무작위였던 값이 **의미 있는 모델 파라미터(가중치·편향)와 일부 버퍼**로 바뀝니다. **이것이 다음 절에서 저장할 "학습 결과"** 입니다.

---

## 3. 모델 저장과 불러오기 (이번 DAY의 핵심)

### 3.1 왜, 무엇을 위해 저장하는가 🟦 (강의 PDF)

학습이 끝난 뒤 **컴퓨터를 끄면 메모리의 학습 결과는 사라집니다.** 그래서 파일로 보관하며, 이를 **모델 저장(Model Saving)** 이라고 합니다. 5시간 학습한 모델을 저장해 두면, 다음 날 다시 5시간 학습할 필요 없이 불러와 바로 씁니다. 가장 큰 이유는 **학습 시간 절약**입니다.

저장은 "무엇을 위해 저장하는가"에 따라 필요한 항목이 다릅니다. 🟩 *(보충: 입문자가 가장 자주 헷갈리는 구분입니다.)*

| 저장 목적 | 함께 필요한 것 |
|---|---|
| **추론(예측) 재사용** | 모델 구조 + 학습된 모델 상태 + 전처리 방법 + 클래스 매핑 |
| **학습 재개** | 위 + 옵티마이저 상태 · 마지막 epoch (· scheduler 등) |
| **다른 환경 배포** | 위 + 라이브러리 버전 · 입출력 형식 · 배포 형식 검토 |

> 🟦 비유 — 모델 저장은 "공부한 내용을 책으로 기록", 불러오기는 "책을 다시 꺼내 활용"하는 것과 같습니다. *(이 비유는 여기서 한 번만 사용합니다.)*

### 3.2 무엇이 저장되는가 — `state_dict`의 정확한 의미 🟦+🟩

저장의 핵심 대상은 모델이 학습한 **파라미터**이지만, 정확히는 그뿐만이 아닙니다.

> 🟩 **(보충) 정정** — PyTorch의 `state_dict()`는 단순히 "가중치·편향만" 담는 객체가 아니라, 모델의 **학습 가능한 파라미터와 영속 버퍼(persistent buffer)** 를 이름과 함께 담은 딕셔너리입니다. 예를 들어 **BatchNorm의 running mean/variance** 같은 상태도 버퍼로 포함됩니다. 다만 **계층 구조 코드는 포함하지 않으므로**, 불러올 때 같은 구조의 모델을 먼저 만들어야 합니다.

또 실무에서는 모델 파일만으로는 부족합니다. 같은 결과를 재현하려면 **전처리 방법, 클래스 매핑, 하이퍼파라미터**도 함께 보관합니다. 🟦 작은 메타데이터 파일을 곁들이는 것이 실용적입니다. 🟩

```python
# 🟩 (보충) 모델과 함께 저장하면 좋은 메타데이터 예시 (직접 실행해 확인한 코드)
import json

metadata = {
    "input_shape": [224, 224, 3],   # 전처리 시 이미지 크기
    "pixel_scale": "0_to_1",        # 픽셀값 정규화 방식
    "class_names": ["cat", "dog", "rabbit"],   # 0,1,2 → 사람이 읽는 이름
}
with open("model_meta.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
```

> 🟩 모델 출력은 보통 `0, 1, 2` 같은 숫자라서, 어떤 클래스인지 적어 두지 않으면 사람이 해석할 수 없습니다. 정확한 재현에는 **코드·패키지 버전**까지 함께 관리합니다.

### 3.3 PyTorch 방식 — 학습된 상태 저장(state_dict) 🟦 (강의 PDF)

PyTorch는 보통 **구조는 코드로 유지**하고 **학습된 상태만 파일로 저장**합니다.

> 🟦 **프레임워크: PyTorch** · 작업: 회귀 · 데이터: `y = 2x` 토이 데이터

```python
# [PyTorch] 모델 학습 후 저장
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)   # 🟩 재현성을 위해 시드 고정 (PDF에는 없는 보충)

# 입력 X, 정답 y (정답은 항상 입력의 2배 → y = 2x)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # shape (4, 1), dtype float32
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # shape (4, 1), dtype float32

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)   # in_features=1, out_features=1
    def forward(self, x):
        return self.linear(x)           # 입력 (N,1) → 출력 (N,1)

model = SimpleModel()
criterion = nn.MSELoss()                       # 회귀 → 평균제곱오차
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    pred = model(X)                 # 순전파
    loss = criterion(pred, y)       # 손실 계산
    optimizer.zero_grad()           # 이전 기울기 초기화
    loss.backward()                 # 역전파(자동미분)
    optimizer.step()                # 가중치 갱신

# 학습된 모델 상태 저장 → pytorch_model.pth 생성
torch.save(model.state_dict(), "pytorch_model.pth")
print("PyTorch 모델 저장 완료")
```

**불러오기**는 ① 같은 구조의 모델을 만들고 → ② 저장된 상태를 끼워 넣는 순서입니다.

```python
# [PyTorch] 불러오기 및 예측
import torch
import torch.nn as nn

# 저장할 때와 "똑같은 구조"를 다시 정의해야 한다 (state_dict에는 구조가 없음)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

loaded_model = SimpleModel()
state_dict = torch.load(
    "pytorch_model.pth",
    map_location="cpu",     # 🟩 GPU에서 저장한 파일도 CPU에서 안전하게 로드
    weights_only=True,      # 🟩 역직렬화 범위를 제한하는 보안 옵션
)
loaded_model.load_state_dict(state_dict)
loaded_model.eval()         # 예측 모드로 전환

new_data = torch.tensor([[5.0]])     # shape (1, 1)
with torch.no_grad():                # 예측에는 기울기 계산이 불필요
    result = loaded_model(new_data)

print("예측 결과:", result.item())
```

**결과 해석** — 데이터가 `y = 2x`이므로 모델은 `w ≈ 2, b ≈ 0`을 학습합니다. 따라서 `x=5`의 예측은 **10에 가까워질 것으로 기대**됩니다. 정확한 값은 실행 환경·초기화에 따라 조금 달라질 수 있습니다.

> ⚠️ **(실행 안내)** 작성 환경에 PyTorch가 없어 위 코드를 **실행하지 못했습니다.** 숫자를 임의로 적지 않았으니 직접 실행해 확인하세요. 또한 `x=5`는 훈련 데이터(1~4) **범위 밖의 외삽**이며, 저장·복원 동작을 보여 주는 토이 예시일 뿐 모델 성능 평가가 아닙니다.

> 🟩 **(보충) 초보자가 놓치기 쉬운 점**
> - **`weights_only=True`** 는 "파일에 가중치만 들어 있다"는 뜻이 **아닙니다.** `torch.load()`가 역직렬화할 객체 범위를 제한하는 옵션입니다. **출처가 불확실한 `.pth`는 이 옵션과 무관하게 불러오지 않는 것이 원칙**입니다.
> - **`model.eval()`** — Dropout·BatchNorm은 학습/추론 동작이 다릅니다. 예측 전에 `eval()`로 전환해야 일관된 결과가 나옵니다.
> - **복원 검증** — 저장 직전과 불러온 직후 모델에 같은 입력을 넣어 결과가 같은지 확인하면 안전합니다: `torch.testing.assert_close(before, after)`.

### 3.4 추론용 저장 vs 학습 재개용 체크포인트 🟩 (보충)

위 코드는 **예측(추론)용 복원**에는 충분합니다. 하지만 멈춘 지점부터 **학습을 정확히 이어가려면** 옵티마이저 상태와 epoch도 필요합니다. PDF는 이 둘을 모두 "추가 학습"으로 묶어 소개하지만, 실무에서는 구분합니다.

| 용어 | 의미 |
|---|---|
| **학습 재개(Resume)** | 같은 작업을 중단 지점부터 그대로 이어서 학습 |
| **추가 학습(Continued)** | 같은 모델을 새 데이터까지 포함해 더 학습 |
| **미세조정(Fine-Tuning)** | **사전학습 모델**을 새 데이터·새 작업에 맞게 적응 (일부 층 고정/낮은 학습률 등) |

> 🟦 PDF는 "기존 모델을 불러와 새 데이터로 이어 학습"하는 활용을 Fine-Tuning이라고 간단히 소개합니다. 🟩 정확히는 위 세 가지를 구분하는 편이 좋습니다.

```python
# 🟩 (보충) 학습 재개용 체크포인트 — 추론용보다 더 많은 상태를 저장
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.detach().cpu(),
}, "checkpoint.pth")

# 불러와서 이어서 학습
checkpoint = torch.load("checkpoint.pth", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
model.train()   # 다시 학습 모드
```

> 🟩 scheduler나 mixed precision을 썼다면 그 상태도 함께 저장합니다. **예측만 할 거라면** `model.state_dict()`만 저장하는 3.3 방식이 더 간단합니다.

### 3.5 TensorFlow/Keras 방식 — 모델 전체 저장 🟦 (강의 PDF)

> 🟦 **프레임워크: TensorFlow / Keras** — 여기서 프레임워크가 **바뀝니다.** 같은 회귀 문제를 Keras로 풉니다.

Keras는 `model.save()` 한 번으로 **구조 + 가중치 + compile 정보**를 하나의 `.keras` 파일에 담습니다.

```python
# [TensorFlow/Keras] 학습 후 저장
import tensorflow as tf
import numpy as np

# 🟩 모델 기본 자료형(float32)과 맞추기 위해 dtype 명시 (안 하면 float64가 됨)
X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
y = np.array([[2.0], [4.0], [6.0], [8.0]], dtype=np.float32)

# 🟩 최신 Keras 권장 방식: 입력 shape을 Input 층으로 먼저 선언
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss="mse",
)

model.fit(X, y, epochs=1000, verbose=0)   # verbose=0: 학습 로그 숨김
model.save("tensorflow_model.keras")      # 모델 "전체" 저장
print("TensorFlow 모델 저장 완료")
```

```python
# [TensorFlow/Keras] 불러오기 및 예측
import tensorflow as tf
import numpy as np

# 구조를 다시 정의할 필요 없이 파일 하나로 복원
loaded_model = tf.keras.models.load_model("tensorflow_model.keras")

new_data = np.array([[5.0]], dtype=np.float32)   # shape (1, 1)
result = loaded_model.predict(new_data)          # 출력 shape (1, 1)

print("예측 결과:", result[0][0])
```

**결과 해석** — 같은 `y=2x` 데이터이므로 예측 역시 **10에 가까워질 것으로 기대**됩니다. PyTorch와 달리 불러올 때 **구조를 다시 작성하지 않아도** 됩니다(`.keras` 파일에 구조가 들어 있기 때문).

> ⚠️ **(실행 안내)** 이 환경에는 TensorFlow도 없어 실행하지 못했습니다. 예측 숫자는 직접 실행해 확인하세요.

> 🟩 **(보충)** `.keras`는 모델 복원에 필요한 정보를 함께 저장하지만, **모델 밖에서 한 전처리, 클래스 이름, 실행 환경까지 자동 저장되는 것은 아닙니다.** 사용자 정의 층·함수가 있으면 등록 또는 `custom_objects` 설정이 필요할 수 있습니다.

### 3.6 두 방식 비교 🟦+🟩

| 구분 | PyTorch `state_dict` 저장 | Keras `.keras` 전체 저장 |
|---|---|---|
| 주로 저장하는 것 | 파라미터 + 영속 버퍼 | 구조 · 가중치 · compile 정보 |
| 구조 코드 | **별도로 필요** | 직렬화 가능한 구조는 파일에서 복원 |
| 불러오기 | 같은 구조 생성 후 `load_state_dict()` | `load_model()` |
| 학습 재개 | 옵티마이저·epoch를 별도 체크포인트에 추가 | 옵티마이저 상태 포함 가능(재개 조건 확인 필요) |
| 주의점 | device 매핑, 구조 일치, 신뢰 가능한 파일 | 사용자 정의 객체, 버전 호환성, 외부 전처리 |

> 🟩 **(보충)** PDF는 PyTorch=연구·개발, TensorFlow=서비스 배포로 요약합니다. 이는 **입문용 경향 설명**일 뿐 고정된 구분은 아닙니다. 두 프레임워크 모두 연구·제품·추론 배포에 쓰이며, 실제 선택은 팀의 기술 스택과 배포 환경에 따라 달라집니다.

저장된 모델은 **예측 수행 · 추가 학습 · 웹 서비스 배포**에 활용됩니다. 🟦 대규모 AI 서비스도 학습된 모델 상태를 추론 인프라에 올려 요청을 처리하며, 실제로는 여러 서버와 최적화된 배포 시스템을 함께 사용합니다. 🟩

---

## 4. 컴퓨터비전 모델 활용

딥러닝의 대표 응용인 **컴퓨터비전(Computer Vision)** 입니다. 모델로 "이미지"를 다루려면 먼저 **컴퓨터가 이미지를 어떻게 보는지** 알아야 합니다.

### 4.1 컴퓨터비전은 무엇을 하는가 🟦 (강의 PDF)

사람은 사진을 한 번에 이해하지만, 컴퓨터는 작은 영역을 하나씩 분석해 장면을 이해합니다. 주요 작업은 다음과 같습니다. 🟩 *(겹치기 쉬운 용어를 입문자 기준으로 구분)*

| 작업 | 하는 일 |
|---|---|
| 이미지 분류(Classification) | 이미지 **전체**에 라벨 예측 ("코끼리가 있다") |
| 객체 검출(Object Detection) | 객체의 **클래스 + 위치(bounding box)** 예측 |
| 영상 분할(Segmentation) | **픽셀마다** 클래스 예측 (객체 경계 추출) |
| 객체 추적(Tracking) | 영상 **프레임 사이**에서 같은 객체를 이어 추적 |
| 이미지 캡셔닝(Captioning) | 이미지 전체를 **문장**으로 설명 |
| Dense Captioning | 여러 영역을 찾아 **영역별**로 설명 생성 |
| 화질 개선(Enhancement) | 노이즈 제거·초해상도 등 **품질 향상** |

대표 모델로 **CNN, Faster R-CNN, YOLO, SSD, Mask R-CNN, Vision Transformer(ViT), BLIP, GPT-4o 계열 멀티모달** 등이 PDF에 소개됩니다.

> 🟩 **(보충)** 이 목록은 **PDF 시점의 대표 예시**이며, CNN 계열 구조·객체 검출기·분할 모델·Transformer·멀티모달이 **서로 다른 종류로 섞여** 있습니다. "현재 가장 많이 쓰임" 같은 표현은 시간에 민감하므로 고정 순위처럼 받아들이지 않는 것이 좋습니다. 응용 분야는 자율주행·의료영상·얼굴인식·스마트공장·CCTV 분석 등입니다.

### 4.2 컴퓨터는 이미지를 "숫자"로 본다 🟦 + 🟩 실습

먼저 이미지가 어떻게 만들어지는지 짧게 봅니다. 🟦 디지털 카메라는 **피사체에서 반사된 빛 → 렌즈 → 이미지 센서(전기 신호) → ISP(색 복원·노이즈 제거·화질 개선) → JPG 등 파일**의 과정을 거칩니다. 노이즈·색 왜곡·해상도 한계의 원인을 알아야 영상 처리 알고리즘을 설계할 수 있기 때문에, 컴퓨터비전은 이 과정을 이해하는 데서 출발합니다.

만들어진 이미지는 **픽셀(pixel)** 이 격자처럼 배열된 형태입니다. 🟦

- **그레이스케일(흑백)** : 픽셀 1개가 밝기값 하나 — `0`(검정) ~ `255`(흰색), 1바이트
- **컬러(트루컬러)** : 픽셀 1개가 **R, G, B** 세 값(각 0~255)

즉 컴퓨터에게 이미지는 **숫자 배열**일 뿐입니다. 프레임워크 없이 NumPy로 직접 확인할 수 있습니다.

> 🟩 **프레임워크 무관: NumPy** — 아래는 직접 실행해 확인한 예제입니다.

```python
import numpy as np

# 4x4 흑백 이미지 = 밝기값(0~255) 격자
gray = np.array([
    [  0,  64, 128, 255],
    [ 32,  96, 160, 224],
    [ 64, 128, 192, 255],
    [255, 192, 128,  64],
], dtype=np.uint8)
print("흑백 shape:", gray.shape, "dtype:", gray.dtype)   # (4, 4) uint8

# 2x2 컬러 이미지 = 픽셀마다 [R, G, B]
color = np.array([
    [[255, 0, 0], [0, 255, 0]],     # 빨강, 초록
    [[0, 0, 255], [255, 255, 0]],   # 파랑, 노랑
], dtype=np.uint8)
print("컬러 shape:", color.shape)              # (2, 2, 3)  ← 마지막 3이 RGB
print("좌상단 픽셀 RGB:", color[0, 0])          # [255 0 0] = 빨강
```

**실행 결과(확인됨):**

```text
흑백 shape: (4, 4) dtype: uint8
컬러 shape: (2, 2, 3)
좌상단 픽셀 RGB: [255   0   0]
```

**해석** — 흑백은 `(높이, 너비)`, 컬러는 `(높이, 너비, 3)` 배열입니다. 딥러닝 입력 텐서의 shape(예: `(배치, 높이, 너비, 채널)`)이 왜 그렇게 생겼는지가 여기서 출발합니다.

**영상 용량**도 이 숫자 개수로 정해집니다. 🟦

```text
512 × 512 흑백        = 262,144 Bytes
1920 × 1080 컬러(×3)  = 6,220,800 Bytes ≈ 5.93 MB
```

> 🟩 **(보충)** 위 값은 **압축 전, 메모리에 펼친 배열의 크기**입니다. 실제 JPG·PNG **파일의 디스크 용량은 압축률과 이미지 내용에 따라 달라집니다**(JPG=손실 압축, PNG=무손실 압축, BMP=무압축, GIF=256색). 그래서 해상도가 커지면 메모리·계산량이 빠르게 늘어 압축이 필요합니다.

### 4.3 OpenCV 입문 🟦 (강의 PDF)

**OpenCV(Open Source Computer Vision Library)** 는 이미지·영상 처리를 위한 대표적인 무료 오픈소스 라이브러리입니다. Python·C++ 등 여러 언어와 OS를 지원하고, 이미지 입출력·필터링·객체/얼굴 인식·딥러닝 모델 실행까지 폭넓게 제공합니다.

**설치:**

```bash
pip install opencv-python
```

> 🟩 **(보충) 패키지 선택 주의** — 비슷한 이름이 여러 개 있으니 **하나만** 설치하세요. `opencv-python`(기본+GUI), `opencv-contrib-python`(추가 모듈 포함), `opencv-python-headless`(서버용, **GUI 창 없음**). 설치 가능한 버전은 바뀌므로 공식 패키지 페이지를 확인하세요.

기본 4개 함수만 먼저 익히면 됩니다: `cv2.imread`(읽기), `cv2.imshow`(창 출력), `cv2.imwrite`(저장), `cv2.waitKey`(키 입력 대기). `imread`의 `flags`는 `IMREAD_COLOR`(컬러 BGR, 기본) / `IMREAD_GRAYSCALE`(흑백) / `IMREAD_UNCHANGED`(투명도 포함)를 고릅니다.

```python
# [OpenCV] 이미지 읽기 → 출력 → 저장 (실패 처리 포함)
import cv2

input_path, output_path = "input.jpg", "output.png"

img = cv2.imread(input_path, cv2.IMREAD_COLOR)   # img: NumPy 배열 (H, W, 3)
if img is None:                                  # 🟩 경로/형식 오류 시 None 반환
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {input_path}")

try:
    cv2.imshow("window", img)     # 창에 출력
    cv2.waitKey(0)                # 0 = 키를 누를 때까지 대기 (이게 없으면 창이 안 보임)
finally:
    cv2.destroyAllWindows()       # 🟩 어떤 경우든 창 정리

if not cv2.imwrite(output_path, img):   # 🟩 저장 성공 여부 확인
    raise OSError(f"이미지를 저장하지 못했습니다: {output_path}")
```

> ⚠️ **(실행 안내)** 이 환경에 OpenCV(`cv2`)가 없어 실행하지 못했습니다. 위 흐름은 PDF의 API 설명을 정리·보강한 것으로, 실제 이미지 파일과 함께 직접 실행해 확인하세요.

> 🟩 **(보충) 꼭 기억할 함정**
> - **`cv2.imread` 실패는 예외가 아니라 `None` 반환** 입니다. 위처럼 `if img is None` 검사를 빼면 초보자가 자주 만나는 assertion 오류가 납니다.
> - **`cv2.imshow`는 데스크톱 GUI 창** 입니다. 서버·Colab·일부 Jupyter·`opencv-python-headless`에서는 동작하지 않을 수 있습니다. 이럴 땐 Matplotlib로 출력합니다.
> - **OpenCV는 색 순서가 BGR** 입니다. Matplotlib(RGB)로 보여줄 땐 변환이 필요합니다.

```python
# [OpenCV + Matplotlib] GUI 없이 이미지 보기 (BGR → RGB 변환 필수)
import cv2
import matplotlib.pyplot as plt

img_bgr = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
if img_bgr is None:
    raise FileNotFoundError("input.jpg를 읽을 수 없습니다.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # 변환 안 하면 색이 뒤집혀 보임
plt.imshow(img_rgb)
plt.title("OpenCV image")
plt.axis("off")
plt.show()
```

**동영상 다루기** 🟦 — OpenCV는 `cv2.VideoCapture`로 동영상·카메라 프레임을 한 장씩 받아옵니다. PDF가 5쪽에 걸쳐 다루는 핵심 흐름은 **열기 → 준비 확인 → 프레임 반복 → 자원 해제** 입니다.

```python
# [OpenCV] 동영상(또는 카메라) 프레임 읽기 루프
import cv2

cap = cv2.VideoCapture("input.mp4")    # 기본 카메라는 0
if not cap.isOpened():
    raise OSError("동영상 또는 카메라를 열 수 없습니다.")

try:
    while True:
        ret, frame = cap.read()        # (성공 여부, 프레임)
        if not ret:
            break                      # 파일 끝 / 연결 끊김 / 읽기 실패
        cv2.imshow("video", frame)
        if cv2.waitKey(1) == ord("q"): # q 키로 종료 (ESC=27, ENTER=13)
            break
finally:
    cap.release()                      # 🟩 파일/카메라 자원 해제 (필수)
    cv2.destroyAllWindows()
```

> 🟩 영상 저장은 `cv2.VideoWriter`와 코덱 코드(`cv2.VideoWriter_fourcc(*"DIVX")` 등)로 합니다. 프레임 속성은 `cap.get(cv2.CAP_PROP_FRAME_WIDTH)`처럼 조회합니다. 입문 단계에서는 위 **읽기 루프와 `release()`** 까지만 확실히 익혀도 충분합니다.

---

## 5. 자주 하는 실수

- **불러올 모델 구조를 안 만들고 `load_state_dict` 호출(PyTorch)** — `state_dict`에는 값만 있고 구조가 없습니다. 같은 구조를 먼저 만드세요.
- **`weights_only=True`를 안전 보증으로 오해** — 역직렬화 범위 제한 옵션일 뿐입니다. 출처 불확실한 파일은 불러오지 마세요.
- **추론용 저장으로 학습을 재개하려 함** — 옵티마이저·epoch가 없으면 정확히 이어서 학습할 수 없습니다. 체크포인트로 저장하세요.
- **예측 전에 `model.eval()` / `torch.no_grad()` 누락** — Dropout·BatchNorm 결과가 흔들리고 불필요하게 느려집니다.
- **모델만 저장하고 전처리·클래스 정보 누락** — 가중치가 멀쩡해도 입력 가공이 다르면 예측이 어긋납니다.
- **두 프레임워크 파일 혼용** — `.pth`는 Keras `load_model`로 못 엽니다. 저장한 프레임워크로 불러오세요.
- **OpenCV `imread` 실패 미검사 / BGR↔RGB 혼동 / `waitKey` 누락** — 오류가 나거나 색이 뒤집히거나 창이 안 뜹니다.
- **실행하지 않은 정확도·예측값을 단정적으로 적기** — 직접 실행해 확인한 값만 신뢰하세요.

---

## 6. DAY6 핵심 정리

```text
모델 저장 (이번 DAY의 핵심)
  - 저장 대상: 학습된 파라미터 + 영속 버퍼 (+ 전처리·클래스·하이퍼파라미터)
  - 목적 구분: 추론 재사용 / 학습 재개(옵티마이저·epoch 추가) / 배포
  - PyTorch: state_dict(.pth) → 같은 구조 코드 필요, map_location·weights_only·eval()
  - Keras:   model.save(.keras) → 파일 하나로 구조까지 복원
  - 두 프레임워크는 연구/배포로 고정되지 않음

오차 역전파 (DAY5 복습)
  - loss.backward()가 gradient 자동 계산 → 경사하강법이 파라미터 수정
  - 손실은 매 epoch 단조 감소하지 않음, validation도 함께 확인

컴퓨터비전 · OpenCV
  - 컴퓨터에게 이미지 = 숫자 배열 (흑백 (H,W), 컬러 (H,W,3), 0~255)
  - 작업: 분류·검출·분할·추적·캡셔닝 (작업마다 적합한 모델이 다름)
  - OpenCV: imread(실패 검사) / imshow·waitKey / imwrite, 색은 BGR
  - 동영상: VideoCapture → read 루프 → release
```

> 다음 DAY 주제는 이번 강의 PDF에서 확인할 수 없어 따로 예고하지 않습니다.

---

## 참고 자료

- 강의 자료: `DAY6_딥러닝_모델저장및활용.pdf` (교과목 2 — 데이터 분석과 머신러닝/딥러닝, 단원 3)
- PyTorch 공식 문서 — *Saving and Loading Models*, `Module.state_dict`, `torch.load`
- Keras 공식 문서 — *Save, serialize, and export models*, *The Sequential model*, *Transfer learning & fine-tuning*
- OpenCV 공식 문서 — *Image file reading and writing*, *High-level GUI*, `VideoCapture` (https://docs.opencv.org/4.x/)
- OpenCV Python 패키지 — https://pypi.org/project/opencv-python/
