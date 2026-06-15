<!--
============================================================
기획 문서 (드래프트 메타) — 게시 전 이 주석 블록은 제거 예정
============================================================

[작업 모드] Generate (Stage 1: 초안)
[1차 출처] sources/DAY6_딥러닝_모델저장및활용.pdf (총 44페이지)
[분야] 딥러닝 / [독자] 파이썬 기본은 알지만 딥러닝은 처음
[이전 편] DAY1(개념·설치) · DAY2(신경망·계층구조) · DAY3(데이터 처리·데이터셋) · DAY4(네트워크 모델 설계) · DAY5(모델 훈련 및 평가)

[프레임워크 결정]
  - 이번 PDF 1장(모델 저장)은 PyTorch와 TensorFlow/Keras 두 가지를 "비교 목적"으로 모두 제시함.
    → 두 프레임워크를 모두 싣되, "프레임워크가 전환된다"는 점을 명시(딥러닝 체크리스트 1).
    → DAY2·DAY4는 PyTorch, DAY5 완성 예제는 Keras였음. 이번에는 둘 다.
  - 2장(오차 역전파)은 수학·개념 위주 → 프레임워크 무관. 자동미분 연결만 PyTorch 한 줄 언급.
  - 4장(컴퓨터비전/OpenCV)은 OpenCV(cv2) 기반. "이미지=숫자 배열" 직관 예제는 NumPy로 작성(실행 확인).

[실행 환경 한계 — 임의 결과값 금지]
  - 이 환경에 torch / tensorflow / cv2 모두 미설치. → 저장 예제·예측값·OpenCV 코드는 "직접 실행" 안내, 수치 임의 기입 금지.
  - 실행해서 확인한 값만 사용:
      · 순전파 y=x*w=2*3=6, MSE Loss=(10-6)²=16            (PDF p14와 일치)
      · Loss=(10-2w)² → dLoss/dw=-4(10-2w)=8w-40 (w=0:-40, w=5:0)  (PDF p15 보충 미분)
      · 영상 용량 512×512=262,144 Bytes, 1920×1080×3=6,220,800 Bytes(≈5.93MB)  (PDF p31과 일치)
      · NumPy 이미지 배열: gray (4,4) uint8 / color (2,2,3) uint8  (실행 확인)
  - 회귀 토이데이터는 y=2x이므로 학습 후 w≈2, b≈0 → x=5 예측은 10 부근. "9.9~10.0"은 범위로만 표기.

------------------------------------------------------------
PDF 페이지 → 핵심 주제 지도 (전 44페이지 확인 완료, 이미지 전용 페이지 포함)
------------------------------------------------------------
p1-2    표지 / 목차 (3장: ①모델 저장·활용 ②오차 역전파법 ③컴퓨터비전 모델 활용)
p3      [1장] 모델 저장 이유: 학습 결과=가중치·편향, 종료 시 메모리 소멸 → 파일 저장, 예측 활용
p4      추가 학습(Fine-Tuning), 웹 배포(ChatGPT 등), 저장 과정 흐름도
p5      PyTorch 저장=가중치만(state_dict, torch.save / load_state_dict), TensorFlow 저장=모델 전체(model.save)
p6      .keras 파일=구조+가중치+손실+옵티마이저, load_model, PyTorch vs TensorFlow 비교표
p7      실무 동반 저장 정보(전처리·클래스·하이퍼파라미터), 학생 비유
p8      1장 요약 + [PyTorch 저장 예제 코드 시작] import, X/y, SimpleModel 정의
p9      forward, MSELoss, SGD, 1000 epoch 학습 루프, torch.save(state_dict)
p10     저장완료 출력 / [불러오기] 동일 구조 재정의, load_state_dict(weights_only=True), eval()
p11     예측 출력 / [TensorFlow 저장 예제] Sequential Dense(1, input_shape=(1,)), compile(SGD, mse)
p12     fit→save / load_model→predict / 1·2장 코드 요약
p13     [2장] 오차 역전파법 정의·필요성(무작위 초기화), 학습 7단계
p14     순전파 예시 y=x*w=6, 손실 MSE=(10-6)²=16, 역전파 개념
p15     (이미지) 미분 필요성 Loss=(10-2w)², dLoss/dw, 경사하강 식 w_new=w_old-η·∂Loss/∂w
p16     연쇄법칙 z=xw,a=sigmoid(z),Loss=(y-a)², dLoss/dw=dLoss/da·da/dz·dz/dw, 책임 분배 예시
p17     Epoch↑ → 손실↓(5.2→0.2), 역전파 장점(빠름·대규모·자동미분 loss.backward())
p18     [3장] 컴퓨터비전 정의(위키백과 인용 이미지)
p19-20  (이미지) Dense Captioning 예시 사진: 객체탐지 색상박스, 자동 캡션 문장
p21-22  Dense Captioning 기술, CV 처리 5단계(입력→탐지→특징→인식→캡션)
p23     대표 딥러닝 모델(CNN/Faster R-CNN/YOLO/SSD/Mask R-CNN/ViT/BLIP/GPT-4o), CV 작업 표
p24-25  (이미지) CV 연구 분야: 화질개선/객체검출/분할/인식/추적 + 응용
p26     (이미지) 머신비전·AI 서비스(Amazon Go 등)
p27     (이미지) 디지털 카메라 영상 획득 과정(ISP), 왜 영상 생성 원리를 알아야 하나
p28     (이미지) 영상 표현: 픽셀=2차원 행렬, 좌표계, M×N 행렬
p29     (이미지) 그레이스케일 0~255(1 byte), unsigned char, 픽셀값 예시
p30     (이미지) 트루컬러 RGB 값 분포 예시
p31     (이미지) 영상 용량 계산(512²=262,144B, 1920×1080×3≈5.93MB), 파일 형식 BMP/JPG/GIF/PNG
p32-34  (이미지) OpenCV 소개·주요기능·활용분야·크로스플랫폼
p35     (이미지) OpenCV 모듈, OpenCV-Python 설치(pip install opencv-python)
p36     (이미지) OpenCV 프로그래밍 기초, cv2.imread(filename, flags) + flags 표(COLOR/GRAYSCALE/UNCHANGED)
p37     (이미지) cv2.imwrite, namedWindow, destroyWindow/destroyAllWindows
p38     (이미지) moveWindow/resizeWindow, cv2.imshow(데이터타입별 출력)
p39     (이미지) cv2.waitKey + while True 종료 루프, 특수키 코드(ESC=27,ENTER=13,TAB=9), 도움말
p40     (이미지) Matplotlib 영상 출력, BGR↔RGB(cv2.cvtColor), cv2.VideoCapture 클래스
p41-44  (이미지) OpenCV 동영상 API: VideoCapture(open/isOpened/read/get/set), VideoWriter(fourcc/open/write)

[검토 메모]
  - 1장 코드는 PDF에 텍스트로 온전히 들어있음(p8-12). 출력층/라벨/손실 조합: 회귀-raw output-float-MSE로 일관 ✓.
  - 2장은 DAY5의 경사하강·역전파와 내용이 겹침 → "복습+자동미분 연결"로 압축, 분량 절제.
  - 3장 OpenCV API는 사실상 레퍼런스 슬라이드. 입문 범위에서 핵심 함수(imread/imshow/imwrite/waitKey)만 실습으로 정리하고,
    동영상(VideoCapture/Writer)·전체 flag 표는 "더 보기"로 간략 처리.
  - 다음 DAY 주제는 PDF·파일에서 확인 불가 → 예고하지 않음.
============================================================
-->

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

이번 글에서 다루는 것:

1. **오차 역전파법(Backpropagation)** — 신경망이 "무엇을" 학습해 저장하는지 다시 정리 (DAY5 복습 + 자동미분 연결)
2. **모델 저장과 불러오기** — 학습한 가중치를 파일로 보관하고 다시 꺼내 쓰는 법 (PyTorch·TensorFlow) ← 이번 DAY의 제목
3. **컴퓨터비전 모델 활용** — 컴퓨터가 이미지를 어떻게 "숫자"로 보는지, 그리고 OpenCV 입문

> 🟩 **(보충) 순서를 바꾼 이유** — 강의 PDF는 "모델 저장 → 역전파 → 컴퓨터비전" 순서이지만, 이 글은 **"역전파(무엇을 학습하는가) → 저장(학습 결과를 보관)"** 순으로 재배치했습니다. 저장하는 대상인 *가중치·편향*이 어떻게 만들어지는지 먼저 알면, 왜 그것을 저장하는지가 자연스럽게 이해되기 때문입니다.

---

## 2. 신경망은 무엇을 학습하는가 — 오차 역전파법

### 2.1 왜 필요한가 🟦 (강의 PDF)

딥러닝 모델은 처음에 **가중치(Weight)를 무작위 값**으로 시작합니다. 그래서 학습 초기에는 고양이 사진을 넣어도 "강아지"라고 틀리게 예측합니다.

```text
입력: 고양이 사진
실제 정답: 고양이
모델 예측: 강아지   → 오차(Error) 발생
```

이때 모델은 **"어떤 가중치가, 얼마나 잘못됐는가?"** 를 알아내야 가중치를 고칠 수 있습니다. 이 계산을 해 주는 알고리즘이 바로 **오차 역전파법(Backpropagation)** 입니다.

> 🟦 비유: 시험을 본 학생이 틀린 문제를 보고 "왜 틀렸는지" 분석해 다음 시험에서 같은 실수를 줄이는 과정과 같습니다.

### 2.2 동작 원리: 학습 한 사이클 🟦 (강의 PDF)

딥러닝 학습은 아래 순서를 **반복**합니다.

```text
1. 입력 데이터
      ↓
2. 순전파(Forward Propagation)  — 입력 → 출력 방향으로 예측 계산
      ↓
3. 예측 결과
      ↓
4. 손실함수 계산              — 예측이 정답에서 얼마나 벗어났나
      ↓
5. 역전파(Backpropagation)     — 오차를 거꾸로 전달, 각 가중치의 "책임" 계산
      ↓
6. 가중치 수정                — 경사하강법으로 업데이트
      ↓
7. 반복(다음 Epoch)
```

**① 순전파 예시** — 입력 `x=2`, 가중치 `w=3`이면 출력은 `y = x × w = 6`.

**② 손실(MSE)** — 실제값이 `10`이라면, 평균제곱오차는

```text
Loss = (실제값 - 예측값)² = (10 - 6)² = 16
```

손실값이 클수록 더 많이 틀렸다는 뜻입니다.

**③ 역전파** — 출력층의 오차를 입력층 방향으로 **거꾸로(Backward)** 전달(Propagation)하면서, 각 가중치가 오차에 얼마나 기여했는지 계산합니다. 그래서 이름이 "역(逆) + 전파"입니다.

### 2.3 미분과 경사하강법 🟦 (강의 PDF)

가중치를 **올려야 할지 내려야 할지, 얼마나** 바꿔야 할지 알려면 손실함수를 가중치로 **미분**합니다. 미분값(기울기)은 *손실이 증가하는 방향*을 가리킵니다.

예를 들어 손실이 `Loss = (10 - 2w)²` 라면, 가중치 `w`에 대한 기울기는:

```text
dLoss/dw = -4 × (10 - 2w) = 8w - 40
```

> 🟩 **(보충) 직접 계산해 보기** — 합성함수 미분으로 `dLoss/dw = 2·(10-2w)·(-2) = -4(10-2w)`. 값을 넣어 보면:
> | w | Loss | dLoss/dw(기울기) | 해석 |
> |---|---|---|---|
> | 0 | 100 | -40 | 기울기가 음수 → w를 **늘려야** 손실이 준다 |
> | 2 | 36 | -24 | 아직 음수 → 더 늘린다 |
> | 5 | 0 | 0 | 기울기 0 → **최소점 도착** |
>
> (위 표의 값은 직접 계산해 확인한 값입니다. PDF는 `dLoss/dw`까지만 제시합니다.)

기울기를 구했으면, **경사하강법(Gradient Descent)** 으로 가중치를 한 걸음 옮깁니다.

```text
w_new = w_old - η × (∂Loss/∂w)
```

- `w` : 가중치
- `η`(에타) : 학습률(Learning Rate) — 한 번에 얼마나 움직일지
- `∂Loss/∂w` : 기울기

즉 **역전파는 기울기를 계산**하고, **경사하강법은 그 기울기로 가중치를 수정**합니다. (학습률과 경사하강법의 직관은 DAY5에서 NumPy로 자세히 다뤘습니다.)

### 2.4 연쇄법칙과 자동미분 🟦 (강의 PDF)

신경망은 여러 층이 겹쳐 있습니다(입력층 → 은닉층1 → 은닉층2 → 출력층). 오차를 앞쪽 층까지 전달하려면 각 단계의 미분을 **곱해서 연결**해야 하는데, 이때 쓰는 규칙이 **연쇄법칙(Chain Rule)** 입니다.

```text
z = x·w
a = sigmoid(z)
Loss = (y - a)²

dLoss/dw = dLoss/da × da/dz × dz/dw   ← 미분을 연결(곱)
```

직관적으로는 "책임 분배"입니다. 출력층 오차가 `0.8`이고 은닉층 뉴런 A가 오차의 60%에 영향을 줬다면, A의 책임은 `0.8 × 0.6 = 0.48`로 봅니다.

> 🟦 **(강의 PDF) 좋은 소식** — 이 복잡한 연쇄법칙 미분을 직접 손으로 풀 필요는 없습니다. **PyTorch·TensorFlow가 자동미분(Autograd)** 으로 계산해 줍니다. PyTorch에서는

```python
loss.backward()   # 이 한 줄이 모든 가중치의 기울기를 자동 계산
```

학습이 반복되면 손실값이 점점 줄어듭니다(예: `5.2 → 3.8 → 2.1 → 0.9 → 0.2`). 이것이 모델이 정답에 가까워진다는 신호입니다.

> 🟩 **(보충) 핵심 연결고리** — 학습이 끝나면 무작위였던 가중치가 **의미 있는 값**으로 바뀝니다. **이 가중치·편향 값이 바로 "모델이 학습한 결과"이고, 다음 절에서 저장할 대상**입니다.

---

## 3. 모델 저장과 불러오기 (이번 DAY의 핵심)

### 3.1 왜 저장하는가 🟦 (강의 PDF)

학습이 끝나면 모델은 가중치·편향이라는 값을 갖습니다. 그런데 **컴퓨터를 끄면 메모리에 있던 이 값들은 사라집니다.** 그래서 파일로 보관해야 하며, 이를 **모델 저장(Model Saving)** 이라고 합니다.

예를 들어 고양이/강아지 분류 모델을 5시간 학습했다면:

```text
저장 안 함  → 다음 날 다시 5시간 처음부터 학습
저장 함     → 파일을 불러오기만 하면 즉시 사용
```

가장 큰 이유는 **학습 시간 절약**입니다. 저장해 둔 모델로 할 수 있는 일:

| 활용 | 설명 | 예시 |
|---|---|---|
| **예측 수행** | 새 데이터를 분류·예측 | 손글씨 인식, 주택가격 예측 |
| **추가 학습(Fine-Tuning)** | 기존 모델에 새 데이터로 이어 학습 | 1만 개 학습한 모델에 1천 개 추가 |
| **웹 서비스 배포** | 서버에 올려 사용자에게 제공 | ChatGPT, 얼굴 인식, 번역 |

> 🟦 비유 — 모델 저장은 "공부한 내용을 책으로 기록", 모델 불러오기는 "책을 다시 꺼내 활용"하는 것과 같습니다.

### 3.2 무엇을 저장하는가 🟦 (강의 PDF)

저장의 핵심 대상은 **가중치와 편향**이지만, 실무에서는 모델 파일만 저장하지 않습니다. 같은 결과를 안정적으로 재현하려면 아래 정보도 함께 보관합니다.

- **데이터 전처리 정보** — 예: 이미지 크기 224×224, 픽셀값 0~1 정규화
- **클래스 정보** — 예: `0=고양이, 1=강아지, 2=토끼`
- **학습 설정(하이퍼파라미터)** — 예: 학습률 0.001, Epoch 100, Batch Size 32

> 🟩 **(보충) 왜 전처리·클래스 정보까지?** — 모델은 "학습할 때와 똑같이 가공된 입력"만 제대로 해석합니다. 예측할 때 전처리를 다르게 하면 가중치가 멀쩡해도 엉뚱한 답이 나옵니다. 또 모델 출력은 보통 `0,1,2` 같은 **숫자**라서, 그것이 어떤 클래스인지 적어 두지 않으면 사람이 해석할 수 없습니다.

### 3.3 PyTorch 방식 — 가중치만 저장(state_dict) 🟦 (강의 PDF)

PyTorch는 보통 **모델 구조는 코드로 유지**하고, **학습된 가중치만 파일로 저장**합니다.

```text
모델 = 구조(코드) + 가중치(weight) + 편향(bias)
        └ 코드에 보관      └────── 파일에 저장(state_dict) ──────┘
```

> 🟦 **프레임워크: PyTorch** · 작업: 회귀 · 데이터: `y = 2x` 토이 데이터

```python
# [PyTorch] 모델 저장 예제
import torch
import torch.nn as nn
import torch.optim as optim

# 입력 X, 정답 y (정답은 항상 입력의 2배 → y = 2x)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # shape (4, 1), dtype float32
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # shape (4, 1), dtype float32

# 입력 1개 → 출력 1개를 만드는 아주 단순한 선형 회귀 모델
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)   # in_features=1, out_features=1

    def forward(self, x):
        return self.linear(x)           # 입력 (N,1) → 출력 (N,1)

model = SimpleModel()
criterion = nn.MSELoss()                      # 회귀 → 평균제곱오차
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
for epoch in range(1000):
    pred = model(X)                 # 순전파
    loss = criterion(pred, y)       # 손실 계산
    optimizer.zero_grad()           # 이전 기울기 초기화
    loss.backward()                 # 역전파(자동미분)
    optimizer.step()                # 가중치 갱신

# 학습된 "가중치만" 저장 → pytorch_model.pth 파일 생성
torch.save(model.state_dict(), "pytorch_model.pth")
print("PyTorch 모델 저장 완료")
```

**불러오기**는 ① 같은 구조의 모델을 먼저 만들고 → ② 저장된 가중치를 끼워 넣는 순서입니다.

```python
# [PyTorch] 모델 불러오기 및 예측
import torch
import torch.nn as nn

# 저장할 때와 "똑같은 구조"를 다시 정의해야 한다
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

loaded_model = SimpleModel()                                  # 빈(무작위) 모델
loaded_model.load_state_dict(
    torch.load("pytorch_model.pth", weights_only=True))       # 저장된 가중치 주입
loaded_model.eval()                                           # 예측 모드로 전환

new_data = torch.tensor([[5.0]])          # shape (1, 1)
with torch.no_grad():                     # 예측에는 기울기 계산이 불필요
    result = loaded_model(new_data)

print("예측 결과:", result.item())
```

**결과 해석** — 데이터가 `y = 2x`이므로 모델은 `w ≈ 2, b ≈ 0`을 학습합니다. 따라서 `x=5` 입력의 예측은 **10에 매우 가까운 값**(대략 9.9~10.0 부근)이 나옵니다.

> ⚠️ **(실행 안내)** 이 글의 작성 환경에는 PyTorch가 설치되어 있지 않아 위 코드를 **실행하지 못했습니다.** 정확한 숫자는 임의로 적지 않았으니 직접 실행해 확인하세요. 무작위 초기화 때문에 실행마다 소수점 아래가 조금씩 다를 수 있습니다.

> 🟩 **(보충) 초보자가 놓치기 쉬운 3가지**
> - **`load_state_dict`는 구조가 먼저** 있어야 합니다. `state_dict`에는 "값"만 있고 "구조"는 없기 때문입니다. (PyTorch 공식 문서도 같은 모델 인스턴스를 먼저 만들라고 안내합니다.)
> - **`model.eval()`** — Dropout·BatchNorm은 학습할 때와 예측할 때 동작이 다릅니다. 예측 전에 `eval()`로 전환해야 일관된 결과가 나옵니다.
> - **`weights_only=True`** — 신뢰할 수 없는 `.pth` 파일을 그냥 불러오면 보안 위험이 있습니다. 가중치만 불러올 때는 이 옵션을 켜는 것이 안전합니다.

### 3.4 TensorFlow/Keras 방식 — 모델 전체 저장 🟦 (강의 PDF)

> 🟦 **프레임워크: TensorFlow / Keras** — 여기서 프레임워크가 **바뀝니다.** 같은 회귀 문제를 Keras로 다시 풉니다.

TensorFlow/Keras는 `model.save()` 한 번으로 **구조 + 가중치 + 손실함수 + 옵티마이저 정보**를 하나의 `.keras` 파일에 담습니다.

```python
# [TensorFlow/Keras] 모델 저장 예제
import tensorflow as tf
import numpy as np

X = np.array([[1.0], [2.0], [3.0], [4.0]])   # shape (4, 1)
y = np.array([[2.0], [4.0], [6.0], [8.0]])   # shape (4, 1)

# 입력 1개 → 출력 1개 Dense 계층
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss="mse"
)

model.fit(X, y, epochs=1000, verbose=0)   # verbose=0: 학습 로그 숨김

# 모델 "전체" 저장 → tensorflow_model.keras 파일 생성
model.save("tensorflow_model.keras")
print("TensorFlow 모델 저장 완료")
```

```python
# [TensorFlow/Keras] 모델 불러오기 및 예측
import tensorflow as tf
import numpy as np

# 구조를 다시 정의할 필요 없이 파일 하나로 복원
loaded_model = tf.keras.models.load_model("tensorflow_model.keras")

new_data = np.array([[5.0]])               # shape (1, 1)
result = loaded_model.predict(new_data)    # 출력 shape (1, 1)

print("예측 결과:", result[0][0])
```

**결과 해석** — PyTorch 예제와 같은 `y=2x` 데이터이므로 예측 역시 **10 부근**입니다. PyTorch와 달리 **불러올 때 모델 구조를 다시 작성하지 않아도** 됩니다(`.keras` 파일에 구조가 들어 있기 때문).

> ⚠️ **(실행 안내)** 이 환경에는 TensorFlow도 설치되어 있지 않아 실행하지 못했습니다. 예측 숫자는 직접 실행해 확인하세요.

> 🟩 **(보충) 최신 Keras에서의 작은 주의** — 위 코드의 `input_shape=(1,)`는 강의 PDF 그대로입니다. 최신 Keras에서는 `tf.keras.Input(shape=(1,))`을 첫 줄에 두는 방식을 권장하며, 그대로 두면 경고가 뜰 수 있으나 동작에는 문제가 없습니다.

### 3.5 두 방식 비교 🟦 (강의 PDF)

| 구분 | PyTorch | TensorFlow/Keras |
|---|---|---|
| 저장 대상 | 가중치 위주(`state_dict`) | 모델 전체 |
| 저장 파일 | `.pth` | `.keras` |
| 불러올 때 | **모델 구조 코드 필요** | 파일만으로 바로 복원 |
| 성격 | 유연함 | 사용 편리 |
| 주로 쓰는 곳 | 연구·개발 | 서비스 배포 |

> 🟩 **(보충)** "어느 쪽이 옳다"기보다 **습관과 목적의 차이**입니다. PyTorch도 모델 전체를 저장할 수 있지만, 구조는 코드로 관리하고 가중치만 저장하는 `state_dict` 방식을 공식적으로 권장합니다. 두 경우 모두 **전처리·클래스·하이퍼파라미터(3.2)** 를 함께 보관해야 재현이 안정적입니다.

---

## 4. 컴퓨터비전 모델 활용

마지막 장은 딥러닝의 대표 응용인 **컴퓨터비전(Computer Vision)** 입니다. 모델을 저장·배포해서 실제로 "이미지"를 다루려면, 먼저 **컴퓨터가 이미지를 어떻게 보는지** 알아야 합니다.

### 4.1 컴퓨터비전은 무엇을 하는가 🟦 (강의 PDF)

사람은 사진을 한 번 보면 "코끼리 위에 사람이 타고 있다"를 바로 이해하지만, 컴퓨터는 작은 영역을 하나씩 분석해 전체 장면을 이해합니다. 컴퓨터비전의 주요 작업:

| 작업 | 하는 일 |
|---|---|
| 이미지 분류(Image Classification) | "이 사진에 코끼리가 있다" |
| 객체 검출(Object Detection) | 코끼리·사람·공의 **위치(상자)** 찾기 |
| 객체 인식(Object Recognition) | 그 영역이 **무엇인지** 판별 |
| 이미지 캡셔닝(Image Captioning) | "남자가 코끼리를 타고 있다" 같은 설명문 생성 |
| 장면 이해(Scene Understanding) | 전체 상황 파악 |
| Dense Captioning | 여러 영역을 **세밀하게** 설명 |

대표 딥러닝 모델로는 **CNN, Faster R-CNN, YOLO, SSD, Mask R-CNN, Vision Transformer(ViT), BLIP, GPT-4o 계열 멀티모달** 등이 있습니다. 이들은 수백만 장의 이미지로 "사람·코끼리·공이 어떤 모양인가"를 학습합니다.

> 🟩 **(보충) 작업별로 모델이 다릅니다** — "위치만" 필요하면 객체 검출(YOLO 등), "경계까지" 필요하면 분할(Mask R-CNN, U-Net), "종류 판별"이면 분류(ResNet, ViT)를 씁니다. 응용 분야는 자율주행·의료영상·얼굴인식·스마트공장·CCTV 분석 등입니다.

### 4.2 컴퓨터는 이미지를 "숫자"로 본다 🟦 (강의 PDF) + 🟩 (보충 실습)

이미지는 **픽셀(pixel)** 이 바둑판처럼 배열된 2차원 격자입니다.

- **그레이스케일(흑백)** : 픽셀 1개가 밝기값 하나 — `0`(검정) ~ `255`(흰색), 1바이트(`unsigned char`)
- **컬러(트루컬러)** : 픽셀 1개가 **R, G, B** 세 값(각 0~255)

즉 컴퓨터에게 이미지는 **숫자 배열**일 뿐입니다. 이 개념은 프레임워크 없이 NumPy로 직접 확인할 수 있습니다.

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

**해석** — 흑백 이미지는 `(높이, 너비)`, 컬러 이미지는 `(높이, 너비, 3)` 형태의 배열입니다. 딥러닝 입력 텐서의 shape이 왜 그렇게 생겼는지(예: `(배치, 높이, 너비, 채널)`)가 바로 여기서 출발합니다.

**영상 용량**도 이 숫자 개수로 계산됩니다. 🟦

```text
512 × 512 흑백        = 262,144 Bytes
1920 × 1080 컬러(×3)  = 6,220,800 Bytes ≈ 5.93 MB
```

> 🟩 **(보충)** 그래서 같은 장면이라도 해상도가 커지면 용량과 계산량이 급격히 늘어납니다. 압축 파일 형식(JPG=손실 압축, PNG=무손실 압축, BMP=무압축, GIF=256색)이 필요한 이유입니다.

### 4.3 OpenCV 입문 🟦 (강의 PDF)

**OpenCV(Open Source Computer Vision Library)** 는 이미지·영상 처리를 위한 대표적인 무료 오픈소스 라이브러리입니다. Python·C++·Java 등 여러 언어와 Windows·Linux·Mac 등 여러 OS를 지원하고, 이미지 읽기/저장, 필터링, 객체·얼굴 인식, 딥러닝 모델 실행까지 폭넓게 제공합니다.

**설치:**

```bash
pip install opencv-python
```

가장 기본이 되는 4개 함수만 먼저 익히면 됩니다.

| 함수 | 역할 |
|---|---|
| `cv2.imread(filename, flags)` | 이미지 파일 읽기 → NumPy 배열 반환 |
| `cv2.imshow(winname, img)` | 창에 이미지 출력 |
| `cv2.imwrite(filename, img)` | 이미지 파일로 저장 (성공 시 `True`) |
| `cv2.waitKey(delay)` | 키 입력 대기 (창 표시에 필수) |

`imread`의 `flags`는 읽기 방식을 정합니다.

| flags | 의미 |
|---|---|
| `cv2.IMREAD_COLOR` | 컬러(BGR)로 읽기 (기본값) |
| `cv2.IMREAD_GRAYSCALE` | 흑백으로 읽기 |
| `cv2.IMREAD_UNCHANGED` | 투명도(alpha) 채널까지 그대로 |

```python
# [OpenCV] 이미지 읽기 → 출력 → 저장 (기본 흐름)
import cv2

img = cv2.imread("input.jpg", cv2.IMREAD_COLOR)   # img: NumPy 배열 (H, W, 3)

cv2.imshow("window", img)     # 창에 출력
cv2.waitKey(0)                # 0 = 키를 누를 때까지 무한 대기
cv2.destroyAllWindows()       # 모든 창 닫기

cv2.imwrite("output.png", img)   # 다른 형식으로 저장
```

> ⚠️ **(실행 안내)** 이 환경에는 OpenCV(`cv2`)가 설치되어 있지 않아 위 코드를 실행하지 못했습니다. 위 흐름은 강의 PDF의 API 설명을 정리한 것으로, 실제 이미지 파일(`input.jpg`)과 함께 직접 실행해 확인하세요.

> 🟩 **(보충) 꼭 기억할 함정 2가지**
> - **`cv2.waitKey()`가 없으면 창이 안 보입니다.** `imshow`만 호출하면 창이 그려지기 전에 프로그램이 끝나 버립니다. 키 입력을 기다리는 동안 화면이 유지됩니다. (종료 루프 예: `if cv2.waitKey(1) == ord('q'): break` — 특수키 코드는 ESC=27, ENTER=13, TAB=9)
> - **OpenCV는 색 순서가 BGR입니다.** Matplotlib(RGB)로 보여줄 때 색이 이상하면 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`로 변환하세요.

> 📌 **더 보기 (이 글 범위 밖, 강의 PDF에 포함)** — OpenCV는 동영상도 다룹니다. `cv2.VideoCapture`로 카메라·동영상 프레임을 한 장씩 받아오고(`read()`), `cv2.VideoWriter`로 코덱(fourcc, 예: `'DIVX'`, `'XVID'`, `'MJPG'`)을 지정해 영상으로 저장합니다. 입문 단계에서는 정지 이미지부터 익힌 뒤 넘어가는 것을 권합니다.

---

## 5. 자주 하는 실수

- **불러올 모델 구조를 안 만들고 `load_state_dict` 호출(PyTorch)** — `state_dict`에는 값만 있습니다. 같은 구조의 클래스를 먼저 만들어야 합니다.
- **예측 전에 `model.eval()` / `torch.no_grad()` 누락** — Dropout·BatchNorm이 있으면 결과가 흔들리고, 불필요한 기울기 계산으로 느려집니다.
- **모델만 저장하고 전처리·클래스 정보는 안 저장** — 가중치가 멀쩡해도 입력 가공이 달라지면 예측이 어긋납니다.
- **두 프레임워크 코드 혼용** — PyTorch로 저장한 `.pth`는 Keras `load_model`로 못 엽니다. 저장한 프레임워크로 불러오세요.
- **OpenCV에서 `cv2.waitKey()` 빼먹기 / BGR↔RGB 혼동** — 창이 안 뜨거나 색이 뒤집혀 보입니다.
- **실행하지 않은 정확도·예측값을 글에 단정적으로 적기** — 직접 실행해 확인한 값만 신뢰하세요.

---

## 6. DAY6 핵심 정리

```text
오차 역전파법
  - 순전파→손실→역전파→가중치 갱신을 반복하며 학습
  - 역전파=기울기 계산, 경사하강법=가중치 수정
  - 연쇄법칙 미분은 loss.backward()가 자동으로 처리

모델 저장
  - 저장 대상: 학습된 가중치·편향(+ 전처리·클래스·하이퍼파라미터)
  - PyTorch: state_dict(.pth) → 불러올 때 같은 구조 코드 필요, eval()·no_grad()
  - TensorFlow/Keras: model.save(.keras) → 파일 하나로 구조까지 복원
  - 활용: 예측 / 추가 학습 / 웹 배포

컴퓨터비전 · OpenCV
  - 컴퓨터에게 이미지 = 숫자 배열 (흑백 (H,W), 컬러 (H,W,3), 0~255)
  - 작업: 분류·검출·분할·인식·캡셔닝 (작업마다 적합한 모델이 다름)
  - OpenCV 기본: imread / imshow / imwrite / waitKey, 색은 BGR
```

> 다음 DAY 주제는 이번 강의 PDF에서 확인할 수 없어 따로 예고하지 않습니다.

---

## 참고 자료

- 강의 자료: `DAY6_딥러닝_모델저장및활용.pdf` (교과목 2 — 데이터 분석과 머신러닝/딥러닝, 단원 3)
- PyTorch 공식 문서 — *Saving and Loading Models* (state_dict 저장·복원, 구조 먼저 생성)
- TensorFlow 공식 문서 — *Save and load models* (`model.save` / `load_model`)
- OpenCV 공식 문서 — https://docs.opencv.org/master/
