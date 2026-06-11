# DAY4 딥러닝 네트워크 모델 설계 - Codex 통합 검토안

## 전체 평가

초안은 PDF 28쪽의 세 축인 **네트워크 설계 절차**, **출력층과 손실함수 조합**, **대표 딥러닝 모델의 발전 흐름**을 대부분 빠짐없이 반영했다. 특히 PDF의 수식 이미지가 있는 p4, p5, p10~p15를 직접 확인한 결과, 초안의 뉴런 계산식, ReLU, MSE, MAE, Huber, BCE, Cross Entropy, KL 발산, 가중치 갱신식은 원문 취지와 일치하며 Markdown 수식도 정상적인 LaTeX 형태다.

PyTorch로 프레임워크를 통일하고 `CrossEntropyLoss`와 raw logits, 클래스 인덱스 라벨의 관계를 설명한 방향도 좋다. NumPy 수치 예시는 실행 결과가 맞고, Python 코드 블록은 문법상 정상이다. 다만 현재 상태를 그대로 최종 포스트로 사용하기에는 다음 문제가 남아 있다.

1. "끝까지 연결한 예제"가 실제로는 무작위 데이터에 대한 **한 번의 가중치 갱신**만 보여 주며, 데이터 분할ㆍ검증ㆍ테스트ㆍ평가 지표가 없다.
2. 문제 유형별 표에 PyTorch가 요구하는 **정확한 tensor shape와 dtype**이 충분히 드러나지 않는다.
3. `KLDivLoss` 설명에 기본 `reduction="mean"`의 함정과 함수 인자 순서가 빠져 있다.
4. 여러 코드 블록이 앞 블록의 `np` 또는 `nn` import에 암묵적으로 의존한다.
5. `pip install torch`는 OS와 CPU/CUDA 환경에 따라 달라지는 설치를 지나치게 단순화한다.

따라서 **높은 우선순위 항목을 반영한 뒤 게시 권장**이다. 원본 PDF와 초안은 수정하지 않았다.

### 검증 범위

- PDF: 28쪽 전체의 목차와 텍스트를 확인하고, 수식ㆍ도식 중심 페이지 p4, p5, p10~p15를 렌더링하여 직접 확인
- 초안: UTF-8, 제목 구조, 표, 링크, 코드 fence, LaTeX 수식 확인
- 코드: Python 코드 블록 10개의 문법 확인, NumPy 수치 예시 실행 확인
- PyTorch 코드: 현재 환경에 `torch`가 없어 실제 실행하지 못했으며 정적 검토만 수행
- 공식 문서 확인: PyTorch 설치 안내, `CrossEntropyLoss`, `BCEWithLogitsLoss`, `KLDivLoss`

## 높은 우선순위의 오류와 수정사항

### 1. "끝까지 연결한 예제"라는 설명과 실제 코드 범위가 다름

- 초안 위치: 5절, 290~335행
- 관련 PDF: p6의 12단계 중 "모델 학습ㆍ평가", "성능 개선ㆍ재설계"

코드는 모델 정의, 무작위 데이터 생성, 순전파, 역전파, 한 번의 갱신까지는 올바르다. 그러나 epoch 반복, train/validation/test 분할, 검증 손실 또는 평가 지표, 최종 테스트가 없으므로 "다중 분류를 끝까지 연결한 예제"라고 부르면 입문자가 이것을 완전한 학습 절차로 오해할 수 있다.

이 예제는 작은 shape/dtype 확인용 예제로 유지하되 제목과 설명을 **"한 스텝 학습 구조 확인"**으로 바꾸는 것이 가장 자연스럽다. 이어서 실제 프로젝트에서는 다음 원칙이 필요하다고 명시해야 한다.

- 학습 데이터로 가중치를 학습한다.
- 검증 데이터로 epoch 수, 학습률, 모델 구조를 선택한다.
- 테스트 데이터는 모든 선택이 끝난 뒤 최종 성능 확인에 한 번 사용한다.
- 표준화, 결측치 대치, 어휘 구축처럼 데이터에서 학습되는 전처리는 train에만 `fit`하고 validation/test에는 `transform`만 한다.
- Early Stopping은 validation 지표를 기준으로 하며 test 지표를 기준으로 삼지 않는다.

현재 예제는 완전한 학습 예제로 확대하기보다 범위를 정직하게 제한하는 편이 DAY4의 중심인 "출력층ㆍ라벨ㆍ손실 조합"을 흐리지 않는다.

### 2. 출력층ㆍ라벨ㆍ손실 표에 shape와 dtype 계약을 추가해야 함

- 초안 위치: 154~170행, 337~355행, 456~460행
- 관련 PDF: p7~p9, p13~p16

현재 표의 "실수", "0/1", "정수 클래스 인덱스"만으로는 PyTorch 런타임 오류를 예방하기 어렵다. 특히 `BCEWithLogitsLoss`는 logits와 target의 shape가 같아야 하고 target은 보통 부동소수점이어야 한다.

입문용 기본 조합은 다음처럼 명시하는 것이 안전하다.

| 문제 | 모델 출력 | 타깃 | 손실 |
|---|---|---|---|
| 회귀 1개 값 | logits/예측값 `(N, 1)`, float | `(N, 1)`, float | `MSELoss`, `L1Loss`, `HuberLoss` |
| 이진 분류 | logits `(N, 1)`, float | `(N, 1)`, float, 값 0~1 | `BCEWithLogitsLoss` |
| 다중 분류 | logits `(N, C)`, float | `(N,)`, `torch.long`, 값 0~C-1 | `CrossEntropyLoss` |
| 다중 라벨 | logits `(N, C)`, float | `(N, C)`, float, 각 원소 0~1 | `BCEWithLogitsLoss` |

회귀 타깃이 `(N,)`인 데이터셋도 만들 수 있지만, 예제의 출력이 `(N, 1)`이면 두 텐서 shape를 `(N, 1)`로 맞추는 원칙을 먼저 가르치는 편이 좋다.

### 3. `KLDivLoss`의 PyTorch 동작 설명이 불완전함

- 초안 위치: 266~275행, 441행
- 관련 PDF: p15~p16

초안은 입력이 log-확률이고 target이 확률이라는 점은 잘 적었다. 그러나 PyTorch 공식 문서상 다음 두 가지가 중요하다.

- 함수 호출은 `criterion(model_output, target)` 순서지만, 수학식 `D_KL(P || Q)`에서는 관측 분포 `P`가 target, 모델 분포 `Q`가 input에 해당한다.
- 기본값 `reduction="mean"`은 수학적 KL 발산 값과 일치하지 않는다. 수학식에 맞춘 배치 평균에는 `reduction="batchmean"`을 사용해야 한다.

이 항목은 버전 민감 API 설명이므로 공식 문서를 함께 연결해야 한다.

공식 문서: <https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html>

### 4. 코드 블록의 실행 단위를 명확히 해야 함

- 초안 위치: 178행 이후의 NumPy 예시, 210~264행의 손실함수 예시

NumPy의 BCE와 CE 블록은 앞선 블록의 `import numpy as np`에 의존하고, PyTorch 손실함수 블록은 210행의 `import torch.nn as nn`에 의존한다. 각 블록만 복사해 실행하는 독자는 `NameError`를 만날 수 있다.

다음 중 한 방법으로 통일해야 한다.

1. 각 독립 코드 블록에 필요한 import를 넣는다.
2. 4절 시작에 "이 절의 코드 블록은 위에서 아래 순서대로 실행한다"고 명시한다.
3. 짧은 API 나열은 코드 fence 대신 인라인 코드 또는 "부분 코드"라고 표시한다.

입문 글에서는 첫 번째 방법이 가장 안전하다.

### 5. PyTorch 설치 명령은 공식 설치 선택기로 안내해야 함

- 초안 위치: 297행

`pip install torch`는 CPU에서 간단히 시험할 때 동작할 수 있지만, Windows/Linux, Python 버전, CUDA/ROCm, CPU 전용 환경에 따라 공식 명령이 달라진다. 코드 주석에 단일 명령을 확정적으로 쓰기보다 다음처럼 바꿔야 한다.

> PyTorch 설치 명령은 운영체제와 GPU 환경에 따라 달라집니다. 공식 설치 선택기에서 OS, 패키지 관리자, 연산 플랫폼을 고른 뒤 표시되는 명령을 사용하세요.

공식 설치 선택기: <https://pytorch.org/get-started/locally/>

### 6. 데이터 분할ㆍ누수ㆍ평가 원칙이 빠져 있음

- 초안 위치: 5절과 6.2절
- 관련 PDF: p6 "모델 학습ㆍ평가", p5 과적합 방지

현재 예제는 생성 데이터의 한 스텝 계산이므로 그 자체에서 실제 데이터 누수는 발생하지 않는다. 그러나 독자가 이를 실제 데이터에 적용할 때 필요한 안전장치가 전혀 없다.

반드시 짧게라도 다음을 추가해야 한다.

- supervised learning에서는 train/validation/test를 먼저 나눈다.
- 스케일러, 결측치 처리, 특징 선택 등 학습형 전처리는 train에만 맞춘다.
- Data Augmentation은 일반적으로 train에만 적용한다.
- 모델과 하이퍼파라미터 선택은 validation으로 수행한다.
- test는 최종 평가용이며 반복 선택에 사용하지 않는다.
- 분류는 정확도만 보지 말고 불균형 여부에 따라 precision, recall, F1, ROC-AUC/PR-AUC 등을 고른다.
- 회귀는 MAE, RMSE 등 문제 비용과 단위에 맞는 지표를 고른다.

## PDF 기준 누락 내용

### 반드시 보완할 누락

1. **p6의 학습ㆍ평가 단계에 대응하는 실무 설명**
   - 12단계 목록은 옮겼지만 실제 예제에는 validation/test와 평가 지표가 없다.

2. **p15~p16 `KLDivLoss`의 실행 가능한 PyTorch 사용 형태**
   - 수식과 개념은 있으나 `log_softmax`, `batchmean`, 인자 순서가 빠졌다.

3. **p7~p9 출력층 설계를 PyTorch tensor 계약으로 연결하는 설명**
   - 문제별 노드 수와 활성화는 포함했지만 이진ㆍ다중 라벨 target의 shape/dtype이 본문 표에 없다.

### 선택적으로 보완할 누락

1. **p4 Softmax 수식**
   - 초안은 Softmax의 역할과 `CrossEntropyLoss` 내부 처리를 충분히 설명한다. DAY3에서 이미 다뤘다면 수식 반복은 생략 가능하되, "Softmax 수식과 확률 해석은 DAY3 참고"처럼 의도적 생략임을 표시하면 좋다.

2. **p17 퍼셉트론의 `y=f(sum(wx)+b)`**
   - 동일한 핵심 계산식이 2.3절에 이미 있으므로 모델 표에서 반복하지 않아도 된다.

3. **p20 합성곱 필터 계산 그림, p21 LSTM 게이트 도식, p22 AutoEncoder 구조 그림**
   - 글의 중심이 모델 카탈로그가 아니라 설계와 손실함수이므로 현재처럼 한 줄 요약으로 줄인 것은 타당하다. 빠진 핵심이라기보다 의도적 범위 축소로 볼 수 있다.

4. **p24의 대표 GAN 종류**
   - `DCGAN`, `CycleGAN`, `StyleGAN`이 초안에서 생략되었다. 입문 범위에서는 필수는 아니며, 모델 이름 나열을 늘리지 않는 편이 더 읽기 쉽다.

## 더 자세히 설명할 내용

### 1. "입력층 노드 수 = 특성 수"의 적용 범위

- 초안 위치: 109~117행, 453행

이 설명은 고정 길이 벡터를 넣는 MLP에서는 맞지만 CNN, RNN, Transformer까지 일반화하면 오해가 생긴다. 이미지는 보통 `(N, C, H, W)`, 시퀀스는 `(N, L, D)` 같은 구조를 유지하고 모델의 입력 채널 수나 임베딩 차원을 설계한다.

다음처럼 범위를 제한해야 한다.

> 표 형태 데이터를 MLP에 넣는 경우 입력층의 `in_features`는 샘플당 특성 수와 같습니다. 이미지ㆍ문장ㆍ시계열은 채널, 길이, 높이, 너비 같은 축을 유지하므로 단순히 "특성 수 = 입력 노드 수"로 설명하지 않습니다.

### 2. 은닉층 개수에 대한 수치적 일반화

- 초안 위치: 117행

"단순한 문제는 1~2개, 복잡한 이미지ㆍ자연어는 수십~수백 층"은 구조 선택을 문제 복잡도 하나로 결정하는 것처럼 보인다. 데이터 양, 사전학습 모델 사용 여부, 계산 예산, skip connection, 정규화 등도 영향을 준다.

> 층 수와 너비는 문제 복잡도뿐 아니라 데이터 규모, 계산 자원, 과적합 정도, 사용할 아키텍처에 따라 달라집니다. 처음에는 작은 기준 모델에서 시작해 validation 성능을 보며 늘리는 편이 안전합니다.

### 3. `CrossEntropyLoss`의 타깃 형식 표현

- 초안 위치: 168~170행, 438행

입문용 단일 정답 분류에서는 `(N,)`의 `torch.long` 클래스 인덱스를 권장하는 설명이 맞다. 다만 최신 PyTorch는 label smoothing, mixup 같은 경우를 위해 클래스 확률 형태의 float target도 지원한다. "정수 인덱스만 받는다"는 절대 표현은 다음처럼 완화한다.

> 일반적인 단일 정답 다중 분류에서는 `(N,)` 모양의 `torch.long` 클래스 인덱스를 사용합니다. PyTorch는 soft label을 위한 클래스 확률 target도 지원하지만, 입문 예제는 클래스 인덱스 방식을 사용합니다.

공식 문서: <https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>

### 4. BCE 수식의 확률과 API의 logits 구분

- 초안 위치: 217~234행

BCE 수식의 `hat y`는 0~1 확률이지만, `BCEWithLogitsLoss`에 실제로 넘기는 모델 출력은 Sigmoid 전의 logits다. 현재 주석으로 일부 설명했지만 수식 직후 한 문장으로 연결하면 더 명확하다.

> 위 수식의 `hat y`는 Sigmoid를 통과한 확률입니다. 하지만 `BCEWithLogitsLoss`에는 확률이 아니라 Sigmoid 전 logits를 전달하며, 손실함수가 내부에서 Sigmoid와 BCE를 안정적으로 함께 계산합니다.

### 5. `model.train()`과 `model.eval()`의 의미

- 초안 위치: 375~386행

Dropout만 언급하지 말고 BatchNorm의 동작도 학습/평가 모드에 따라 달라진다고 설명해야 한다. 또한 `model.eval()`은 gradient 계산을 자동으로 끄지 않으므로 평가 시 `torch.no_grad()` 또는 `torch.inference_mode()`도 함께 사용해야 한다.

### 6. Batch Normalization 설명

- 초안 위치: 382행

"층의 입력 분포를 정규화해 학습을 안정ㆍ가속"은 입문 요약으로 이해할 수 있으나, 가속이 항상 보장되는 것처럼 읽힐 수 있다.

> 미니배치의 중간 활성값을 정규화하고 학습 가능한 scaleㆍshift를 적용합니다. 많은 모델에서 학습을 안정시키는 데 도움이 되지만 효과는 구조와 배치 크기에 따라 달라집니다.

### 7. 모델 카탈로그의 조건부 표현

- 초안 위치: 398~410행

다음 표현은 단정적으로 읽히지 않도록 완화하는 것이 좋다.

- DNN "대량 데이터 필요" -> 깊은 모델은 보통 더 많은 데이터ㆍ정규화ㆍ사전학습의 도움을 필요로 할 수 있음
- GRU "학습 빠름, 성능은 LSTM과 유사" -> 파라미터가 적어 더 빠를 수 있으며 성능은 데이터에 따라 다름
- Transformer "긴 문장에 강함" -> RNN보다 장거리 의존성을 직접 연결하지만 attention 비용과 context 길이 제한이 있음
- Diffusion의 "DALL-E" -> 모든 버전을 한 방식으로 묶지 말고 구체적 버전을 밝히거나 예시에서 제외

## 유용한 추가 내용

### 1. 평가 지표 선택 표

PDF의 중심은 손실함수지만 입문자는 손실과 평가 지표를 같은 것으로 오해하기 쉽다. 다음 짧은 표를 보충 내용으로 넣으면 좋다.

| 문제 | 학습 손실 예 | 평가 지표 예 |
|---|---|---|
| 회귀 | MSE, MAE, Huber | MAE, RMSE, R2 |
| 균형 잡힌 분류 | Cross Entropy | Accuracy, macro/weighted F1 |
| 불균형 이진 분류 | BCEWithLogitsLoss | Precision, Recall, F1, PR-AUC |
| 다중 라벨 | BCEWithLogitsLoss | micro/macro F1, label별 precision/recall |

손실은 gradient를 계산하기 위한 학습 목표이고, 평가지표는 실제 업무 목표에 맞춰 성능을 해석하는 기준이라는 차이도 한 문장으로 설명한다.

### 2. 클래스 불균형 주의

`CrossEntropyLoss(weight=...)`와 `BCEWithLogitsLoss(pos_weight=...)`가 존재한다는 정도만 소개할 수 있다. 다만 가중치 설정법까지 깊게 들어가면 DAY4 범위를 벗어나므로 "불균형에서는 accuracy만으로 판단하지 않는다"를 우선한다.

### 3. seed와 재현성 범위

초안의 "`torch.manual_seed(42)`가 완전한 일치를 보장하지 않는다"는 주석은 좋다. GPU 연산, 라이브러리 버전, 비결정적 연산 때문에 결과가 달라질 수 있다는 한 문장을 덧붙이면 충분하다.

### 4. raw logits에서 예측값으로 바꾸는 최소 예

손실 계산에는 Softmax를 넣지 않되, 추론 시 해석 방법을 한 번 보여 주면 입문자가 혼동하지 않는다.

```python
model.eval()
with torch.no_grad():
    logits = model(X)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = probabilities.argmax(dim=1)
```

## 줄이거나 제거할 내용

### 1. 게시 전 기획 주석 제거

- 초안 위치: 1~67행

초안 제작 과정, 실행 환경, 페이지 지도는 최종 게시물 독자에게 필요하지 않다. HTML 주석이라 렌더링되지 않더라도 원고에는 남지 않게 제거한다.

### 2. 실행 불가 안내의 반복 축소

- 초안 위치: 78행, 294행

PyTorch를 실행하지 못했다는 고지는 정직하고 필요하다. 다만 서두와 코드 직전에 같은 내용을 길게 두 번 적기보다, 코드 직전에 한 번 명확히 남기고 서두에서는 짧게 줄여도 된다.

### 3. 모델 이름 나열의 밀도

7절은 PDF 범위를 충실히 담지만 모델 이름이 많아 글의 핵심인 출력층ㆍ손실 설계를 흐릴 수 있다. 현재 표는 유지하되 각 모델의 추가 예시를 더 늘리지 않는 것이 좋다. p20ㆍp24의 세부 모델명 누락을 모두 보충할 필요는 없다.

### 4. "가장 많이", "최적", "강함" 표현

다음 표현은 시점과 조건에 따라 달라지므로 완화한다.

- 95행 "최적의 신경망 구조" -> "문제에 적절한 후보 구조"
- 238행 "가장 많이 씁니다" -> "대표적으로 사용합니다"
- 406행 "긴 문장에 강함" -> "장거리 관계를 직접 모델링할 수 있음"

## 바로 붙여 넣을 수 있는 수정 블록

### 블록 1. 문제별 출력ㆍ라벨ㆍ손실 표 교체

```markdown
### 3.1 한눈에 보는 짝 맞추기 표

아래 shape에서 `N`은 배치 크기, `C`는 클래스 개수입니다.

| 문제 유형 | 모델의 raw 출력 | 라벨 형식 | PyTorch 손실함수 |
|---|---|---|---|
| **회귀(값 1개)** | `(N, 1)` float, 출력 활성화 없음 | `(N, 1)` float | `nn.MSELoss` / `nn.L1Loss` / `nn.HuberLoss` |
| **이진 분류** | logits `(N, 1)` float | `(N, 1)` float, 값 0~1 | `nn.BCEWithLogitsLoss` |
| **다중 분류** | logits `(N, C)` float | `(N,)` `torch.long`, 값 0~C-1 | `nn.CrossEntropyLoss` |
| **다중 라벨 분류** | logits `(N, C)` float | `(N, C)` float, 각 원소 0~1 | `nn.BCEWithLogitsLoss` |

`CrossEntropyLoss`에는 Softmax 전 logits를, `BCEWithLogitsLoss`에는 Sigmoid 전 logits를 전달합니다. 확률이 필요할 때만 평가ㆍ추론 단계에서 각각 `softmax` 또는 `sigmoid`를 적용합니다.

> 일반적인 단일 정답 다중 분류에서는 `(N,)` 모양의 `torch.long` 클래스 인덱스를 사용합니다. PyTorch는 soft label을 위한 클래스 확률 target도 지원하지만, 이 글의 입문 예제는 클래스 인덱스 방식을 사용합니다.
```

### 블록 2. 한 스텝 예제의 범위와 데이터 분할 설명

```markdown
## 5. 하나로 합치기 - PyTorch 한 스텝 학습 구조 확인

아래 코드는 다중 분류에서 **입력 -> 모델 -> logits -> 손실 -> 역전파 -> 가중치 갱신**이 어떻게 연결되는지 한 스텝으로 확인하는 예제입니다. 무작위로 만든 8개 샘플만 사용하므로 실제 성능을 평가하는 완전한 학습 예제는 아닙니다.

실제 프로젝트에서는 데이터를 먼저 train/validation/test로 나눕니다. 모델은 train으로 학습하고, 구조ㆍ학습률ㆍepoch 수와 Early Stopping 시점은 validation으로 선택합니다. test는 모든 선택이 끝난 뒤 최종 성능을 확인할 때 사용합니다. 스케일러나 결측치 대치처럼 데이터에서 값을 학습하는 전처리는 train에만 `fit`하고 validation/test에는 `transform`만 적용해야 데이터 누수를 막을 수 있습니다.
```

### 블록 3. 학습 모드와 평가 모드

```python
# 학습 단계
model.train()
logits = model(X_train)
loss = criterion(logits, y_train)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 검증ㆍ평가 단계: Dropout과 BatchNorm을 평가 모드로 전환
model.eval()
with torch.no_grad():
    val_logits = model(X_val)
    val_loss = criterion(val_logits, y_val)
    val_pred = val_logits.argmax(dim=1)
    val_accuracy = (val_pred == y_val).float().mean()
```

```markdown
`model.eval()`은 Dropout과 BatchNorm 등의 동작 모드를 바꾸지만 gradient 계산까지 끄지는 않습니다. 평가 시에는 `torch.no_grad()` 또는 `torch.inference_mode()`를 함께 사용합니다.
```

### 블록 4. `KLDivLoss` 교체 설명과 코드

```markdown
### KLDivLoss를 PyTorch에서 사용할 때

수학식 `D_KL(P || Q)`에서 `P`는 기준 분포, `Q`는 모델 분포입니다. 그러나 PyTorch 손실함수의 호출 순서는 일반적인 손실함수처럼 `criterion(model_output, target)`입니다. 따라서 첫 번째 인자에는 모델 분포 `Q`의 log-확률을, 두 번째 인자에는 기준 분포 `P`의 확률을 넣습니다.

또한 기본 `reduction="mean"`은 수학적 KL 발산 값과 같지 않으므로, 배치 단위의 수학적 정의에 맞추려면 `reduction="batchmean"`을 사용합니다.
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.randn(3, 5)              # 모델의 raw 출력 Q
target_logits = torch.randn(3, 5)       # 예시용 기준 분포 P

log_q = F.log_softmax(logits, dim=1)    # input: log-확률
p = F.softmax(target_logits, dim=1)     # target: 확률

criterion = nn.KLDivLoss(reduction="batchmean")
loss = criterion(log_q, p)
print(loss.item())
```

### 블록 5. 설치 안내 교체

```markdown
> **PyTorch 설치**  
> 설치 명령은 운영체제와 CPU/CUDA 환경에 따라 달라집니다. [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 OS, 패키지 관리자, 연산 플랫폼을 선택한 뒤 표시되는 명령을 사용하세요.
```

코드 첫 줄의 `# 프레임워크: PyTorch (설치: pip install torch)`는 다음처럼 바꾼다.

```python
# 프레임워크: PyTorch
# 설치 명령은 https://pytorch.org/get-started/locally/ 에서 환경별로 확인
```

### 블록 6. 입력층 설명 교체

```markdown
**표 형태 데이터를 MLP에 넣는 경우**, 입력층의 `in_features`는 샘플당 특성 수와 같습니다. 예를 들어 학생 한 명을 `(국어, 영어, 수학)` 3개 특성으로 표현하면 `in_features=3`입니다.

이미지ㆍ문장ㆍ시계열은 다릅니다. 이미지는 보통 `(N, C, H, W)`, 시퀀스는 `(N, L, D)`처럼 여러 축을 유지하므로 "입력 노드 수 = 특성 수"라는 규칙을 그대로 적용하지 않습니다. 이때는 입력 채널 수, 시퀀스 길이, 임베딩 차원 등 해당 구조에 맞는 입력 크기를 설계합니다.
```

### 블록 7. 평가 지표와 손실의 차이

```markdown
### 손실함수와 평가지표는 역할이 다르다

손실함수는 역전파로 가중치를 업데이트하기 위한 학습 목표이고, 평가지표는 실제 문제에서 모델 성능을 해석하는 기준입니다.

| 문제 | 학습 손실 예 | 평가 지표 예 |
|---|---|---|
| 회귀 | MSE, MAE, Huber | MAE, RMSE, R2 |
| 균형 잡힌 분류 | Cross Entropy | Accuracy, F1 |
| 불균형 이진 분류 | BCEWithLogitsLoss | Precision, Recall, F1, PR-AUC |
| 다중 라벨 | BCEWithLogitsLoss | micro/macro F1, label별 precision/recall |

지표는 데이터와 업무 비용에 맞춰 고릅니다. 예를 들어 양성 사례가 매우 드물면 accuracy가 높아도 좋은 모델이 아닐 수 있으므로 precision과 recall을 함께 확인해야 합니다.
```

## 우선순위 표

| 우선순위 | 항목 | 조치 |
|---|---|---|
| 높음 | 5절이 완전한 학습 예제처럼 서술됨 | "한 스텝 구조 확인"으로 범위를 수정하고 train/validation/test 역할 추가 |
| 높음 | 문제별 shapeㆍdtype 계약 부족 | 출력ㆍ타깃ㆍ손실 표를 구체적인 tensor 계약으로 교체 |
| 높음 | `KLDivLoss`의 `batchmean`과 인자 순서 누락 | 공식 문서 기준 설명과 실행 코드 추가 |
| 높음 | 데이터 누수ㆍ평가 방식 누락 | train에만 전처리 fit, validation 선택, test 최종 1회 원칙 추가 |
| 높음 | 코드 블록이 `np`, `nn`을 암묵적으로 재사용 | 각 블록에 import 추가 또는 순차 실행/부분 코드 표시 |
| 높음 | `pip install torch` 단일 설치 안내 | 공식 설치 선택기로 교체 |
| 중간 | "입력 노드 수 = 특성 수"를 전체 DL에 일반화 | MLP 표 데이터에 한정하고 이미지ㆍ시퀀스 shape 설명 |
| 중간 | `CrossEntropyLoss`가 정수 인덱스만 받는다는 표현 | 일반 입문 방식임을 밝히고 확률 target 지원을 짧게 언급 |
| 중간 | BCE 수식의 확률과 API logits 관계 | 수식 직후 명시 |
| 중간 | `model.eval()` 설명 부족 | BatchNorm과 `no_grad()`까지 설명 |
| 중간 | BatchNormㆍGRUㆍTransformer 등의 단정 표현 | 조건부 표현으로 완화 |
| 낮음 | Softmax 수식, 세부 CNN/GAN 그림ㆍ모델명 생략 | DAY 범위를 고려해 생략 가능, 필요 시 이전 글 연결 |
| 낮음 | 게시용이 아닌 기획 주석과 반복 고지 | 최종본에서 제거ㆍ축소 |

## 최종 권고

현재 초안은 PDF 핵심 범위와 수식을 충실히 담았고, 프레임워크 혼용이나 출력층ㆍ손실함수의 치명적인 조합 오류는 없다. 그러나 **한 스텝 예제의 범위 표시, tensor shape/dtype 표, `KLDivLoss`, 데이터 분할ㆍ누수ㆍ평가 원칙, 설치 안내, 코드 블록 import**는 게시 전에 반드시 수정해야 한다.

위 높은 우선순위 항목을 반영하면 DAY4 입문 글로 게시하기에 충분하다. 모델 카탈로그의 세부 그림과 추가 모델명은 범위를 넓히지 않기 위해 현재처럼 간결하게 유지하는 것을 권장한다.

## 확인한 공식 문서

- PyTorch 설치 선택기: <https://pytorch.org/get-started/locally/>
- `CrossEntropyLoss`: <https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
- `BCEWithLogitsLoss`: <https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>
- `KLDivLoss`: <https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html>
