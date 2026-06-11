# DAY3 딥러닝 데이터 처리와 데이터셋 Codex 통합 검토안

## 1. 전체 평가

초안은 PDF의 네 개 대단원인 **데이터셋 이해, MNIST, CIFAR-10, 다차원 배열과 행렬**을 모두 포함하고 있으며, 강의 내용과 실무 보충을 `🟦`와 `🟩`로 구분한 점이 좋습니다. 특히 PDF의 Softmax 예시 수치 불일치와 CIFAR-100 클래스 예시를 그대로 반복하지 않고 실제 계산·공식 클래스 목록에 맞게 정정한 판단은 적절합니다.

PDF는 총 56쪽이며, **35~56쪽은 페이지당 추출 텍스트가 약 11자에 불과한 이미지 전용 구간**입니다. 이 구간을 전부 렌더링해 확인한 결과, 초안은 행렬 단원의 큰 흐름은 복원했지만 여러 성질과 블록 행렬 계산을 상당히 축약했습니다. 입문 글의 범위를 고려하면 모든 증명과 예제를 옮길 필요는 없지만, PDF 핵심 범위를 충족하려면 최소 성질 몇 개는 보강하는 편이 좋습니다.

현재 초안은 전체적으로 완성도가 높지만 다음 항목은 최종본 전에 반드시 수정해야 합니다.

1. `to_categorical()` 결과를 `float32`라고 적은 dtype 불일치
2. Windows를 포함해 “GPU가 있으면 자동 사용”한다고 읽히는 device 설명
3. test 데이터를 오버피팅 진단에 포함한 표와 `validation_split`의 실제 선택 방식 설명 부족
4. MLP가 2차원 입력을 받을 수 없고 Flatten이 공간 정보를 없앤다는 과도한 단정

### 페이지별 범위 지도

| PDF 범위 | PDF 핵심 내용 | 초안 상태 | 조치 |
|---|---|---|---|
| p.3~7 | 데이터 중요성, 수집·분석·가공, train/validation/test, 언더·오버피팅 | 대부분 반영 | 데이터 처리 흐름과 test 역할 표현 보완 |
| p.8~14 | 모델 제작 5단계, MLP, compile/fit, epoch/batch, loss/accuracy 곡선 | 대부분 반영 | batch size 단정 완화, accuracy 곡선 코드 추가 |
| p.15~22 | MNIST 구조, Flatten, One-Hot, 784→512→256→128→10 | 반영 | MLP 입력과 One-Hot의 절대 표현 수정 |
| p.23~28 | Softmax 정의·계산·예측·수식, Sigmoid 비교 | 잘 반영 | 확률 해석의 한계 한 줄 보충 가능 |
| p.29~34 | CIFAR-10 클래스·shape·난이도·샘플 이미지, CIFAR-100 | 잘 반영 | Flatten의 공간 구조 설명 수정, 샘플 시각화 추가 권장 |
| p.35~43 | 행렬 종류, 덧셈·스칼라곱·영행렬, 곱셈 성질, 전치·내적 | 부분 반영 | 영행렬과 핵심 전치·곱셈 성질 추가 |
| p.44~49 | 분할행렬과 네 가지 곱셈 관점, 블록 덧셈·곱셈 | 개념만 요약 | 블록 곱셈 예시 1개 또는 계산 조건 추가 |
| p.50~56 | 연립방정식↔행렬식, 역행렬·가역 성질, 해, 영공간 | 핵심 반영 | 역행렬 유일성과 대표 가역 성질은 선택 보강 |

### 검증 결과

- Markdown의 Python 코드 블록 10개는 모두 문법 분석을 통과했습니다.
- TensorFlow가 설치되지 않은 환경이어서 MNIST/CIFAR 학습 코드는 실행하지 못했습니다.
- NumPy 예제 7개는 순서대로 실행했으며 초안의 행렬·Softmax·One-Hot 출력과 일치했습니다.
- 행렬 단원의 LaTeX 수식을 PDF p.35~56과 다시 대조했습니다. 현재 역행렬 공식과 영공간 예제의 계산은 정확하지만, 행렬 곱의 일반식·전치 성질·블록 행렬 곱이 수식으로 제시되지 않았고 영공간의 정의역 차원이 생략되어 있습니다.
- MNIST/CIFAR 데이터 shape, `validation_split`, 설치·GPU 조건은 현재 공식 문서와 대조했습니다.
- 코드 fence 수는 짝수이고 UTF-8 손상 문자는 확인되지 않았습니다.

---

## 2. 높은 우선순위의 오류와 수정사항

### H1. `to_categorical()`의 dtype 표기가 실제 코드와 다름

**위치:** 초안 293~294행, 329행

현재 Keras의 NumPy 입력 경로에서 `to_categorical()`은 내부적으로 dtype을 지정하지 않은 `np.zeros()`를 사용하므로 일반적으로 `float64` 배열을 반환합니다. 초안은 별도 변환 없이 정답 `y`를 `float32`라고 적고 있어 코드와 설명이 일치하지 않습니다.

학습 자체는 프레임워크 내부 변환으로 동작할 수 있지만, dtype을 명시적으로 가르치는 글에서는 실제 배열과 설명이 같아야 합니다.

**수정:**

```python
y_train = to_categorical(y_train, 10).astype("float32")
y_test  = to_categorical(y_test, 10).astype("float32")
```

CIFAR-10 부분 코드에도 같은 수정을 적용해야 합니다.

공식 확인: [Keras `to_categorical` 구현](https://github.com/keras-team/keras/blob/master/keras/src/utils/numerical_utils.py)

### H2. device 설명이 현재 Windows GPU 지원 조건을 누락함

**위치:** 초안 331행

현재 문장:

```text
장치(device): Keras가 자동 선택(GPU 있으면 GPU, 없으면 CPU)
```

이 문장은 컴퓨터에 GPU 하드웨어만 있으면 자동으로 사용할 수 있다는 뜻으로 읽힙니다. 실제로는 **TensorFlow가 인식하는 호환 장치와 올바른 드라이버·설치 구성이 있어야** 합니다. 특히 TensorFlow 공식 설치 문서에 따르면 네이티브 Windows의 NVIDIA GPU 지원은 TensorFlow 2.10이 마지막이며, 최신 TensorFlow에서 GPU를 사용하려면 일반적으로 WSL2 구성이 필요합니다.

**권장 문장:**

```text
장치(device): TensorFlow가 현재 환경에서 인식한 호환 장치에 연산을 배치합니다.
GPU 하드웨어가 있어도 드라이버와 TensorFlow 설치 구성이 맞지 않으면 CPU를 사용합니다.
네이티브 Windows의 최신 TensorFlow는 NVIDIA GPU를 직접 지원하지 않으므로 GPU 실습은 WSL2 등 공식 설치 조건을 확인해야 합니다.
```

확인 코드:

```python
print(tf.config.list_physical_devices("GPU"))
```

공식 확인: [TensorFlow pip 설치 및 GPU 지원 조건](https://www.tensorflow.org/install/pip)

### H3. test를 언더피팅·오버피팅 진단에 사용하는 것처럼 보임

**위치:** 초안 129~134행

표의 열 제목이 `검증/시험 성능`으로 되어 있습니다. 그러나 바로 앞에서는 test를 모든 선택이 끝난 뒤 한 번만 사용한다고 설명합니다. 언더피팅·오버피팅을 반복해서 진단하고 epoch나 모델 크기를 바꾸는 데 사용하는 것은 **validation 성능**이어야 합니다. test 결과를 보고 설정을 변경하면 test가 사실상 validation 역할을 하게 됩니다.

**수정:**

- 표의 `검증/시험 성능`을 `검증 성능`으로 변경
- 표 아래에 “test는 이 진단과 설정 선택에 사용하지 않는다”는 문장 추가

### H4. `validation_split=0.2`가 무작위·계층 분할처럼 설명됨

**위치:** 초안 312~317행

Keras의 `validation_split`은 전달된 배열의 **마지막 일부를 검증 데이터로 먼저 떼고, 그 뒤에 학습 데이터를 shuffle**합니다. 따라서 데이터가 클래스·시간·그룹 순서로 정렬되어 있다면 편향된 검증셋이 될 수 있습니다. `set_random_seed(42)`만으로 이 선택 방식이 무작위 분할로 바뀌지는 않습니다.

MNIST 원본 훈련 배열은 실습에 사용할 수 있지만, 일반 원칙을 함께 가르치려면 다음 중 하나가 필요합니다.

- fit 전에 고정 시드로 훈련 배열 순서를 섞은 뒤 `validation_split` 사용
- 별도의 `X_val`, `y_val`을 만들고 `validation_data=(X_val, y_val)`로 전달
- 일반 분류 데이터에서는 가능한 경우 클래스 비율을 유지하는 stratified split 사용

공식 확인: [Keras `Model.fit`의 `validation_split`](https://keras.io/api/models/model_training_apis/)

### H5. MLP와 Flatten 설명이 기술적으로 너무 강함

**위치:** 초안 201행, 414행

현재 표현:

```text
기본 MLP는 2차원 이미지(28×28)를 그대로 받지 못합니다.
Flatten하면 상하좌우 위치 같은 공간 정보가 사라집니다.
```

이번 `Input(shape=(784,))` MLP 구성은 1차원 특징 벡터를 기대하므로 Flatten이 필요하지만, `Dense` 계층 자체가 모든 고차원 입력을 문법적으로 거부하는 것은 아닙니다. 또한 Flatten은 픽셀값을 삭제하지 않습니다. 다만 2차원 이웃 관계가 **구조로 명시되지 않고**, 완전 연결층은 CNN처럼 지역성과 이동 특성을 활용하는 귀납적 편향이 없습니다.

**권장 표현:**

```text
이번 실습의 MLP는 입력 shape을 `(784,)`로 설계했으므로 28×28 이미지를 784개 특징의 벡터로 펼칩니다.
Flatten은 픽셀값 자체를 버리지는 않지만, 높이·너비의 2차원 이웃 구조를 모델 입력에 명시적으로 보존하지 않습니다.
CNN은 이 지역적 구조를 활용하도록 설계되어 일반적인 이미지 분류에 더 적합합니다.
```

### H6. One-Hot이 반드시 필요하다는 앞 설명과 뒤 설명이 충돌함

**위치:** 초안 207행과 334행

207행은 출력이 10개이므로 정답도 반드시 One-Hot이어야 한다고 설명하지만, 334행에서는 정수 라벨과 `sparse_categorical_crossentropy`도 가능하다고 올바르게 설명합니다.

**수정:** One-Hot은 모델 출력 때문에 무조건 필요한 것이 아니라 **선택한 손실함수의 라벨 형식에 맞추기 위해 사용하는 방식**이라고 설명해야 합니다.

---

## 3. PDF 기준 누락 내용

### 3.1 데이터 수집 → 분석 → 가공 흐름

**PDF p.4**

PDF는 모델 학습 이전에 데이터를 수집하고, 특징을 분석하고, 학습 가능한 형태로 가공하는 흐름을 제시합니다. 초안은 데이터 품질과 다양성은 설명하지만 이 세 단계가 명시적으로 연결되지 않습니다.

다음 한 문장을 추가하면 충분합니다.

```text
모델 학습 전에는 데이터를 수집하고, 분포·오류·결측값을 분석한 뒤, 모델이 처리할 수 있는 숫자와 일정한 shape으로 가공합니다.
```

### 3.2 모델 구조에 하나의 정답이 없다는 설명

**PDF p.21**

PDF의 784→512→256→128→10 구조는 예시이며 층 수와 노드 수에 확정적인 정답이 없다고 강조합니다. 초안은 해당 구조를 소개하지만 이 점이 빠져 있어 독자가 강의 구조를 정답처럼 외울 수 있습니다.

### 3.3 Accuracy의 정의와 accuracy 학습 곡선

**PDF p.13~14**

초안은 Accuracy가 높을수록 좋다고만 설명하고, 학습 곡선 코드는 loss만 그립니다. 다음 내용을 보강해야 PDF 설명과 코드가 일치합니다.

- Accuracy = 전체 예측 중 정답을 맞힌 비율
- train/validation accuracy를 함께 그림
- 한 epoch마다 반드시 단조 증가하는 것은 아니며 전체 추세를 봄

### 3.4 CIFAR-10 샘플 이미지 확인

**PDF p.34**

PDF는 작은 샘플 이미지를 보여 주며 데이터의 형태와 클래스별 시각적 다양성을 확인하게 합니다. 초안은 클래스 목록과 난이도만 설명하므로, 학습 전에 몇 장을 라벨과 함께 표시하는 코드가 있으면 PDF의 교육 의도가 더 잘 살아납니다.

### 3.5 행렬 핵심 성질

**PDF p.35~43**

다음 항목이 초안에서 빠졌습니다.

- 같은 크기에서 대응 성분이 같을 때 두 행렬이 같다는 정의
- 영행렬과 `A + O = A`
- 행렬 곱의 결합법칙·분배법칙과 일반적으로 성립하지 않는 교환법칙
- 단위행렬의 크기에 따른 `I_m A = A = A I_n`
- 전치의 핵심 성질 `(AB)^T = B^T A^T`

모든 성질을 긴 증명으로 넣을 필요는 없지만, 전치 시 곱의 순서가 바뀐다는 성질은 향후 신경망 수식 이해에도 유용하므로 넣는 편이 좋습니다.

### 3.6 분할행렬 계산의 실제 예

**PDF p.44~49**

초안은 행렬-열, 행-행렬, 행-열, 열-행 관점을 이름만 소개합니다. 최소한 다음 중 하나는 예로 보여 주는 편이 좋습니다.

- 결과 행렬의 한 성분은 왼쪽 행과 오른쪽 열의 내적이라는 관점
- 열-행 외적들의 합으로 행렬 곱을 표현하는 관점
- 블록 행렬 곱은 각 블록의 shape이 맞을 때 일반 행렬 곱처럼 계산한다는 예

### 3.7 역행렬의 유일성과 대표 성질

**PDF p.52~54**

초안은 역행렬 정의와 2×2 공식은 다루지만 역행렬이 존재하면 유일하다는 점과 `(AB)^{-1}=B^{-1}A^{-1}` 같은 대표 성질은 생략했습니다. 입문 범위를 고려하면 필수는 아니며, 한 줄 요약 또는 접이식 보충으로 충분합니다.

### 3.8 행렬 수식 표기와 변수 일치

**PDF p.35~56 / 초안 418~508행**

수학적으로 잘못된 핵심 공식은 없지만 다음 표기 문제를 수정해야 합니다.

1. `P = np.array(...)`를 제곱한 코드의 주석이 `= A²`로 되어 있습니다. `= P²`로 바꿔야 코드 변수와 설명이 일치합니다.
2. 행렬 곱은 결과값만 보여 주고 일반식이 없습니다. PDF p.40의 정의대로 `c_{ij}`가 왼쪽 행과 오른쪽 열의 내적이라는 식을 넣어야 합니다.
3. 내적 `uᵀv`가 Python 주석에만 있습니다. 본문 수식으로 분리하면 위첨자와 합 기호가 정상적으로 렌더링됩니다.
4. 영공간은 `AX=0`을 만족하는 값의 모임이라고만 되어 있습니다. `A\in\mathbb{R}^{m\times n}`일 때 `x\in\mathbb{R}^{n}`이라는 차원과 `\operatorname{Null}(A)\subseteq\mathbb{R}^{n}`을 명시해야 정의가 완전합니다.
5. 역행렬 정의에서는 `A`가 정사각행렬이라는 조건을 먼저 밝혀야 합니다.
6. 긴 행렬과 집합 표현은 인라인 `$...$`보다 독립된 `$$...$$` 표시 수식으로 두는 편이 Velog에서 잘리지 않고 읽기 쉽습니다.

초안의 현재 수식:

```latex
A=\begin{bmatrix}a&b\\c&d\end{bmatrix},\quad
A^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix},\quad
\det(A)=ad-bc
```

은 정확합니다. 예제 `A=[[1,2],[3,5]]`의 `det(A)=-1`, `A^{-1}=[[-5,2],[3,-1]]`, 연립방정식의 해 `(3,-1)`, 영공간 예제 `\{(-2t,t)\mid t\in\mathbb{R}\}`도 모두 계산상 맞습니다.

---

## 4. 더 자세히 설명할 내용

### 4.1 데이터 양보다 대표성과 라벨 품질이 중요함

**위치:** 초안 109~111행

“많이 공부할수록 잘한다”, “데이터가 다양할수록 실제 상황에서도 정확해진다”는 표현은 방향은 맞지만 보장이 아닙니다. 잘못된 라벨, 실제 환경과 다른 데이터, 중복 데이터가 많으면 양이 늘어도 성능이 좋아지지 않을 수 있습니다.

권장 표현:

```text
실제 환경을 잘 대표하고 라벨 품질이 좋은 데이터를 충분히 확보하면 일반화 성능이 좋아질 가능성이 큽니다.
```

### 4.2 언더피팅·오버피팅의 대응 방법

현재 표는 언더피팅에 “학습 더, 모델 키우기”만 제시하고 오버피팅 대응은 없습니다.

- 언더피팅: 모델 용량, 특징, 학습률, 학습 시간 점검
- 오버피팅: 더 다양한 훈련 데이터, 데이터 증강, 정규화, dropout, 조기 종료 등을 검토
- 어느 방법도 항상 해결을 보장하지 않음

### 4.3 Batch Size의 트레이드오프

**위치:** 초안 169~174행

큰 batch가 항상 더 빠르고 작은 batch가 항상 더 느린 것은 아닙니다. 실제 속도는 하드웨어, 데이터 파이프라인, 모델 크기에 따라 달라집니다. 또한 batch 크기는 메모리뿐 아니라 gradient의 변동성과 일반화에도 영향을 줍니다.

`32, 64, 128`은 PDF가 제시한 흔한 시작값으로 표시하되 보편적 정답처럼 읽히지 않게 해야 합니다.

### 4.4 Accuracy가 적절한 조건

MNIST는 클래스 수가 균형적인 편이므로 Accuracy를 기본 지표로 쓰는 것이 적절합니다. 다만 일반화된 설명에서는 클래스 불균형이나 오류 비용이 다른 문제에서는 precision, recall, F1, confusion matrix 등이 필요할 수 있다고 한 줄 덧붙이면 좋습니다.

### 4.5 Softmax 값의 해석

Softmax 출력의 합은 1이지만, 값이 실제 정답 확률로 잘 보정되어 있다는 보장은 없습니다. 입문 글에서는 “클래스별 상대 점수를 0~1로 정규화한 값이며 가장 큰 값을 예측 클래스로 선택한다” 정도로 표현하면 충분합니다.

### 4.6 “딥러닝의 모든 계산은 행렬 연산” 표현

**위치:** 초안 420행

신경망의 핵심 선형 계산은 행렬 곱이지만 활성화, 정규화, 합성곱, reduction 등 다른 텐서 연산도 사용합니다.

권장 표현:

```text
딥러닝의 많은 핵심 계산, 특히 Dense 층의 가중합은 행렬 연산으로 표현됩니다.
```

---

## 5. 유용한 추가 내용

### 5.1 MNIST 샘플과 라벨 시각화

전처리 전에 원본 shape과 픽셀 범위를 출력하고 샘플 몇 장을 표시하면 이미지가 숫자 배열이라는 설명을 직접 확인할 수 있습니다.

### 5.2 모델 파라미터 수 해석

`model.summary()` 출력에서 각 층의 파라미터 수가 `(입력 수 + bias 1개) × 출력 노드 수`로 계산된다는 예를 하나 들면 DAY2의 계층 구조와 자연스럽게 연결됩니다.

### 5.3 CIFAR-10의 알려진 라벨 노이즈

현재 Keras 공식 문서는 CIFAR-10에 소량의 잘못된 라벨이 알려져 있다고 안내합니다. 필수는 아니지만 “공개 데이터셋도 완벽하지 않다”는 데이터 품질 사례로 짧게 추가할 수 있습니다.

공식 확인: [Keras CIFAR-10 데이터셋](https://keras.io/api/datasets/cifar10/)

### 5.4 데이터셋 출처와 이용 조건

MNIST와 CIFAR를 실제 프로젝트에서 사용할 독자를 위해 출처·라이선스·인용 안내를 참고 자료에 한 줄씩 추가할 수 있습니다.

- [Keras MNIST 데이터셋](https://keras.io/api/datasets/mnist/)
- [CIFAR-10/CIFAR-100 공식 페이지](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 6. 줄이거나 제거할 내용

### 6.1 문서 맨 앞의 기획 메타 주석

1~75행의 HTML 주석은 작성 이력으로는 유용하지만 최종 게시물에는 필요하지 않습니다. 예정대로 최종본에서 제거해야 합니다.

### 6.2 핵심 정리의 반복

9장의 요약은 유용하지만 본문 문장을 상당 부분 다시 나열합니다. 최종 글이 길다고 판단되면 다음만 남겨도 됩니다.

- 데이터 분할의 역할
- MNIST 전처리 shape
- 출력층·라벨·손실함수 조합
- CIFAR-10과 MNIST의 차이
- Dense와 행렬 곱의 연결

### 6.3 행렬의 고급 성질은 더 늘리지 않아도 됨

PDF의 분할행렬과 가역행렬 성질을 모두 전개하면 글의 핵심인 데이터 처리 흐름이 흐려질 수 있습니다. 누락 항목 중 핵심 성질만 표로 추가하고, 증명과 반복 계산은 생략하는 현재 방향이 적절합니다.

### 6.4 “CIFAR-100 정정” 문단의 어조

내용은 정확하지만 “비유일 뿐”이라고 단정하기보다 “공식 클래스 목록에는 해당 개 품종이 없으므로, 세분화 개념을 설명하기 위한 예시로 이해하는 편이 안전하다”로 쓰면 출처 오류를 교정하면서도 어조가 부드럽습니다.

---

## 7. 바로 붙여 넣을 수 있는 수정 블록

### 블록 A. train/validation/test와 과적합 표 교체

```markdown
| 상태 | 훈련 성능 | 검증 성능 | 해석 |
|---|---|---|---|
| **언더피팅** | 낮음 | 낮음 | 모델이 패턴을 충분히 학습하지 못한 상태 |
| **적절한 학습** | 높음 | 높음 | 훈련에서 배운 패턴이 검증 데이터에도 비교적 잘 적용됨 |
| **오버피팅** | 매우 높음 | 상대적으로 낮거나 악화 | 훈련 데이터에 지나치게 맞춰져 새 데이터에 일반화되지 못함 |

언더피팅과 오버피팅은 **훈련·검증 성능의 차이와 변화 추세**로 진단합니다.
test 데이터는 이 진단이나 설정 선택에 사용하지 않고, 모든 선택이 끝난 뒤 최종 성능을 확인할 때만 사용합니다.
```

### 블록 B. Flatten과 One-Hot 설명 교체

```markdown
이번 실습의 MLP는 입력 shape을 `(784,)`로 설계했으므로 28×28 이미지를 784개 특징의 벡터로 펼칩니다.
Flatten은 픽셀값을 없애지는 않지만, 높이·너비의 2차원 이웃 구조를 입력 shape에 명시적으로 보존하지 않습니다.

One-Hot Encoding은 **선택한 손실함수가 One-Hot 라벨을 요구할 때** 사용합니다.
이 글의 예제는 `categorical_crossentropy`를 사용하므로 정수 5를 `[0,0,0,0,0,1,0,0,0,0]`으로 바꿉니다.
정수 라벨을 그대로 유지하려면 출력층은 동일하게 Softmax를 사용하고 손실을 `sparse_categorical_crossentropy`로 선택할 수 있습니다.
```

### 블록 C. MNIST 코드의 dtype·검증 선택·device 보정

```python
# 프레임워크: TensorFlow / Keras
# 설치 방법과 지원 Python/GPU 조건은 실행 시점의 TensorFlow 공식 문서를 확인하세요.
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical

tf.keras.utils.set_random_seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0
y_train = to_categorical(y_train, 10).astype("float32")
y_test = to_categorical(y_test, 10).astype("float32")

# validation_split은 배열의 마지막 부분을 떼므로 먼저 순서를 재현 가능하게 섞습니다.
rng = np.random.default_rng(42)
order = rng.permutation(len(X_train))
X_train = X_train[order]
y_train = y_train[order]

model = Sequential([
    Input(shape=(784,)),
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
)

# test는 모델과 설정 선택이 끝난 뒤 최종 평가에만 사용합니다.
test_loss, test_acc = model.evaluate(X_test, y_test)
print("test loss =", test_loss, "test acc =", test_acc)

print("TensorFlow가 인식한 GPU:", tf.config.list_physical_devices("GPU"))
```

> 일반적인 분류 데이터에서는 가능한 경우 별도의 stratified split으로 `X_val`, `y_val`을 만들고 `validation_data=(X_val, y_val)`을 사용하는 방법이 더 명확합니다.

### 블록 D. loss와 accuracy 학습 곡선

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()
```

```markdown
곡선은 매 epoch마다 완벽하게 한 방향으로 움직이지 않고 조금씩 흔들릴 수 있습니다.
훈련 loss는 계속 내려가는데 validation loss가 상승하거나, 훈련 accuracy만 높아지고 validation accuracy가 정체되면 오버피팅을 의심합니다.
```

### 블록 E. CIFAR-10 부분 코드 dtype 수정

```python
y_train = to_categorical(y_train, 10).astype("float32")
y_test  = to_categorical(y_test, 10).astype("float32")
```

### 블록 F. 행렬 단원 수식 전체 보강

아래 블록은 초안 7장의 행렬 설명에 그대로 붙여 넣을 수 있도록 PDF p.35~56의 핵심 수식을 입문 범위로 정리한 것입니다.

```markdown
#### 행렬과 shape

$m$개의 행과 $n$개의 열을 가진 행렬 $A$를 다음처럼 씁니다.

$$
A=[a_{ij}]\in\mathbb{R}^{m\times n}
$$

$a_{ij}$는 $i$번째 행, $j$번째 열의 성분입니다. 두 행렬은 shape이 같고 모든 대응 성분이 같을 때 같은 행렬입니다.

#### 덧셈·스칼라곱·영행렬

같은 shape의 행렬 $A=[a_{ij}]$, $B=[b_{ij}]$와 스칼라 $c$에 대해

$$
A+B=[a_{ij}+b_{ij}],\qquad cA=[ca_{ij}]
$$

모든 성분이 0인 영행렬을 $O$라고 하면 다음이 성립합니다.

$$
A+O=A,\qquad A+(-A)=O
$$

#### 행렬 곱

$$
A\in\mathbb{R}^{m\times n},\qquad
B\in\mathbb{R}^{n\times r}
$$

일 때만 $AB$를 계산할 수 있으며 결과 shape은 $(m,r)$입니다.

$$
C=AB\in\mathbb{R}^{m\times r},\qquad
c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}
$$

즉, 결과의 $c_{ij}$는 $A$의 $i$번째 행과 $B$의 $j$번째 열을 내적한 값입니다.

행렬 곱은 일반적으로 교환법칙이 성립하지 않습니다.

$$
AB\neq BA
$$

하지만 shape이 맞으면 결합법칙과 분배법칙은 성립합니다.

$$
A(BC)=(AB)C
$$

$$
A(B+C)=AB+AC
$$

크기가 맞는 단위행렬에 대해서는

$$
I_mA=A=AI_n
$$

입니다.

Dense 층의 가중합도 같은 shape 규칙을 따릅니다.

$$
X\in\mathbb{R}^{B\times 784},\qquad
W\in\mathbb{R}^{784\times 512},\qquad
b\in\mathbb{R}^{512}
$$

$$
Z=XW+b\in\mathbb{R}^{B\times 512}
$$

여기서 $B$는 batch size이며, $b$는 각 샘플에 broadcasting됩니다.

#### 내적과 전치

두 열벡터 $u,v\in\mathbb{R}^{n}$의 내적은

$$
u^\mathsf{T}v=\sum_{k=1}^{n}u_kv_k
$$

입니다. 전치는 행과 열을 바꾸며 다음 성질을 가집니다.

$$
(A^\mathsf{T})^\mathsf{T}=A
$$

$$
(A+B)^\mathsf{T}=A^\mathsf{T}+B^\mathsf{T}
$$

$$
(AB)^\mathsf{T}=B^\mathsf{T}A^\mathsf{T}
$$

마지막 식에서는 **곱의 순서가 반대로 바뀐다**는 점이 중요합니다.

#### 행렬 곱을 열과 외적으로 보는 방법

$B=[b_1\ b_2\ \cdots\ b_r]$처럼 열벡터로 나누면

$$
AB=[Ab_1\ Ab_2\ \cdots\ Ab_r]
$$

로 볼 수 있습니다.

또 $A=[a_1\ a_2\ \cdots\ a_n]$을 열벡터로, $B$의 각 행을 $b_k^\mathsf{T}$로 쓰면

$$
AB=\sum_{k=1}^{n}a_kb_k^\mathsf{T}
$$

처럼 외적들의 합으로 표현할 수 있습니다.

#### 분할행렬

shape이 서로 맞는 블록으로 행렬을 나누면 일반 행렬처럼 계산할 수 있습니다.

$$
X=
\begin{bmatrix}
A&B\\
C&D
\end{bmatrix},
\qquad
Y=
\begin{bmatrix}
E&F\\
G&H
\end{bmatrix}
$$

$$
X+Y=
\begin{bmatrix}
A+E&B+F\\
C+G&D+H
\end{bmatrix}
$$

$$
XY=
\begin{bmatrix}
AE+BG&AF+BH\\
CE+DG&CF+DH
\end{bmatrix}
$$

각 블록의 덧셈과 곱셈이 정의될 수 있도록 내부 shape이 맞아야 합니다.

#### 역행렬

정사각행렬 $A\in\mathbb{R}^{n\times n}$에 대해

$$
AA^{-1}=A^{-1}A=I_n
$$

을 만족하는 행렬 $A^{-1}$이 존재하면 $A$를 가역행렬이라고 합니다. 역행렬이 존재하면 그 역행렬은 유일합니다.

2×2 행렬에서는

$$
A=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix},
\qquad
\det(A)=ad-bc
$$

이고, $\det(A)\neq0$일 때

$$
A^{-1}
=
\frac{1}{ad-bc}
\begin{bmatrix}
d&-b\\
-c&a
\end{bmatrix}
$$

입니다. 가역행렬의 곱은 다음 성질을 가집니다.

$$
(AB)^{-1}=B^{-1}A^{-1}
$$

#### 선형연립방정식

선형연립방정식은

$$
Ax=b
$$

로 나타낼 수 있습니다. $A$가 가역이면

$$
x=A^{-1}b
$$

이지만, 실제 수치 계산에서는 역행렬을 직접 구하기보다 `np.linalg.solve(A, b)`를 권장합니다.

#### 영공간

$A\in\mathbb{R}^{m\times n}$에 대해 영공간은

$$
\operatorname{Null}(A)
=
\left\{
x\in\mathbb{R}^{n}\mid Ax=0
\right\}
\subseteq\mathbb{R}^{n}
$$

입니다.

예를 들어

$$
A=
\begin{bmatrix}
1&2\\
-2&-4
\end{bmatrix}
$$

이면 $x_1+2x_2=0$이므로

$$
\operatorname{Null}(A)
=
\left\{
\begin{bmatrix}
-2t\\
t
\end{bmatrix}
\;\middle|\;
t\in\mathbb{R}
\right\}
$$

입니다.
```

NumPy 거듭제곱 예제의 주석도 다음처럼 고쳐야 합니다.

```python
P = np.array([[6, 7], [0, 1]])
print(P @ P)     # [[36 49] [0 1]] = P²
```

### 블록 G. 버전에 민감한 설치 안내

```markdown
> ⚠️ **설치 안내는 실행 시점의 공식 문서를 확인하세요**
>
> CPU 실습의 기본 설치 명령은 `python -m pip install tensorflow`입니다.
> 지원 Python 버전과 GPU 설치 방법은 운영체제와 TensorFlow 버전에 따라 달라집니다.
> 특히 최신 TensorFlow의 NVIDIA GPU를 네이티브 Windows에서 바로 사용할 수 있다고 가정하면 안 되며, WSL2 등 공식 지원 구성을 확인해야 합니다.
>
> 공식 문서: https://www.tensorflow.org/install/pip
```

---

## 8. 우선순위 표

| 우선순위 | 항목 | 조치 |
|---|---|---|
| 높음 | `to_categorical()` 뒤 `y`를 `float32`라고 잘못 표기 | `.astype("float32")` 추가 |
| 높음 | GPU가 있으면 Keras가 자동 사용한다는 device 설명 | 호환 설치·인식 조건과 Windows 제한 명시 |
| 높음 | 언더·오버피팅 표에서 test 성능을 함께 사용 | validation만 사용하도록 수정 |
| 높음 | `validation_split`이 마지막 샘플을 선택하는 동작 누락 | 사전 shuffle 또는 명시적 validation 데이터 사용 |
| 높음 | MLP가 2D를 받을 수 없고 Flatten이 공간 정보를 없앤다는 단정 | 이번 모델의 입력 계약과 공간 구조 약화로 표현 |
| 높음 | One-Hot이 반드시 필요하다는 설명 | 선택한 loss에 따른 라벨 형식으로 수정 |
| 중간 | PDF p.13~14 Accuracy 정의와 곡선 코드 누락 | train/validation accuracy 그래프 추가 |
| 중간 | PDF p.35~43 행렬 핵심 성질 누락 | 영행렬·곱셈·전치 성질 표 추가 |
| 중간 | 행렬 곱·분할행렬·영공간의 수식과 shape 조건 부족 | 블록 F의 표시 수식 추가 |
| 중간 | `P @ P` 출력 주석이 `A²`로 표기됨 | `P²`로 변수명 수정 |
| 중간 | 데이터 다양성, batch size, 95% 정확도의 단정 | 조건부 표현으로 완화 |
| 중간 | PDF p.4 수집→분석→가공 흐름 누락 | 짧은 단계 설명 추가 |
| 중간 | CIFAR 샘플 시각화 부재 | 라벨과 함께 5~10장 표시 |
| 낮음 | 분할행렬·가역행렬 세부 성질 축약 | 핵심 예시 1개만 추가하거나 현재 수준 유지 |
| 낮음 | 데이터셋 라이선스·CIFAR 라벨 노이즈 | 참고 자료 또는 보충 상자에 추가 |
| 낮음 | 기획 메타 주석과 긴 요약 | 최종본에서 제거·축약 |

---

## 9. 최종 권고

**현재 초안은 수정 후 게시 가능한 수준**입니다.

PDF의 네 대단원과 주요 실습 흐름은 모두 들어 있으며, 이미지 전용인 p.35~56도 핵심 개념은 복원했습니다. 프레임워크는 TensorFlow/Keras로 일관되고 NumPy는 선형대수 실습으로 명확히 분리되어 있습니다. Softmax·Sigmoid, 라벨 형식, 손실함수의 조합도 큰 틀에서 정확하며, 고정 상수 `÷255`와 학습 통계 기반 전처리의 누수 차이도 잘 설명했습니다.

다만 최종본을 만들기 전에 우선순위 **높음 6건**을 먼저 반영해야 합니다. 그중 dtype, device, validation/test 역할은 코드 실행과 평가 신뢰성에 직접 관련되므로 반드시 수정해야 합니다. 이후 Accuracy 곡선과 행렬 핵심 성질을 보강하면 PDF 충실도와 입문자 이해도가 모두 충분한 수준이 됩니다.

### 이번 검토에 사용한 현재 공식 문서

- [TensorFlow pip 설치 및 플랫폼별 GPU 조건](https://www.tensorflow.org/install/pip)
- [Keras Model.fit API](https://keras.io/api/models/model_training_apis/)
- [Keras MNIST 데이터셋](https://keras.io/api/datasets/mnist/)
- [Keras CIFAR-10 데이터셋](https://keras.io/api/datasets/cifar10/)
- [TensorFlow `to_categorical` API](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical)
- [CIFAR-10/CIFAR-100 공식 페이지](https://www.cs.toronto.edu/~kriz/cifar.html)
