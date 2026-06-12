# DAY5 딥러닝 모델 훈련 및 평가 - Codex 통합 검토안

## 전체 평가

초안은 PDF 58쪽의 네 영역을 모두 반영했다.

1. 모델 훈련과 평가(p3~p8)
2. 기울기ㆍ미분ㆍ경사하강법(p9~p16)
3. TensorFlowㆍKeras 소개와 MNIST 코드(p17~p20)
4. 합성곱의 직관부터 CNN 구조와 배치 정규화까지(p21~p58)

특히 텍스트 추출량이 매우 적은 p17~p58은 모든 페이지의 이미지 내용을 직접 확인했다. 초안의 이동평균 값, 저역ㆍ고역통과 필터, 합성곱 출력 크기, 채널과 커널, 곱셈 횟수, Pooling, Padding, Stride, Bias, Batch Normalization 설명은 대체로 PDF의 교육 흐름을 충실히 재구성했다. PDF의 수식 이미지에 있는 MSE, Cross Entropy, 가중치 갱신, Accuracy, Precision, Recall, F1, MAE, RMSE, R²도 초안에 정상적으로 들어가 있다.

NumPy 코드의 출력값은 다시 실행해 확인했으며 모두 초안과 일치한다. Python 코드 블록 7개도 문법상 정상이다. 프레임워크를 NumPy 직관 실습과 TensorFlow/Keras 완성 예제로 구분한 점도 좋다.

다만 게시 전 반드시 수정해야 할 핵심 문제가 있다.

- 글에서 validation의 중요성을 강조하지만 실제 MNIST 코드에는 validation이 없다.
- PDF의 예전 MNIST 코드와 현재 TensorFlow 공식 quickstart의 logits 방식이 다르지만, 초안은 현재 공식 예제와 같은 코드처럼 설명한다.
- 손실이 작으면 좋은 모델, 기울기가 0이면 최적점이라는 설명이 일반적인 딥러닝 문제에서도 항상 성립하는 것처럼 쓰였다.
- CNN 출력 크기와 same padding 설명에 stride, 홀수 커널, dilation 조건이 빠져 있다.
- TensorFlow 설치와 Windows GPU 지원은 현재 공식 문서를 확인해야 하는 버전 민감 정보다.

따라서 **높은 우선순위 수정 후 게시 권장**이다. 원본 PDF와 초안은 수정하지 않았다.

### 검증 범위

- PDF: 전체 58쪽의 목차와 텍스트 계층 확인
- 이미지 중심 PDF: p17~p58 전 페이지와 p3~p16의 수식 이미지 직접 확인
- 초안: UTF-8, 제목, 표, 링크, LaTeX 수식, 코드 fence 확인
- 코드: Python 코드 블록 7개 문법 확인
- 실행: 경사하강법, 학습률 비교, 이동평균, LPF, HPF, 출력 크기, 곱셈 횟수 확인
- TensorFlow: 현재 환경에 설치되지 않아 MNIST 코드는 정적 검토만 수행
- 공식 문서 확인일: 2026-06-12

## 높은 우선순위의 오류와 수정사항

### 1. validation을 설명하지만 MNIST 코드에서는 사용하지 않음

- 초안 위치: 256~266행, 291~298행, 316~357행
- 관련 PDF: p6~p8

4절은 train/validation/test의 역할을 올바르게 설명하고, 과적합은 훈련 곡선과 검증 곡선을 함께 봐야 한다고 적었다. 그러나 유일한 완성형 코드인 MNIST 예제는 다음처럼 train과 test만 사용한다.

```python
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

이 코드는 PDF p20의 짧은 예제를 재현한 것으로는 맞지만, 글의 중심인 **모델 훈련 및 평가**를 실습하는 코드로는 부족하다. validation이 없으므로 과적합 여부를 확인할 수 없고, 학습 곡선과 검증 곡선을 비교한다는 앞 설명도 실행되지 않는다.

다음 사항을 반영해야 한다.

- `model.fit()`에 validation 데이터를 제공한다.
- 훈련 loss/accuracy와 validation loss/accuracy를 분리해 기록한다.
- test는 학습과 모델 선택이 끝난 뒤 한 번 평가한다.
- `evaluate()` 반환값을 변수로 받아 의미를 설명한다.

입문 예제에서는 `validation_split=0.1`을 사용할 수 있다. 보다 엄밀한 분류 실험에서는 먼저 stratified split을 만들어 `validation_data=(x_val, y_val)`로 넘기는 방식을 권장한다.

### 2. PDF의 MNIST 코드와 현재 공식 quickstart를 구분해야 함

- 초안 위치: 316~357행
- 관련 PDF: p20

초안의 `Dense(10, activation="softmax")`와 문자열 손실 `sparse_categorical_crossentropy` 조합은 서로 호환되며 실행 가능한 방식이다. 그러나 현재 TensorFlow 공식 beginner quickstart는 마지막 층에서 softmax를 제거한 **raw logits**와 다음 손실을 사용한다.

```python
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

현재 공식 문서는 softmax를 모델에 포함하는 방식보다 logits에서 수치적으로 안정적인 손실 계산을 하는 방식을 권장한다. 또한 현재 문서에서도 `Flatten(input_shape=(28, 28))`는 동작하지만, Keras는 Sequential 모델의 첫 층으로 `Input(shape=(28, 28))`을 사용하는 방식을 권장하며 기존 표현에 경고를 출력한다.

따라서 다음 중 하나를 명확히 선택해야 한다.

1. PDF 코드 보존: "PDF에 실린 당시 quickstart 코드이며 현재 공식 예제와는 차이가 있다"고 표시
2. 현재 방식으로 갱신: `Input`, logits, `from_logits=True`를 사용하고 PDF 코드에서 현대화한 보충 예제라고 표시

DAY4에서 raw logits와 손실 조합을 이미 배웠으므로 두 번째 방법이 시리즈 연결상 더 좋다.

공식 문서:

- TensorFlow beginner quickstart: <https://www.tensorflow.org/tutorials/quickstart/beginner>
- Keras Sequential: <https://keras.io/api/models/sequential/>

### 3. 손실이 작으면 항상 좋은 모델이라는 표현을 제한해야 함

- 초안 위치: 130~138행
- 관련 PDF: p4~p5

`손실이 작을수록 좋은 모델`이라는 문장은 **같은 데이터, 같은 손실함수, 같은 평가 조건**에서만 제한적으로 해석해야 한다. 훈련 손실이 낮더라도 validation/test 성능이 나쁘면 과적합 모델일 수 있다. 서로 다른 손실함수의 숫자를 직접 비교하는 것도 의미가 없다.

다음처럼 고쳐야 한다.

> 같은 데이터와 같은 손실함수를 기준으로 볼 때 손실이 작을수록 해당 목적에 더 잘 맞춘 것입니다. 다만 훈련 손실만 작다고 새로운 데이터에서도 좋은 모델은 아니므로 validation/test 지표를 함께 확인해야 합니다.

### 4. `기울기=0`을 최적점으로 단정하면 안 됨

- 초안 위치: 164~180행
- 관련 PDF: p11~p14

초안은 1차원 볼록 이차함수 예제에서는 올바르다. 하지만 실제 딥러닝의 비볼록 손실 표면에서는 기울기가 0인 지점이 전역 최솟값뿐 아니라 지역 최솟값, 극댓값, 안장점일 수 있다. 또한 기울기가 0에 가까워도 평평한 구간일 뿐 좋은 해라는 보장은 없다.

PDF의 직관을 유지하되 다음처럼 조건을 붙여야 한다.

> 아래 설명은 하나의 가중치를 가진 단순한 1차원 손실 곡선의 직관입니다. 실제 신경망에서는 기울기가 0인 지점이 항상 전역 최적점은 아니며, 경사 기반 최적화는 충분히 낮은 손실을 주는 파라미터를 반복적으로 찾습니다.

`기울기가 양수이면 왼쪽, 음수이면 오른쪽`도 1차원 설명임을 밝혀야 한다. 여러 파라미터에서는 gradient vector의 반대 방향으로 이동한다.

### 5. CNN 출력 크기 공식의 성립 조건이 빠짐

- 초안 위치: 443~465행, 535행, 560행
- 관련 PDF: p33~p34, p49~p53

초안의 함수는 dilation이 1일 때 올바르다.

```python
return (H - K + 2*P) // S + 1
```

그러나 표에는 다음 조건이 생략되어 있다.

- `P=(K-1)/2`로 크기를 유지하는 설명은 stride 1, dilation 1, 홀수 커널일 때 성립한다.
- stride 2라고 항상 정확히 `H/2`가 되는 것은 아니다. 커널, 패딩, 입력 크기의 조합에 따라 floor가 적용된다.
- 높이와 너비의 값이 다르면 각각 별도로 계산한다.

입문 범위에서는 dilation을 깊게 설명할 필요는 없지만, 현재 공식이 **dilation=1을 가정**한다고 적어야 한다.

또한 PDF p34의 `Warping`은 반대쪽 값을 순환시키는 동작을 가리키므로 일반적인 용어는 **Wrapping 또는 circular padding**이다. PDF 표기를 그대로 보존하려면 다음처럼 정정한다.

> PDF에는 `Warping`으로 적혀 있지만, 반대편 값을 순환해 채우는 일반적인 용어는 `Wrapping` 또는 `circular padding`입니다.

### 6. TensorFlow 설치 안내를 운영체제와 GPU 환경별로 구분해야 함

- 초안 위치: 320~323행

`pip install tensorflow`은 CPU 실습의 간단한 출발점으로는 가능하지만, 설치 가능 Python 버전과 GPU 방법은 계속 바뀐다. 현재 공식 설치 문서 기준으로 Windows Native의 공식 GPU 지원은 TensorFlow 2.10 이후 제공되지 않으며, 최신 GPU 설치는 WSL2 사용 안내가 중심이다.

2026-06-12 공식 설치 페이지는 TensorFlow 2.21 기준 Python 3.10~3.13 예시를 안내한다. 버전 숫자를 본문에 고정하기보다 공식 설치 페이지에서 현재 조합을 확인하도록 안내해야 한다.

공식 설치 문서: <https://www.tensorflow.org/install/pip>

## PDF 기준 누락 내용

### 반드시 보완할 누락

1. **p6~p8의 validation 역할을 실제 코드로 연결**
   - 개념 설명은 있으나 MNIST 예제에 validation이 없다.

2. **p8의 과적합ㆍ과소적합을 실제 곡선으로 판별하는 방법**
   - "훈련↓, 검증↑" 문장은 있으나 `history.history`를 확인하거나 그리는 코드가 없다.

3. **p20 MNIST 코드의 현재 API 차이**
   - PDF 당시 코드와 현재 공식 quickstart를 구분해야 한다. 이는 PDF 내용 누락이라기보다 버전 변화에 대한 필수 보정이다.

### 충분히 반영된 내용

| PDF 범위 | 초안 반영 상태 |
|---|---|
| p3~p6 순전파ㆍ손실ㆍ역전파ㆍEpochㆍBatchㆍOptimizer | 반영됨 |
| p7 분류 지표 수식 | 반영됨 |
| p8 회귀 지표ㆍ과적합ㆍ과소적합 | 개념 반영, 실습 연결 부족 |
| p9~p16 경사하강법 직관과 갱신식 | 반영됨 |
| p17~p19 TensorFlowㆍKerasㆍPyTorch 소개 | 반영됨, 현재 Keras 3 보정 필요 |
| p21~p25 이동평균ㆍ커널ㆍLPFㆍHPF | 반영됨 |
| p26~p34 픽셀ㆍRGBㆍ필터ㆍPadding | 반영됨, `Warping` 용어 정정 필요 |
| p35~p44 CNN 채널ㆍFeature Mapㆍ연산량 | 반영됨 |
| p45~p53 FCㆍ활성화ㆍPoolingㆍStrideㆍBias | 반영됨 |
| p54~p58 Batch Normalization과 요약 | 반영됨, 효과의 단정 완화 필요 |

### 의도적으로 줄여도 되는 PDF 세부 내용

- p29~p32의 포토샵 예시와 PrewittㆍSobel 커널 전체 숫자
- p43의 6중 반복문 그림
- p47의 SigmoidㆍTanh 그래프를 모두 재현하는 부분
- p54~p56의 Batch Normalization 유도 과정

초안은 위 세부 내용을 핵심 개념 위주로 요약했으며, 입문 글의 범위를 고려하면 적절하다.

## 더 자세히 설명할 내용

### 1. Keras 예제의 dtype과 장치

- 초안 위치: 320~355행

`mnist.load_data()`의 이미지 배열은 정수형이며, NumPy에서 `/ 255.0`을 수행하면 환경에 따라 `float64` 배열이 된다. TensorFlow가 모델 입력에서 변환할 수 있지만, 메모리와 dtype을 명확히 하기 위해 다음처럼 `float32`로 변환하는 편이 좋다.

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

라벨은 `(N,)` 모양의 정수 클래스 인덱스다. 입력, 출력, 라벨 정보를 다음처럼 완성한다.

| 항목 | shape | dtype/의미 |
|---|---|---|
| `x_train` | `(60000, 28, 28)` | `float32`, 0~1 |
| `y_train` | `(60000,)` | 정수 클래스 인덱스 0~9 |
| logits | `(batch, 10)` | `float32` |
| loss target | `(batch,)` | 정수 라벨 |

TensorFlow는 사용 가능한 장치를 자동 배치한다. CPU에서도 실행되며 GPU가 반드시 필요한 예제는 아니다. GPU 확인이 필요하면 `tf.config.list_physical_devices("GPU")`를 사용할 수 있다.

### 2. 정규화와 데이터 누수의 차이

MNIST의 `/ 255.0`은 미리 정해진 상수로 나누는 변환이라 test 데이터에서 통계를 학습하지 않으며 누수가 아니다. 반면 평균ㆍ표준편차, 최솟값ㆍ최댓값, vocabulary 등을 데이터에서 추정하는 전처리는 train에만 `fit`하고 validation/test에는 같은 변환을 적용해야 한다.

이 구분을 한 문장 넣으면 DAY3의 데이터 처리 원칙과 잘 연결된다.

### 3. 분류 지표의 multiclass 평균 방식

- 초안 위치: 268~281행

TPㆍFPㆍFNㆍTN 공식은 이진 분류의 직관이다. MNIST처럼 클래스가 10개면 클래스별 지표를 계산한 뒤 `macro`, `micro`, `weighted` 방식 등으로 평균한다. 불균형 데이터에서 F1 하나만 자동으로 선택하는 것이 아니라 다음을 고려해야 한다.

- 놓치면 큰 문제가 되는 양성 사례: Recall
- 거짓 경보 비용이 큰 문제: Precision
- 양성 클래스가 희소하고 순위ㆍ확률 품질이 중요: PR-AUC
- 클래스별 성능 차이: confusion matrix와 macro F1

MNIST는 비교적 균형 잡힌 다중 분류이므로 첫 입문 예제에서 accuracy를 쓰는 것은 적절하다.

### 4. R²의 범위와 해석

- 초안 위치: 283~289행, 551행

R²는 1이 최선이지만 반드시 0~1 사이에 있지는 않다. test 데이터에서는 음수가 될 수 있으며, 이는 단순히 정답 평균을 예측하는 기준보다도 못할 수 있다는 뜻이다. 또한 높은 R²가 인과관계나 모든 오차가 작다는 뜻은 아니다.

다음 문장으로 보완한다.

> R²는 1이 최선이며 test 데이터에서는 음수가 나올 수도 있습니다. 값의 단위가 있는 MAEㆍRMSE와 함께 해석해야 합니다.

### 5. 학습률 `0.001`은 보편적 정답이 아님

- 초안 위치: 246행

Adam의 흔한 초기값 예시로 `0.001`을 소개할 수 있지만, 모델ㆍ배치 크기ㆍ스케줄러ㆍ정규화에 따라 달라진다.

> Adam에서 `0.001`은 자주 쓰이는 출발점 중 하나지만, validation 곡선을 보며 조정해야 합니다.

### 6. 현재 Keras의 위치

- 초안 위치: 304~314행
- 관련 PDF: p17~p19

PDF의 Keras 설명은 TensorFlow를 쉽게 쓰는 고수준 API라는 당시 맥락이다. 현재 Keras 3은 TensorFlow뿐 아니라 JAXㆍPyTorch 백엔드도 지원하는 multi-backend API다. 이 글의 코드는 `tf.keras`를 쓰므로 **TensorFlow 백엔드의 Keras 예제**라고 명확히 하면 된다.

공식 Keras 3 문서: <https://keras.io/keras_3/>

### 7. CNN 코드가 아니라 MLP 기준선임을 명시

- 초안 위치: 316~355행

MNIST 입력을 `Flatten`한 뒤 Dense 층에 넣는 p20 예제는 이미지 데이터로 학습하지만 CNN은 아니다. 이후 6절에서 CNN을 설명하므로 다음 문장을 추가해야 혼동이 없다.

> 이 예제는 이미지를 1차원으로 펼쳐 Dense 층에 넣는 MLP 기준선이며, 아직 합성곱층을 사용한 CNN은 아닙니다.

### 8. 연산량 공식의 범위

- 초안 위치: 482~500행

`M x R x C x N x K x K`는 일반적으로 **샘플 하나에 대한 한 합성곱층의 곱셈 수**를 단순화한 값이다. 배치 크기, bias 덧셈, 활성화, 메모리 이동 등은 포함하지 않는다.

`GPU가 필요합니다`는 절대 표현이다. CPU에서도 계산할 수 있지만 GPUㆍTPU 같은 가속기가 실용적인 학습 시간을 만드는 데 유리하다고 수정한다.

### 9. Batch Normalization 효과의 표현

- 초안 위치: 520~525행
- 관련 PDF: p54~p57

공변량 이동은 Batch Normalization의 역사적 동기지만, 그것이 성능 향상의 유일하거나 확정된 원인처럼 설명하지 않는 편이 안전하다. Batch Normalization이 과적합을 줄이거나 큰 학습률을 항상 허용하는 것도 보장되지 않는다.

> PDF는 입력 분포 변화 완화와 학습 안정화를 중심으로 설명합니다. 실제 효과는 모델 구조, 배치 크기, 데이터에 따라 달라지며 Dropout을 항상 대체하지는 않습니다.

### 10. 합성곱과 cross-correlation

NumPy의 `np.convolve`는 수학적 convolution처럼 커널을 뒤집지만, 딥러닝 프레임워크의 `Conv2D`는 보통 커널을 뒤집지 않는 cross-correlation 형태로 계산한다. 학습 과정에서 필터 값 자체를 배우므로 실무에서는 관습적으로 convolution이라고 부른다.

현재 예제의 이동평균과 HPF 커널은 대칭이라 뒤집어도 결과가 같으므로 출력값에는 문제가 없다. 이는 유용한 보충이지만 입문 범위를 고려해 한 문장만 추가하면 충분하다.

## 유용한 추가 내용

### 1. 훈련ㆍ검증 곡선의 최소 해석

MNIST 예제에서 `history.history`로 다음을 확인하도록 한다.

- train loss와 validation loss가 함께 감소: 학습 진행
- train loss는 감소하지만 validation loss가 상승: 과적합 신호
- 둘 다 높고 거의 줄지 않음: 과소적합 또는 학습 설정 문제

### 2. EarlyStopping의 validation 기준

PDF에 Early Stopping이 나오므로 다음 정도의 짧은 예제가 유용하다.

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)
```

test loss를 monitor로 사용하지 않는다는 점을 함께 적는다.

### 3. 재현성 범위

완전한 재현성을 보장한다고 쓰지 않는 범위에서 다음을 추가할 수 있다.

```python
tf.keras.utils.set_random_seed(42)
```

하드웨어와 일부 연산에 따라 결과가 조금 달라질 수 있다고 설명한다.

### 4. 이미지 배열 순서

현재 MLP 예제는 `(N, 28, 28)`을 Flatten한다. 실제 Keras CNN에 `Conv2D`를 넣을 때는 채널 축을 추가한 `(N, 28, 28, 1)`의 NHWC 형태가 필요하다는 연결 설명이 유용하다.

## 줄이거나 제거할 내용

### 1. 게시 전 기획 주석 제거

- 초안 위치: 1~73행

페이지 지도와 실행 기록은 제작 과정에 유용하지만 최종 게시물에서는 제거한다.

### 2. "이것이 학습의 전부" 완화

- 초안 위치: 214행

실제 학습에는 데이터 샘플링, 정규화, 정규화 기법, 최적화 상태, 스케줄러 등이 있으므로 다음처럼 줄인다.

> 이것이 경사 기반 학습의 핵심 원리입니다.

### 3. 포토샵 피부 보정 단정

- 초안 위치: 441행

단순 blur가 LPF의 예인 것은 맞지만 실제 피부 보정은 경계 보존 필터나 다른 비선형 처리를 함께 쓸 수 있다. `바로 저역통과 필터의 결과` 대신 다음처럼 완화한다.

> 단순한 이미지 blur는 저역통과 필터의 대표적인 예입니다.

### 4. 최종 요약의 길이

8절은 본문 전체를 거의 다시 나열한다. CNN 세부 항목을 3~4개로 묶으면 반복을 줄일 수 있다. 다만 PDF 범위가 넓으므로 현재 길이도 게시를 막는 문제는 아니다.

### 5. 프레임워크 "3대" 표현

`3대 프레임워크`는 PDF의 분류임을 명시하고, 현재 생태계의 고정된 공식 순위처럼 표현하지 않는다.

## 바로 붙여 넣을 수 있는 수정 블록

### 블록 1. 손실과 기울기 설명 교체

```markdown
예측과 실제 정답의 차이를 수치로 표현한 것이 **손실 함수(Loss Function)** 입니다. 같은 데이터와 같은 손실함수를 기준으로 볼 때 손실이 작을수록 해당 학습 목표에 더 잘 맞춘 것입니다. 다만 **훈련 손실만 작다고 새로운 데이터에서도 좋은 모델은 아닙니다.** 과적합 여부를 확인하려면 validation loss와 실제 평가지표를 함께 봐야 합니다.

> 아래 경사하강법 설명은 하나의 가중치를 가진 단순한 1차원 손실 곡선의 직관입니다. 실제 신경망에서는 기울기가 0인 지점이 항상 전역 최적점은 아니며, 지역 최솟값ㆍ극댓값ㆍ안장점일 수도 있습니다. 여러 파라미터가 있을 때는 하나의 왼쪽ㆍ오른쪽이 아니라 **gradient vector의 반대 방향**으로 이동합니다.
```

### 블록 2. 현재 방식의 Keras MNIST 예제

```python
# TensorFlow 설치 방법은 운영체제와 CPU/GPU 환경에 따라 다릅니다.
# https://www.tensorflow.org/install/pip 에서 현재 명령을 확인하세요.
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

# 1) 데이터 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 고정된 255로 나누는 변환은 데이터 통계를 학습하지 않으므로 누수가 아닙니다.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 2) MLP 기준선: 마지막 층은 softmax 전 raw logits 10개를 출력
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

# 정수 라벨 + raw logits 조합
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=["accuracy"],
)

# train 일부를 validation으로 사용하고, test는 여기서 사용하지 않음
history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=32,
)

# 모든 학습ㆍ선택이 끝난 뒤 test를 최종 평가
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_accuracy:.4f}")
```

```markdown
이 코드는 이미지를 `Flatten`해 Dense 층에 넣는 **MLP 기준선**이며 아직 CNN은 아닙니다. 모델 출력은 `(batch, 10)` 모양의 raw logits이고, 라벨은 `(batch,)` 모양의 정수 클래스 인덱스입니다. 확률이 필요할 때만 `tf.nn.softmax(logits, axis=1)`를 적용합니다.

PDF p20의 코드는 출력층에 `softmax`를 넣고 `sparse_categorical_crossentropy`를 사용하는 당시 quickstart 방식입니다. 그 조합도 서로 호환되지만, 위 코드는 현재 TensorFlow 공식 quickstart의 logits 방식을 반영한 보충 예제입니다.
```

### 블록 3. 훈련ㆍ검증 곡선

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```

```markdown
train loss는 계속 감소하지만 validation loss가 다시 증가하기 시작하면 과적합 신호일 수 있습니다. 이때 epoch 수, 모델 크기, Dropout, 데이터 증강, Early Stopping 등을 validation 기준으로 조정합니다. test 결과를 보며 조정하면 test 정보가 모델 선택에 새어 들어갑니다.
```

### 블록 4. shapeㆍdtypeㆍ장치 표 교체

```markdown
| 항목 | shape | dtype/설명 |
|---|---|---|
| `x_train` | `(60000, 28, 28)` | `float32`, 픽셀 범위 0~1 |
| `y_train` | `(60000,)` | 정수 클래스 인덱스 0~9 |
| `Flatten` 후 | `(batch, 784)` | 이미지 공간 구조를 한 줄로 펼침 |
| 모델 출력 | `(batch, 10)` | softmax 전 raw logits |
| 손실 | `SparseCategoricalCrossentropy(from_logits=True)` | 정수 라벨과 logits 조합 |

TensorFlow는 사용 가능한 CPU/GPU에 연산을 자동 배치합니다. GPU가 없어도 이 MNIST 예제는 CPU로 실행할 수 있습니다.
```

### 블록 5. 평가 지표 설명 보완

```markdown
위 TPㆍFPㆍFNㆍTN 공식은 먼저 **이진 분류**를 기준으로 이해하면 쉽습니다. MNIST 같은 다중 분류에서는 클래스마다 지표를 계산한 뒤 macroㆍmicroㆍweighted 방식 등으로 평균할 수 있습니다.

- 놓치면 위험한 양성을 최대한 찾는 것이 중요하면 **Recall**
- 거짓 경보를 줄이는 것이 중요하면 **Precision**
- 두 기준의 균형이 필요하면 **F1**
- 희소한 양성 클래스의 확률 순위를 평가하려면 **PR-AUC**

MNIST는 비교적 균형 잡힌 다중 분류이므로 첫 예제에서는 accuracy를 기본 지표로 사용해도 적절합니다.

R²는 1이 최선이지만 test 데이터에서 음수가 나올 수도 있습니다. 따라서 실제 단위로 해석되는 MAEㆍRMSE와 함께 확인합니다.
```

### 블록 6. PaddingㆍStride 설명 교체

```markdown
아래 식은 **dilation=1**인 경우의 한 축 출력 크기입니다. 높이와 너비는 각각 따로 계산합니다.

$$
R = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1
$$

- `valid`: `P=0`
- stride 1, dilation 1, 홀수 커널에서 `P=(K-1)/2`이면 입력과 같은 크기를 유지
- stride 2는 출력 크기를 대략 절반으로 줄이지만, 정확한 값은 입력ㆍ커널ㆍ패딩 조합과 floor 계산에 따라 달라짐

> PDF에는 반대편 값을 순환해 채우는 방식을 `Warping`으로 적었지만, 일반적인 용어는 **Wrapping** 또는 **circular padding**입니다.
```

### 블록 7. TensorFlowㆍKeras 현재 상태 안내

```markdown
> **버전 주의**
>
> PDF는 Keras를 TensorFlow를 쉽게 사용하는 고수준 도구로 설명합니다. 현재 `tf.keras`는 TensorFlow에 통합된 Keras API이고, 별도의 Keras 3은 TensorFlowㆍJAXㆍPyTorch 백엔드를 지원합니다. 이 글의 코드는 **TensorFlow 백엔드의 `tf.keras`** 를 사용합니다.
>
> TensorFlow 설치 가능 Python 버전과 GPU 설치 방법은 바뀔 수 있습니다. 특히 Windows Native의 최신 공식 GPU 지원에는 제한이 있으므로 [TensorFlow 공식 설치 문서](https://www.tensorflow.org/install/pip)에서 현재 환경에 맞는 명령을 확인하세요.
```

### 블록 8. CNN 연산량과 Batch Normalization 표현

```markdown
PDF의 곱셈 횟수 식은 **샘플 하나가 한 합성곱층을 통과할 때** 필요한 곱셈을 단순화해 계산한 것입니다. 배치 크기, bias 덧셈, 활성화, 메모리 이동 비용은 포함하지 않습니다. CPU로도 계산할 수 있지만 연산량이 커질수록 GPUㆍTPU 같은 가속기가 실용적인 학습 시간에 유리합니다.

PDF는 Batch Normalization을 층의 입력 분포 변화를 완화하고 학습을 안정시키는 방법으로 설명합니다. 다만 큰 학습률을 항상 사용할 수 있게 하거나 과적합을 항상 줄이는 것은 아니며, Dropout을 반드시 대체하지도 않습니다. 효과는 모델 구조와 배치 크기에 따라 달라집니다.
```

## 우선순위 표

| 우선순위 | 항목 | 조치 |
|---|---|---|
| 높음 | validation 설명과 MNIST 코드 불일치 | validation 추가, history 기록, test 최종 1회 평가 |
| 높음 | PDF 코드와 현재 공식 quickstart 혼동 | PDF snapshot과 현재 logits 방식을 구분 |
| 높음 | 손실이 작으면 좋은 모델이라는 단정 | train loss와 일반화 성능을 분리해 설명 |
| 높음 | 기울기 0을 최적점으로 단정 | 1차원 볼록 예제의 직관임을 밝히고 안장점 등 언급 |
| 높음 | PaddingㆍStride 공식 조건 누락 | stride, 홀수 커널, dilation 조건과 floor 효과 명시 |
| 높음 | 단일 `pip install tensorflow` 안내 | 공식 설치 문서와 Windows GPU 제한 안내 |
| 중간 | MNIST dtypeㆍdevice 설명 부족 | `float32`, 정수 라벨, TensorFlow 자동 장치 배치 명시 |
| 중간 | 두 번째 경사하강법 코드가 앞 블록의 `f`, `grad`에 의존 | 순차 실행 표시 또는 함수 재정의 |
| 중간 | multiclass PrecisionㆍRecallㆍF1 평균 방식 누락 | macro/micro/weighted 개념 짧게 추가 |
| 중간 | R²를 단순히 1에 가까울수록 좋다고만 설명 | 음수 가능성과 MAEㆍRMSE 병행 해석 추가 |
| 중간 | Keras를 TensorFlow 전용처럼 설명 | Keras 3 multi-backend와 `tf.keras` 구분 |
| 중간 | MNIST 예제가 CNN처럼 오해될 수 있음 | Flatten+Dense MLP 기준선임을 명시 |
| 중간 | `Warping` 용어 | Wrapping/circular padding으로 정정 |
| 중간 | 합성곱 연산량 때문에 GPU가 필요하다는 단정 | CPU 가능, 가속기가 실용적이라고 완화 |
| 중간 | Batch Normalization 효과 단정 | PDF 주장과 실무 caveat를 분리 |
| 낮음 | convolution과 cross-correlation 차이 | 한 문장 보충 |
| 낮음 | 포토샵 피부 보정이 바로 LPF라는 표현 | 단순 blur 예시로 범위 제한 |
| 낮음 | 게시용이 아닌 기획 주석과 긴 최종 요약 | 최종본에서 제거ㆍ축소 |

## 최종 권고

초안은 이미지 기반 p17~p58을 포함한 PDF 전체 범위를 매우 충실하게 반영했고, NumPy 수치 예시와 CNN 수식도 대체로 정확하다. 그러나 글의 핵심 주제인 **훈련과 평가**에서 validation이 코드에 없고, PDF 당시 Keras 코드와 현재 공식 API가 구분되지 않은 점은 반드시 수정해야 한다.

우선 다음 여섯 항목을 반영한다.

1. MNIST 예제에 validation과 `history` 추가
2. test는 최종 1회 평가로 분리
3. 현재 권장 `Input + logits + from_logits=True` 조합 또는 PDF snapshot 표시
4. 손실ㆍ기울기 설명의 단정 완화
5. PaddingㆍStride 공식 조건과 Wrapping 용어 수정
6. TensorFlow 설치를 공식 문서로 안내

이후 dtypeㆍdevice, R², multiclass 지표, Batch Normalization caveat를 보완하면 입문자가 이론과 실행을 안전하게 연결할 수 있는 DAY5 글이 된다.

## 확인한 공식 문서

- TensorFlow beginner quickstart: <https://www.tensorflow.org/tutorials/quickstart/beginner>
- TensorFlow pip 설치: <https://www.tensorflow.org/install/pip>
- Keras 3 소개: <https://keras.io/keras_3/>
- Keras Sequential API: <https://keras.io/api/models/sequential/>
