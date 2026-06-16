# DAY7 딥러닝 다중퍼셉트론 - Codex 통합 검토안

## 전체 평가

초안은 PDF 72쪽의 큰 흐름인 **MLP → Optimizer → RNN/LSTM 항공 여행자 수 예측 → GAN 손글씨 숫자 생성**을 모두 포함하고 있습니다. 특히 PDF에 실행 코드가 거의 없다는 점을 밝히고 Keras 기반 재구성 코드로 보충한 방향은 적절합니다. MLP의 XOR, 활성화 함수, 옵티마이저 비교, LSTM 입력 shape `(samples, timesteps, features)`, GAN의 생성자/감별자 구조도도 입문자에게 대체로 이해하기 좋습니다.

다만 게시 전 반드시 고쳐야 할 기술적 위험이 있습니다. 가장 큰 문제는 **LSTM 시계열 예제에서 `MinMaxScaler`를 전체 시계열에 먼저 `fit_transform`한 뒤 train/test를 나누는 데이터 누수**입니다. 초안에 주의 문구가 있지만, 본문 코드는 여전히 누수 있는 절차를 보여줍니다. 또한 Keras `model.fit()`의 `shuffle=True` 기본값, LSTM 회귀 출력층을 `sigmoid`로 고정했을 때의 예측 범위 제한, GAN에서 `trainable` 변경과 `compile()`의 관계를 명확히 보완해야 합니다.

검증 범위는 다음과 같습니다.

- PDF 전체 72쪽 텍스트 추출 및 렌더링 확인
- 텍스트 추출이 적은 이미지 중심 페이지 p6, p13, p16, p20, p23, p27~p52, p54~p72 직접 확인
- 초안 Markdown 구조와 Python 코드 블록 5개 문법 검사
- NumPy/scikit-learn LSTM 전처리 예제 실행 확인: `X_train (105, 12, 1)`, `X_test (27, 12, 1)`
- 현재 로컬 환경 확인: `tensorflow`/`keras` 미설치, `numpy`/`sklearn`/`pandas`/`seaborn`/`matplotlib` 설치됨
- 공식 문서 확인 필요 항목: Keras `fit(shuffle=True)`, Keras `trainable`/`compile()`, Keras `train_on_batch`, Keras `LeakyReLU(negative_slope=...)`, Keras MNIST, scikit-learn `MinMaxScaler`, TensorFlow 설치

## 높은 우선순위의 오류와 수정사항

### 1. LSTM 예제의 데이터 누수: 전체 시계열에 `fit_transform` 후 분할

- 초안 위치: 4.3, 4.4
- PDF 관련: p42~p47
- 문제: 현재 코드는 `scaler.fit_transform(series)`를 전체 144개월 데이터에 적용한 뒤 윈도우를 만들고 train/test를 나눕니다. 이러면 test 기간의 최솟값/최댓값 정보가 학습 전처리에 반영됩니다.
- 영향: 초보자는 "시계열은 순서대로 분할하면 충분하다"고 오해할 수 있습니다. 실제로는 스케일러도 학습 구간에만 `fit`해야 합니다.
- 수정: PDF의 흐름을 설명할 때는 "강의 단순화 버전"으로 남기되, 실행 코드는 train 구간에만 `fit`하는 버전으로 바꾸는 것을 권장합니다.

### 2. LSTM `model.fit()`에서 `shuffle=False`가 빠짐

- 초안 위치: 4.4
- 공식 문서 확인: Keras `Model.fit`의 기본값은 `shuffle=True`이며, NumPy 배열 입력은 매 epoch마다 전역 셔플됩니다.
- 문제: 본문에는 "셔플하지 않고 시간 순서대로 분할"이라고 되어 있지만, 실제 학습 호출은 `shuffle=False`를 지정하지 않습니다.
- 수정: `model.fit(..., shuffle=False)`를 명시하세요. 시계열 윈도우를 독립 샘플로 볼 수 있더라도, 입문 글에서는 설명과 코드가 일치해야 합니다.

### 3. LSTM 출력층 `sigmoid`와 train-only 스케일링의 충돌 가능성

- 초안 위치: 4.4, 7장 요약
- PDF 관련: p47은 MinMax `[0,1]` 입력에 맞춰 출력층 `sigmoid`, 손실 `MSE`를 사용합니다.
- 문제: train 구간에만 `MinMaxScaler`를 fit하면 미래/test 값이 train 최대값보다 커질 때 scaled target이 1보다 커질 수 있습니다. 이때 `Dense(1, activation="sigmoid")`는 예측을 0~1로 제한하므로 상승 추세를 과소예측할 수 있습니다.
- 수정 방향:
  - PDF 재현 설명: `sigmoid + MSE`는 PDF 설계라고 명시
  - 실전 보충 코드: 회귀 예측은 `Dense(1)` 선형 출력 + `MSE` 또는 `MAE`를 권장
  - 평가: 원래 단위로 되돌린 뒤 `MAE`, `RMSE`를 함께 제시

### 4. 검증/테스트 역할이 부족함

- 초안 위치: 4.4
- 문제: LSTM은 train/test만 있고 validation이 없습니다. `epochs=100`처럼 epoch를 정하면 검증 손실 없이 과적합 여부를 확인하기 어렵습니다.
- 수정: 시간 순서를 지켜 train/validation/test를 나누거나, 최소한 "이 예제는 설명용이라 validation을 생략했다"고 명시하세요. 게시용 예제라면 chronological train/val/test 분할을 넣는 편이 좋습니다.

### 5. GAN의 `trainable` 변경과 `compile()` 관계 설명이 부정확해질 수 있음

- 초안 위치: 5.4
- PDF 관련: p69
- 공식 문서 확인: Keras는 trainable 변수 구성이 `compile()` 시점에 결정됩니다. `compile()` 후 `layer.trainable`을 바꾸면 다시 `compile()`해야 변경이 반영됩니다.
- 현재 코드 흐름은 흔한 GAN 예제 패턴에 가깝지만, 루프 안에서 `discriminator.trainable = True/False`를 바꾸는 부분은 초보자에게 "그때그때 즉시 반영된다"고 오해를 줄 수 있습니다.
- 수정: 감별자는 `trainable=True` 상태에서 먼저 compile하고, 결합 GAN은 감별자를 `trainable=False`로 둔 뒤 compile한다는 순서를 본문에 분명히 설명하세요. 루프 안의 토글은 제거하거나 "이미 compile된 두 모델의 학습 그래프가 다르다"는 주석을 달아야 합니다.

### 6. "LSTM은 기울기 소실에서 자유롭다"는 표현은 과장

- 초안 위치: 4.2
- 문제: LSTM은 장기 의존성 학습을 돕고 기울기 소실을 완화하지만, 완전히 자유롭다고 말하면 과장입니다.
- 수정: "기울기 소실을 완화한다", "기본 RNN보다 장기 정보를 더 잘 보존하도록 설계됐다"로 바꾸세요.

### 7. 게시 전 HTML 계획 주석 제거 필요

- 초안 위치: 문서 맨 앞 계획 주석
- 문제: 초안 상단에 PDF 페이지맵과 작성 계획이 HTML 주석으로 들어 있습니다. Velog 게시 전에는 제거해야 합니다.
- 수정: 최종 포스트에는 제목부터 시작하고, 계획 주석은 남기지 않는 것이 좋습니다.

## PDF 기준 누락 내용

### MLP/활성화/학습 흐름

- p7의 출력층 구분이 더 선명해질 수 있습니다. 이진 분류는 노드 1개+sigmoid, 다중 클래스는 클래스 수만큼 출력+softmax라는 대응은 있지만, 라벨 형식까지는 부족합니다.
- p9의 Softmax 수식과 "출력 합이 1" 설명은 포함되어 있으나, `sparse_categorical_crossentropy`는 정수 라벨, `categorical_crossentropy`는 원-핫 라벨이라는 실전 연결을 추가하면 좋습니다.
- p10의 MSE/Cross Entropy 수식은 잘 반영됐지만, 회귀/분류 문제별 손실 선택 기준을 한 문장 더 넣으면 입문자가 덜 헷갈립니다.

### Optimizer

- p15~p18의 learning rate가 너무 작거나 클 때의 그래프적 직관은 초안에 반영되어 있으나, "학습률이 너무 크면 발산할 수 있다"를 조금 더 명시하면 좋습니다.
- p21의 AdaGrad/RMSProp 수식, p50의 RMSProp 설명은 표 수준으로 축약되어 있습니다. RNN/LSTM 예제에서 RMSProp을 쓰는 이유를 "최근 기울기 크기에 맞춰 학습률을 조절한다" 정도로 짧게 보충하세요.
- p24의 AdamW/Lion은 시점에 민감한 최신성 표현입니다. PDF 내용으로 소개하되 "현재도 항상 표준"처럼 단정하지 말고, "대규모 모델에서 널리 쓰인다", "논문/프레임워크 지원 여부는 확인 필요"로 낮추세요.

### RNN/LSTM

- p26~p29의 RNN 활용 예시가 빠져 있습니다. 이미지 캡셔닝, 문장 검색, one-to-many/many-to-one/many-to-many 유형을 한 단락으로 요약하면 PDF 흐름이 더 잘 살아납니다.
- p35~p36의 LSTM 게이트 설명 중 PDF는 "cell state로 다음 단에 전해질 양"과 "현재 단계의 출력"을 나누어 보여줍니다. 초안의 3개 게이트 요약은 입문자에게는 괜찮지만, PDF의 그림을 반영하려면 "강의 그림은 게이트 동작을 단계별로 나눠 보여준다"고 덧붙이세요.
- p37~p41의 항공 여행자 데이터 배경과 그래프 해석이 축약되어 있습니다. "1949~1960년 월별 승객 수, 144개 시점, 추세와 계절성이 보이는 작은 시계열"이라고 명시하세요.
- p43의 정규화 종류(Standard/MinMax/Robust/Normalizer) 비교가 거의 빠져 있습니다. 실습에 쓰는 것은 MinMax지만, 다른 정규화가 있다는 PDF 표를 2~3줄로 보완하세요.
- p48의 `flatten()` 필요성은 반영되어 있으나, `(n, 1)`을 `(n,)`으로 바꾸는 이유를 "seaborn lineplot이 1차원 시리즈를 기대하기 때문"이라고 조금 더 풀면 좋습니다.

### GAN

- p54~p57의 GAN 활용 예시가 누락되었습니다. 스케치→제품 이미지, 도메인 변환, 얼굴/나이 변환, video-to-video synthesis를 간단히 열거하세요.
- p60의 핵심 훈련 원리, 즉 "생성자를 학습할 때는 감별자를 고정하고, 감별자를 학습할 때는 생성자 쪽으로 업데이트가 흘러가지 않게 한다"는 내용은 있지만, Keras 코드의 `compile()` 순서와 연결이 약합니다.
- p61~p62의 원래 GAN 알고리즘과 minimax 손실은 입문 글에서 전체 수식을 깊게 다룰 필요는 없습니다. 그래도 `D(x)`는 진짜 이미지 확률, `D(G(z))`는 생성 이미지가 진짜로 판별될 확률이라는 해석은 추가하면 좋습니다.
- p70의 `fit` vs `train_on_batch` 비교표는 설명으로만 압축되어 있습니다. GAN은 두 모델을 번갈아 한 배치씩 업데이트하기 때문에 `train_on_batch`가 쓰인다는 점을 표 또는 짧은 비교로 복원하세요.

## 더 자세히 설명할 내용

- **train/validation/test**: 시계열은 무작위 분할이 아니라 시간순 분할을 사용하고, 스케일러도 train 구간에만 `fit`해야 한다는 점을 코드와 설명 모두에서 맞추세요.
- **LSTM tensor shape**: `(샘플 수, 12개월, 특성 1개)`는 잘 설명되어 있습니다. 여기에 `dtype`은 보통 `float32`가 안전하다는 점을 추가하면 TensorFlow/Keras 예제와 더 잘 맞습니다.
- **device/GPU**: 초안은 TensorFlow/Keras 기준이므로 PyTorch처럼 `.to(device)`는 필요 없습니다. 대신 GPU 사용은 TensorFlow 설치/런타임 설정에 의존한다고 짧게 안내하세요.
- **출력층·활성화·손실함수 조합**:
  - XOR 이진 분류: `Dense(1, sigmoid)` + 0/1 라벨 + `binary_crossentropy`는 적절
  - 다중 클래스: `Dense(num_classes, softmax)` + 정수 라벨이면 `sparse_categorical_crossentropy`, 원-핫이면 `categorical_crossentropy`
  - 시계열 회귀: PDF 재현은 `sigmoid + mse`, 실전 보충은 `linear + mse/mae` 권장
  - GAN 감별자: `Dense(1, sigmoid)` + 진짜/가짜 1/0 라벨 + `binary_crossentropy`는 적절
- **평가 지표**:
  - XOR: toy 예제라 accuracy만으로 충분하나 일반화 성능을 말하면 안 됨
  - LSTM 회귀: loss 외에 원 단위 MAE/RMSE 필요
  - GAN: 초안처럼 생성 이미지 시각 확인은 입문 단계에 적절. FID 같은 지표는 너무 깊으므로 선택 보충 정도만 가능
- **실행 순서**: 4.3의 전처리 블록과 4.4의 모델 블록이 이어지는 구조입니다. `series`, `scaler`, `X_train`, `X_test`가 앞 블록에서 정의된다는 점을 문장으로 명확히 하세요.

## 유용한 추가 내용

- "PDF 재현 코드"와 "실전에서 더 안전한 코드"를 나누면 좋습니다. 특히 LSTM은 PDF 설계를 존중하면서도 데이터 누수 없는 절차를 보여주는 것이 중요합니다.
- 설치 안내는 짧게만 넣되 TensorFlow는 Windows/GPU 조건이 민감하므로 공식 문서 확인을 권장하세요. 현재 TensorFlow 공식 설치 문서는 Windows native GPU 지원이 TensorFlow 2.10 이후 제한된다고 안내합니다.
- Keras 3 문서 기준 `LeakyReLU` 인자는 `negative_slope`입니다. `tf.keras.layers.LeakyReLU(0.01)`보다 `tf.keras.layers.LeakyReLU(negative_slope=0.01)`가 초보자에게 명확합니다.
- MNIST `load_data()`는 공식적으로 `(60000, 28, 28)` train과 `(10000, 28, 28)` test를 반환하며 dtype은 `uint8`, 픽셀 범위는 0~255입니다. 초안의 reshape/정규화 설명과 잘 맞습니다.
- 재현성은 선택 사항입니다. GAN 예제에 `tf.keras.utils.set_random_seed(42)` 같은 줄을 넣으면 결과가 매번 달라지는 이유를 줄일 수 있지만, GPU에서는 완전 동일 결과가 보장되지 않을 수 있습니다.

## 줄이거나 제거할 내용

- 상단 HTML 계획 주석은 최종 게시물에서 제거하세요.
- "AdamW는 사실상 표준", "Lion은 최신 optimizer" 같은 표현은 시점에 민감합니다. PDF 내용으로 소개하되 단정 강도를 낮추세요.
- "LSTM은 기울기 소실에서 자유롭다"는 표현은 제거하거나 완화하세요.
- GAN 수식/알고리즘을 너무 깊게 확장할 필요는 없습니다. 입문 글에서는 `D(x)`, `D(G(z))`, 진짜/가짜 라벨, 번갈아 학습 정도면 충분합니다.
- RNN/LSTM의 게이트 수식을 모두 전개하면 DAY7의 범위가 과도하게 깊어질 수 있습니다. 대신 PDF 그림의 단계와 직관 중심으로 보완하세요.

## 바로 붙여 넣을 수 있는 수정 블록

### 1. LSTM 절 설명 교체 블록

```markdown
> 🟩 **실습 주의: 시계열 전처리의 데이터 누수**
>
> 강의 PDF는 MinMax 정규화와 12개월 윈도우 만들기를 직관적으로 보여주는 데 초점을 둡니다. 실제 예측 실험에서는 test 기간의 최솟값/최댓값을 미리 알면 안 되므로, `MinMaxScaler`는 반드시 **학습 구간에만 `fit`** 하고 validation/test 구간에는 `transform`만 적용합니다.
>
> 또한 Keras `model.fit()`은 NumPy 배열 입력에서 기본값이 `shuffle=True`입니다. 시계열 예제에서는 설명과 코드가 일치하도록 `shuffle=False`를 명시합니다.
```

### 2. 누수 없는 LSTM 전처리 + 학습 예시

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 예: pandas로 읽은 월별 승객 수라고 가정
# series = pd.read_csv("AirPassengers.csv")["#Passengers"].values.astype("float32")
series = np.arange(1, 145, dtype="float32")

W = 12
raw = series.reshape(-1, 1)

# 시간 순서 유지: train / validation / test
n = len(raw)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_raw = raw[:train_end]
val_raw = raw[train_end - W:val_end]   # 첫 validation 윈도우가 과거 W개월을 볼 수 있게 겹침 허용
test_raw = raw[val_end - W:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
val_scaled = scaler.transform(val_raw)
test_scaled = scaler.transform(test_raw)

def make_windows(values, window=12):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window, 0])
        y.append(values[i + window, 0])
    X = np.array(X, dtype="float32").reshape(-1, window, 1)
    y = np.array(y, dtype="float32").reshape(-1, 1)
    return X, y

X_train, y_train = make_windows(train_scaled, W)
X_val, y_val = make_windows(val_scaled, W)
X_test, y_test = make_windows(test_scaled, W)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(W, 1)),
    tf.keras.layers.LSTM(300),
    # PDF는 MinMax[0,1]에 맞춰 sigmoid를 사용하지만,
    # train-only scaler에서는 미래 값이 1을 넘을 수 있어 회귀 출력은 선형이 더 안전합니다.
    tf.keras.layers.Dense(1),
])

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=1,
    shuffle=False,
    verbose=0,
)

pred_scaled = model.predict(X_test, verbose=0)
pred = scaler.inverse_transform(pred_scaled)
truth = scaler.inverse_transform(y_test)

mae = mean_absolute_error(truth, pred)
rmse = mean_squared_error(truth, pred) ** 0.5
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
```

### 3. PDF 설계와 실전 보충을 구분하는 문장

```markdown
PDF 설계도는 `MinMax[0,1] → LSTM(300) → Dense(1, sigmoid) → MSE` 흐름을 보여줍니다. 이는 강의용으로 정규화된 출력 범위를 직관적으로 맞춘 예입니다. 다만 실제 시계열 예측에서는 스케일러를 학습 구간에만 맞추면 미래 값이 1보다 커질 수 있으므로, 회귀 문제에서는 `Dense(1)`처럼 선형 출력층을 쓰는 편이 더 안전합니다.
```

### 4. GAN `trainable`/`compile` 설명 보강 블록

```markdown
> 🟩 **GAN에서 `trainable`과 `compile()` 순서**
>
> Keras에서는 어떤 가중치를 학습할지(`trainable`)가 `compile()` 시점에 학습 함수로 고정됩니다. 그래서 GAN 예제는 보통 다음 순서를 지킵니다.
>
> 1. 감별자 단독 모델은 `trainable=True` 상태에서 `compile()`한다.
> 2. 결합 GAN을 만들기 전 감별자를 `trainable=False`로 바꾼다.
> 3. 결합 GAN을 `compile()`한다. 이 모델로 학습할 때는 생성자만 업데이트된다.
>
> 루프 안에서 감별자 단독 모델을 `train_on_batch`로 학습하고, 결합 GAN을 `train_on_batch`로 학습하는 것은 서로 다른 compile 결과를 사용하는 것입니다.
```

### 5. GAN 코드의 LeakyReLU와 주석 수정

```python
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100,)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(negative_slope=0.01),
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(negative_slope=0.01),
    tf.keras.layers.Dense(784, activation="sigmoid"),
    tf.keras.layers.Reshape((28, 28, 1)),
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(negative_slope=0.01),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

### 6. 과장 표현 수정 블록

```markdown
기본 RNN은 시점이 멀어질수록 기울기 소실(vanishing gradient) 때문에 오래전 정보를 학습하기 어렵습니다. LSTM은 셀 상태(cell state)와 게이트(gate)를 사용해 중요한 정보를 더 오래 보존하도록 설계되어, 기본 RNN보다 장기 의존성 문제를 완화합니다.
```

### 7. PDF 누락 보완용 짧은 블록

```markdown
PDF는 RNN의 활용 예시도 함께 보여줍니다. 하나의 이미지에서 여러 단어를 만드는 이미지 캡셔닝은 one-to-many, 문장을 보고 하나의 판단을 내리는 감성 분석이나 검색은 many-to-one, 문장을 문장으로 바꾸는 번역은 many-to-many 구조로 볼 수 있습니다. 이번 실습의 항공 여행자 수 예측은 과거 12개월을 보고 다음 1개월 값을 예측하므로 many-to-one에 가깝습니다.

GAN의 활용 예로는 스케치를 제품 이미지로 바꾸기, 사진과 그림의 도메인 변환, 얼굴 속성 변환, video-to-video synthesis 등이 소개됩니다. 이 글의 MNIST 예제는 그중 가장 단순한 형태로, 100차원 잡음에서 28×28 흑백 숫자 이미지를 만들어 보는 실습입니다.
```

### 8. 설치/공식 문서 확인 안내 블록

```markdown
> 🟨 **설치 참고**
>
> 이 글의 실행 코드는 TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib, seaborn을 사용합니다. TensorFlow는 운영체제와 GPU 사용 여부에 따라 설치 방법이 달라지므로 게시 시점의 공식 설치 문서를 확인하는 것이 안전합니다. 특히 Windows native GPU 지원은 TensorFlow 버전에 따라 제한이 있으므로, GPU 학습을 기대한다면 WSL2 또는 Colab 같은 환경도 함께 고려하세요.
```

## 우선순위 표

| 우선순위 | 항목 | 위치 | 조치 |
|---|---|---|---|
| 높음 | LSTM `MinMaxScaler` 전체 데이터 `fit_transform`으로 인한 데이터 누수 | 4.3, 4.4 | train 구간에만 `fit`, val/test는 `transform` |
| 높음 | Keras `fit()` 기본 셔플과 시계열 설명 불일치 | 4.4 | `shuffle=False` 명시 |
| 높음 | LSTM 회귀 출력 `sigmoid`의 예측 범위 제한 | 4.4, 7장 | PDF 재현과 실전 보충 분리, 선형 출력 권장 |
| 높음 | validation/test 역할 부족, 회귀 평가 지표 부족 | 4.4 | 시간순 validation 추가, MAE/RMSE 출력 |
| 높음 | GAN `trainable` 변경과 `compile()` 관계 설명 부족 | 5.4 | compile 순서와 두 학습 모델의 차이 설명 |
| 높음 | LSTM이 기울기 소실에서 자유롭다는 과장 | 4.2 | "완화한다"로 수정 |
| 중간 | PDF p26~p29 RNN 유형/활용 예시 누락 | 4.1 | one-to-many/many-to-one/many-to-many 간단 보완 |
| 중간 | PDF p43 정규화 종류 비교 누락 | 4.3 | Standard/MinMax/Robust/Normalizer 한 줄 비교 |
| 중간 | PDF p54~p57 GAN 활용 예시 누락 | 5.1 | 활용 사례 1단락 추가 |
| 중간 | AdamW/Lion 최신성 단정 | 3.3, 7장 | 시점 민감 표현 완화 |
| 중간 | `LeakyReLU(0.01)` 인자명 생략 | 5.4 | `negative_slope=0.01`로 수정 |
| 중간 | TensorFlow 설치/버전 민감성 안내 부족 | 참고/실습 전 | 공식 문서 확인 필요 표시 |
| 낮음 | 상단 HTML 계획 주석 | 문서 맨 앞 | 최종 게시 전 제거 |
| 낮음 | GAN 지표 확장 | 5장 | FID 등은 선택 보충, 본문 필수 아님 |

## 최종 권고

현재 초안은 PDF의 핵심 주제를 넓게 반영했지만, **그대로 게시하기에는 LSTM 시계열 전처리와 GAN 학습 코드 설명의 기술적 위험이 큽니다.** 특히 데이터 누수는 입문자에게 잘못된 습관을 만들 수 있으므로 반드시 수정해야 합니다.

권고는 **높은 우선순위 수정 후 게시**입니다. 수정 순서는 다음이 좋습니다.

1. LSTM 코드를 train-only scaler, `shuffle=False`, validation, MAE/RMSE 포함 버전으로 교체
2. PDF의 `sigmoid + MSE` 설계와 실전 보충의 `linear + MSE/MAE` 차이를 명시
3. GAN `trainable`/`compile()` 설명 보강
4. LSTM 기울기 소실 과장 표현 완화
5. PDF의 RNN 활용 예시, 정규화 종류, GAN 활용 예시를 짧게 보완
6. 상단 계획 주석 제거 및 TensorFlow 설치/공식 문서 확인 안내 추가

원본 PDF와 초안은 수정하지 않았습니다.

## 공식 문서 확인 메모

- Keras Model training APIs: `fit(shuffle=True)`, `train_on_batch`, trainable variables and `compile()` timing  
  https://keras.io/api/models/model_training_apis/
- Keras FAQ: validation split and shuffle behavior  
  https://keras.io/getting_started/faq/
- Keras LSTM layer: input/output shape and 기본 활성화  
  https://keras.io/api/layers/recurrent_layers/lstm/
- Keras LeakyReLU layer: `negative_slope` 인자  
  https://keras.io/api/layers/activation_layers/leaky_relu/
- Keras MNIST dataset: shape, dtype, pixel range  
  https://keras.io/api/datasets/mnist/
- scikit-learn MinMaxScaler: `fit`, `transform`, `inverse_transform`, train data 기준 min/max  
  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
- TensorFlow pip installation: OS/GPU별 설치 주의  
  https://www.tensorflow.org/install/pip
