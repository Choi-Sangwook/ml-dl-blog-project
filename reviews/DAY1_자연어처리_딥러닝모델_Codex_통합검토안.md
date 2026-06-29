# DAY1 자연어처리 딥러닝모델 Codex 통합검토안

검토 대상:

- 기준 PDF: `sources/DAY1_자연어처리_딥러닝모델.pdf`
- 초안: `drafts/DAY1_자연어처리_딥러닝모델_초안_v1.md`
- 대상 독자: 파이썬 기본을 알고 있지만 자연어처리 딥러닝을 처음 배우는 사람

## 1. 전체 평가

초안은 PDF의 큰 목차인 `1. 자연어 처리 딥러닝 모델`, `2. 자연어 처리를 위한 CNN`, `3. 자연어 처리를 위한 RNN`을 모두 반영하고 있으며, DNN부터 GPT까지의 모델 설명, 발전 과정, 비교표도 PDF의 흐름과 대체로 일치한다. 특히 PDF에 없는 보충 설명을 `🟩`로 구분하고, TensorFlow/PyTorch 미설치 환경에서 CNN/RNN 코드를 실행하지 않았다고 명시한 점은 적절하다. 거짓 출력 금지 원칙도 전반적으로 잘 지켰다.

다만 최종본으로 가기 전 반드시 고쳐야 할 항목이 있다. 첫째, NumPy 1D 합성곱 예제는 "실제 실행 출력"이라고 되어 있지만 현재 코드가 출력하지 않는 줄들이 출력 블록에 포함되어 있어 코드와 출력이 불일치한다. 둘째, RNN 절은 PDF의 `test_iter`를 `val_dataloader()`로 쓰는 레거시 구조를 그대로 보여 주면서도, 실제 검증/테스트 분리 문제를 충분히 경고하지 않는다. 셋째, RNN 코드를 "원문 그대로 인용"이라고 표현하지만 실제로는 초안 작성자가 정리한 부분 발췌이므로 출처 표기가 과하다.

PDF는 31페이지이며 PyMuPDF 기준으로 전 페이지에서 텍스트가 추출되었다. 사용자가 안내한 대로 이미지 전용 코드 페이지는 없는 것으로 보고 OCR 복원은 수행하지 않았다.

## 2. 높은 우선순위의 오류와 수정사항

### High 1. NumPy 1D 합성곱 코드와 출력이 일치하지 않음

- 위치: 초안 5.3, 대략 222-257행
- 문제: 코드 블록은 `피처 맵`과 `GlobalMaxPooling` 두 줄만 출력한다. 그러나 출력 블록에는 `문장 임베딩 행렬 shape`, 단어별 벡터, 창별 점수까지 포함되어 있다.
- 왜 중요한가: "실행 결과 O"라고 표시한 보충 코드에서 코드와 출력이 다르면, 거짓 출력은 아니더라도 독자가 그대로 실행했을 때 결과가 달라져 신뢰가 떨어진다.
- 수정: 출력 블록을 줄이거나, 코드에 실제 출력 줄을 모두 생성하는 `print()`를 추가해야 한다.

### High 2. RNN에서 test 데이터와 validation 데이터 역할이 섞임

- 위치: 초안 6.5, `val_dataloader()`가 `test_iter`를 반환하는 부분 및 "validation accuracy" 설명
- PDF 근거: p.30-31은 `val_dataloader(self): return test_iter`, 평가 지표 `train accuracy`, `validation accuracy`를 제시한다.
- 문제: 강의 코드 흐름은 `train/test`만 분리한 뒤 `test_iter`를 검증용으로 쓰는 형태다. 초안은 이를 그대로 보여 주지만, 실제 모델 개발에서는 검증 데이터와 테스트 데이터의 역할을 분리해야 한다는 경고가 부족하다.
- 왜 중요한가: 하이퍼파라미터 조정, 조기 종료, 모델 선택을 `test_iter` 성능으로 반복하면 테스트 세트가 사실상 검증 세트가 되어 최종 성능 추정이 낙관적으로 치우칠 수 있다.
- 수정: "PDF의 예제는 입문용 단순화이며, 실제 실습에서는 train/validation/test를 분리한다"는 경고를 High 우선순위로 넣어야 한다.

### High 3. "강의 원문 그대로 인용" 표현이 부정확함

- 위치: 초안 6.1, 6.2, 6.4, 6.5
- 문제: RNN 코드는 PDF 코드의 전체 원문이 아니라 초안에서 일부를 묶고 생략한 "강의 기반 부분 발췌/요약"이다. 예를 들어 PDF p.25에는 `print(vars(train.examples[0]))`와 `print(vars(train.examples[0])['label'])`이 함께 나오지만 초안 코드에는 일부만 남아 있다.
- 왜 중요한가: PDF에 없는 정리·생략·주석을 PDF 원문처럼 보이게 하면 출처 구분 원칙을 약화한다.
- 수정: "강의 원문 코드" 대신 "강의 흐름을 옮긴 레거시 코드 요약" 또는 "PDF 기반 부분 발췌"로 바꾼다.

### High 4. RNN 임베딩 텐서와 device 처리 설명이 부족함

- 위치: 초안 6.4, `self.embedding = embedding`, `x = self.embedding[X]`, `x = x.to(self.device)`
- 문제: `text_field.vocab.vectors`를 그대로 넘기면 `embedding`은 보통 `nn.Embedding` 모듈이 아니라 텐서다. `BucketIterator`가 배치를 GPU로 올렸는데 임베딩 텐서가 CPU에 남아 있으면 `self.embedding[X]` 단계에서 디바이스 불일치가 날 수 있다. `x.to(self.device)`는 인덱싱 이후라 너무 늦다.
- 왜 중요한가: 초안은 이 코드를 미실행 레거시 코드로 표시했지만, 입문자는 그대로 옮겨 실행할 수 있다.
- 수정: 이 코드는 "현대 환경에서 비실행/구조 이해용"이라고 더 강하게 표시하고, 실제 구현에서는 `nn.Embedding.from_pretrained(...)`를 모델 모듈로 등록하거나 입력과 임베딩을 같은 device에 둬야 한다고 덧붙인다.

## 3. PDF 기준 누락 내용

| PDF 페이지/주제 | 초안 반영 상태 | 권장 조치 |
|---|---|---|
| p.3 자연어 처리 정의와 예시 목록 | 부분 반영 | "NLP란 무엇인가" 정의와 문장 분류, 감성 분석, 번역, 요약, 질의응답, 챗봇, 음성 인식, 문장 생성 예시를 초반에 짧게 추가 |
| p.17 CNN 기본 연산 중 Flattening, Softmax | 일부 누락 | CNN 기본 흐름에 `Flattening`과 `Softmax`를 명시. 텍스트 CNN에서는 `GlobalMaxPooling1D`가 flattening 역할을 일부 대체할 수 있음을 설명 |
| p.19 컨볼루션 층을 깊게 만드는 이유 | 누락 | "깊어질수록 복잡한 특징, 고차원 잠재 특징을 학습할 수 있다"를 CNN 절에 추가 |
| p.20 유니크 값 확인 | 거의 누락 | 스팸 분류 전처리 단계에 `unique`/라벨 종류 확인을 한 줄 추가 |
| p.21 `word_to_index`, `vocab_size`, `max_len`, 학습 데이터 shape 출력 | 요약만 있음 | 입문자가 출력을 해석할 수 있도록 각 값의 의미를 2-3줄 설명 |
| p.26 지원되는 사전학습 임베딩 전체 목록 | 일부 반영 | 현재 초안은 대표 예시만 언급하므로 충분함. 전체 목록은 필요하면 접기/요약으로 처리 |
| p.29 `val_accuracy = pl.metrics.Accuracy()` | 누락 | 레거시 코드 요약에 train/val accuracy가 모두 있었다고 설명하되, 최신은 `torchmetrics`로 이전한다고 명시 |
| p.30 Training / Validation / Test 함수 재정의 | Test 설명 부족 | PDF 제목에는 Test가 포함되지만 코드 추출은 train/validation 중심임을 설명하고, 실제 프로젝트에서는 test를 최종 평가에만 사용한다고 보충 |

## 4. 더 자세히 설명할 내용

- `Tokenizer.fit_on_texts()`는 반드시 훈련 데이터에만 적용한다는 설명은 좋다. 여기에 `texts_to_sequences()`는 검증/테스트 데이터에도 적용하지만, 이때 새 단어는 훈련 vocabulary 기준으로 처리된다는 점을 덧붙이면 더 명확하다.
- 스팸 분류에서 `accuracy` 한계와 `precision·recall·F1` 필요성을 설명한 점은 적절하다. 다만 Keras 예제 코드의 `metrics=["accuracy"]`와 prose가 어긋나므로 `Precision`, `Recall`, AUC 또는 scikit-learn의 `classification_report`를 함께 보여 주거나, "코드는 구조 예시라 accuracy만 넣었다"는 단서를 추가한다.
- `CrossEntropyLoss` 앞에 softmax를 넣지 않는다는 설명은 정확하다. PyTorch 공식 문서는 `CrossEntropyLoss`가 `LogSoftmax`와 `NLLLoss` 조합과 동등하다고 설명한다. 이 설명은 유지한다.
- `permute(1, 0, 2)` 설명은 좋다. PyTorch `nn.LSTM`의 기본 `batch_first=False`는 `(seq_len, batch, feature)` 입력을 기대하고, `batch_first=True`이면 `(batch, seq, feature)`를 사용할 수 있다는 공식 문서 기준 설명과 일치한다.
- Transformer 설명의 "Attention만 사용"은 PDF 표현이지만, 입문자에게는 "RNN처럼 순차적으로 넘기는 구조 대신 attention 중심 구조를 사용한다"가 더 안전하다. Transformer에는 feed-forward layer, residual/normalization 등도 있기 때문이다.

## 5. 유용한 추가 내용

- PDF에는 없는 `🟩` 보충인 데이터 누수, 불균형 평가 지표, 레거시 API 경고, 프레임워크 혼용 주의는 모두 유용하다.
- `torchtext.legacy` 경고는 필요하다. 공식 torchtext 문서는 현재 TorchText 개발이 중단되었고 0.18 릴리스가 마지막 안정 릴리스라고 밝힌다. 또한 현재 안정판 문서의 package reference에는 `torchtext.legacy`, `Field`, `BucketIterator`가 나타나지 않는다.
- PyTorch Lightning 관련 경고도 정확하다. Lightning 2.x 업그레이드 가이드는 `Trainer(gpus=...)` 대신 `devices`를 사용하라고 안내하며, 최신 Trainer 문서는 `Trainer(accelerator="gpu", devices=...)` 형태를 보여 준다.
- `pl.metrics.Accuracy` 경고도 정확하다. PyTorch Lightning 1.3 문서는 `pytorch_lightning.metrics`가 별도 패키지 TorchMetrics로 이동했고 v1.5에서 제거될 예정이라고 안내했다.

## 6. 줄이거나 제거할 내용

- `현재 대부분의 최신 서비스는 Transformer 계열...`은 PDF p.16에 있는 설명이지만, "대부분", "최신"은 시간에 따라 달라지는 표현이다. 최종본에서는 "강의에서는 현재 많은 NLP 서비스가 Transformer 계열을 기반으로 한다고 설명한다"처럼 출처와 시점을 분명히 하는 편이 안전하다.
- RNN 레거시 코드를 너무 길게 유지할 필요는 없다. 이 글의 목적은 실행 가능한 최신 RNN 튜토리얼이 아니라 PDF 기반 개념 정리이므로, 레거시 코드는 "강의 흐름 파악용"으로 짧게 두고 최신 구현은 별도 DAY 또는 보충 링크로 넘기는 편이 좋다.
- `pip install pytorch-lightning torchtext`는 현재 초안의 레거시 경고와 충돌한다. Lightning 최신 문서는 `python -m pip install lightning`을 안내한다. torchtext는 개발 중단 상태이므로 "강의 코드 재현 목적 외에는 새 프로젝트에 권장하지 않음"을 붙인다.
- Keras 코드의 `Embedding(..., input_length=max_len)`은 Keras 3 공식 `Embedding` 시그니처에 별도 인자로 문서화되어 있지 않다. 최신 Keras 스타일로는 `keras.Input(shape=(max_len,))`를 먼저 두고 `Embedding(input_dim, output_dim)`을 쓰는 편이 더 안전하다.

## 7. 바로 붙여 넣을 수 있는 수정 블록

### 7.1 NumPy 1D 합성곱 코드 교체

```python
import numpy as np

np.random.seed(42)

words = ["정말", "재미있는", "영화", "였다", "!"]
emb_dim = 4

embeddings = np.round(np.random.rand(len(words), emb_dim), 2)
kernel = np.round(np.random.rand(2, emb_dim), 2)

print("문장 임베딩 행렬 shape:", embeddings.shape)
for word, vector in zip(words, embeddings):
    print(f"{word:>6} -> {vector}")

feature_map = []
for i in range(len(words) - 2 + 1):  # stride=1
    window = embeddings[i:i + 2]
    score = round(float(np.sum(window * kernel)), 2)
    feature_map.append(score)
    print(f"창[{words[i]} {words[i + 1]}] 점수 = {score}")

print("피처 맵:", feature_map)
print("GlobalMaxPooling(가장 강한 패턴):", max(feature_map))
```

### 7.2 RNN 레거시 코드 안내 문구 교체

```markdown
> 🟦 **PDF 기반 레거시 코드 요약**: 아래 코드는 강의가 사용한 `torchtext.legacy`, `BucketIterator`, `pytorch_lightning` 흐름을 이해하기 쉽게 묶은 것입니다. 원문 전체를 그대로 복사한 것이 아니며, 현재 최신 환경에서는 그대로 실행되지 않을 수 있습니다.
>
> 🟩 **실습 주의**: 이 예제는 강의 흐름상 `test_iter`를 `val_dataloader()`에서 사용하지만, 실제 모델 개발에서는 `train / validation / test`를 분리해야 합니다. 검증 데이터는 학습 중 모델 선택과 조기 종료에 쓰고, 테스트 데이터는 마지막 최종 평가에만 사용합니다.
```

### 7.3 RNN device/embedding 주의 문구 추가

```markdown
> 🟩 **device 주의**: `text_field.vocab.vectors`는 보통 텐서이므로 `self.embedding[X]`처럼 바로 인덱싱하면 입력 `X`와 임베딩 텐서가 서로 다른 device에 있을 때 오류가 날 수 있습니다. 실제 구현에서는 `nn.Embedding.from_pretrained(...)`로 임베딩을 모델 모듈에 등록하고 `model.to(device)`로 함께 이동시키는 방식이 더 안전합니다.
```

### 7.4 CNN 누락 내용 추가

```markdown
🟦 PDF의 CNN 기본 흐름은 다음처럼 정리할 수 있습니다.

```text
Convolution → Pooling → Flattening 또는 Global Pooling → Fully Connected Layer → Softmax/Sigmoid
```

- **Flattening**: 2차원 또는 3차원 특징 맵을 완전 연결층에 넣기 쉬운 1차원 구조로 펼치는 과정입니다.
- **텍스트 CNN의 GlobalMaxPooling1D**: 모든 위치의 특징 중 가장 강한 반응만 남기므로, 전통적인 flattening보다 파라미터 수를 줄이는 역할을 할 수 있습니다.
- **컨볼루션 층을 깊게 만드는 이유**: 얕은 층은 단순한 패턴을, 깊은 층은 더 복잡하고 추상적인 특징을 학습할 수 있습니다. 다만 깊어진다고 항상 성능이 좋아지는 것은 아니며 데이터 크기와 과적합 관리가 함께 필요합니다.
```

### 7.5 Keras 예제 최신화

```python
import keras
from keras import layers

vocab_size = 10000
max_len = 200

model = keras.Sequential([
    keras.Input(shape=(max_len,), dtype="int32"),
    layers.Embedding(input_dim=vocab_size, output_dim=128),
    layers.Dropout(0.3),
    layers.Conv1D(filters=128, kernel_size=5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)
```

### 7.6 설치 명령 문구 교체

```markdown
설치 참고:

- 강의 당시 코드 재현: `pip install pytorch-lightning torchtext`
- 최신 Lightning 학습 코드 작성: `python -m pip install lightning torchmetrics`
- PyTorch는 OS, Python 버전, CPU/GPU 환경에 따라 설치 명령이 달라지므로 PyTorch 공식 설치 선택기를 확인합니다.

> 🟩 `torchtext`는 현재 개발이 중단된 라이브러리이므로, 새 프로젝트에서는 `torchtext.legacy` 기반 예제를 그대로 시작하지 않는 편이 좋습니다.
```

## 8. 우선순위 표

| 우선순위 | 항목 | 위치 | 조치 |
|---|---|---|---|
| High | NumPy 예제 코드와 출력 불일치 | 5.3 | 코드에 출력문을 추가하거나 출력 블록 축소 |
| High | `test_iter`를 validation으로 사용하는 문제 경고 부족 | 6.5 | train/validation/test 역할 분리 설명 추가 |
| High | "원문 그대로 인용" 표현 부정확 | 6.1-6.5 | "PDF 기반 레거시 코드 요약/부분 발췌"로 변경 |
| High | RNN 임베딩 텐서/device 오류 가능성 | 6.4 | `nn.Embedding.from_pretrained` 또는 구조 이해용 명시 |
| Medium | NLP 정의와 예시 목록 부족 | 2장 초반 | p.3 정의/예시를 짧게 추가 |
| Medium | CNN의 Flattening, Softmax, 깊은 컨볼루션 이유 누락 | 5.1-5.2 | p.17, p.19 내용 추가 |
| Medium | 불균형 평가 지표 설명과 Keras 코드 metric 불일치 | 5.4 | Precision/Recall metric 또는 classification_report 추가 |
| Medium | 설치 명령이 최신 권장 방식과 충돌 | 6.3 | `lightning`, `torchmetrics`, PyTorch 공식 선택기 기준으로 보정 |
| Medium | Keras `Embedding(input_length=...)` 버전 민감 | 5.4 | `keras.Input(shape=(max_len,))` 방식으로 교체 |
| Low | 사전학습 임베딩 전체 목록 축약 | 6.2 | 현재처럼 대표 예시만 유지 가능 |
| Low | `현재 대부분의 최신 서비스` 표현 | 4.12 | PDF 설명임을 밝히고 "많은" 등으로 완화 |

## 9. 공식 문서 기준 점검 메모

- torchtext: 공식 문서와 GitHub README는 TorchText 개발이 중단되었고 0.18 릴리스가 마지막 안정 릴리스라고 안내한다. 현재 안정판 문서에는 `torchtext.legacy`, `Field`, `BucketIterator`가 주요 package reference에 없다.
- PyTorch Lightning: 2.x 업그레이드 문서는 `Trainer(gpus=...)`를 `devices`로 바꾸라고 안내한다. 현재 Trainer 문서는 `Trainer(accelerator="gpu", devices=...)` 형태를 사용한다.
- Lightning metrics: PyTorch Lightning 1.3 문서는 `pytorch_lightning.metrics`가 TorchMetrics로 이동했고 v1.5에서 제거 예정이라고 안내했다. 초안의 `pl.metrics.Accuracy` 경고는 타당하다.
- PyTorch `CrossEntropyLoss`: 공식 문서는 클래스 인덱스 target 사용 시 `LogSoftmax`와 `NLLLoss` 조합과 동등하다고 설명한다. 따라서 모델 앞에 별도 softmax를 넣지 말라는 초안 설명은 정확하다.
- PyTorch `nn.LSTM`: 기본 `batch_first=False`에서는 입력 shape가 `(seq_len, batch, feature)`이고, `batch_first=True`에서는 `(batch, seq, feature)`이다. 초안의 `permute(1, 0, 2)` 설명은 정확하다.
- Keras `Conv1D`: 공식 문서는 기본 `channels_last` 입력 shape를 `(batch, steps, channels)`로 설명한다. 초안의 텍스트 CNN 입력 설명과 부합한다.
- Keras `BinaryCrossentropy`: `from_logits=False`일 때 확률값을 기대한다. 따라서 `Dense(1, activation="sigmoid")` + `binary_crossentropy` 조합은 타당하다.

참고한 공식 문서:

- https://docs.pytorch.org/text/stable/index.html
- https://github.com/pytorch/text
- https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html
- https://lightning.ai/docs/pytorch/stable/common/trainer.html
- https://lightning.ai/docs/pytorch/stable/starter/installation.html
- https://pytorch-lightning.readthedocs.io/en/1.3.8/extensions/metrics.html
- https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html
- https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
- https://keras.io/api/layers/convolution_layers/convolution1d/
- https://keras.io/api/layers/core_layers/embedding/
- https://keras.io/api/losses/probabilistic_losses/

## 10. 최종 권고

초안은 PDF 범위를 대부분 충실히 반영하고 있고, 보충 설명의 방향도 입문자에게 유용하다. 다만 최종본으로 넘기기 전에 High 항목 4개는 반드시 수정해야 한다. 특히 실행 결과 불일치와 validation/test 혼용 경고는 학습 신뢰도와 평가 원칙에 직접 영향을 준다.

권고: **High 항목을 먼저 수정한 뒤 최종본 작성 가능**. Medium 항목 중 CNN 누락 내용, 설치 명령, Keras 예제 최신화까지 함께 반영하면 입문자용 Velog 글로 안정성이 높아진다.

