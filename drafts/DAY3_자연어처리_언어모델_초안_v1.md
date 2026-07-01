# 🤗 자연어처리(NLP) 완전 입문 가이드 — DAY3. 자연어 처리를 위한 언어 모델 (BERT)

> **시리즈**: 파이썬 기본만 있는 사람을 위한 자연어처리(NLP) 입문
> **교과목**: 초거대언어모델(LLM) · 단원 2 — 자연어 딥러닝
> **이전 편**: DAY1. 자연어 처리를 위한 딥러닝 모델 → DAY2. 텍스트 분류

> 📝 DAY1에서 DNN~Transformer 모델 지도를, DAY2에서 텍스트 분류를 봤습니다. 이번 DAY3에서는 그중 **현재 가장 널리 쓰이는 사전훈련 언어 모델 BERT**를 다룹니다. 핵심은 **"이미 학습된 거대한 모델을 내려받아, 내 문제에 맞게 살짝 더 학습(파인튜닝)한다"** 는 것입니다. 영화평 감성(이진), 뉴스 카테고리(다중), IMDB 감성 분류를 **Hugging Face Transformers**로 실습합니다.

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 개념·코드·출력
> - 🟩 **(보충)** : 입문자를 위해 글쓴이가 덧붙인 설명·주의점·수정
>
> ⚠️ **코드 실행 안내**: 이 강의는 **코드 중심**이지만, 작성 환경에 **PyTorch·transformers가 설치되어 있지 않아** 코드를 **직접 실행하지 않았습니다.** PDF에 이미 나온 출력(데이터 shape, `value_counts` 등)은 🟦로 인용하며, **출력값을 새로 지어내지 않았습니다.** 토큰화 개념 데모(§6.3)만 순수 파이썬으로 **직접 실행한 실제 출력**입니다.
> 또한 PDF 코드 일부에 **버전·논리 불일치**(예: `train_test_split` 반환값 개수, BERT-Base인데 `layer.23` 사용)가 있어, 해당 부분은 🟩로 **바로잡아** 설명합니다.

---

## 1. 이번 DAY에서 배우는 것

- 🟦 BERT가 무엇이고 왜 강력한가 (사전훈련 + 미세조정)
- 🟦 BERT-Base와 BERT-Large의 구조 차이(L, D, A, 파라미터 수)
- 🟦 전이학습(Transfer Learning)과 레이어 동결(Freeze) 전략
- 🟦 Hugging Face Transformers로 BERT를 내려받아 파인튜닝하는 전체 흐름
- 🟦 실습: 이진 분류(영화평), 다중 분류(뉴스 8종), 감성 분류(IMDB)

---

## 2. BERT란? 🟦

**BERT(Bidirectional Encoder Representations from Transformers)** 는 구글이 **2018년**에 공개한 **사전 훈련된(pre-trained) 언어 모델**입니다.

- 🟦 등장과 동시에 다양한 자연어 처리 문제에서 **매우 높은 성능**을 보여, NLP에 한 획을 그은 모델로 평가받습니다.
- 🟦 학습 규모(강의 설명): 약 **33억 단어**를 **약 4일간** 학습. **위키피디아(약 25억 단어)** 와 **BookCorpus(약 8억 단어)** 의 **레이블 없는** 텍스트로 사전 훈련.
- 🟦 학습 방식: **레이블이 없는 데이터로 먼저 학습(사전훈련)** → **레이블이 있는 데이터로 추가 학습(미세조정, Fine-Tuning)**. 미세조정에서 하이퍼파라미터를 다시 조정합니다.

> 🟩 **핵심 직관**: BERT는 "언어의 일반 상식"을 먼저 대량으로 익힌 모델입니다. 그래서 우리는 **처음부터 학습할 필요 없이**, 이 모델을 가져와 **내 작은 데이터로 조금만 더 학습**하면 됩니다. 강의의 비유처럼 *"영어를 잘하는 사람이 비슷한 다른 언어를 빨리 배우는 것"* 과 같습니다.

> 🟩 **주의(과장 완화)**: PDF의 "최고의 성능"은 **2018년 당시 여러 벤치마크 기준**입니다. 이후 더 좋은 모델들이 많이 나왔으므로, "지금도 항상 최고"라는 뜻은 아닙니다.

---

## 3. BERT의 구조 🟦

BERT는 Transformer의 **인코더(Encoder)** 를 여러 층 쌓은 구조입니다. 크기에 따라 두 종류가 있습니다.

🟦 변수 설명:
- **L** = 인코더 층(Encoder Layer)의 개수
- **D** = `d_model`(은닉 벡터)의 크기
- **A** = Self-Attention Head의 개수

| 모델 | L (층) | D (은닉 크기) | A (헤드) | 파라미터 |
|---|---|---|---|---|
| **BERT-Base** | 12 | 768 | 12 | 약 1억 1천만(110M) |
| **BERT-Large** | 24 | 1024 | 16 | 약 3억 4천만(340M) |

- 🟦 BERT-Large는 Base보다 헤드 수·은닉 크기·전체 파라미터가 모두 더 큽니다.
- 🟦 **Hugging Face**에서 무료로 내려받을 수 있고, 약 **110여 개 언어**(한국어 포함) 모델이 제공됩니다.
- 🟦 내려받은 모델을 **미세조정**해 영화 리뷰 분류, 스팸 메일 분류 등 다양한 응용에 활용합니다.

---

## 4. 전이학습과 미세조정 전략 🟦

**전이학습(Transfer Learning)**: 이미 학습된 모델의 지식을 가져와 새 문제에 적용하는 방법. BERT 파인튜닝이 대표적입니다.

🟦 학습할 층을 고르는 방법은 PyTorch의 `requires_grad`로 정합니다.
- `requires_grad = True` → 해당 층의 파라미터를 **학습(변경)**
- `requires_grad = False` → 해당 층을 **동결(Freeze)**, 변경하지 않음

🟦 강의가 제시하는 세 가지 동결 전략(`tl_strategy`):

| 전략 | 학습하는 부분 | 설명 |
|---|---|---|
| 전략 1 | (없음) 모든 BERT 층 Freeze | 분류기 위쪽만 학습 |
| 전략 2 | Pooler만 학습 | 나머지 BERT는 동결 |
| 전략 3 | Pooler + 마지막 Encoder 층 | 가장 흔한 절충안 |

```python
# (PDF) 전략 3: Pooler와 마지막 Encoder Layer만 학습
for name, param in model.bert.named_parameters():
    if (not name.startswith("pooler")) and ("layer.23" not in name):
        param.requires_grad = False
```

> 🟩 **중요(버전·모델 불일치 주의)**: 위 코드의 `"layer.23"`은 **BERT-Large(24층, 인덱스 0~23)의 마지막 층**을 가리킵니다. 그런데 §6에서 쓰는 한국어 모델 `kykim/bert-kor-base`는 **BERT-Base(12층, 인덱스 0~11)** 입니다. Base에는 `layer.23`이 없으므로, 이 코드를 그대로 쓰면 **마지막 층까지 전부 동결**되어 의도와 달라집니다. **Base라면 `"layer.11"`** 로 바꿔야 "마지막 Encoder 층만 학습"이 됩니다. 모델 층 수에 맞춰 인덱스를 확인하세요.

> 🟩 **왜 동결하나?**: 데이터가 적을 때 BERT 전체를 학습하면 **과적합**되거나 학습이 불안정할 수 있습니다. 일부만 학습하면 빠르고 안정적입니다. 단, "어떤 전략이 항상 최고"는 아니며 데이터 양에 따라 다릅니다.

---

## 5. Hugging Face와 Transformers 🟦

- 🟦 **Hugging Face**: 사전훈련 모델을 공유하는 **모델 허브(Model Hub)**. 모델·데이터셋·언어별 모델을 검색하고 내려받아 파인튜닝할 수 있습니다. (https://huggingface.co/models)
- 🟦 **Transformers** 라이브러리 특징:
  1. 자연어 처리를 위한 **범용 아키텍처** 제공
  2. **BERT 계열, GPT 계열** 등 다양한 언어 모델 제공
  3. **PyTorch, TensorFlow** 등 여러 프레임워크 지원
  4. **Python 패키지**로 설치·사용이 쉬움

```bash
!pip install transformers
```

> 🟦 **Down Stream Task**: 파인튜닝으로 **수행하려는 실제 작업**. 예) Text Classification, Question Answering, Summarization 등.

> 🟩 이 글의 모든 실습 코드는 **PyTorch + Transformers(`Trainer` API)** 로 일관됩니다. 데이터 경로 `/content/mnt/MyDrive/...`는 **Google Colab(구글 드라이브 마운트)** 환경 기준입니다. 로컬에서는 경로를 바꾸세요.

---

## 6. 실습 ① — BERT 이진 분류 (영화평 감성) 🟦

> **프레임워크: PyTorch + Hugging Face Transformers** / 모델: `kykim/bert-kor-base`(한국어, BERT-Base)

### 6.1 데이터 준비와 분리 🟦

```python
import pandas as pd

dataset = pd.read_table("/content/mnt/MyDrive/ratings.txt")
dataset.columns = ['id', 'review', 'sentiment']   # sentiment: 0=부정, 1=긍정
dataset.head()
```

🟦 예시 데이터(PDF):

| id | review | sentiment |
|---|---|---|
| 9976970 | 더빙이 실망스럽네요. | 0 |
| 3819312 | 오버 연기조차 가볍지 않구나 | 1 |
| 10265843 | 너무재밌었다. 그래서 보는 것을 추천한다 | 1 |

🟦 **Train/Test → Train/Validation** 두 단계로 나눕니다. 분류 문제이므로 **층화 샘플링(`stratify`)** 으로 클래스 비율을 유지합니다.

```python
from sklearn.model_selection import train_test_split

# 1) 전체를 Train 80% / Test 20%
train_idx, test_idx = train_test_split(
    dataset.index, test_size=0.2,
    stratify=dataset.sentiment, random_state=42)

train_set = dataset.iloc[train_idx].reset_index(drop=True)
test_set  = dataset.iloc[test_idx]

# 2) Train을 다시 Train 70% / Validation 30%
tr_idx, val_idx = train_test_split(
    train_set.index, test_size=0.3,
    stratify=train_set.sentiment, random_state=42)

valid_set = train_set.iloc[val_idx]
train_set = train_set.iloc[tr_idx]
```

🟦 결과 shape(PDF): `train (84000, 3)`, `valid (36000, 3)`, `test (30000, 3)`

> 🟩 **수정(반환값 개수)**: PDF 코드는 `train_idx, test_idx, _, _ = train_test_split(dataset.index, ...)` 처럼 **4개**로 받습니다. 하지만 `train_test_split`에 **배열을 1개**만 넘기면 결과는 **2개**(`train_idx, test_idx`)입니다. 4개는 `X, y`처럼 **2개를 넘길 때** 나옵니다. 위 코드처럼 **2개로 받아야** 오류가 없습니다.
>
> 🟩 **데이터 누수 방지(중요)**: 반드시 **나눈 뒤** 각 세트에 토크나이저를 적용합니다. Validation·Test 정보가 학습에 섞이면 안 됩니다. 또 **Test는 마지막 평가에만** 쓰고, 모델 선택·조기 종료에는 **Validation**을 씁니다.

### 6.2 Dataset 클래스 정의 🟦

🟦 `torch.utils.data.Dataset`을 상속해 세 함수를 재정의합니다.

| 함수 | 역할 |
|---|---|
| `__init__()` | Text, Label, Tokenizer, 최대 토큰 길이 설정 |
| `__len__()` | 전체 데이터 개수 반환 |
| `__getitem__()` | 특정 index의 `input_ids`, `attention_mask`, `label` 반환 |

```python
import torch
from transformers import BertTokenizerFast

bert_model_name = "kykim/bert-kor-base"
tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, reviews, sentiments, tokenizer):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.reviews[idx],
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.sentiments[idx]),
        }

train_set_dataset = BertDataset(
    train_set.review.tolist(), train_set.sentiment.tolist(), tokenizer)
valid_set_dataset = BertDataset(
    valid_set.review.tolist(), valid_set.sentiment.tolist(), tokenizer)
```

> 🟩 `__getitem__`의 구체적 토큰화 부분은 PDF에 상세히 나오지 않아 입문자가 실행 가능하도록 보충했습니다(🟩). 핵심 반환값(`input_ids`, `attention_mask`, `labels`)은 PDF 설명과 같습니다.

### 6.3 🟩 토큰화·어텐션 마스크 개념 데모 (실행 결과 O)

BERT 입력의 두 핵심인 **`input_ids`(단어를 숫자로)** 와 **`attention_mask`(진짜 토큰=1, 패딩=0)** 를 순수 파이썬으로 흉내 내 봅니다. (실제 BERT는 `[CLS]`·`[SEP]` 특수 토큰과 WordPiece 서브워드를 쓰며, 아래는 **개념 이해용**입니다.)

```python
reviews = ["더빙이 실망스럽네요", "너무 재밌었다 추천"]
vocab = {"[PAD]":0, "[CLS]":1, "[SEP]":2,
         "더빙이":5,"실망":6,"스럽네요":7,"너무":8,"재밌었다":9,"추천":10}

def fake_tokenize(text):
    return [vocab.get(w, 6) for w in text.split()]

max_len = 6
for r in reviews:
    ids = [vocab["[CLS]"]] + fake_tokenize(r) + [vocab["[SEP]"]]
    pad = max_len - len(ids)
    attention_mask = [1]*len(ids) + [0]*pad
    ids = ids + [vocab["[PAD]"]]*pad
    print(f"문장: {r}")
    print(f"  input_ids      = {ids}")
    print(f"  attention_mask = {attention_mask}")
```

실제 실행 출력:

```text
문장: 더빙이 실망스럽네요
  input_ids      = [1, 5, 6, 2, 0, 0]
  attention_mask = [1, 1, 1, 1, 0, 0]
문장: 너무 재밌었다 추천
  input_ids      = [1, 8, 9, 10, 2, 0]
  attention_mask = [1, 1, 1, 1, 1, 0]
```

> 🟩 **해석**: 문장 앞뒤에 `[CLS]`(1)·`[SEP]`(2)가 붙고, 길이를 `max_len=6`으로 맞추려고 **`[PAD]`(0)** 로 채웁니다. `attention_mask`는 **패딩 자리를 0** 으로 표시해, BERT가 의미 없는 패딩에 집중하지 않게 합니다.

### 6.4 모델 다운로드와 미세조정 설정 🟦

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(bert_model_name)

# 미세조정 전략 (예: 전략 3 — Pooler + 마지막 Encoder 층만 학습)
# ⚠️ bert-kor-base는 BERT-Base(12층)이므로 "layer.11"이 마지막 층
for name, param in model.bert.named_parameters():
    if (not name.startswith("pooler")) and ("layer.11" not in name):
        param.requires_grad = False
```

### 6.5 학습·평가 (Trainer) 🟦

🟦 `TrainingArguments`로 하이퍼파라미터를 설정하고 `Trainer`로 학습/예측합니다.

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="steps",     # ⚠️ 구버전 명칭은 evaluation_strategy
    eval_steps=500,            # 🟩 steps 평가 시 간격 지정 필요
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set_dataset,
    eval_dataset=valid_set_dataset,
    compute_metrics=compute_metrics,   # 🟩 별도 정의 필요(아래)
)
trainer.train()
```

🟦 **예측(평가)**: 마지막에 **Test 데이터**로 예측합니다.

```python
test_set_dataset = BertDataset(
    test_set.review.tolist(), test_set.sentiment.tolist(), tokenizer)
trainer.predict(test_set_dataset)
```

> 🟩 **`compute_metrics`는 PDF에 정의가 없습니다.** 직접 만들어야 합니다. 이진 분류라 정확도뿐 아니라 precision·recall·F1을 함께 보는 것이 좋습니다.

```python
# 🟩 보충: compute_metrics 예시
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": accuracy_score(labels, preds),
            "precision": p, "recall": r, "f1": f1}
```

> 🟩 **버전 주의**: 최신 `transformers`는 `evaluation_strategy` 대신 **`eval_strategy`** 를 씁니다(구버전은 전자). 설치 버전에 맞춰 사용하세요.

---

## 7. 실습 ② — BERT 다중 분류 (뉴스 8종) 🟦

이진 분류와 **거의 같고**, 클래스 수만 늘어납니다. 달라지는 부분만 봅니다.

🟦 데이터: 뉴스 기사(`news.csv`), **8개 카테고리**(0~7), 각 200개씩 균형 데이터.

```python
import pandas as pd
dataset = pd.read_csv("/content/mnt/MyDrive/news.csv")
dataset.category.value_counts()   # 🟦 각 카테고리 200개
```

🟦 결과 shape(PDF): `train (1024, 3)`, `valid (256, 3)`, `test (320, 3)`

🟦 **핵심 차이: `num_labels`** 를 클래스 수로 지정합니다.

```python
from transformers import BertForSequenceClassification

num_labels = 8   # 분류할 카테고리 개수
model = BertForSequenceClassification.from_pretrained(
    bert_model_name, num_labels=num_labels)
```

> 🟩 다중 분류는 출력층이 **클래스 수만큼의 logits** 이 되고, 손실은 내부적으로 **교차 엔트로피**가 쓰입니다(`BertForSequenceClassification`이 자동 처리). Dataset 클래스는 `review/sentiment` 대신 `news/category`를 쓰도록만 바꾸면 됩니다. `compute_metrics`의 average는 `"macro"` 등으로 바꿔 클래스별 균형을 봅니다.

---

## 8. 실습 ③ — BERT 감성 분류 (IMDB, 영어) 🟦

🟦 영어 데이터 **IMDB**(영화 리뷰 긍/부정), 모델 `bert-base-uncased`(영어).

🟦 **모델 구조 확인**: 내려받은 BERT의 층 이름을 출력해 볼 수 있습니다.

```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

for name, param in model.bert.named_parameters():
    print(name)
```

🟦 출력되는 구조(요약):
- **Embedding Layer**: `word_embeddings`, `position_embeddings`, `token_type_embeddings`, `LayerNorm`
- **Encoder × 12**: `encoder.layer.0 … 11`의 `attention.self.query/key/value`, `output.dense` 등
- **Classification(Pooler) Layer**: `pooler.dense`

> 🟩 이름의 `query/key/value`는 Self-Attention의 핵심 3요소입니다(DAY1 Attention 참고). `encoder.layer.0~11`이 **12층**이라는 점에서 이 모델이 **BERT-Base** 임을 확인할 수 있습니다 → 동결 전략에서 마지막 층은 `layer.11`.

🟦 데이터 분리·Dataset·Trainer 흐름은 §6과 동일합니다(`bert-base-uncased` 토크나이저 사용, **모델과 호환되는 토크나이저** 사용 주의).

> 🟩 **PDF 출력값 주의(불일치)**: PDF는 IMDB 절에서도 shape를 `(84000,3)/(36000,3)/(30000,3)`으로 적었는데, 이는 **§6(영화평) 수치를 재사용**한 것으로 보입니다. 실제 **IMDB는 약 5만 행**이고 열도 `review, sentiment` 2개입니다. 직접 실행하면 다른 값이 나옵니다.
>
> 🟩 **stratify 권장**: PDF의 IMDB 분리 코드는 `train_test_split(dataset.index, test_size=0.2)` 로 **`stratify`가 빠져** 있습니다(본문은 "층화 샘플링"이라고 함). 분류에서는 §6처럼 **`stratify=dataset.sentiment`** 를 넣어 클래스 비율을 유지하는 편이 좋습니다.

---

## 9. 입문자가 자주 하는 실수 🟩

1. **토크나이저-모델 불일치**: `bert-base-uncased` 모델에 다른 토크나이저 사용. → 모델과 **같은 이름**의 토크나이저를 쓰기.
2. **동결 인덱스 오류**: BERT-Base인데 `layer.23` 동결. → 층 수 확인 후 `layer.11`.
3. **`train_test_split` 반환값**: 배열 1개를 넘기고 4개로 받기. → 1개 → 2개 반환.
4. **데이터 누수**: 나누기 전에 토크나이저/통계를 전체에 적용. → 분리 후 적용, Test는 최종 평가만.
5. **불균형/다중 분류에서 정확도만 보기**: → precision·recall·F1(다중은 macro).
6. **버전 인자**: `evaluation_strategy` vs `eval_strategy`. → 설치한 `transformers` 버전 확인.
7. **PDF 출력값 맹신**: IMDB shape 등 일부는 앞 절 수치 재사용. → 직접 실행해 확인.

---

## 10. DAY3 핵심 정리

```text
BERT
  - 구글 2018, 사전훈련(레이블 없음) → 미세조정(레이블 있음)
  - Base: L12/D768/A12/110M   Large: L24/D1024/A16/340M
  - Transformer 인코더 기반, 양방향 문맥 이해

파인튜닝 (Hugging Face Transformers)
  1) 데이터 분리: Train/Test → Train/Valid (stratify)
  2) Dataset: __init__/__len__/__getitem__ → input_ids, attention_mask, labels
  3) Tokenizer/Model: from_pretrained (모델·토크나이저 짝 맞추기)
  4) Freeze 전략: requires_grad (Base는 layer.11)
  5) TrainingArguments + Trainer + compute_metrics
  6) trainer.predict(test) 로 최종 평가

작업별 차이
  - 이진: num_labels 기본(2)
  - 다중: num_labels=8
  - 평가지표: 정확도 + precision/recall/F1
```

> 🟩 다음 DAY 주제는 강의 자료에서 확인되지 않아 적지 않습니다(추측 금지).

---

## 참고 자료

- 🟦 강의 자료: 교과목 3 「초거대언어모델(LLM)」 · 단원 2 「자연어 딥러닝」 — DAY3. 자연어 처리를 위한 언어 모델 (21p)
- 🟩 Hugging Face 모델 허브 — https://huggingface.co/models
- 🟩 Hugging Face Transformers 문서 — `BertForSequenceClassification`, `Trainer`, `TrainingArguments`
- 🟩 한국어 BERT 예시 — `kykim/bert-kor-base` (Hugging Face)
- 🟩 scikit-learn `train_test_split` 공식 문서
