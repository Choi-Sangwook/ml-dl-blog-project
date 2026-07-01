# DAY3 자연어처리 언어모델 초안 통합 검토안

- 기준 PDF: `sources/DAY3_자연어처리_언어모델.pdf`
- 검토 초안: `drafts/DAY3_자연어처리_언어모델_초안_v1.md`
- 출력 파일: `reviews/DAY3_자연어처리_언어모델_Codex_통합검토안.md`
- 대상 독자: 파이썬 기본은 알고 있지만 자연어처리/딥러닝을 처음 배우는 사람
- 처리 방식: PyMuPDF로 PDF 21페이지 텍스트를 페이지별 추출해 대조했다. 페이지 1은 표지, 페이지 2는 목차이고, 나머지 페이지도 텍스트 레이어가 충분히 추출되어 이미지 전용 코드 페이지 복구는 필요하지 않았다.
- 코드 실행 여부: `transformers`/`torch` 기반 실습 코드는 실행하지 않았다. scikit-learn `train_test_split` 반환값만 로컬에서 최소 예제로 확인했다.

## 1. 전체 평가

초안은 PDF의 큰 흐름인 `1. BERT 개념·구조`, `2. 이진 분류`, `3. 다중 분류`, `4. 감성 분류`를 모두 반영하고, PDF에 있는 오류 가능성을 보충 표시(🟩)로 분리하려는 방향도 좋다. 특히 BERT의 공개 시점, 사전훈련 말뭉치 규모, Base/Large 구조, `train_test_split` 반환값, `layer.23` 불일치, IMDB shape 불일치, Test 최종 평가 원칙은 전반적으로 타당하다.

다만 최종본으로 가려면 코드 실행 순서와 레이블 처리에서 몇 가지 High 수정이 필요하다. 현재 초안의 학습 코드 블록은 `compute_metrics`가 정의되기 전에 `Trainer(..., compute_metrics=compute_metrics)`를 호출하고, `model_dir`도 먼저 정의하지 않아 그대로 순차 실행하면 오류가 난다. 또한 IMDB의 `sentiment`가 문자열(`positive`/`negative`)인 일반적인 CSV를 그대로 `torch.tensor(...)`에 넣는 흐름은 실패하므로 0/1 정수 라벨 매핑을 명시해야 한다.

PDF/보충 구분은 대체로 명확하지만, PDF 원본 코드와 수정된 실행 코드를 한 코드블록 안에서 섞어 🟦처럼 보이게 하는 곳이 있다. 수정 코드 자체는 독자에게 유용하지만, "PDF 원문"이 아니라 "PDF 기반 수정본"임을 더 분명히 표시해야 한다.

## 2. PDF 페이지-주제 맵

| PDF 페이지 | PDF 주제 | 초안 반영 상태 | 조치 |
|---|---|---|---|
| p.1 | 표지: DAY3 자연어 처리를 위한 언어 모델 | 반영 | 유지 |
| p.2 | 목차: BERT 개념/구조, 이진 분류, 다중 분류, 감성 분류 | 반영 | 유지 |
| p.3-p.4 | BERT 정의, 2018 Google, 33억 단어, Wikipedia+BookCorpus, Base/Large 구조, Hugging Face | 거의 반영 | "현재 가장 널리" 표현 완화 |
| p.5-p.7 | 한국어 영화평 이진 분류 데이터 로드, train/test/valid 분리, Dataset 역할, tokenizer 다운로드 | 반영, 일부 수정 보충 | 수정 코드 표기 정리, 실행 순서 보완 |
| p.8-p.10 | `BertForSequenceClassification`, `requires_grad` 동결 전략, `TrainingArguments`, Test predict | 반영 | `model_dir`, `compute_metrics`, `eval_strategy` 버전 처리 보완 |
| p.11-p.13 | 뉴스 8종 다중 분류, 데이터 출처, label별 200개, `num_labels=8`, TrainingArguments | 핵심 반영, 출처 일부 누락 | 뉴스 데이터 출처와 다중 metrics 코드 추가 |
| p.14-p.16 | Hugging Face/Transformers 설명, 모델 다운로드, 구조와 파라미터 출력, 동결 전략 | 재배치해 반영 | 파라미터 출력 설명은 선택적으로 1문장 추가 |
| p.17-p.21 | IMDB 감성 분류, `bert-base-uncased`, 호환 tokenizer, split, Dataset, TrainingArguments, Test predict | 반영, PDF 오류 지적 포함 | IMDB 라벨 매핑과 shape 설명 강화 |

## 3. 높은 우선순위의 오류와 수정사항

### High 1. `compute_metrics`와 `model_dir` 정의 순서 오류

- 위치: 초안 §6.5 학습·평가 코드
- 문제: `Trainer(..., compute_metrics=compute_metrics)`가 나오기 전에 `compute_metrics`가 정의되지 않았다. `TrainingArguments(output_dir=model_dir, ...)`도 `model_dir` 정의가 없다.
- 영향: 독자가 위에서 아래로 실행하면 `NameError: name 'model_dir' is not defined` 또는 `NameError: name 'compute_metrics' is not defined`가 발생한다.
- 수정: `model_dir`와 `compute_metrics`를 `TrainingArguments`/`Trainer`보다 먼저 둔다.

### High 2. IMDB `sentiment` 문자열 라벨 처리 누락

- 위치: 초안 §8, §6.2 Dataset 재사용 설명
- 문제: PDF의 IMDB 파일명은 흔히 쓰이는 `IMDB Dataset.csv` 흐름이고, 이 데이터는 보통 `review`, `sentiment` 두 열과 `positive`/`negative` 문자열 라벨을 갖는다. 초안처럼 `torch.tensor(self.sentiments[idx])`에 문자열이 들어가면 텐서 변환이 실패한다.
- 영향: IMDB 실습이 바로 실행되지 않는다. `BertForSequenceClassification`의 단일 라벨 분류는 정수 class id 라벨을 기대한다.
- 수정: IMDB에서는 `dataset["label"] = dataset["sentiment"].map({"negative": 0, "positive": 1})`를 먼저 만들고 Dataset에는 `label`을 넘긴다. Dataset 반환도 `torch.tensor(..., dtype=torch.long)`로 명시한다.

### High 3. PDF 원문 코드와 수정 코드 표기 혼합

- 위치: 초안 §6.1 `train_test_split`, §6.5 `TrainingArguments(eval_strategy=...)`, §6.4 `layer.11` 동결 코드
- 문제: 설명은 🟦로 시작하지만 코드블록은 PDF 원문이 아니라 초안 작성자가 고친 실행 가능 코드다. 의도는 좋지만, 독자는 어떤 줄이 PDF 원문이고 어떤 줄이 보충 수정인지 혼동할 수 있다.
- 영향: PDF 대조 기준에서 "PDF에 나온 코드"와 "수정 제안 코드"의 경계가 흐려진다.
- 수정: 코드블록 제목을 `🟩 PDF 코드의 수정본`처럼 바꾸고, PDF 원문 오류는 짧은 인용/요약으로 따로 적는다.

### High 4. `evaluation_strategy`와 `eval_strategy`는 버전 고정 설명이 필요

- 위치: 초안 §6.5, §9
- 검증: PDF는 `evaluation_strategy="steps"`를 사용한다. Hugging Face Transformers 최신 문서 계열에서는 평가 전략 관련 이름이 `eval_strategy`로 노출되고, 구버전 문서/코드에서는 `evaluation_strategy`가 쓰였다.
- 문제: 초안의 "최신 `transformers`는 `evaluation_strategy` 대신 `eval_strategy`"라는 방향은 타당하지만, 독자가 설치한 버전에 따라 어느 쪽이 동작하는지 달라진다.
- 수정: 실행 코드에는 한 가지 버전을 명확히 택하고, "설치 버전에 따라 둘 중 하나만 사용"이라고 적는다. 가장 안전한 입문자 문구는 `transformers.__version__` 확인 후 맞춰 쓰라는 안내다.

## 4. PDF 기준 누락 내용

### Medium 1. 뉴스 다중 분류 데이터 출처가 축약됨

- PDF: p.11에서 뉴스 데이터 출처를 공공데이터포털, 한국언론진흥재단 뉴스빅데이터 메타데이터, 올림픽 2021로 설명한다.
- 초안: `news.csv`, 8개 카테고리, 각 200개만 반영했다.
- 조치: 데이터 출처를 한 문장으로 추가하면 PDF 충실도가 좋아진다.

### Low 1. 모델 파라미터 출력 설명이 짧음

- PDF: p.16에서 `for name, param in model.bert.named_parameters(): print(name, param)`로 각 Layer의 Weight/Bias Tensor가 출력된다고 설명한다.
- 초안: 구조 이름 출력 요약은 있으나 Weight/Bias Tensor 출력 의미는 거의 없다.
- 조치: 입문자에게 "이 출력은 학습 가능한 파라미터 이름과 텐서값/shape를 확인하는 용도"라는 1문장만 추가하면 충분하다. 긴 텐서 출력 예시는 불필요하다.

## 5. 더 자세히 설명할 내용

### Medium 1. `BertForSequenceClassification`의 출력층과 손실

- 초안은 다중 분류에서 "출력층이 클래스 수만큼의 logits, 손실은 내부적으로 교차 엔트로피"라고 잘 적었다.
- 보완: 이진 분류도 `num_labels=2`인 단일 라벨 분류로 보면 logits shape가 `(batch_size, 2)`이고, `labels`가 들어오면 모델이 classification loss를 반환한다는 점을 §6에 먼저 설명하면 좋다.
- 공식 문서 근거: Hugging Face BERT 문서에서 `BertForSequenceClassification`은 `labels` 제공 시 classification/regression loss를 반환한다고 설명한다.

### Medium 2. 데이터 분리와 토크나이저 적용의 누수 설명을 더 정확히

- 초안: "반드시 나눈 뒤 각 세트에 토크나이저를 적용"이라고 했다.
- 평가: 방향은 안전하지만, BERT의 `from_pretrained` 토크나이저는 이미 학습된 고정 vocabulary를 쓰므로 `fit`되는 전처리와 같은 의미의 데이터 누수는 아니다. 다만 Dataset 객체 생성과 평가 절차는 train/valid/test를 분리한 뒤 해야 한다.
- 수정: "새 vocabulary를 학습하는 경우에는 train에만 fit해야 하고, BERT의 사전학습 tokenizer는 고정 vocab이므로 분리 후 각 split에 같은 tokenizer를 적용한다"로 정밀화한다.

### Medium 3. 다중 분류 평가 지표 코드

- 초안은 `average="macro"` 권장을 말로만 제시한다.
- 보완: 8개 뉴스 분류용 `compute_metrics_multiclass` 예시를 붙이면 정확도 외 precision/recall/F1 요구를 더 잘 충족한다.
- 공식 문서 근거: scikit-learn `precision_recall_fscore_support`는 multiclass에서 `average="macro"`를 각 label별 metric의 단순 평균으로 계산한다고 설명한다.

## 6. 유용한 추가 내용

### 유지 권장

- 🟩 `train_test_split` 반환값 개수 보충: 사실상 맞다. scikit-learn 공식 문서는 반환 리스트 길이가 `2 * len(arrays)`라고 설명한다. 로컬 최소 예제에서도 배열 1개 입력은 2개, 배열 2개 입력은 4개를 반환했다.
- 🟩 `layer.23` 대 `layer.11` 지적: 맞다. `kykim/bert-kor-base` config는 `num_hidden_layers=12`, `hidden_size=768`, `num_attention_heads=12`이므로 마지막 encoder layer 인덱스는 11이다. `bert-base-uncased`도 12층이다.
- 🟩 IMDB shape 지적: 타당하다. PDF p.18의 `(84000, 3)`, `(36000, 3)`, `(30000, 3)`은 p.6의 한국어 영화평 split 결과와 동일하며, `IMDB Dataset.csv` 흐름의 일반적 데이터 규모/열 구조와 맞지 않는다. "앞 절 수치 재사용으로 보인다"는 표현처럼 단정 대신 추정형을 유지하면 좋다.
- 🟩 토큰화 개념 데모: 실제 BERT 토큰화가 아니라는 한계를 명시했으므로 입문자 보충으로 유용하다.
- 🟩 PyTorch + Transformers 일관성 안내: PDF가 Hugging Face/Trainer 중심이므로 적절하다.

## 7. 줄이거나 제거할 내용

### Medium 1. "현재 가장 널리 쓰이는" 표현 완화

- 위치: 초안 도입부
- 문제: BERT는 중요하고 널리 쓰인 모델이지만, 2026년 기준 "현재 가장 널리 쓰이는"은 시간에 민감하고 검증이 필요한 표현이다.
- 수정: "대표적인 사전훈련 언어 모델 중 하나" 또는 "현대 NLP를 이해할 때 반드시 거치는 대표 모델" 정도가 안전하다.

### Low 1. "처음부터 학습할 필요 없이" 표현 완화

- 위치: §2 핵심 직관
- 문제: 실무적으로는 대개 사전학습 모델을 가져오면 되지만, 도메인/언어/라이선스/성능 요구에 따라 사전학습이나 추가 pretraining이 필요할 수 있다.
- 수정: "대부분의 입문 실습에서는 처음부터 사전훈련하지 않고"로 제한한다.

### Low 2. "일부만 학습하면 빠르고 안정적" 표현 완화

- 위치: §4 동결 설명
- 문제: 일부 동결은 메모리/속도/과적합 면에서 유리할 수 있지만, 항상 안정적이거나 성능이 좋지는 않다.
- 수정: "상대적으로 빠르고 과적합 위험을 줄일 수 있다"로 바꾼다.

## 8. 코드 검토 상세

| 항목 | 평가 | 수정 필요 |
|---|---|---|
| import | `pandas`, `train_test_split`, `torch`, `BertTokenizerFast`, `BertForSequenceClassification`, `TrainingArguments`, `Trainer`, metrics import가 대체로 포함됨 | `compute_metrics`를 Trainer보다 먼저 배치 |
| 변수 정의 | `bert_model_name`은 §6.2에서 정의되어 이후 사용 가능 | `model_dir` 누락 |
| 실행 순서 | Dataset -> model -> args -> trainer 흐름은 적절 | `compute_metrics` 후정의 문제 |
| 출력 해석 | PDF 출력과 직접 실행 출력 구분은 대체로 적절 | PDF 수정 코드와 원문 코드 표기 분리 |
| 미실행 처리 | `torch`/`transformers` 미설치로 미실행이라고 밝힌 것은 적절 | 실행하지 않은 출력은 계속 만들지 말 것 |
| 데이터 split | stratify, valid/test 분리, Test 최종 평가 원칙을 잘 지적 | IMDB split에도 `stratify=dataset.label` 반영 |
| tokenizer/model 호환 | `bert-base-uncased`와 `bert-base-uncased` tokenizer 호환 주의 반영 | 한국어/영어 모델별 tokenizer 변수명 혼동 주의 |
| labels/loss | 다중 분류 자동 CE 설명은 좋음 | 모든 labels를 `torch.long` 정수로 명시 |
| freeze | `requires_grad` 전략과 Base 마지막 층 `layer.11` 지적은 맞음 | "pooler만"이 아니라 classifier head는 기본적으로 학습됨을 함께 설명하면 더 정확 |

## 9. 사실관계 검증

### BERT 기본 사실

- Google이 2018년에 공개한 사전훈련 언어 모델: 맞다.
- 사전훈련 말뭉치: BookCorpus 8억 단어 + English Wikipedia 25억 단어 = 약 33억 단어: PDF와 BERT 논문 기준으로 맞다.
- BERT-Base: L=12, D/hidden size=768, attention heads=12, 110M: 맞다.
- BERT-Large: L=24, hidden size=1024, attention heads=16, 340M: 맞다.
- 초안의 "2018년 당시 여러 벤치마크 기준" 완화는 적절하다. PDF의 "최고의 성능"을 현재까지 항상 최고라고 확대하지 않은 점은 좋다.

### 초안이 지적한 PDF 코드 불일치

| 지적 | 검증 결과 | 우선순위 |
|---|---|---|
| `train_test_split`에 배열 1개를 넘기면 반환값 2개 | 맞다. 공식 문서상 반환 길이는 `2 * len(arrays)`이다. PDF p.5의 4개 언패킹은 그대로 실행하면 오류다. | High |
| `bert-kor-base`는 BERT-Base라 `layer.23`이 아니라 `layer.11` | 맞다. `kykim/bert-kor-base` config는 12 hidden layers다. | High |
| IMDB shape `(84000/36000/30000)`은 앞 절 재사용으로 보임 | 타당하다. PDF p.18 shape가 p.6과 동일하고 IMDB CSV 흐름과 맞지 않는다. 추정형 표현 유지 권장. | Medium |
| `evaluation_strategy` vs `eval_strategy` 버전 차이 | 대체로 맞다. PDF는 구버전식 `evaluation_strategy`, 최신 문서/코드 계열은 `eval_strategy`를 쓴다. 설치 버전을 명시해야 한다. | High |

## 10. 바로 붙여 넣을 수 있는 수정 블록

### 10.1 도입부 표현 완화

```markdown
> 📝 DAY1에서 DNN~Transformer 모델 지도를, DAY2에서 텍스트 분류를 봤습니다.
> 이번 DAY3에서는 현대 NLP를 이해할 때 반드시 거치는 대표적인 사전훈련 언어 모델인 **BERT**를 다룹니다.
> 핵심은 **"이미 대규모 텍스트로 학습된 모델을 내려받아, 내 문제에 맞게 추가 학습(파인튜닝)한다"** 는 것입니다.
```

### 10.2 PDF 코드와 수정 코드 표기 분리

```markdown
🟦 PDF는 `train_test_split(dataset.index, ...)`처럼 입력 배열을 1개만 넘긴 뒤
`train_idx, test_idx, _, _` 네 변수로 받는 형태를 제시합니다.

🟩 그러나 scikit-learn의 `train_test_split`은 입력 배열이 1개이면
`train`, `test` 두 개만 반환합니다. 실행 가능한 수정본은 아래처럼 2개로 받습니다.
```

```python
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    dataset.index,
    test_size=0.2,
    stratify=dataset.sentiment,
    random_state=42,
)
```

### 10.3 `compute_metrics`와 `model_dir` 선정의

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer

model_dir = "./bert-kor-sentiment"

def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": p,
        "recall": r,
        "f1": f1,
    }

training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="steps",  # 구버전 transformers에서는 evaluation_strategy 사용
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set_dataset,
    eval_dataset=valid_set_dataset,
    compute_metrics=compute_metrics_binary,
)
```

### 10.4 Dataset 라벨 dtype 명시

```python
class BertDataset(torch.utils.data.Dataset):
    def __init__(self, reviews, sentiments, tokenizer, max_len=512):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.reviews[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.sentiments[idx], dtype=torch.long),
        }
```

### 10.5 IMDB 라벨 매핑과 stratify 수정

```python
dataset = pd.read_csv("/content/mnt/MyDrive/IMDB Dataset.csv")
dataset["label"] = dataset["sentiment"].map({"negative": 0, "positive": 1})

train_idx, test_idx = train_test_split(
    dataset.index,
    test_size=0.2,
    stratify=dataset.label,
    random_state=42,
)

train_set = dataset.iloc[train_idx].reset_index(drop=True)
test_set = dataset.iloc[test_idx].reset_index(drop=True)

tr_idx, val_idx = train_test_split(
    train_set.index,
    test_size=0.3,
    stratify=train_set.label,
    random_state=42,
)

valid_set = train_set.iloc[val_idx].reset_index(drop=True)
train_set = train_set.iloc[tr_idx].reset_index(drop=True)
```

```python
train_set_dataset = BertDataset(
    train_set.review.tolist(), train_set.label.tolist(), tokenizer
)
valid_set_dataset = BertDataset(
    valid_set.review.tolist(), valid_set.label.tolist(), tokenizer
)
test_set_dataset = BertDataset(
    test_set.review.tolist(), test_set.label.tolist(), tokenizer
)
```

### 10.6 다중 분류 metrics 예시

```python
def compute_metrics_multiclass(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
    }
```

### 10.7 토크나이저 누수 설명 정밀화

```markdown
> 🟩 데이터 누수 주의:
> BERT의 `from_pretrained` 토크나이저는 이미 학습된 고정 vocabulary를 사용하므로,
> 이 예제에서 토크나이저를 불러오는 행위 자체가 내 train/test 데이터에 fit되는 것은 아닙니다.
> 그래도 데이터셋 객체와 평가는 Train/Validation/Test를 먼저 나눈 뒤 각각 만들고,
> Test는 모델 선택이나 하이퍼파라미터 조정에 쓰지 않고 마지막 평가에만 사용합니다.
> 만약 새 vocabulary나 전처리 통계를 직접 학습한다면 반드시 Train 데이터에만 fit해야 합니다.
```

## 11. 우선순위 표

| 우선순위 | 항목 | 이유 | 권장 조치 |
|---|---|---|---|
| High | `compute_metrics`, `model_dir` 정의 순서 | 그대로 실행 시 NameError | Trainer 생성 전 정의 |
| High | IMDB 문자열 라벨 처리 | `torch.tensor("positive")` 형태는 실패 | 0/1 label 매핑 후 `dtype=torch.long` |
| High | PDF 원문/수정 코드 표기 혼합 | source fidelity 저하 | 수정 코드는 🟩 "수정본"으로 명시 |
| High | `evaluation_strategy`/`eval_strategy` 버전 | 설치 버전에 따라 코드 실행 실패 | 버전 확인 문구와 한 가지 실행 코드 선택 |
| Medium | 뉴스 데이터 출처 누락 | PDF p.11 내용 일부 누락 | 공공데이터포털/한국언론진흥재단 출처 1문장 추가 |
| Medium | 다중 분류 metrics 코드 부재 | 말로만 macro F1 권장 | `compute_metrics_multiclass` 추가 |
| Medium | 토크나이저 누수 설명 과단정 | BERT tokenizer는 고정 vocab | "fit 전처리"와 "고정 tokenizer 적용" 구분 |
| Medium | 출력층/손실 설명 위치 | 이진 분류 초반 설명 부족 | `num_labels=2`, logits, labels, loss 설명 추가 |
| Low | 모델 파라미터 출력 설명 | PDF p.16 세부 내용 축약 | Weight/Bias Tensor 확인 용도 1문장 추가 |
| Low | 과장 표현 | "현재 가장 널리", "처음부터 필요 없이" | 대표 모델/대부분 입문 실습으로 완화 |

## 12. 최종 권고

초안은 PDF 핵심 내용과 주요 오류 지적을 잘 반영했으므로 구조를 크게 바꿀 필요는 없다. 다만 최종본 전에는 High 항목을 먼저 고쳐야 한다. 특히 `compute_metrics` 정의 순서, `model_dir`, IMDB 라벨 매핑, `labels` dtype, PDF 원문과 수정 코드의 표기 분리는 실행 가능성과 출처 충실도에 직접 영향을 준다.

중간 우선순위로는 뉴스 데이터 출처, 다중 분류 macro metrics 코드, 토크나이저 누수 설명 정밀화를 반영하길 권한다. 낮은 우선순위 항목은 문장 완화와 짧은 설명 추가 수준이면 충분하다.

## 참고한 공식/원문 자료

- PDF 원문: `sources/DAY3_자연어처리_언어모델.pdf`
- scikit-learn `train_test_split`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- scikit-learn `precision_recall_fscore_support`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
- Hugging Face `BertForSequenceClassification`: https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertForSequenceClassification
- Hugging Face `TrainingArguments`: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
- `kykim/bert-kor-base` config: https://huggingface.co/kykim/bert-kor-base/blob/main/config.json
- `google-bert/bert-base-uncased` model card/config: https://huggingface.co/google-bert/bert-base-uncased
- BERT 논문: https://arxiv.org/abs/1810.04805
