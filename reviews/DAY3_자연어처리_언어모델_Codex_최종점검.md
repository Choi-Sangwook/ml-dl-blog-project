# DAY3 자연어처리 언어모델 Codex 최종점검

- 기준 PDF: `sources/DAY3_자연어처리_언어모델.pdf`
- 기존 통합검토안: `reviews/DAY3_자연어처리_언어모델_Codex_통합검토안.md`
- 최종 포스트: `posts/DAY3_자연어처리_언어모델_최종본.md`
- 점검 방식: PyMuPDF로 PDF 21페이지 텍스트를 페이지별 추출하고, 전체 페이지를 썸네일 렌더링해 코드/표/구조 누락 여부를 확인했다. PDF 전 페이지에 텍스트 레이어가 충분했고 이미지 전용 페이지는 없었다.
- 코드 실행 여부: `torch`, `transformers`가 현재 환경에 설치되어 있지 않아 BERT 학습 코드는 실행하지 못했다. 대신 모든 Python 코드블록은 `ast.parse`로 문법을 확인했고, 순수 파이썬 토큰화 데모는 최종본의 코드블록을 그대로 추출해 실행했다.

## 게시 가능

조건부 판정: 통합검토안의 High/Medium 대부분과 PDF 핵심 내용은 최종본에 반영되었다. 다만 아래 "게시 전 반드시 수정"의 IMDB 토크나이저 순서 1건은 코드 흐름과 모델-토크나이저 호환성에 직접 영향을 주므로, 그 1건 수정 후 게시 가능하다.

### 통합검토안 High 4건 반영 상태

| 항목 | 반영 여부 | 확인 내용 |
|---|---:|---|
| High1. `compute_metrics`, `model_dir` 선정의 | 반영 | §6.5에서 `model_dir`와 `compute_metrics_binary`를 `TrainingArguments`/`Trainer`보다 먼저 정의한다. 위에서 아래로 실행할 때 이 부분의 `NameError` 위험은 해소되었다. |
| High2. IMDB 문자열 라벨 매핑 + `torch.long` | 대부분 반영 | §8에서 `negative/positive`를 0/1로 매핑하고, §6.2 Dataset이 `labels: torch.tensor(..., dtype=torch.long)`를 반환한다. 단, IMDB Dataset 생성 전 영어 토크나이저 정의 순서는 아래 필수 수정 대상이다. |
| High3. PDF 원문 코드와 실행용 수정본 표기 분리 | 반영 | `train_test_split`, `layer.23`, `TrainingArguments`, IMDB shape 등은 PDF 원문/출력은 🟦, 실행 가능 수정본은 🟩로 분리되어 있다. |
| High4. `evaluation_strategy`/`eval_strategy` 버전 안내 + 실행코드 택1 | 반영 | 실행 코드에는 `eval_strategy="steps"` 하나를 택했고, 구버전은 `evaluation_strategy`를 쓰라는 안내와 `transformers.__version__` 확인 문구가 있다. Hugging Face 현재 문서도 `Trainer`/`TrainingArguments` API와 버전별 문서를 제공하며, 현재 문서 검색 기준 `eval_strategy` 계열 안내가 더 맞다. |

### Medium 반영 상태

| 항목 | 반영 여부 | 확인 내용 |
|---|---:|---|
| 뉴스 데이터 출처 | 반영 | §7에 공공데이터포털, 한국언론진흥재단 뉴스빅데이터 메타데이터, 올림픽 2021, `data.go.kr`가 들어갔다. |
| 다중분류 macro metrics 코드 | 반영 | §7에 `compute_metrics_multiclass`가 있고 `average="macro"`, `macro_precision`, `macro_recall`, `macro_f1`를 반환한다. |
| 토크나이저 누수 설명 정밀화 | 반영 | §6.1에서 BERT `from_pretrained` 토크나이저는 고정 vocabulary라 데이터에 새로 `fit`되는 전처리가 아니며, split별 Dataset 생성과 Test 최종 평가 원칙을 구분한다. |
| 이진분류 출력층/손실 설명 | 반영 | §6.4에서 `num_labels=2`, logits shape `(batch_size, 2)`, `labels` 제공 시 교차 엔트로피 손실 자동 계산을 설명한다. Hugging Face BERT 문서의 sequence classification 출력 설명과 부합한다. |

### PDF 핵심 내용 누락 여부

최종본은 PDF의 핵심 흐름을 모두 포함한다.

| PDF 범위 | 핵심 내용 | 최종본 상태 |
|---|---|---|
| p.1-p.4 | BERT 개념, 2018 Google, 33억 단어, Wikipedia/BookCorpus, Base/Large 구조, Hugging Face | 반영 |
| p.5-p.10 | 한국어 영화평 이진 분류, split, Dataset, tokenizer, `BertForSequenceClassification`, `requires_grad`, `TrainingArguments`, Test predict | 반영, PDF 코드 오류는 🟩 수정본으로 보완 |
| p.11-p.13 | 뉴스 8종 다중 분류, 데이터 출처, label별 200개, `num_labels=8`, 학습 인자 | 반영 |
| p.14-p.16 | Hugging Face/Transformers, 모델 다운로드, 구조/파라미터 출력, 동결 전략 | 반영 |
| p.17-p.21 | IMDB 감성 분류, `bert-base-uncased`, 호환 tokenizer, split, Dataset, TrainingArguments, Test predict | 대체로 반영, 토크나이저 정의 순서만 필수 수정 필요 |

### 기술/마크다운 상태

- 과도한 단정은 대체로 완화되었다. "최고의 성능"은 2018년 당시 기준으로 제한했고, "항상 최고"가 아니라고 명시했다.
- `layer.11` 지적은 정확하다. PDF의 `layer.23`은 Large 24층 기준 마지막 레이어이고, 최종본의 실습 모델 `kykim/bert-kor-base`와 `bert-base-uncased`는 Base 계열 12층이므로 마지막 encoder layer는 `layer.11`이다.
- `requires_grad`, `input_ids`, `attention_mask`, `labels(torch.long)`, `num_labels`, logits/CE 설명은 큰 기술 오류 없이 일관된다.
- Markdown 코드 fence는 36개로 짝이 맞고, Python 코드블록 15개는 모두 문법 파싱을 통과했다.
- UTF-8로 읽었을 때 한글 깨짐(`U+FFFD` replacement character)이나 명백한 mojibake는 없었다.
- 표 구조와 링크 문법은 정상이다.
- 최종본은 실행하지 않은 BERT 학습 결과를 새로 만들어 쓰지 않고, PDF 출력 또는 직접 실행한 토큰화 데모만 구분해 표시한다.

## 게시 전 반드시 수정

### 1. IMDB 절에서 영어 토크나이저를 Dataset 생성보다 먼저 정의해야 함

- 위치: `posts/DAY3_자연어처리_언어모델_최종본.md` §8, IMDB 라벨 매핑 코드블록 직후
- 문제: IMDB 코드블록은 `train_set_dataset = BertDataset(..., tokenizer)`를 먼저 실행하고, 그 다음 코드블록에서 `tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")`를 정의한다.
- 영향:
  - §8만 따로 실행하면 `tokenizer`가 아직 없어 `NameError`가 난다.
  - 글 전체를 위에서 아래로 실행하면 §6.2에서 이미 정의된 한국어 `kykim/bert-kor-base` 토크나이저가 남아 있어 NameError는 피하지만, IMDB 영어 모델 `bert-base-uncased`와 맞지 않는 토크나이저로 Dataset을 만들 수 있다.
  - 이는 사용자가 요청한 "tokenizer-모델 호환"과 "위에서 아래로 실행 시 정의 순서" 점검에서 실제 게시 전 수정이 필요한 항목이다.

권장 수정 방향:

```python
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_set_dataset = BertDataset(
    train_set.review.tolist(), train_set.label.tolist(), tokenizer)
valid_set_dataset = BertDataset(
    valid_set.review.tolist(), valid_set.label.tolist(), tokenizer)
test_set_dataset = BertDataset(
    test_set.review.tolist(), test_set.label.tolist(), tokenizer)
```

또는 기존 `bert_token_model = "bert-base-uncased"` / `bert_model_name = bert_token_model` 흐름을 먼저 둔 뒤 Dataset을 생성하면 된다.

## 선택적으로 개선

1. `eval_strategy` 설명의 "최신 버전" 표현은 시간에 민감하다. 현재 Hugging Face 공식 문서 기준으로는 괜찮지만, 게시 후 오래 유지될 글이라면 "현재 문서 기준" 또는 "설치된 `transformers` 버전에 따라"로 더 안전하게 완화할 수 있다. 확인한 공식 문서: https://huggingface.co/docs/transformers/main_classes/trainer

2. IMDB 절의 Dataset 생성 예시는 `train_set_dataset`만 만든다. §6과 동일 흐름이라고 설명되어 있어 치명적 누락은 아니지만, 게시용 코드 일관성을 높이려면 `valid_set_dataset`, `test_set_dataset`까지 함께 제시하는 편이 낫다.

3. 뉴스 데이터 출처는 PDF 핵심을 반영했지만, 실제 독자가 데이터를 찾으려면 `data.go.kr` 일반 링크보다 검색어(예: "한국언론진흥재단 뉴스빅데이터 메타데이터 올림픽 2021")를 한 문장 덧붙이면 재현성이 좋아진다.

4. BERT 학습 코드는 현재 환경에서 실행하지 못했다(`torch`, `transformers` 미설치). 최종 게시 직전 실제 Colab/로컬 환경에서 최소 1 epoch 또는 작은 샘플로 `BertDataset` 생성, `Trainer` 초기화, `trainer.predict()`까지 한 번 확인하면 좋다.

최종 결론: 현재 최종본은 핵심 반영 상태가 좋지만, IMDB 토크나이저 정의 순서 1건은 게시 전 반드시 수정해야 한다. 이 1건을 고치면 게시 가능하다.
