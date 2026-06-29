# 🤖 자연어처리(NLP) 완전 입문 가이드 — DAY1. 자연어 처리를 위한 딥러닝 모델

> **시리즈**: 파이썬 기본만 있는 사람을 위한 자연어처리(NLP) 입문
> **교과목**: 초거대언어모델(LLM) · 단원 2 — 자연어 딥러닝
> **이전 편(단원 1)**: 자연어 처리 절차와 형태소 분석 → 워드 임베딩

> 📝 단원 1에서는 문장을 **토큰·형태소로 쪼개고**, 그 단어를 **숫자(워드 임베딩)로 바꾸는** 준비 과정을 배웠습니다. 이번 단원 2의 첫 글에서는, 그렇게 숫자로 바뀐 문장을 실제로 학습하는 **딥러닝 모델들** — DNN, CNN, RNN, LSTM, GRU, Seq2Seq, Attention, Transformer, BERT, GPT — 을 **한눈에 정리**하고, 그중 CNN과 RNN으로 문장을 분류하는 방법까지 살펴봅니다.

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 개념·코드·표
> - 🟩 **(보충)** : 입문자를 위해 글쓴이가 덧붙인 설명·실습 코드·실무 주의점
>
> ⚠️ **코드 실행 안내**: 이 글의 작성 환경에는 **TensorFlow·PyTorch가 설치되어 있지 않아** CNN(Keras)·RNN(PyTorch) 코드는 **직접 실행하지 않았고, 출력값을 지어내지 않았습니다**(직접 실행 시 환경에 따라 결과가 달라집니다).
> 반면 **NumPy로 만든 1D 합성곱 데모**(§5.3)는 추가 설치가 필요 없어 **글쓴이가 직접 실행한 실제 출력**입니다.
> 또한 강의의 RNN 코드는 `torchtext.legacy` 등 **현재 버전에서 제거·폐기된 API**를 사용합니다. 해당 부분은 **강의 흐름을 옮긴 레거시 코드 요약**으로 제시하며, 지금 그대로는 실행되지 않는다는 점을 분명히 표시했습니다.

---

## 1. 이번 DAY에서 배우는 것

이번 글의 목표는 **"자연어 처리에 어떤 딥러닝 모델들이 있고, 각각 무엇을 잘하는가"** 를 정리하는 것입니다.

- 🟦 자연어 처리(NLP)란 무엇이고, 왜 딥러닝이 필요한가
- 🟦 문장이 모델에 들어가기까지의 전체 처리 과정
- 🟦 대표 딥러닝 모델 10가지(DNN ~ GPT)의 특징과 발전 흐름
- 🟦 문장 분류를 위한 **CNN** 구조와 실습 흐름
- 🟦 순서가 있는 데이터를 다루는 **RNN/LSTM/GRU**와 PyTorch Lightning 실습 흐름

> 🟩 **참고**: 이번 강의는 **여러 모델을 빠르게 훑는 "지도(map)" 성격**의 자료입니다. 각 모델의 깊은 수학·내부 구조보다는 **"언제 무엇을 쓰는지"** 를 잡는 데 초점을 둡니다. 개별 모델의 자세한 구현은 이후 DAY에서 다룹니다.

---

## 2. 자연어 처리란? 그리고 왜 딥러닝이 필요한가 🟦

**자연어 처리(NLP, Natural Language Processing)** 는 사람이 쓰는 언어(한국어, 영어 등)를 컴퓨터가 **이해·분석·생성**하도록 하는 기술입니다. 과거에는 사람이 직접 규칙을 만들어 프로그램을 짰지만, 지금은 대부분 **대량의 문장을 학습해 스스로 언어의 특징을 배우는** 딥러닝 방식이 쓰입니다.

🟦 자연어 처리가 쓰이는 대표적인 일:

- 문장 분류 · 감성 분석 · 기계 번역 · 문서 요약 · 질의응답 · 챗봇 · 음성 인식 · 문장 생성

그렇다면 왜 단순한 규칙이 아니라 딥러닝이 필요할까요? 사람의 언어는 **같은 글자라도 상황에 따라 의미가 달라지기** 때문입니다.

> 🟦 (강의 예시)
> - "**배**가 아프다" → 배 = 신체
> - "**배**를 먹었다" → 배 = 과일

컴퓨터는 글자만 보고는 이 둘을 구분하기 어렵습니다. 그래서 딥러닝 모델은 단어 하나만 보는 것이 아니라 **문맥(Context)**, **주변 단어**, **문장의 구조** 를 함께 학습해 의미를 추론합니다.

---

## 3. 문장이 모델에 들어가기까지: 전체 과정 🟦

강의가 제시하는 자연어 처리 딥러닝의 전체 파이프라인입니다.

```text
문장 입력
  ↓ 텍스트 전처리
  ↓ 형태소 분석
  ↓ 토큰(Token) 생성
  ↓ 단어 번호 부여(정수 인코딩)
  ↓ 임베딩(Embedding)        ← 단어 번호를 의미 벡터로 변환
  ↓ 딥러닝 모델
  출력(분류/번역/생성 등)
```

> 🟦 (강의 예시) "오늘 점심을 먹었다."
> 1. **토큰**: `오늘`, `점심`, `먹었다`
> 2. **번호 부여**: `오늘→13`, `점심→85`, `먹었다→271`
> 3. **벡터 변환(임베딩)**: `13 → [0.34, 0.27, …, 0.18]`
> 4. 이 벡터들이 **딥러닝 모델의 입력**이 됩니다.

> 🟩 **연결 고리**: 1~4단계 중 **전처리·형태소·토큰**은 단원 1(형태소 분석), **번호 부여·임베딩**은 단원 1(워드 임베딩)에서 다뤘습니다. 이번 글은 그 결과를 받아 처리하는 **"딥러닝 모델" 부분**에 집중합니다.

---

## 4. 자연어 처리 대표 딥러닝 모델 총정리 🟦

강의가 소개하는 핵심 모델들을 발전 순서대로 정리합니다. 큰 흐름은 **"순서를 모르는 모델 → 순서를 기억하는 모델 → 중요한 곳에 집중하는 모델 → 병렬로 한 번에 보는 모델"** 입니다.

### 4.1 DNN (Deep Neural Network)

- 가장 기본적인 신경망. `입력층 → 은닉층 → 은닉층 → 출력층` 구조.
- **장점**: 구조가 간단하고 구현이 쉬움.
- **단점**: **문장의 순서를 모른다.**
  - 🟦 예: "나는 밥을 먹었다."와 "먹었다 나는 밥을."을 거의 같게 본다.

### 4.2 CNN (Convolutional Neural Network)

- 원래 **이미지 분석용**이지만 자연어 처리에도 많이 쓰임.
- **연속된 단어 패턴(n-gram)** 을 잘 찾는다.
  - 🟦 예: "정말 재미있는 영화"에서 "재미있는 영화"를 하나의 특징으로 학습.
- **활용**: 감성 분석, 문장 분류, 스팸 메일 분류. **속도가 빠르다.**
- 👉 자세한 구조와 실습은 **§5**에서 다룹니다.

### 4.3 RNN (Recurrent Neural Network)

- **문장의 순서를 기억**하는 모델. 앞 단어에서 본 정보를 다음 단어로 전달.
  - 🟦 예: `오늘 → 날씨가 → 좋다` 처럼 앞 정보를 이어받음.
- **단점**: **긴 문장**의 앞부분을 잘 기억하지 못함(장기 의존성 문제).

### 4.4 LSTM (Long Short-Term Memory)

- RNN의 "긴 문장을 못 기억하는" 단점을 보완.
- **핵심 아이디어**: 필요한 정보만 남기고 불필요한 정보는 버린다.
- 🟦 구조의 구성 요소: `Forget Gate → Input Gate → Cell State → Output Gate`
- **활용**: 번역, 음성 인식 등 긴 문장 처리.

### 4.5 GRU (Gated Recurrent Unit)

- **LSTM을 단순화**한 모델.

| 구분 | 게이트 수 |
|---|---|
| LSTM | 3개 |
| GRU | 2개 |

- **장점**: 속도가 빠르고 메모리를 적게 쓰며, 성능은 LSTM과 **대체로 비슷**.

### 4.6 Seq2Seq (Sequence to Sequence)

- **한 문장을 다른 문장으로 바꾸는** 모델. `Encoder → Context Vector → Decoder` 구조.
  - 🟦 예: "안녕하세요." → "Hello."
- **활용**: 번역, 요약, 챗봇.

### 4.7 Attention

- Seq2Seq의 단점(긴 문장을 **중간 벡터 하나**에 다 담기 어려움)을 보완.
- **필요한 단어를 다시 본다** = 중요한 단어에 가중치를 더 준다.
  - 🟦 예: "나는 어제 친구와 영화를 보았다."에서 "영화"를 번역할 때 "영화" 단어에 집중.
- **장점**: 긴 문장에서도 성능 향상.

### 4.8 Transformer

- **현재 자연어 처리의 핵심 모델.** RNN처럼 단어를 앞에서부터 하나씩 넘기는 구조 대신, **Attention을 중심으로** 문장을 처리한다.
- 🟦 구조: `입력 → Embedding → Positional Encoding → Multi-Head Attention → Feed Forward → 출력`
- **장점**: 매우 빠르고 **병렬 처리**가 가능하며 긴 문장도 잘 처리.
- 🟩 **보충**: RNN은 단어를 **앞에서부터 하나씩** 처리해 병렬화가 어렵지만, Transformer는 **문장 전체를 한 번에** 보기 때문에 학습이 빠릅니다. 대신 단어 순서 정보를 잃지 않으려고 **Positional Encoding**으로 위치를 따로 알려 줍니다. (구조에는 Attention 외에도 Feed-Forward 층, 잔차 연결·정규화 등이 함께 들어갑니다. "Attention만 쓴다"는 것은 *순환 구조를 쓰지 않는다*는 의미로 이해하면 됩니다.)

### 4.9 BERT (Bidirectional Encoder Representations from Transformers)

- Transformer의 **인코더(Encoder)** 구조를 사용.
- **특징**: 문장의 **양쪽 방향을 동시에** 읽어 문맥을 이해.
  - 🟦 예: "은행에 갔다."에서 앞뒤 문맥을 모두 참고해 '은행'의 의미를 파악.
- **활용**: 문장 분류, 질의응답, 개체명 인식, 검색.

### 4.10 GPT (Generative Pre-trained Transformer)

- Transformer의 **디코더(Decoder)** 구조를 사용하는 **생성형** 언어 모델.
- **동작**: 앞 단어들을 보고 **다음 단어를 예측**하는 것을 반복.
  - 🟦 예: "오늘은 날씨가" → "좋다" → "오늘은 날씨가 좋다. 그래서" → … (계속 생성)
- **활용**: 챗봇, 문장 생성, 코드 생성, 번역, 요약.

### 4.11 모델 발전 과정 🟦

```text
DNN → CNN → RNN → LSTM → GRU → Seq2Seq → Attention → Transformer
                                                         ├── BERT (이해)
                                                         └── GPT  (생성)
```

### 4.12 모델별 특징 비교표 🟦

| 모델 | 순서 정보 | 긴 문장 | 병렬 처리 | 주요 활용 |
|---|---|---|---|---|
| DNN | ✗ | ✗ | ○ | 기본 분류 |
| CNN | 부분적 | △ | ○ | 문장 분류, 감성 분석 |
| RNN | ○ | ✗ | ✗ | 시계열, 간단한 문장 |
| LSTM | ○ | ○ | ✗ | 번역, 음성 인식 |
| GRU | ○ | ○ | ✗ | 실시간 분석 |
| Seq2Seq | ○ | ○ | ✗ | 기계 번역 |
| Attention | ○ | ○ | 부분적 | 번역, 요약 |
| Transformer | ○(위치 정보 사용) | ○ | ○ | 대부분의 NLP 작업 |
| BERT | ○ | ○ | ○ | 문장 이해, 검색, 분류 |
| GPT | ○ | ○ | ○ | 텍스트 생성, 챗봇, 코드 생성 |

> 🟦 **실제 활용 사례**: 감성 분석(리뷰 긍·부정), 기계 번역, 챗봇·AI 비서, 문서 요약, 검색 시스템, 음성 비서, 코드 생성 보조 등. 강의는 **현재 많은 NLP 서비스가 Transformer 계열(BERT·GPT)을 기반으로 동작한다**고 설명합니다.

> 🟩 **주의(과장 금지)**: "Transformer가 항상 최고", "BERT가 모든 문제에서 우월" 같은 단정은 피하세요. 데이터 크기·과제·자원에 따라 **작은 CNN/RNN이 더 적합**할 때도 많습니다. 비교표의 ○/✗는 **경향**이지 절대 규칙이 아니며, "대부분·최신" 같은 표현은 시점에 따라 달라집니다.

---

## 5. 자연어 처리를 위한 CNN 🟦

> **이 절의 프레임워크: Keras / TensorFlow** (강의 §2 기준)

### 5.1 CNN의 기본 개념과 흐름

CNN(합성곱 신경망)은 **대상의 특징을 인식하는 방식을 모델링**한 신경망입니다. 원래 이미지용이지만, **연속된 단어 패턴**을 잡는 능력 덕분에 문장 분류에도 쓰입니다.

🟦 CNN의 기본 연산과 전체 흐름:

```text
Convolution → Pooling → Flattening 또는 Global Pooling → Fully Connected Layer → Softmax/Sigmoid
```

1. **컨볼루션(Convolution)**: 필터를 옮겨 가며 **특징을 추출**
2. **풀링(Pooling)**: 추출한 특징을 **압축**(대표값만 남김)
3. **플래트닝(Flattening)**: 2·3차원 특징 맵을 완전 연결층에 넣기 쉽게 **1차원으로 펼침**
4. **완전 연결층(Fully Connected Layer)**: 압축된 특징으로 **최종 분류**
5. **소프트맥스(Softmax)/시그모이드(Sigmoid)**: 어떤 분류에 속하는지 확률로 출력

> 🟩 **보충**: 텍스트 CNN에서는 `GlobalMaxPooling1D`가 전통적인 Flattening 역할을 일부 대신합니다. 모든 위치의 특징 중 **가장 강한 반응만** 남기므로 파라미터 수를 줄이는 효과도 있습니다.

### 5.2 컨볼루션 · 패딩 · 스트라이드 🟦

- **컨볼루션**: 입력 위에서 작은 **필터**를 이동시키며 곱하고 더해 특징 맵을 만든다. (예: 4×4 이미지가 2×2로 압축, **파라미터 공유**)
- **패딩(Padding)**: 입력 **주변을 특정 값(보통 0)으로 채워** 출력 크기를 조정하고 **가장자리 정보를 보존**.
- **스트라이드(Stride)**: 필터를 **몇 칸씩** 이동할지.

🟦 **출력 크기 공식** (입력 `n`, 필터 `f`, 스트라이드 `s`):

```text
출력 크기 = ((n - f) / s) + 1
```

| 입력 n | 필터 f | 스트라이드 s | 계산 | 출력 |
|---|---|---|---|---|
| 7 | 3 | 1 | ((7-3)/1)+1 = 5 | 5 |
| 7 | 3 | 2 | ((7-3)/2)+1 = 3 | 3 |
| 7 | 3 | 3 | ((7-3)/3)+1 = **2.33** | 2 |

> 🟩 **주의**: 마지막 행처럼 결과가 **딱 떨어지지 않으면(2.33)**, 실제 출력 크기는 **내림(floor)** 해서 2가 됩니다. 보통은 입력·필터·스트라이드를 **나누어떨어지게** 설계하거나, **패딩**으로 크기를 맞춥니다.

🟦 **풀링**: 컨볼루션 다음에 수행하며, 영역에서 **최댓값**을 고르는 **Max Pooling**과 **평균값**을 쓰는 **Average Pooling**이 있습니다. 다중 분류 출력에는 **Softmax**와 **Softmax Cross Entropy** 목적함수를 씁니다.

> 🟩 **컨볼루션 층을 깊게 쌓는 이유**: 얕은 층은 단순한 패턴을, 깊은 층은 더 복잡하고 추상적인(고차원 잠재) 특징을 학습할 수 있습니다. 다만 **깊다고 항상 좋은 것은 아니며**, 데이터 크기와 과적합 관리가 함께 필요합니다.

### 5.3 🟩 NumPy로 1D 합성곱 직접 보기 (실행 결과 O)

CNN이 "문장에서 패턴을 잡는다"는 말을 코드로 확인해 봅니다. **단어 5개짜리 문장**을 4차원 임베딩으로 표현하고, **2개 단어(2-gram)** 를 보는 필터를 한 칸씩 옮기며 점수를 계산합니다. (텍스트의 CNN은 이미지의 2D 합성곱과 달리 **시간축으로만 움직이는 1D 합성곱**을 씁니다.)

```python
import numpy as np

np.random.seed(42)

words = ["정말", "재미있는", "영화", "였다", "!"]
emb_dim = 4

embeddings = np.round(np.random.rand(len(words), emb_dim), 2)  # (단어수, 임베딩차원)
kernel = np.round(np.random.rand(2, emb_dim), 2)               # 2-gram 필터: (커널크기, 임베딩차원)

print("문장 임베딩 행렬 shape:", embeddings.shape)
for word, vector in zip(words, embeddings):
    print(f"{word:>6} -> {vector}")

feature_map = []
for i in range(len(words) - 2 + 1):   # stride=1
    window = embeddings[i:i + 2]      # 연속된 두 단어 (2, 4)
    score = round(float(np.sum(window * kernel)), 2)   # 원소별 곱 후 전부 더함
    feature_map.append(score)
    print(f"창[{words[i]} {words[i + 1]}] 점수 = {score}")

print("피처 맵:", feature_map)
print("GlobalMaxPooling(가장 강한 패턴):", max(feature_map))
```

실제 실행 출력:

```text
문장 임베딩 행렬 shape: (5, 4)
    정말 -> [0.37 0.95 0.73 0.6 ]
  재미있는 -> [0.16 0.16 0.06 0.87]
    영화 -> [0.6  0.71 0.02 0.97]
    였다 -> [0.83 0.21 0.18 0.18]
     ! -> [0.3  0.52 0.43 0.29]
창[정말 재미있는] 점수 = 1.45
창[재미있는 영화] 점수 = 1.79
창[영화 였다] 점수 = 1.51
창[였다 !] 점수 = 1.44
피처 맵: [1.45, 1.79, 1.51, 1.44]
GlobalMaxPooling(가장 강한 패턴): 1.79
```

> 🟩 **해석**: 필터를 옮길 때마다 **연속된 두 단어**의 점수가 하나씩 나옵니다(피처 맵). 그중 **가장 큰 값(1.79)** 만 남기는 것이 **GlobalMaxPooling** 입니다. 즉 "이 문장에서 **가장 강하게 반응한 2-gram 패턴**"만 뽑아 분류에 쓰는 셈입니다. 실제 CNN에서는 이 필터(가중치)가 **학습으로** 정해집니다. (여기 숫자는 `seed=42`로 만든 임의 값입니다.)

### 5.4 CNN 문장 분류 파이프라인 🟦

강의는 **정상 메일 vs 스팸 메일** 분류를 예로 듭니다.

```text
Word Embedding → Convolution → Max Pooling → Classification
```

🟦 실습 흐름(스팸 분류기):

1. **데이터 불러오기** → 불필요한 열 삭제
2. 영문/문자 데이터를 숫자로 변환
3. `data.info()`, 결측값 확인, **고유값(unique)·라벨 종류 확인**, **중복 제거**(`drop_duplicates()`)
4. 라벨 비율 확인(`value_counts()`) — 예: 정상 메일 약 80%
5. **`train_test_split()`** 로 훈련/평가 분리
6. **`Tokenizer`** 생성 → `fit_on_texts()` → `texts_to_sequences()`
7. 빈도가 적은 단어 제거, **`pad_sequences()`** 로 길이 맞추기(패딩)
8. 모델 구성 → 학습(`fit`) → 평가(`evaluate`)

🟩 6~7단계에서 나오는 값들의 의미(입문자용):

| 값 | 의미 |
|---|---|
| `word_to_index` | 단어 → 정수 번호 매핑 사전 |
| `vocab_size` | 사전에 들어 있는 **단어 종류 수**(임베딩 입력 크기) |
| `max_len` | 패딩으로 맞출 **문장 최대 길이**(모델 입력 길이) |
| 학습 데이터 shape | `(문장 수, max_len)` — 정수 인코딩된 문장들의 크기 |

> 🟩 **중요(데이터 누수 방지)**: `Tokenizer`(단어 사전)와 빈도 통계는 **반드시 훈련 데이터에만 `fit`** 해야 합니다. 평가 데이터까지 포함해 사전을 만들면 **테스트 정보가 학습에 새어 드는(data leakage)** 문제가 생깁니다. 강의 순서(5번 분리 → 6번 토크나이저)가 이 점에서 올바릅니다. 단, `texts_to_sequences()`는 검증·테스트 데이터에도 적용하되, **새 단어는 훈련 사전 기준**으로 처리(사전에 없으면 제외/미등록)된다는 점을 기억하세요.

🟦 모델 구성 요소(강의 기준):

| 구성 | 역할 |
|---|---|
| `Embedding` | 단어 번호 → 의미 벡터 |
| `Dropout(0.3)` | 과적합 방지(뉴런 일부를 임의로 끔) |
| `Conv1D` (+ Kernel Size) | n-gram 패턴 추출 |
| `GlobalMaxPooling1D` | 가장 강한 특징만 남김 |
| `Dense` | 최종 분류 |
| `EarlyStopping` | 검증 성능이 안 좋아지면 조기 종료 |
| `ModelCheckpoint` | 가장 좋은 모델 저장 |

🟦 **마지막 활성화 함수와 손실함수**(중요):

| 문제 | 마지막 활성화 | 손실함수(Keras) |
|---|---|---|
| 이진 분류(스팸 O/X) | `sigmoid` | `binary_crossentropy` |
| 다중 분류 | `softmax` | `categorical_crossentropy` 또는 `sparse_categorical_crossentropy` |

🟩 **참고용 Keras 코드(이 글에서는 미실행)** — 강의가 글로 설명한 구성을 입문자가 보기 쉽게 코드로 옮긴 것입니다. 실제 데이터·환경에 맞춰 수정해야 하며, **출력값은 지어내지 않았습니다.**

```python
# (보충) 환경: TensorFlow/Keras 필요 — 이 글에서는 실행하지 않음
import keras
from keras import layers

vocab_size = 10000   # 단어 사전 크기(훈련 데이터로 정함)
max_len = 200        # pad_sequences로 맞춘 문장 길이

model = keras.Sequential([
    keras.Input(shape=(max_len,), dtype="int32"),
    layers.Embedding(input_dim=vocab_size, output_dim=128),
    layers.Dropout(0.3),
    layers.Conv1D(filters=128, kernel_size=5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(1, activation="sigmoid"),     # 이진 분류(스팸 O/X)
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",                # 출력 sigmoid와 짝이 맞음
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)
model.summary()
```

> 🟩 **주의(평가)**: 스팸 분류처럼 **클래스가 불균형**(정상 80% : 스팸 20%)할 때는 **정확도(accuracy)만으로 판단하면 안 됩니다.** "전부 정상"이라고만 찍어도 정확도 80%가 나오기 때문입니다. 그래서 위 코드처럼 **precision·recall**(필요하면 F1·AUC)을 함께 보거나, 학습 후 scikit-learn의 `classification_report`로 클래스별 성능을 확인하세요. (강의는 "정확도가 높음"이라고만 언급하므로, 이 부분은 보충이 필요합니다.)

---

## 6. 자연어 처리를 위한 RNN 🟦

> **이 절의 프레임워크: PyTorch + torchtext + PyTorch Lightning** (강의 §3 기준)
> ⚠️ §5(CNN)는 Keras였지만, 이 절부터는 **PyTorch 계열로 프레임워크가 바뀝니다.** 두 절의 코드를 섞지 마세요.

### 6.1 ⚠️ 먼저 읽어 주세요 — 레거시 API 주의 🟩

강의의 RNN 실습은 **`torchtext.legacy`** 의 `Field`, `BucketIterator`, 그리고 `pl.metrics.Accuracy` 등을 사용합니다. 그런데 이 API들은 **현재 버전에서 제거·폐기**되었습니다.

- `torchtext.legacy`(`Field`, `BucketIterator`, `IMDB.splits`)는 **torchtext 0.12(2022) 이후 제거**되었고, torchtext 라이브러리 자체도 현재 **개발이 중단**되어 0.18이 마지막 안정 릴리스입니다. (현재 안정판 문서의 package reference에 `Field`, `BucketIterator`가 없습니다.)
- `pl.metrics.Accuracy`는 **`torchmetrics`** 패키지로 분리되었습니다(Lightning v1.5에서 제거).
- `pl.Trainer(gpus=1)`의 `gpus=` 인자도 폐기되어, 최신 버전은 `accelerator="gpu", devices=1` 형태를 씁니다.

설치 참고:

- 강의 당시 코드 재현: `pip install pytorch-lightning torchtext`
- 최신 Lightning 학습 코드 작성: `python -m pip install lightning torchmetrics`
- PyTorch는 OS·Python 버전·CPU/GPU 환경에 따라 설치 명령이 다르므로 **PyTorch 공식 설치 선택기**를 확인하세요.

> 🟩 `torchtext`는 현재 개발이 중단된 라이브러리이므로, **새 프로젝트에서는 `torchtext.legacy` 기반 예제를 그대로 시작하지 않는 편이 좋습니다.** 아래 코드는 **강의 학습 흐름을 이해하기 위한 요약**으로만 보세요. 실제로 돌리려면 최신 PyTorch 방식(`torch.utils.data.Dataset`/`DataLoader`, Hugging Face `datasets`/`tokenizers` 등)으로 옮겨야 합니다. 이 글에서는 **거짓 출력 대신 구조 설명**에 집중합니다.

### 6.2 데이터 준비 흐름(개념) 🟦

강의의 RNN 데이터 준비는 **IMDB 영화 리뷰(긍정/부정)** 를 예로 합니다.

```text
① Field 정의 (text / label 형식 지정)
② Dataset 생성 (IMDB를 train/test로 분리)
③ Vocabulary 생성 (사전 만들기 + 사전학습 임베딩 연결)
④ DataLoader 구현 (BucketIterator로 배치 묶기)
```

> 🟦 **PDF 기반 레거시 코드 요약**: 아래 코드는 강의가 사용한 `torchtext.legacy`, `BucketIterator`, `pytorch_lightning` 흐름을 이해하기 쉽게 묶은 것입니다. **원문 전체를 그대로 복사한 것이 아니며**, 현재 최신 환경에서는 그대로 실행되지 않을 수 있습니다.

```python
# ① Field 정의  (PDF 기반 레거시 요약 — 미실행)
from torchtext.legacy.data import Field

text_field = Field(sequential=True, include_lengths=True, fix_length=200)
label_field = Field(sequential=False)

# ② Dataset 생성
from torchtext.legacy.datasets import IMDB
train, test = IMDB.splits(text_field, label_field)
print(vars(train.examples[0]))            # 첫 번째 데이터(text 토큰 리스트)
print(vars(train.examples[0])['label'])   # 예: pos

# ③ Vocabulary 생성 (사전학습 임베딩 fasttext.simple.300d 사용)
text_field.build_vocab(train, vectors='fasttext.simple.300d')
label_field.build_vocab(train)

# ④ DataLoader 구현
import torch
from torchtext.legacy.data import BucketIterator

device = 'cuda' if torch.cuda.is_available() else 'cpu'   # GPU 없으면 CPU
train_iter, test_iter = BucketIterator.splits(
    (train, test), batch_size=32, device=device)
```

> 🟦 **사전학습 임베딩**: 강의는 `fasttext.simple.300d` 외에도 `glove.6B.100d`, `glove.840B.300d` 등 여러 사전학습 임베딩을 지원한다고 소개합니다. 단어를 **이미 학습된 의미 벡터**로 시작하면, 적은 데이터로도 성능이 좋아질 수 있습니다.
>
> 🟩 **좋은 습관**: `device = 'cuda' if torch.cuda.is_available() else 'cpu'` 처럼 **GPU가 있는지 먼저 확인**한 뒤 장치를 정하는 패턴은 지금도 그대로 권장됩니다. GPU가 없는 컴퓨터에서도 CPU로 동작합니다.

### 6.3 PyTorch Lightning이란? 🟦

- 🟦 **PyTorch Lightning**: PyTorch의 학습/평가 코드를 **더 간결하게** 쓰도록 도와주는 High-Level 라이브러리.
- 🟦 관계: **TensorFlow ↔ Keras** 관계와 비슷합니다(저수준 ↔ 고수준).

🟦 (강의 비교) 직접 `for epoch …` 루프를 짜는 대신:

```python
# 기존 PyTorch: 학습 루프를 직접 작성
for epoch in range(num_epochs):
    for x, y in mnist_train:
        logits = pytorch_model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        # optimizer.zero_grad(), optimizer.step() 등도 직접 호출

# PyTorch Lightning: 핵심만 정의하면 학습 루프는 자동
model = LightningMNISTClassifier()
trainer = pl.Trainer()
trainer.fit(model)
```

### 6.4 LSTM 모델 구조(개념) 🟦

강의는 `LightningModule`을 상속해 LSTM 모델을 정의합니다. **부분 발췌(원문에 `...`로 생략된 부분 포함)** 이므로 **구조 이해용**으로만 보세요.

```python
# PDF 기반 레거시 요약 — 부분 발췌, 미실행 (구조 이해용)
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(pl.LightningModule):
    def __init__(self, embedding, lstm_input_size=300, lstm_hidden_size=...):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(...)
        self.lin = nn.Linear(...)
        self.loss_function = nn.CrossEntropyLoss()
        # self.train_accuracy = pl.metrics.Accuracy()  # ⚠️ 현재는 torchmetrics로 분리됨

    def forward(self, X):
        x = self.embedding[X]
        x = x.to(self.device)
        x = x.permute(1, 0, 2)     # (배치, 길이, 임베딩) → (길이, 배치, 임베딩)
        x, _ = self.lstm(x)
        x = F.elu(x.permute(1, 0, 2))
        x = self.lin(x)
        x = x.sum(dim=1)           # 시퀀스 방향으로 합쳐 문장 하나의 결과로
        return x
```

> 🟩 **shape 읽는 법(입문자용)**: `permute(1, 0, 2)`는 차원의 **순서를 바꾸는** 연산입니다. 여기서는 `(배치, 문장길이, 임베딩차원)`을 `(문장길이, 배치, 임베딩차원)`으로 바꿉니다. PyTorch `nn.LSTM`의 기본 설정(`batch_first=False`)이 **"길이(시간축)가 맨 앞"** 인 `(seq_len, batch, feature)` 형식을 기대하기 때문입니다. (최신 코드는 `batch_first=True`로 `(batch, seq, feature)`를 쓰는 경우가 많습니다.)
>
> 🟩 **손실함수 짝 확인**: 출력이 **클래스 점수(logits)** 이고 손실이 `nn.CrossEntropyLoss()`이면 **모델 마지막에 softmax를 넣지 않습니다.** `CrossEntropyLoss`가 내부에서 `LogSoftmax`+`NLLLoss` 계산을 함께 처리하기 때문입니다(중복 적용 금지).
>
> 🟩 **device 주의**: `text_field.vocab.vectors`는 보통 **`nn.Embedding` 모듈이 아니라 텐서**입니다. 그래서 `self.embedding[X]`처럼 바로 인덱싱하면, 배치 `X`가 GPU에 올라가 있을 때 임베딩 텐서는 CPU에 남아 **device 불일치 오류**가 날 수 있습니다(여기서 `x.to(self.device)`는 인덱싱 *이후*라 너무 늦습니다). 실제 구현에서는 `nn.Embedding.from_pretrained(...)`로 임베딩을 **모델 모듈에 등록**하고 `model.to(device)`로 함께 이동시키는 방식이 더 안전합니다. **이 코드는 현대 환경에서 비실행·구조 이해용**으로 보세요.

### 6.5 학습 단계와 옵티마이저(개념) 🟦

```python
# PDF 기반 레거시 요약 — 부분 발췌, 미실행
def training_step(self, batch, batch_idx):
    # ... 손실 계산
    loss = self.loss_function(...)
    self.log("train_loss", loss)
    return loss

def validation_step(self, batch, batch_idx):
    # ... 검증 정확도 계산
    self.log("val_acc", ...)

def train_dataloader(self):
    return train_iter

def val_dataloader(self):
    return test_iter        # ⚠️ 강의는 검증용으로 test_iter를 사용 (아래 주의 참고)

def configure_optimizers(self):
    from torch.optim import Adam
    return Adam(self.parameters(), lr=0.01)

# 모델 생성 및 학습
model = RNNModel(text_field.vocab.vectors)
trainer = pl.Trainer(gpus=1, max_epochs=3)   # ⚠️ 최신: accelerator="gpu", devices=1
trainer.fit(model)
```

🟦 **평가 지표**: `train accuracy`와 `validation accuracy`를 함께 봅니다. (강의 학습 결과 예시: `loss = 0.396` 부근)

> 🟩 **중요(검증/테스트 분리)**: 위 예제는 강의 흐름상 `test_iter`를 `val_dataloader()`에서 그대로 사용하지만, **실제 모델 개발에서는 `train / validation / test`를 분리**해야 합니다. **검증 데이터**는 학습 중 조기 종료·모델 선택에 쓰고, **테스트 데이터**는 **마지막 최종 평가에만** 사용합니다. 테스트 성능을 보며 하이퍼파라미터를 반복 조정하면, 테스트 세트가 사실상 검증 세트가 되어 **최종 성능이 낙관적으로 부풀려집니다.** (PDF 제목에는 Test 단계가 있지만 추출된 코드는 train/validation 중심입니다.)
>
> 🟩 **훈련/검증을 같이 보는 이유**: 훈련 정확도만 오르고 **검증 정확도가 떨어지면 과적합(overfitting)** 신호입니다. 그래서 두 값을 **함께** 추적하고, `EarlyStopping`이나 `Dropout` 같은 장치로 과적합을 줄입니다. 단, 어떤 기법도 과적합을 **"완전히" 없애지는 못합니다.**

---

## 7. 입문자가 자주 하는 실수 🟩

1. **프레임워크 혼용**: CNN 절(Keras)과 RNN 절(PyTorch) 코드를 한 파일에 섞기. → 한 모델은 한 프레임워크로.
2. **토크나이저를 전체 데이터로 학습**: `train_test_split` **전에** 사전을 만들면 데이터 누수. → 항상 훈련 데이터로만 `fit`.
3. **검증·테스트 혼용**: `test_iter`로 모델을 고르고 다시 그 `test_iter`로 최종 평가. → validation과 test를 분리.
4. **불균형 데이터에서 정확도만 보기**: 스팸 20% 문제에서 정확도 80%는 의미가 약함. → precision·recall·F1.
5. **softmax 이중 적용**: `CrossEntropyLoss` 앞에 softmax를 또 넣기. → 손실함수가 내부 처리하므로 빼기.
6. **GPU 가정**: `device='cuda'`로 고정. → `torch.cuda.is_available()`로 확인 후 결정.
7. **폐기된 API를 그대로 실행**: `torchtext.legacy`는 현재 동작하지 않음. → 최신 데이터 파이프라인으로 이전.

---

## 8. DAY1 핵심 정리

```text
자연어 처리 딥러닝 모델 지도
  - 순서 모름:      DNN, (CNN은 지역 패턴)
  - 순서 기억:      RNN → LSTM → GRU
  - 문장 변환·집중: Seq2Seq → Attention
  - 병렬·문맥:      Transformer → BERT(이해) / GPT(생성)

CNN for NLP (Keras)
  - 1D 합성곱으로 n-gram 패턴 추출 → GlobalMaxPooling → Dense
  - 이진=sigmoid+binary_crossentropy / 다중=softmax+categorical
  - 불균형 데이터는 정확도 외 precision·recall·F1

RNN for NLP (PyTorch + Lightning)
  - Field → Dataset → Vocab(사전학습 임베딩) → BucketIterator
  - LightningModule로 학습 루프 간결화
  - ⚠️ 강의의 torchtext.legacy API는 현재 폐기 → 최신 방식 필요
  - train/validation/test를 분리하고 과적합 관리
```

> 🟩 다음 단계 예고는 강의 자료에서 확인되지 않아 적지 않습니다(추측 금지).

---

## 참고 자료

- 🟦 강의 자료: 교과목 3 「초거대언어모델(LLM)」 · 단원 2 「자연어 딥러닝」 — DAY1. 자연어 처리를 위한 딥러닝 모델 (31p)
- 🟩 PyTorch 공식 문서 — [`torch.nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [`torch.nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- 🟩 PyTorch Lightning 공식 문서 — [`Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html), [설치](https://lightning.ai/docs/pytorch/stable/starter/installation.html)
- 🟩 Keras 공식 문서 — [`Conv1D`](https://keras.io/api/layers/convolution_layers/convolution1d/), [`Embedding`](https://keras.io/api/layers/core_layers/embedding/)
- 🟩 torchtext 개발 중단·legacy 제거 안내 — [PyTorch text 문서](https://docs.pytorch.org/text/stable/index.html), [GitHub](https://github.com/pytorch/text)
