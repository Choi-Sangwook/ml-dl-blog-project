# 🌐 자연어처리(NLP) 완전 입문 가이드 — DAY4. 자연어 처리를 위한 번역 모델 구조

> **시리즈**: 파이썬 기본만 있는 사람을 위한 자연어처리(NLP) 입문
> **교과목**: 초거대언어모델(LLM) · 단원 2 — 자연어 딥러닝
> **이전 편**: DAY3. 자연어 처리를 위한 언어 모델(BERT)

> 📝 DAY3에서 BERT(이해형 모델)를 봤다면, 이번 DAY4는 **문장을 다른 언어의 문장으로 바꾸는 "번역 모델"** 을 다룹니다. 구글 번역과 파파고가 실제로 틀리는 사례에서 출발해, 번역 기술이 **통계 기반(PBMT) → 신경망 기반(GNMT)** 으로 어떻게 발전했는지 보고, 그 핵심 구조인 **Seq2Seq 인코더-디코더**가 무엇이며, 실제로 어떤 코드로 구현되는지(정수 인코딩·패딩·인코더/디코더 LSTM)까지 살펴봅니다.

> 💡 **이 글의 표기 약속**
> - 🟦 **(강의 PDF)** : 강의 자료에 직접 나온 개념·예시·비교·코드
> - 🟩 **(보충)** : 입문자를 위해 글쓴이가 덧붙인 설명·용어·실습 코드
>
> ⚠️ **코드 실행 안내**: 이 글의 작성 환경에는 **TensorFlow·Keras가 설치되어 있지 않아** §7의 인코더/디코더 LSTM 코드는 **직접 실행하지 않았고, 출력값을 지어내지 않았습니다.** 반면 **정수 인코딩·패딩·원-핫 인코딩 데모**(§7.2)는 추가 설치 없이 **직접 실행한 실제 출력**입니다. 또한 `Teacher Forcing`이라는 **용어 자체는 PDF에 없는 🟩 보충**이며, PDF는 그 동작("정답을 한 칸씩 밀어 디코더 입력으로 사용")만 풀어서 설명합니다. 번역기 비교 예시는 **강의 자료 기준(2021년 12월)** 이며, 번역기 성능은 계속 바뀌므로 지금 실행하면 결과가 다를 수 있습니다.

---

## 1. 이번 DAY에서 배우는 것

- 🟦 자동 번역기가 **왜 어려운가** (관용어·고어·사투리·문자 단위 오역)
- 🟦 번역 기술의 발전: **PBMT(통계 기반) → GNMT(신경망 기반)**
- 🟦 **GNMT = Seq2Seq 인코더-디코더** 구조의 동작 원리 (챗봇 예시 포함)
- 🟦 **Seq2Seq**가 정확히 무엇이고, 왜 필요한가 (입출력 길이가 다른 문제)
- 🟦 Seq2Seq 번역기를 만드는 **실제 구현 흐름**: SOS/EOS, 정수 인코딩, 패딩, 원-핫 인코딩, 인코더/디코더 LSTM

---

## 2. 자동 번역기는 왜 어려운가 🟦

강의는 **구글 번역 vs 네이버 파파고** 비교로 시작합니다(🟦 2021년 12월 기준). 두 서비스 모두 **입력창에 원문을 넣으면 결과창에 번역이 나오는 구조**(파파고는 음성 입력·번역 실행 버튼도 제공)로, 이것이 뒤에서 배울 **"입력 시퀀스 → 출력 시퀀스"(Seq2Seq)** 의 사용자 화면입니다. 같은 한국어라도 번역기가 **문맥·표현**을 이해하는 정도에 따라 결과가 갈립니다.

| 한국어 | 구글 | 파파고 | 올바른 번역 | 문제 |
|---|---|---|---|---|
| 영문을 모르겠어. | I don't know English. | I don't know English. | **I don't know the reason.** | 관용 표현(‘이유를 모르겠다’)을 문자 그대로 해석 |
| 씨가 말랐다. | The seeds are dry. | The seeds have dried up. | **Nothing is left.** | 관용어(‘모두 없어졌다’)를 ‘씨앗’으로 직역 |
| 어딜 도망가? | Where are you running? | Where are you going? | (둘 다 맞음) | 문맥에 따라 여러 정답 가능 |
| 뭐하노? | What are you doing? | What are you doing? | (둘 다 맞음) | 경상도 **사투리**도 학습됨 |
| 통촉하여 주시옵소서. | Please contact me. | Please let me know. | **Please understand my situation.** | **고어(옛말)** 학습 데이터 부족 |

> 🟦 **핵심 교훈**:
> - 번역기는 **관용 표현(idiom)** 을 모르면 문자 그대로 번역해 의미가 달라집니다.
> - 문맥에 따라 **정답이 하나가 아닐 수도** 있습니다(‘어딜 도망가?’).
> - 최근 신경망 번역기는 **사투리**도 상당량 학습해 자연스럽게 번역합니다.
> - 반면 **고어·오래된 표현**은 데이터가 부족해 품질이 낮아질 수 있습니다.

---

## 3. 번역기 부작용 사례 🟦

🟦 **해외 사례**: 한국어 리뷰(‘이 스파 진짜 좋아요!’ 같은 긍정 후기)가 자동 번역 과정에서 **한자가 섞이거나 의미 없는 문자**로 깨져 나온 사례. 번역기가 의미를 이해하지 못하고 **문자 단위로 잘못 처리**하면 전혀 다른 결과가 나옵니다.

🟦 **국내 사례(메뉴판 오역)**: 음식 이름을 **의미가 아니라 발음·단어 단위**로 번역한 사례.

| 메뉴 | 오역 | 원인 |
|---|---|---|
| 설렁탕 | **Bear Tang** | ‘설’ → ‘Bear(곰)’ |
| 육회 | **Six Times** | 육(六)→Six, 회(回)→Times |
| 돈가스 정식 | **Money Gas Proper Form** | 돈→Money, 가스→Gas, 정식→Proper Form |
| 돼지 주물럭 | **Massage Pork** | 주물럭→Massage |

> 🟦 기계 번역은 **음식명·고유명사·전통 음식**처럼 사전에 없는 단어를 만나면 문자 단위로 해석해 오류가 납니다.

> 🟦 **개선 예시(PDF 스크린샷)**: 같은 "돼지 주물럭"도 번역기와 시점에 따라 결과가 달라집니다. 구글 번역은 `Pork loin`처럼 돼지 부위는 잡았지만 ‘주물럭’ 조리 의미를 놓쳤고("돼지 부위는 맞으나 주물럭 부분의 번역을 놓침"), 파파고는 `Marinated Grilled Pork`처럼 정확하게 번역했습니다("정확함 / 자연어 데이터를 기반으로 학습 중"). 즉 음식명 번역은 단순 단어 치환보다 **자연어 데이터·문맥 학습**의 영향을 크게 받으며, 최근 신경망 번역기는 과거보다 자연스럽게 번역하는 경우가 많아졌습니다.

---

## 4. 번역 기술의 발전: PBMT → GNMT 🟦

### 4.1 PBMT (Phrase Based Machine Translation, 통계 기반)

🟦 2006년 구글 번역기가 처음 등장했을 때는 인공지능이 아니라 **통계 기반** 방식이었습니다. (초기에는 대중 서비스라기보다 **UN 디지털 문서 분류·외래어 번역** 등 자연어 처리에 활용되었습니다.)

- 인터넷 문서를 **통계적으로 분석**해 단어 등장 **빈도**를 계산
- **가장 확률이 높은 번역**을 선택 (문장 의미를 *이해*하는 게 아님)
- 문장을 **단어·구(Phrase) 단위**로 분리해 번역 → 다시 연결

🟦 예: "나는 학교에 간다."

```text
나는 → I,  학교 → School,  간다 → Go
=> "I school go"   (어순·조사를 고려 못 해 부자연스러움)
```

> 🟦 PBMT는 문장을 전체적으로 이해하지 않고 단어·구 단위로 번역하기 때문에 **어순이 이상하고, 조사가 어색하며, 긴 문장은 거의 번역하지 못했습니다.**

### 4.2 GNMT (Google Neural Machine Translation, 신경망 기반)

🟦 GNMT는 **인공신경망 기반** 번역으로, 문장 **전체 의미를 이해**한 뒤 번역합니다.

```text
문장 전체 → Encoder → 의미 벡터 → Decoder → 번역 문장
```

- 🟦 **문맥·시제·조사·어순**까지 고려 → 자연스러운 번역, 긴 문장 처리, 정확도 향상, **다양한 언어 지원 확대**
- 🟦 **Encoder-Decoder 구조**의 신경망으로 문장 전체 의미를 학습

> 🟩 **"이해"의 뜻**: 여기서 "문장 전체 의미를 이해"는 사람이 뜻을 이해한다는 의미라기보다, 신경망이 **문장 순서와 주변 단어 정보를 이용해 번역에 필요한 의미 표현 벡터를 학습**한다는 뜻으로 보면 됩니다.

> 🟩 **보충(발전 시점)**: Google은 **2016년**에 Google Translate에 신경망 번역(NMT)을 적용하기 시작했습니다. 강의는 GNMT를 **RNN 기반 Seq2Seq 인코더-디코더**로 설명합니다(§5에서 자세히 다룸). 공식 GNMT 논문 기준으로는 **LSTM 기반 인코더-디코더에 attention과 residual connection을 포함**한 시스템이었으며, 이후 **Transformer** 계열에서 attention이 RNN을 **대체**하는 핵심 구조로 더 중심이 되었습니다.

### 4.3 실제 문장 비교 🟦

🟦 원문: *"A run for president is an attempt to be elected to become a president of a country."*

| 방식 | 번역 결과 |
|---|---|
| **PBMT** | "대통령에 대한 실행한 나라의 대통령으로 선출되고 시도되고 있다." (엉망) |
| **GNMT** | "대통령 출마는 한 나라의 대통령이 되려는 시도입니다." (자연스러움) |

> 🟦 `run for president`는 **‘대통령 선거에 출마하다’** 라는 관용 표현입니다. PBMT는 `run`을 ‘실행’으로 직역했지만, GNMT는 문장 전체 의미를 먼저 이해해 자연스럽게 번역합니다.

> 🟦 또 다른 예: **‘은행’** 은 문맥에 따라 `Bank`(금융) 또는 `River bank`(강둑)로 번역될 수 있습니다. GNMT는 **주변 단어를 함께 고려**해 적절한 의미를 고릅니다.

---

## 5. GNMT 자세히: Seq2Seq 인코더-디코더 🟦

🟦 GNMT는 **RNN 계열**의 **Sequence-to-Sequence(Seq2Seq)** 구조입니다.

- **Many-to-Many** 형태: 입력도 출력도 **시계열(순서 있는) 데이터**
- **인코더 / 디코더** 연결 구조

| 구분 | 역할 | 예시(영어→한글) |
|---|---|---|
| **인코더(Encoder)** | 입력 문장의 의미를 **벡터로 압축** | 영어 문장 입력 |
| **디코더(Decoder)** | 압축된 의미로 **새 문장을 생성** | 한글 문장 출력 |

```text
영어 문장 → 인코더 → 의미 정보(벡터) → 디코더 → 한글 문장
```

> 🟩 **핵심**: 번역은 단순 **치환**이 아니라 **‘이해(인코더) → 생성(디코더)’** 의 두 단계입니다.

### 5.1 챗봇 예시로 보는 인코더-디코더 🟦

🟦 GNMT 같은 Seq2Seq 구조는 번역뿐 아니라 **챗봇**에도 쓰입니다. 둘 다 **입력 시퀀스 → 출력 시퀀스**이기 때문입니다.

🟦 (강의 다이어그램) 사용자가 "How is weather today?" 라고 질문하면:

```text
[인코더]  How → is → weather → today?   (질문을 순서대로 읽고 의미를 압축)

[디코더]  <go> → It → is → sunny → today   (입력, 한 칸씩 정답을 밀어 넣음)
          출력: It → is → sunny → today → <eos>
```

> 🟦 사용자가 질문을 입력하면 모델은 그 질문을 이해한 뒤 적절한 답변을 생성합니다. (예: 입력 "오늘 날씨 어때?" → 출력 "오늘은 맑습니다.")
>
> 🟦 챗봇은 **이전 입력과 현재 입력을 함께 고려**할 수 있어, 여러 문장을 연속 입력하면 **앞선 문맥(대화 흐름)** 을 참고해 답합니다. (예: "거기 위치는?" → 앞 문장의 장소를 기억해야 정확히 답변)

> 🟩 **보충(RNN의 한계)**: RNN 기반 모델은 **짧은 문맥은 잘 반영**하지만, 대화나 문장이 **길어지면 앞부분을 완전히 기억하지는 못합니다**(DAY1의 장기 의존성 문제). 이 한계를 보완하는 것이 Attention·Transformer입니다.

### 5.2 GNMT 개념도: 영어 → 불어 번역 🟦

🟦 (강의 다이어그램) 인코더에 `I am a student`를 입력하고, 디코더가 `je suis étudiant`를 생성하는 예시입니다.

```text
1. 인코더 입력 :  I   am   a   student   <s>
2. 디코더 입력 :                          moi  suis  étudiant
3. 디코더 출력 :                          moi  suis  étudiant  </s>
```

- **인코더 입력**: 번역하고 싶은 원문(`I am a student`)을 순서대로 읽어 의미를 벡터로 압축합니다.
- **디코더 입력**: 학습 단계에서 **정답 프랑스어 문장을 한 칸씩 밀어서** 넣습니다. 시작 토큰(`<s>`) 뒤에 정답 단어를 차례로 주고, **다음 단어를 맞히도록** 학습합니다.
- **디코더 출력**: 한 번에 전체가 아니라 **한 단어씩 순서대로** 생성합니다.

> 🟩 **용어 보충 — Teacher Forcing**: 위처럼 학습할 때 디코더에 **직전의 정답 단어**를 넣어 다음 단어를 맞히게 하는 방식을 **Teacher Forcing(교사 강요)** 이라고 부릅니다. 정답이 없는 **추론 단계에서는** 디코더가 **직전에 스스로 생성한 단어**를 다음 입력으로 사용합니다. (본문 설명에서는 `Je t'aime` 같은 예시도 등장하는데, 다이어그램의 `I am a student`/`je suis étudiant`와 별개의 예시입니다.)

🟦 디코더는 각 위치에서 **가능한 여러 단어 후보의 확률**을 계산하고, 그 **확률의 합은 1** 이 됩니다.

```text
[1번째 출력 위치] Je : 0.80,  Tu : 0.10,  Il : 0.05,  기타 : 0.05   (합 = 1)
→ 가장 높은 Je 선택
```

> 🟩 **용어 보충**: 이런 확률 분포를 만들 때 보통 `softmax`를 쓰고, 가장 큰 값을 고르는 동작을 `argmax`라고 부릅니다. 실제 번역 시스템은 매 위치에서 가장 큰 단어 하나만 고르지 않고 **여러 후보 문장을 함께 탐색(beam search)** 하기도 합니다.

🟦 디코더의 입력과 출력을 비교해 **지도학습**을 진행하는 것이 Seq2Seq 기술의 핵심입니다. 예를 들어 번역 정답이 ‘나는 학생이다’라면, 디코더는 한 단어씩 다음 단어를 예측하고 실제 정답과 비교해 오차를 계산하며 그 오차를 줄이도록 학습합니다.

---

## 6. Seq2Seq란 무엇인가 🟦

🟦 **Seq2Seq(Sequence-to-Sequence)** 는 하나의 순서 데이터(시퀀스)를 입력받아 **다른** 순서 데이터로 변환하는 딥러닝 구조입니다.

🟦 예: 번역

```text
입력 시퀀스: 나는 당신을 사랑합니다
출력 시퀀스: I LOVE YOU
```

> 🟦 문장은 단어들이 순서대로 나열된 데이터입니다. 따라서 번역 모델은 단어의 **의미**뿐 아니라 단어의 **순서**도 함께 이해해야 합니다.

### 6.1 Seq2Seq의 기본 구조 🟦

```text
입력 문장 → 인코더(Encoder) → 문장에 대한 압축 정보 → 디코더(Decoder) → 출력 문장
```

🟦 **인코더**: 입력 문장을 읽는 부분. 예를 들어 `나는 / 당신을 / 사랑합니다`가 들어오면, 이 단어들을 순서대로 읽고 문장 전체의 의미를 하나의 내부 정보로 압축합니다. 이 압축 정보는 보통 **Hidden State** 와 **Cell State** 라는 상태값으로 표현되며, **LSTM**을 사용하는 경우 이 두 상태가 디코더로 전달됩니다.

🟦 **디코더**: 인코더가 만든 압축 정보를 바탕으로 새로운 문장을 생성하는 부분. 한국어 문장의 의미를 압축했다면, 디코더는 이를 이용해 영어 문장을 순서대로 생성합니다.

```text
I → LOVE → YOU
```

디코더는 한 번에 전체 문장을 출력하지 않고, **매 시점마다 다음 단어를 예측**합니다.

### 6.2 Seq2Seq가 필요한 이유 🟦

🟦 일반적인 분류 모델은 **입력 하나에 결과 하나**를 출력합니다.

```text
영화 리뷰 → 긍정 / 부정
이미지    → 고양이 / 강아지
```

하지만 번역은 **입력과 출력의 길이가 다를 수 있습니다.**

```text
나는 학생입니다.  (입력 2단어 정도)
→ I am a student.  (출력 4단어)
```

> 🟦 이처럼 **입력 길이와 출력 길이가 다른 문제**에는 Seq2Seq 구조가 적합합니다.

> 🟦 **번역 데이터 구성**: 실습에서는 영어(`src`)와 프랑스어(`tar`) 문장이 매핑된 데이터를 `pd.read_csv()`로 읽습니다. 전체 샘플이 많으면 일부만 사용해 학습 시간을 줄입니다.

---

## 7. Seq2Seq RNN 번역기, 실제로 어떻게 만들까 🟦

> **이 절의 프레임워크: Keras(LSTM)** — 강의가 제시하는 구현 흐름

### 7.1 문장 시작·끝 표시: SOS/EOS 토큰 🟦

🟦 Seq2Seq에서는 문장의 **시작과 끝**을 알려주는 특수 토큰이 필요합니다.

| 토큰 | 의미 | 예 |
|---|---|---|
| **SOS**(Start Of Sequence) | 문장이 시작된다 | `<sos> I love you` |
| **EOS**(End Of Sequence) | 문장이 끝났다 | `I love you <eos>` |

🟦 **디코더 입력과 디코더 출력**: 정답 문장이 `I LOVE YOU`라면,

```text
디코더 입력 : <sos> I LOVE YOU        (앞에 시작 토큰을 붙임)
디코더 출력 : I LOVE YOU <eos>        (뒤에 끝 토큰을 붙임)
```

즉 디코더는 **현재 단어를 보고 다음 단어를 맞히도록** 학습합니다.

```text
입력: <sos> → 출력: I
입력: I     → 출력: LOVE
입력: LOVE  → 출력: YOU
입력: YOU   → 출력: <eos>
```

### 7.2 🟩 정수 인코딩·패딩·원-핫 인코딩 데모 (실행 결과 O)

PDF가 설명하는 **정수 인코딩 → 패딩 → 원-핫 인코딩** 흐름을, PDF와 같은 예시(`I LOVE YOU`)로 직접 실행해 봅니다.

```python
import numpy as np

# 문자 집합(강의 예시와 동일)
vocab = {"<pad>": 0, "I": 1, "LOVE": 2, "YOU": 3, "<sos>": 4, "<eos>": 5}

sentence = ["I", "LOVE", "YOU"]

# 1) 정수 인코딩: 단어를 숫자로
int_encoded = [vocab[w] for w in sentence]
print("정수 인코딩:", sentence, "->", int_encoded)

# 2) 디코더 입력/출력: 정답을 한 칸씩 밀기 (SOS/EOS 부착)
decoder_input = ["<sos>"] + sentence
decoder_target = sentence + ["<eos>"]
print("디코더 입력 :", decoder_input, "->", [vocab[w] for w in decoder_input])
print("디코더 타깃 :", decoder_target, "->", [vocab[w] for w in decoder_target])

# 3) 패딩: 문장 길이를 max_len에 맞춰 <pad>(0)로 채움
def pad(seq, max_len, pad_id=0):
    ids = [vocab[w] for w in seq]
    return ids + [pad_id] * (max_len - len(ids))

max_len = 4
print(f"\n패딩(max_len={max_len}):")
print("  ['I','LOVE','YOU'] ->", pad(["I", "LOVE", "YOU"], max_len))

# 4) 원-핫 인코딩: 정수 인덱스를 (vocab_size,) 크기 벡터로
vocab_size = len(vocab)
def one_hot(idx, size):
    v = np.zeros(size, dtype=int)
    v[idx] = 1
    return v

print(f"\n원-핫 인코딩 (vocab_size={vocab_size}):")
for w in ["I", "LOVE", "YOU"]:
    print(f"  {w:>5} (id={vocab[w]}) -> {one_hot(vocab[w], vocab_size).tolist()}")
```

실제 실행 출력:

```text
정수 인코딩: ['I', 'LOVE', 'YOU'] -> [1, 2, 3]
디코더 입력 : ['<sos>', 'I', 'LOVE', 'YOU'] -> [4, 1, 2, 3]
디코더 타깃 : ['I', 'LOVE', 'YOU', '<eos>'] -> [1, 2, 3, 5]

패딩(max_len=4):
  ['I','LOVE','YOU'] -> [1, 2, 3, 0]

원-핫 인코딩 (vocab_size=6):
      I (id=1) -> [0, 1, 0, 0, 0, 0]
   LOVE (id=2) -> [0, 0, 1, 0, 0, 0]
    YOU (id=3) -> [0, 0, 0, 1, 0, 0]
```

> 🟩 **해석**: `디코더 입력`은 정답 앞에 `<sos>`를 붙인 것, `디코더 타깃`은 정답 뒤에 `<eos>`를 붙인 것입니다. 둘을 나란히 보면 "입력의 각 위치가 타깃의 같은 위치 단어를 맞혀야 한다"(`<sos>→I`, `I→LOVE`, `LOVE→YOU`, `YOU→<eos>`)는 걸 확인할 수 있습니다. 패딩은 문장 길이를 맞춰 배치 학습을 가능하게 하고, 원-핫 인코딩은 정수 인덱스를 신경망이 다루기 쉬운 벡터로 바꿉니다.

### 7.3 RNN 기반 Seq2Seq 번역기 구조 🟦

🟦 RNN 기반 번역기는 다음 흐름으로 만들어집니다.

```text
1. 데이터 읽기
2. 입력 문장과 출력 문장 분리
3. SOS, EOS 토큰 추가
4. 문자 또는 단어 집합 생성
5. 정수 인코딩
6. 패딩
7. 원-핫 인코딩
8. 인코더 LSTM 생성
9. 디코더 LSTM 생성
10. Dense 출력층 생성
11. 모델 학습
12. 번역 결과 예측
```

🟦 **인코더 LSTM**: 입력 문장을 읽고 마지막 상태값을 생성합니다. Keras 기준으로 `return_state=True` 옵션을 쓰면 인코더의 최종 상태를 받을 수 있습니다. 인코더의 시퀀스 출력 자체는 필요 없고, 보통 다음 두 값이 중요합니다.

- `state_h` : 은닉 상태(Hidden State)
- `state_c` : 셀 상태(Cell State)

이 두 값이 디코더로 전달됩니다.

🟦 **디코더 LSTM**: 인코더에서 전달받은 상태값을 초기 상태로 사용합니다(`initial_state=[state_h, state_c]`). 또한 디코더는 문장의 **모든 시점**에서 단어를 예측해야 하므로 `return_sequences=True` 옵션이 필요합니다(마지막 출력만이 아니라 모든 시점의 출력을 반환).

🟩 **참고용 Keras 코드(이 글에서는 미실행)** — PDF가 글로 설명한 구조를 코드로 옮긴 것입니다. 실제 데이터·환경에 맞춰 수정해야 하며, 출력값은 지어내지 않았습니다.

```python
# (보충) 환경: TensorFlow/Keras 필요 — 이 글에서는 실행하지 않음
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

latent_dim = 256          # LSTM 은닉 차원(임의 설정)
src_vocab_size = 10000    # 영어 단어 집합 크기(데이터로 정함)
tar_vocab_size = 12000    # 프랑스어 단어 집합 크기(데이터로 정함)

# 인코더: 시퀀스 출력은 버리고 상태(state_h, state_c)만 사용
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# 디코더: 인코더 상태를 초기 상태로 받고, 모든 시점의 출력을 반환
decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Dense 출력층: 매 시점마다 단어 집합 크기만큼의 확률(다중 클래스 분류)
decoder_dense = Dense(tar_vocab_size, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.summary()
```

🟦 **Dense 출력층과 손실 함수**: 디코더 LSTM의 출력은 Dense 층을 거쳐 최종 단어 확률로 변환됩니다. 예를 들어 프랑스어 단어 집합 크기가 7,000개라면, 매 시점마다 7,000개 단어 중 하나를 선택해야 합니다. 즉 **디코더는 매 시점마다 다중 클래스 분류를 수행**합니다.

```text
현재 시점 출력 → Dense → 단어별 확률 → 가장 높은 확률의 단어 선택
```

디코더는 매 시점마다 단어 집합 중 하나를 맞혀야 하므로 다음 손실 함수를 사용합니다.

| 손실 함수 | 사용 조건 |
|---|---|
| `categorical_crossentropy` | 정답이 **원-핫 인코딩** 형태일 때 |
| `sparse_categorical_crossentropy` | 정답이 **정수 인덱스** 형태일 때 |

> 🟩 **DAY3 연결**: "매 시점마다 단어 집합 중 하나를 고르는 다중 클래스 분류"는 DAY3의 BERT 다중 분류와 원리가 같습니다. 다만 번역은 이 분류를 **문장 길이만큼 반복**한다는 점이 다릅니다.

### 7.4 학습과 예측의 차이 🟦

🟦 **학습 단계**: 정답 문장을 이미 알고 있으므로, 디코더 입력에 정답 문장의 **앞부분**(`<sos> I LOVE`)을 넣고 디코더 출력이 다음 단어(`I LOVE YOU`)를 맞히도록 학습합니다. 이 방식은 모델이 더 안정적으로 학습하도록 도와줍니다(§5.2의 **Teacher Forcing** 보충 참고).

🟦 **예측(추론) 단계**: 정답 문장이 없으므로, 디코더는 **자신이 이전에 예측한 단어**를 다시 다음 입력으로 사용합니다.

```text
1단계: <sos> 입력 → I 예측
2단계: I 입력    → LOVE 예측
3단계: LOVE 입력 → YOU 예측
4단계: YOU 입력  → <eos> 예측   (<eos>가 나오면 문장 생성 종료)
```

> 🟩 **한 문장 요약**: Seq2Seq는 입력 시퀀스를 출력 시퀀스로 변환하는 모델입니다. 번역에서는 `입력 문장 → 인코더가 의미 압축 → 디코더가 번역 문장 생성` 순으로 동작하며, RNN 기반 Seq2Seq에서는 **LSTM**으로 문장의 순서를 기억하고 인코더의 **Hidden State·Cell State**를 디코더에 전달합니다. 디코더는 매 시점마다 단어 하나를 예측하며, 이 과정이 반복되어 최종 번역 문장이 생성됩니다.

---

## 8. 입문자가 기억할 점 🟩

1. **직역의 함정**: 번역기는 관용어·고어·고유명사에서 문자 단위로 직역하면 틀립니다(설렁탕→Bear Tang). 문맥 이해가 핵심.
2. **PBMT vs GNMT**: 통계·단어 단위(PBMT) → 신경망·문장 전체 의미(GNMT). "I school go" vs "학교에 간다".
3. **Encoder-Decoder**: 이해(인코더) → 생성(디코더). 번역·챗봇이 같은 구조.
4. **SOS/EOS + Teacher Forcing**: 디코더 입력엔 `<sos>`+정답을 한 칸 밀어 넣고, 디코더 타깃엔 정답+`<eos>`를 붙인다.
5. **정수 인코딩 → 패딩 → 원-핫**: 문자를 숫자로, 길이를 맞추고, 신경망 입력용 벡터로 변환하는 순서.
6. **인코더는 상태만, 디코더는 매 시점 출력**: `return_state=True`(인코더) / `return_sequences=True`(디코더).
7. **학습 vs 예측**: 학습은 정답을 주지만(Teacher Forcing), 예측은 직전 자기 출력을 다시 입력으로 씀(자기회귀).
8. **번역기 성능은 시점에 따라 변함**: 강의 예시는 2021년 12월 기준입니다.

---

## 9. DAY4 핵심 정리

```text
자동 번역의 어려움
  - 관용어/고어/고유명사 직역 오류, 문맥에 따라 정답 다수

번역 기술 발전
  - PBMT(통계, 단어·구 단위) → GNMT(신경망, 문장 전체 의미)
  - 🟩 GNMT(2016)는 LSTM enc-dec + attention → 이후 Transformer가 attention 중심

Seq2Seq 인코더-디코더
  - 정의: 순서 데이터를 다른 순서 데이터로 변환 (입출력 길이가 달라도 됨)
  - 인코더: 입력 → Hidden/Cell State로 압축
  - 디코더: 상태를 받아 한 단어씩 문장 생성

실제 구현 흐름 (Keras/LSTM)
  - SOS/EOS 토큰 → 정수 인코딩 → 패딩 → 원-핫 인코딩
  - 인코더: return_state=True → state_h, state_c
  - 디코더: initial_state=[state_h, state_c], return_sequences=True
  - Dense(vocab_size, softmax) → categorical/sparse_categorical_crossentropy
  - 학습: Teacher Forcing(정답 한 칸 밀기) / 예측: 자기회귀(직전 출력 재사용)
```

> 🟩 다음 DAY 주제는 강의 자료에서 확인되지 않아 적지 않습니다(추측 금지).

---

## 참고 자료

- 🟦 강의 자료: 교과목 3 「초거대언어모델(LLM)」 · 단원 2 「자연어 딥러닝」 — DAY4. 자연어 처리를 위한 번역 모델 구조 (27p)
- 🟩 Google Blog, "Found in translation: More accurate, fluent sentences in Google Translate"(2016) — https://blog.google/products-and-platforms/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/
- 🟩 GNMT 논문 — "Google's Neural Machine Translation System"(arXiv:1609.08144) — https://arxiv.org/abs/1609.08144
- 🟩 Sutskever et al., "Sequence to Sequence Learning with Neural Networks"(2014) — https://arxiv.org/abs/1409.3215
- 🟩 Keras 공식 문서 — `LSTM`(`return_state`, `return_sequences`), `Dense`
