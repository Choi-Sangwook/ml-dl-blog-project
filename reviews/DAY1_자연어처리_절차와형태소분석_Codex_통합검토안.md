# DAY1 자연어처리 절차와형태소분석 - Codex 통합 검토안

## 전체 평가

초안은 PDF 56쪽의 핵심 흐름인 **자연어 데이터의 특징(p3~10) → 텍스트 데이터 전처리(p11~38) → 형태소 분석(p39~50) → 워드 클라우드 시각화(p51~56)** 를 대부분 충실히 반영합니다. 자연어처리 편을 기존 머신러닝/딥러닝 편과 분리해 DAY1로 시작한다는 설명도 적절합니다.

입문자 관점에서 좋은 점은 영어와 한국어 전처리를 따로 나누고, 정제·정규화·토큰화·불용어 제거·형태소 분석을 코드 흐름으로 연결한 점입니다. 또한 PDF의 LLM 데이터 준비 8단계, 한국어 형태소 분석의 필요성, NLTK/KoNLPy/kss/PyKoSpacing/WordCloud의 역할 구분이 잘 살아 있습니다.

다만 게시 전 반드시 고쳐야 할 실행성 문제가 있습니다. 초안의 일부 `python` 코드 블록에 Jupyter/Colab 전용 `!pip install ...` 명령이 들어 있어 일반 Python 코드로는 문법 오류가 납니다. 또한 NLTK는 패키지 설치만으로 끝나지 않고 `punkt`, `punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng` 같은 데이터 리소스가 필요합니다. 현재 로컬 환경에는 `nltk` 패키지는 있지만 해당 데이터가 없어서 NLTK 토큰화/불용어/품사 태깅 예제는 실행 검증하지 못했습니다.

검증 범위는 다음과 같습니다.

- PDF 전체 56쪽 텍스트 추출 확인
- 텍스트 추출이 적은 p45, p56 및 코드/표가 많은 p25~56 렌더링 직접 확인
- 초안 Markdown UTF-8 정상, 코드 fence 균형 정상
- Python 코드 블록 20개 문법 검사: `!pip install`이 포함된 3개 블록은 일반 Python 기준 문법 오류
- 표준 라이브러리/정규식 일부 실행 확인: `string.punctuation`, 영문 특수문자 제거, 한글 정규식 정제
- 로컬 패키지 확인: `nltk`, `pandas`, `matplotlib` 설치됨 / `konlpy`, `kss`, `pykospacing`, `wordcloud`, `JPype1` 미설치
- NLTK 데이터 확인: `punkt`, `punkt_tab`, `stopwords`, `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng` 모두 현재 로컬에는 없음

## 높은 우선순위의 오류와 수정사항

### 1. `python` 코드 블록 안의 `!pip install`은 일반 Python에서 문법 오류

- 초안 위치: 6.3 PyKoSpacing, 6.4 kss, 8.2 wordcloud
- 문제: 다음 블록들이 `python` fence 안에 `!pip install ...`을 포함합니다.
  - `!pip install git+https://github.com/haven-jeon/PyKoSpacing.git`
  - `!pip install kss`
  - `!pip install wordcloud`
- 영향: Jupyter/Colab에서는 동작하지만, 일반 `.py` 실행이나 Velog 독자가 복사해 로컬 Python에서 실행하면 `SyntaxError`가 납니다.
- 수정: 설치 명령은 `bash` 코드 블록으로 분리하고, Python 코드에는 import부터 넣으세요.

### 2. NLTK 예제는 패키지 외에 데이터 다운로드가 필요함

- 초안 위치: 5.5, 5.6, 5.7
- 문제: 초안은 `punkt_tab`, `averaged_perceptron_tagger_eng`까지 안내한 점은 좋지만, 본문 출력이 "직접 실행"이라고 되어 있는 반면 현재 로컬에는 NLTK 데이터가 없어 재현 검증이 되지 않았습니다.
- 영향: 독자가 `pip install nltk`만 하고 실행하면 `LookupError`가 날 수 있습니다.
- 수정: NLTK 데이터 다운로드가 인터넷 연결을 요구하며, 최초 1회 필요하다는 안내를 코드 앞에 더 분명히 넣으세요. 출력값은 NLTK 버전과 데이터 리소스에 따라 달라질 수 있다고 표시하는 것이 안전합니다.

### 3. 외부 파일 의존성이 있는 코드가 독립 실행 가능해 보임

- 초안 위치: 6.5, 8.2
- 문제:
  - `pd.read_csv('./data/stopword_dict.csv')`는 파일이 없으면 바로 실패합니다.
  - `open('alice_novel.txt', ...)`도 예제 파일이 제공되지 않으면 실패합니다.
- 영향: 입문자는 코드 오류를 라이브러리 문제로 오해할 수 있습니다.
- 수정: "이 파일을 준비해야 실행됨"을 코드 바로 위에 표시하거나, 작은 예시 리스트/문자열로 독립 실행 가능한 대체 코드를 함께 제공하세요.

### 4. KoNLPy/Okt 설명에서 설치 난이도가 낮게 읽힐 수 있음

- 초안 위치: 7.4
- 문제: "Okt는 별도 설치 없이 바로 쓸 수 있어"라는 문장이 있습니다. 앞에서 JDK/JPype1 필요성을 말하긴 하지만, 문장만 따로 보면 `pip install konlpy`만으로 항상 바로 되는 것처럼 읽힙니다.
- 공식 문서 기준: KoNLPy는 Java/JDK, OS/Python bitness, JPype1 등 환경 영향을 받습니다. 특히 Windows에서 `Mecab()`은 KoNLPy 공식 설치 문서상 지원되지 않는다고 안내됩니다.
- 수정: "Okt는 Mecab처럼 별도 형태소 분석기 설치가 필요 없다는 뜻이지, KoNLPy 자체의 Java/JPype 환경 설정은 필요하다"고 명확히 하세요.

### 5. 설치/API가 민감한 도구의 현재성 표시가 더 필요함

- 초안 위치: 참고 자료, 각 설치 절
- 문제: kss는 최신 README 기준 v6 계열에서 `Kss("split_sentences")` 방식이 기본이고, `split_sentences()` 함수 방식은 backward compatibility로 지원됩니다. PyKoSpacing은 TensorFlow/Keras 의존성과 Python 버전 문제가 있고, WordCloud는 Python 버전/휠/컴파일러 영향을 받을 수 있습니다.
- 수정: 각 도구를 "게시 시점 공식 문서 확인 필요"로 표시하세요. 특히 PyKoSpacing은 설치가 오래된 환경에서만 매끄러울 수 있으므로 선택 도구로 유지하는 편이 안전합니다.

## PDF 기준 누락 내용

### 자연어 데이터의 특징(p3~10)

- p3의 정형 데이터 처리 도구(Excel, Python, SQL)는 표에 간단히 반영되어 있습니다.
- p7의 데이터 포맷(대화, 연설문, 소설, 시)과 화자/톤·어조 고려는 일부 반영되어 있으나, 대화 데이터에서 "누가 말했는지"와 이전 맥락이 중요하다는 설명은 더 보강하면 좋습니다.
- p10의 음성 데이터는 STT로 텍스트화한 뒤 분석한다는 흐름이 초안에는 거의 없습니다. 자연어 데이터 유형을 넓게 보여주는 PDF 내용이므로 1~2문장 추가를 권장합니다.

### 텍스트 데이터 전처리(p11~38)

- p18~p19의 아포스트로피/줄임말 처리, `AT&T`, 화폐 표기처럼 "무조건 삭제하면 안 되는 기호"는 잘 반영되어 있습니다.
- p21의 정규표현식으로 길이 1~2 단어를 제거하는 예시는 초안에서 `len(t) >= 3` 방식으로 대체되어 있습니다. 입문자에게는 더 읽기 쉬운 선택이라 괜찮지만, PDF의 regex 예시가 빠졌다는 점은 표시할 수 있습니다.
- p35~p38의 Colab `files.upload()`, `os.chdir()` 흐름은 로컬 경로 방식으로 바뀌었습니다. 이 변경은 실용적이지만, "강의는 Colab 업로드 방식, 본문은 로컬 파일 경로 방식"이라고 구분하면 PDF 충실도가 올라갑니다.

### 형태소 분석(p39~50)

- p45의 분석기별 성능 비교표는 초안에서 "Mecab이 가장 빠르다" 정도로 축약되어 있습니다. 최소한 "속도/결과가 분석기마다 다르다"는 표를 추가하면 좋습니다.
- p47의 KoalaNLP 지원 분석기 표와 `initialize → 분석 → finalize` 흐름은 개념만 반영되어 있습니다. 심화로 둘 수 있지만 PDF 누락 항목으로는 기록해야 합니다.
- p48~p49의 ETRI API 키 발급 절차와 eunjeon 사용자 사전 등록 예시는 축약되어 있습니다. 본문에 모두 넣을 필요는 없지만, "API 키/사용자 사전이 필요할 수 있다"는 실무 주의가 있으면 충분합니다.

### 워드 클라우드(p51~56)

- p51~p56의 4단계 흐름, `Counter`, `WordCloud`, `STOPWORDS`, `figsize=(14,18)`은 대체로 반영되어 있습니다.
- PDF의 `open/readlines/close` 흐름 대신 초안은 `open(...).read()`를 사용합니다. 더 간결한 코드라 문제는 없지만 PDF 원문 코드와 다르다는 점을 "보충 구현"으로 표시해도 좋습니다.

## 더 자세히 설명할 내용

- **정제와 정규화의 차이**: 현재 설명이 좋습니다. 여기에 "삭제하면 정보 손실, 합치면 표현 통일"이라는 판단 기준을 한 번 더 넣으면 입문자가 더 잘 구분합니다.
- **영어 vs 한국어 토큰화 차이**: 표는 좋지만, `학교에`, `밥을`처럼 조사가 붙은 어절을 왜 바로 키워드로 쓰면 안 되는지 예시를 더 넣으면 좋습니다.
- **불용어 제거의 위험**: 초안의 "과하게 지우면 안 된다"는 설명은 적절합니다. 한국어에서는 조사도 목적에 따라 감정/문체 신호가 될 수 있음을 한 문장 더 넣으면 좋습니다.
- **형태소 분석기 출력 차이**: Okt/Kkma/Mecab의 결과가 다르다는 설명은 좋습니다. 다만 품사 태그 체계가 다르므로 결과 예시는 "정답"이 아니라 "분석기 관점"임을 더 강조하세요.
- **서브워드와 형태소의 관계**: LLM 연결 설명은 유용합니다. 단, PDF보다 확장된 보충 설명이므로 계속 🟩 보충으로 유지해야 합니다.

## 유용한 추가 내용

- `pip install` 대신 `python -m pip install ...` 형식도 함께 제시하면 Windows 사용자에게 더 안전합니다.
- NLTK 데이터 다운로드는 아래처럼 한 셀로 묶으면 좋습니다.
  - `nltk.download('punkt')`
  - `nltk.download('punkt_tab')`
  - `nltk.download('stopwords')`
  - `nltk.download('averaged_perceptron_tagger_eng')`
- 한국어 워드클라우드에서는 `font_path`가 사실상 필수라는 설명은 매우 유용합니다. 현재 초안의 보충을 유지하세요.
- `Counter` 예제처럼 외부 패키지 없이 실행 가능한 작은 예제는 입문자에게 좋습니다. 워드클라우드도 외부 파일 없이 작은 문자열 예제로 한 번 더 보여주면 진입 장벽이 낮아집니다.

## 줄이거나 제거할 내용

- 상단 HTML 기획 주석은 게시 전 제거해야 합니다.
- "NLTK는 영어 자연어 처리에 가장 널리 쓰이는"이라는 표현은 약간 강합니다. spaCy 등도 널리 쓰이므로 "대표적인 학습용 라이브러리" 정도로 완화하세요.
- "kss가 표준처럼 쓰입니다"도 "한국어 문장 분리에 자주 쓰이는 도구" 정도가 더 안전합니다.
- KoalaNLP/ETRI/SoyNLP를 너무 깊게 확장하면 DAY1 입문 범위를 벗어납니다. 현재처럼 심화로 짧게 두되, PDF에 나온 핵심 절차만 빠뜨리지 않는 정도가 좋습니다.

## 바로 붙여 넣을 수 있는 수정 블록

### 1. 설치 명령과 Python 코드를 분리

````markdown
설치는 터미널이나 노트북의 별도 셀에서 먼저 실행합니다.

```bash
python -m pip install kss
python -m pip install wordcloud
python -m pip install git+https://github.com/haven-jeon/PyKoSpacing.git
```

설치가 끝난 뒤 Python 코드에서 import합니다.

```python
import kss
from wordcloud import WordCloud, STOPWORDS
from pykospacing import Spacing
```
````

### 2. NLTK 데이터 다운로드 안내

```markdown
> 🟨 **NLTK는 패키지와 데이터가 따로입니다.**
>
> `pip install nltk`는 라이브러리만 설치합니다. 문장 토큰화, 불용어 사전, 품사 태깅을 쓰려면 아래 데이터도 최초 1회 다운로드해야 합니다. 다운로드에는 인터넷 연결이 필요하고, NLTK 버전에 따라 필요한 리소스 이름이 달라질 수 있습니다.
```

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")
```

### 3. KoNLPy/Okt 설치 표현 수정

```markdown
> 🟨 **KoNLPy 설치 주의**
>
> Okt는 Mecab처럼 별도 형태소 분석기 설치가 필요 없어서 입문용으로 쓰기 편합니다. 다만 KoNLPy 자체는 Java/JDK와 JPype1 환경의 영향을 받습니다. 특히 Windows에서 `Mecab()`은 별도 설치가 까다롭거나 지원 범위가 제한될 수 있으므로, 처음에는 `Okt()`로 실행 흐름을 익히는 편이 안전합니다.
```

### 4. 외부 파일 의존성 표시

```markdown
> 🟨 **외부 파일이 필요한 예제**
>
> 아래 코드는 `./data/stopword_dict.csv` 또는 `alice_novel.txt` 파일이 실제로 있어야 실행됩니다. 파일이 없다면 먼저 예시 파일을 만들거나, 본문에 제공된 작은 리스트/문자열 예제로 흐름을 확인하세요.
```

### 5. 작은 문자열로 워드클라우드 흐름 설명

```python
from collections import Counter

words = ["데이터", "인공지능", "데이터", "분석", "데이터", "모델"]
word_count = Counter(words)
print(word_count)

# 출력: Counter({'데이터': 3, '인공지능': 1, '분석': 1, '모델': 1})
```

### 6. 과장 표현 완화

```markdown
NLTK는 영어 자연어 처리 학습과 실습에서 자주 쓰이는 대표적인 파이썬 라이브러리입니다.

kss는 한국어 문장을 문장 단위로 나누는 데 자주 쓰이는 도구입니다.
```

## 우선순위 표

| 우선순위 | 항목 | 위치 | 조치 |
|---|---|---|---|
| 높음 | `python` 코드 블록 안의 `!pip install` 문법 오류 | 6.3, 6.4, 8.2 | 설치는 `bash`, 실행 코드는 `python`으로 분리 |
| 높음 | NLTK 데이터 리소스 의존성 불명확 | 5.5~5.7 | `punkt_tab`, `stopwords`, tagger 데이터 다운로드 안내 강화 |
| 높음 | 외부 파일 의존 코드가 독립 실행처럼 보임 | 6.5, 8.2 | CSV/텍스트 파일 필요 표시 또는 작은 대체 예제 추가 |
| 중간 | KoNLPy/Okt 설치 난이도 표현 | 7.4 | Java/JPype 필요, Okt는 "별도 분석기 설치 없음"으로 표현 |
| 중간 | p45 분석기 성능 비교표 축약 | 7.4 | 분석기별 속도/결과 차이 표 추가 |
| 중간 | KoalaNLP/ETRI 절차 축약 | 7.5 | `initialize → 분석 → finalize`, API 키/사용자 사전 언급 보강 |
| 중간 | 최신 API/설치 문서 확인 필요 | 참고 자료/설치 절 | NLTK, KoNLPy, kss, PyKoSpacing, WordCloud 공식 문서 확인 표시 |
| 낮음 | 상단 HTML 기획 주석 | 문서 맨 앞 | 게시 전 제거 |
| 낮음 | NLTK/kss "가장 널리/표준" 표현 | 5.1, 6.4 | "대표적", "자주 쓰이는"으로 완화 |

## 최종 권고

초안은 PDF의 주요 내용을 폭넓게 반영했고, 영어/한국어 전처리를 입문자에게 맞게 재구성한 점이 좋습니다. 그러나 **코드 블록 실행성 문제는 게시 전 반드시 수정**해야 합니다. 특히 `!pip install`이 들어간 `python` 코드 블록은 일반 Python 기준 문법 오류이므로, 설치 명령과 실행 코드를 분리해야 합니다.

권고는 **높은 우선순위 수정 후 게시**입니다. 수정 순서는 다음이 좋습니다.

1. `!pip install` 블록을 `bash`로 분리
2. NLTK 데이터 다운로드와 인터넷 필요성 명시
3. `stopword_dict.csv`, `alice_novel.txt` 외부 파일 필요 표시
4. KoNLPy/Okt/MeCab 설치 설명 완화
5. p45 분석기 비교와 KoalaNLP/ETRI 절차를 짧게 보강
6. 상단 HTML 기획 주석 제거

원본 PDF와 초안은 수정하지 않았습니다.

## 공식 문서 확인 메모

- NLTK data 설치: https://www.nltk.org/data.html
- KoNLPy 설치: https://konlpy.org/en/latest/install/
- kss README: https://github.com/hyunwoongko/kss
- PyKoSpacing README: https://github.com/haven-jeon/PyKoSpacing
- WordCloud README: https://github.com/amueller/word_cloud
