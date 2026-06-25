# DAY2 자연어처리 워드임베딩 Codex 최종점검

## 게시 가능

아직은 **게시 전 반드시 수정 1건**이 있어서, 그대로는 게시 보류가 좋습니다.

반영 잘 된 부분:

- 기존 검토안의 핵심 수정 대부분 반영됨: `X` 미정의 문제 해결, TF-IDF `smooth_idf` 설명 추가, PCA 설명분산비 `[0.465, 0.303]`로 수정, 데이터 누수 주의 추가, Word2Vec `KeyError`/작은 코퍼스 한계 추가.
- PDF 핵심 내용 누락 없음: 원핫, 워드 벡터, 코사인 유사도, 분산 가설, TF-IDF, LDA, Word2Vec, PCA/t-SNE, 실무 활용까지 반영됨.
- Markdown fence, UTF-8, 표, 링크, Python 코드 문법은 정상입니다.
- 딥러닝 출력층/손실함수/device/tensor 학습 코드는 없어서 해당 없음.

## 게시 전 반드시 수정

1. 문서 간 코사인 유사도 출력 행렬이 실제 실행 결과와 1곳 다릅니다.

최종본 현재 출력:

```text
[0.    0.    0.396 1.   ]
```

실제 재실행 결과:

```text
[0.    0.161 0.396 1.   ]
```

즉 마지막 행 두 번째 값은 `0.0`이 아니라 `0.161`입니다. 코사인 유사도 행렬은 대칭이어야 하므로 2행 4열의 `0.161`과 4행 2열도 같아야 합니다.

## 선택적으로 개선

- `pip install scikit-learn`만 언급하지만 PCA 시각화 코드에는 `matplotlib`도 필요합니다. "대부분"이라고 써서 큰 문제는 아니지만, 시각화 실습 전 `python -m pip install matplotlib`를 한 줄 덧붙이면 더 친절합니다.
- 공식 문서 기준으로 `TfidfVectorizer`의 `smooth_idf=True`, 기본 `token_pattern`, t-SNE `perplexity < n_samples`, gensim `Word2Vec(vector_size, window, min_count, sg...)` 설명은 현재 최종본과 맞습니다.

## 점검 입력

- 기준 PDF: `sources/DAY2_자연어처리를위한워드임베딩.pdf`
- 기존 검토안: `reviews/DAY2_자연어처리_워드임베딩_Codex_통합검토안.md`
- 최종 포스트: `posts/DAY2_자연어처리_워드임베딩_최종본.md`

## 검증 메모

- PDF 12쪽 텍스트 추출 기준 핵심 목차 확인.
- 최종 포스트의 Markdown 코드 fence 균형과 Python 코드 블록 문법 확인.
- 대표 scikit-learn 실습을 로컬 환경에서 재실행.
- 확인 환경: `scikit-learn 1.7.2`, `numpy 2.3.5`, `matplotlib 3.10.6`.
- `gensim`은 현재 로컬 환경에 설치되어 있지 않아 Word2Vec 출력은 검증하지 않았으며, 최종본도 출력값을 지어내지 않고 코드만 제시하고 있습니다.
