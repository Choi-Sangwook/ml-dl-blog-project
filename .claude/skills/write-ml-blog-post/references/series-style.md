# Machine Learning DAY Series Style

Use this reference for the user's established machine-learning series and its following deep-learning series. Preserve newer conventions found in the supplied posts when they conflict with this reference.

## Audience

- Knows basic Python syntax and common data structures.
- Is learning machine learning for the first time.
- Benefits from intuition, small examples, and explicit output interpretation.

## Title

```markdown
# [이모지] [머신러닝|딥러닝] 완전 입문 가이드 - DAY[N]. [주제]
```

Match the punctuation already used in adjacent posts. Use one topic-relevant emoji rather than decorating every heading.

## Recommended Post Shape

```markdown
# [이모지] [머신러닝|딥러닝] 완전 입문 가이드 - DAY[N]. [주제]

> **시리즈**: 파이썬 기본만 있는 사람을 위한 [머신러닝|딥러닝] 입문
> **이전 편**: [DAY와 제목]

---

## 1. 이번 DAY에서 배우는 것

## 2. 핵심 개념 또는 알고리즘 1

### 2.1 왜 필요한가
### 2.2 동작 원리
### 2.3 Python 실습
### 2.4 결과 해석과 주의사항

## 3. 핵심 개념 또는 알고리즘 2

## 4. 알고리즘 선택 또는 비교

## 5. 자주 하는 실수

## 6. DAY[N] 핵심 정리

## 참고 자료
```

Adapt section count and names to the lecture. Do not force empty sections.

## Series Connection

Prefer a compact previous/current/next connection over reproducing a long hardcoded index. If the existing posts use the full history list, preserve that convention and verify every entry from actual files.

Do not infer a next topic when no syllabus or user instruction supports it.

Determine whether the supplied material starts a new deep-learning series or continues numbering from the machine-learning series. Use the filenames, neighboring posts, or explicit user instruction. Do not assume that deep-learning DAY1 must become machine-learning DAY8.

## Completed Topics Snapshot

Treat this as a starting snapshot, not unquestionable current truth.

| DAY | Topic |
|---|---|
| DAY1 | 머신러닝 핵심 개념과 데이터 관리 |
| DAY2 | 회귀 모델과 분류 모델 |
| DAY3 | SVM, KNN, 나이브 베이즈 |
| DAY4 | 결정트리와 회귀트리 |
| DAY5 | 앙상블과 신경망 |
| DAY6 | 차원 축소와 시각화 |
| DAY7 | 비지도학습 알고리즘 |

Verify the title and filename from supplied files before adding a new entry.

The supplied deep-learning lecture files currently form a separate apparent sequence:

| DAY | Topic |
|---|---|
| DAY1 | 딥러닝 이해 및 프레임워크 설치 |
| DAY2 | 신경망 알고리즘과 계층 구조 |
| DAY3 | 데이터 처리와 데이터셋 |

Confirm the user's preferred series title and numbering from the draft when one becomes available.

## Explanation Pattern

For each major algorithm, cover only relevant items:

- one-sentence definition;
- intuitive mechanism;
- important parameters;
- minimal runnable example;
- output interpretation;
- strengths and limitations;
- common mistakes;
- comparison with alternatives.

Use short callouts such as `> 참고:` or `> 주의:`. Do not require a "쉬운 비유" box when a direct explanation is clearer.

## Summary Pattern

Use a concise list or code-block summary when it matches adjacent posts:

```text
핵심 개념
  - 무엇을 해결하는가
  - 언제 사용하는가
  - 무엇을 주의하는가
```

Avoid repeating full sections in the conclusion.

## Output Names

```text
DAY[N]_[머신러닝|딥러닝]_[주제].md
DAY[N]_[머신러닝|딥러닝]_Velog_통합검토안.md
DAY[N]_[머신러닝|딥러닝]_Velog_수정본.md
```

