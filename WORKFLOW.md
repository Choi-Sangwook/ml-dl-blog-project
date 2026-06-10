# ML/DL Blog Workflow

단계별 전체 프롬프트는 [PROMPTS.md](PROMPTS.md)를 사용합니다.

## 1. Add Source Files

Place lecture PDFs in `sources/`.

Example:

```text
sources/DAY1_딥러닝_개념과_프레임워크설치.pdf
```

## 2. Ask Claude Code For The Draft

```text
$write-ml-blog-post를 사용해 sources의 DAY1 PDF를 분석해줘.

파이썬 기본을 아는 딥러닝 입문자를 대상으로 글의 기획과 전체 초안을 작성하고,
drafts/DAY1_딥러닝_개념과_프레임워크설치_초안_v1.md에 저장해줘.

이미지로만 구성된 PDF 페이지도 확인하고, PDF 외 추가 설명은 별도로 구분해줘.
```

## 3. Ask Codex For The Review

```text
$write-ml-blog-post를 사용해 sources의 DAY1 PDF와 drafts의 최신 초안을 대조해줘.

PDF에서 빠진 내용, 잘못되거나 과장된 설명, 더 자세히 설명할 부분,
코드 실행 문제, 프레임워크·텐서 shape·손실함수 조합, 데이터 누수를 검토해줘.

수정 우선순위와 바로 붙여 넣을 수 있는 수정안을 포함한 하나의 통합 검토 문서를
reviews/DAY1_딥러닝_개념과_프레임워크설치_Codex_통합검토안.md에 저장해줘.

PDF와 초안은 수정하지 마.
```

## 4. Ask Claude Code For The Final Post

```text
$write-ml-blog-post를 사용해 sources의 PDF, drafts의 최신 초안,
reviews의 Codex 통합 검토안을 함께 읽어줘.

정확성과 코드 검증에 관한 수정 사항을 우선 반영하고,
입문 범위를 벗어나는 선택적 추가 내용은 필요한 것만 반영해줘.

완성본은 posts/DAY1_딥러닝_개념과_프레임워크설치_최종본.md에 저장하고
기존 초안과 검토안은 변경하지 마.
```

## Project Layout

```text
ml-dl-blog-project/
├─ AGENTS.md
├─ CLAUDE.md
├─ WORKFLOW.md
├─ .agents/
│  └─ skills/
│     └─ write-ml-blog-post/
├─ .claude/
│  └─ skills/
│     └─ write-ml-blog-post/
├─ sources/
├─ drafts/
├─ reviews/
├─ posts/
└─ archive/
```
