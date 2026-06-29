# 자동 파이프라인 (방식 A: Claude Code 오케스트레이터)

`source PDF` 하나만 지정하면 **1~4단계가 자동으로 이어서 실행**되고, 4단계 보고에서 멈춘다.
5단계(재수정)는 사용자가 명시적으로 요청할 때만 실행한다.

- 프롬프트 원본: [PROMPTS.md](PROMPTS.md) (단일 출처. 이 문서는 프롬프트를 복제하지 않고 참조한다)
- 단계별 소유권: `drafts/`·`posts/` = Claude / `reviews/` = Codex (AGENTS.md·CLAUDE.md 규칙)
- Codex 호출 래퍼: [scripts/codex-review.ps1](scripts/codex-review.ps1)

## 실행 트리거

사용자가 다음과 같이 요청하면 이 런북을 따른다:

```text
파이프라인 돌려줘: sources/DAY3_딥러닝_CNN.pdf
```

`[DAY]`, `[분야]`, `[주제]`는 PDF 파일명에서 추출한다. 추출이 모호하면 진행 전에 사용자에게 확인한다.

## 단계 순서

| 단계 | 담당 | 입력 | 출력 | 비고 |
|---|---|---|---|---|
| 1 | **Claude** (직접) | `sources/[PDF]` | `drafts/[DAY]_[분야]_[주제]_초안_v1.md` | PROMPTS.md 1단계 |
| 2 | **Codex** (`codex exec`) | PDF + 초안 | `reviews/[DAY]_[분야]_[주제]_Codex_통합검토안.md` | PROMPTS.md 2단계 |
| 3 | **Claude** (직접) | PDF + 초안 + 검토안 | `posts/[DAY]_[분야]_[주제]_최종본.md` | PROMPTS.md 3단계 |
| 4 | **Codex** (`codex exec`) | PDF + 검토안 + 최종본 | `reviews/[DAY]_[분야]_[주제]_Codex_최종점검.md` | PROMPTS.md 4단계, **여기서 멈춤** |
| 5 | **Claude** (직접) | 최종본 + 최종점검 | `posts/[DAY]_[분야]_[주제]_최종본_v2.md` | PROMPTS.md 5단계, **사용자 요청 시에만** |

## Claude 단계(1·3) 수행 방법

`write-ml-blog-post` 스킬을 사용해 PROMPTS.md의 해당 단계 프롬프트 기준 그대로 작성한다.
이미지 전용 PDF 페이지도 확인하고, PDF 내용과 추가 설명을 구분한다.

## Codex 단계(2·4) 호출 방법

1. PROMPTS.md의 해당 단계 템플릿에 실제 파일 경로를 채워 **프롬프트 파일**로 저장한다
   (임시 경로 사용, 예: 스크래치패드 또는 `.tmp/stageN.txt`).
2. 래퍼를 호출한다:

   ```powershell
   pwsh -File scripts/codex-review.ps1 `
     -PromptFile <프롬프트파일> `
     -OutLast    <코덱스_마지막메시지_저장경로>
   ```

   - Codex 가 PROMPTS.md 지시대로 `reviews/...md` 파일을 **직접** 생성한다.
   - `-OutLast` 에는 Codex 의 요약 메시지가 저장되어, 오케스트레이터가 읽어 사용자에게 보고한다.
   - **스크래치 자동 정리**: Codex 는 PDF 분석 중 프로젝트 루트에 `tmp*` 폴더를 만든다(우리 스크래치패드 규칙을 모름). 래퍼가 실행 전/후 `tmp*` 목록을 비교해 **이번 실행에서 새로 생긴 것만** 삭제한다(기존 폴더는 보존). 디버깅 등으로 남기려면 `-KeepScratch` 를 준다. `.gitignore` 에도 `tmp*` 안전망이 있다.
   - **리다이렉션 금지**: 래퍼 호출 시 `*>` / `2>&1` 로 stdout+stderr 를 파일에 합치지 마라. Windows PowerShell 5.1 은 codex(node) 의 stderr 배너를 오류로 감싸고, 래퍼의 `ErrorActionPreference='Stop'` 과 겹쳐 codex 가 조기 종료된다. 출력을 버리려면 `| Out-Null` 을 쓴다.
3. 4단계 프롬프트는 "최종 포스트를 직접 수정하지 마"를 포함해야 하며, 보고 결과는
   `reviews/[DAY]_..._Codex_최종점검.md` 로 저장하도록 지시한다(게시 가능/반드시 수정/선택 개선 3구역).

## 정지 지점 및 보고

4단계 완료 후 **자동 진행을 멈추고** 사용자에게 다음을 보고한다:

- 각 단계 산출물 경로(초안 / 검토안 / 최종본 / 최종점검)
- 4단계 결론: **게시 가능 / 게시 전 반드시 수정 / 선택적 개선**
- 실행·확인하지 못한 부분(예: 복구 실패한 스크린샷 코드, OCR 불확실 페이지)

"게시 전 반드시 수정" 항목이 있으면 5단계를 권유하되, **실행은 사용자 승인 후**에만 한다.

## 안전 규칙

- 원본 PDF는 절대 수정하지 않는다.
- Claude 는 `reviews/` 파일을 수정하지 않는다(Codex 전용).
- 기존 초안·검토안·최종본을 말없이 덮어쓰지 않는다. 재수정은 버전 증가(`_v2`)로 만든다.
- 단계가 실패하면(예: codex 비정상 종료) 다음 단계로 넘어가지 말고 멈춘 뒤 원인과 함께 보고한다.
- PDF 의미·DAY 번호·다음 DAY 주제가 불확실하면 추측하지 말고 확인한다.
