<#
.SYNOPSIS
  ML/DL 블로그 파이프라인에서 Codex 검토 단계(2단계·4단계)를 비대화형으로 실행하는 얇은 래퍼.

.DESCRIPTION
  프롬프트 본문은 이 스크립트에 두지 않는다. PROMPTS.md 템플릿을 채운 프롬프트 파일을
  -PromptFile 로 넘기면, 이 스크립트는 codex exec 호출 플래그만 일관되게 관리한다.
  - Codex 가 reviews/ 에 검토 문서를 직접 쓸 수 있도록 workspace-write 샌드박스 사용
  - 마지막 요약 메시지는 -OutLast 파일로 저장(오케스트레이터가 읽어 사용자에게 보고)
  - 전체 transcript 는 stdout 으로 스트리밍(필요하면 호출 측에서 Tee/리다이렉트)

.PARAMETER PromptFile
  PROMPTS.md 의 2단계 또는 4단계 템플릿에 실제 파일 경로를 채운 프롬프트 텍스트 파일(UTF-8).

.PARAMETER OutLast
  Codex 의 마지막 메시지(요약/보고)를 저장할 파일 경로.

.PARAMETER Sandbox
  codex 샌드박스 모드. 기본 workspace-write (reviews/ 쓰기 허용). 4단계처럼
  파일을 쓰지 않게 하려면 read-only 로 호출(단, 보고서를 reviews/ 에 저장하려면 write 필요).

.EXAMPLE
  pwsh -File scripts/codex-review.ps1 -PromptFile .tmp/stage2.txt -OutLast .tmp/stage2_last.txt
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$PromptFile,
  [Parameter(Mandatory = $true)][string]$OutLast,
  [ValidateSet('workspace-write', 'read-only', 'danger-full-access')]
  [string]$Sandbox = 'workspace-write',
  # Codex 가 프로젝트 루트에 만드는 tmp* 스크래치를 실행 후 자동 정리한다.
  # 끄려면 -KeepScratch 를 지정.
  [switch]$KeepScratch
)

$ErrorActionPreference = 'Stop'
# 한글 프롬프트가 codex(stdin)로 전달될 때 깨지지 않도록 UTF-8 강제.
# Windows PowerShell 5.1 은 네이티브 exe 파이프에 $OutputEncoding(기본 ASCII)을 사용하므로
# [Console]::OutputEncoding 만으로는 부족하고 $OutputEncoding 도 반드시 UTF-8 로 설정해야 한다.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath $PromptFile)) {
  throw "프롬프트 파일을 찾을 수 없습니다: $PromptFile"
}

# codex CLI 확인
$codex = Get-Command codex -ErrorAction SilentlyContinue
if (-not $codex) {
  throw "codex CLI 를 PATH 에서 찾을 수 없습니다. 'codex --version' 으로 설치를 확인하세요."
}

$prompt = Get-Content -Raw -Encoding UTF8 -LiteralPath $PromptFile

Write-Host "[codex-review] root=$Root sandbox=$Sandbox out=$OutLast" -ForegroundColor Cyan

# Codex 가 PDF 분석 중 루트에 tmp* 스크래치 폴더를 만든다(우리 스크래치패드 규칙을 모름).
# 실행 전 기존 tmp* 목록을 스냅샷해 두고, 실행 후 "새로 생긴 것만" 제거한다.
$scratchBefore = @(Get-ChildItem -LiteralPath $Root -Force -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -like 'tmp*' -or $_.Name -eq '.tmp' } | Select-Object -ExpandProperty Name)

# 프롬프트는 stdin 으로 전달('-'). codex 가 PROMPTS.md 지시대로 reviews/ 에 결과를 직접 저장한다.
$prompt | & codex exec -s $Sandbox -C $Root -o $OutLast --color never --skip-git-repo-check -

$code = $LASTEXITCODE

if (-not $KeepScratch) {
  $scratchAfter = @(Get-ChildItem -LiteralPath $Root -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'tmp*' -or $_.Name -eq '.tmp' } | Select-Object -ExpandProperty Name)
  $new = $scratchAfter | Where-Object { $_ -notin $scratchBefore }
  foreach ($n in $new) {
    try {
      Remove-Item -LiteralPath (Join-Path $Root $n) -Recurse -Force -ErrorAction Stop
      Write-Host "[codex-review] cleaned codex scratch -> $n" -ForegroundColor DarkGray
    }
    catch {
      Write-Host "[codex-review] could not remove scratch '$n': $($_.Exception.Message)" -ForegroundColor Yellow
    }
  }
}

if ($code -ne 0) {
  Write-Host "[codex-review] codex exec FAILED (exit=$code)" -ForegroundColor Red
}
else {
  Write-Host "[codex-review] done. last message -> $OutLast" -ForegroundColor Green
}
exit $code
