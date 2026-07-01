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

# Codex 는 PDF 페이지를 이미지로 렌더링하며 스크래치를 만든다.
# 정책: 자동 삭제하지 않는다. 모든 코덱스 이미지 스크래치는 archive/image/ 한 곳에 모으고,
#       삭제는 사용자가 나중에 일괄로 한다.
# 프롬프트에서도 코덱스에 archive/image/ 아래에 저장하라고 지시하지만,
# 지시를 어겨 루트에 스크래치(tmp_*, .codex_tmp 등)를 남긴 경우 아래 로직이 그것을
# 삭제 대신 archive/image/ 로 "이동"해 루트를 깨끗하게 유지한다.
$ImageDir = Join-Path $Root 'archive/image'
if (-not (Test-Path -LiteralPath $ImageDir)) {
  New-Item -ItemType Directory -Force -Path $ImageDir | Out-Null
}

# 실행 전 루트의 tmp*/.tmp 스냅샷(사용자 폴더 보호: 새로 생긴 것만 대상)
$tmpBefore = @(Get-ChildItem -LiteralPath $Root -Force -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -like 'tmp*' -or $_.Name -eq '.tmp' } | Select-Object -ExpandProperty Name)

# 프롬프트는 stdin 으로 전달('-'). codex 가 PROMPTS.md 지시대로 reviews/ 에 결과를 직접 저장한다.
$prompt | & codex exec -s $Sandbox -C $Root -o $OutLast --color never --skip-git-repo-check -

$code = $LASTEXITCODE

if (-not $KeepScratch) {
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  # (a) 새로 생긴 tmp*/.tmp
  $tmpAfter = @(Get-ChildItem -LiteralPath $Root -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'tmp*' -or $_.Name -eq '.tmp' } | Select-Object -ExpandProperty Name)
  $newTmp = @($tmpAfter | Where-Object { $_ -notin $tmpBefore })
  # (b) .codex_tmp* (재사용되어도 대상). `.codex` 설정 폴더는 제외하려고 `.codex_tmp` 로 좁게 매칭.
  $codexScratch = @(Get-ChildItem -LiteralPath $Root -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like '.codex_tmp*' } | Select-Object -ExpandProperty Name)

  foreach ($n in @($newTmp + $codexScratch | Select-Object -Unique)) {
    $src = Join-Path $Root $n
    $leaf = ($n -replace '^\.', 'dot_')          # 숨김 폴더가 되지 않도록 앞 점 치환
    $dst = Join-Path $ImageDir ("{0}_{1}" -f $stamp, $leaf)
    try {
      Move-Item -LiteralPath $src -Destination $dst -Force -ErrorAction Stop
      Write-Host "[codex-review] moved stray scratch -> archive/image/$(Split-Path $dst -Leaf)" -ForegroundColor DarkGray
    }
    catch {
      Write-Host "[codex-review] could not move scratch '$n': $($_.Exception.Message)" -ForegroundColor Yellow
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
