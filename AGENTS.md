# ML/DL Blog Project

## Purpose

Create beginner-friendly Korean ML/DL Velog posts from lecture PDFs through a three-stage workflow:

1. Claude Code plans and writes the first draft.
2. Codex compares the PDF and draft and writes one integrated review.
3. Claude Code applies the review and writes the final post.

Use the `write-ml-blog-post` skill for all three stages.

## Audience

- The reader knows basic Python.
- The reader is learning machine learning or deep learning for the first time.
- Explain terminology, code output, assumptions, and common mistakes.

## Source Of Truth

- Treat the PDF in `sources/` as the primary lecture source.
- Clearly distinguish PDF-derived content from prerequisite or practical additions.
- Do not describe an outside addition as though it appeared in the PDF.
- Inspect image-only PDF pages because code, formulas, and diagrams may not have a text layer.

## Directory Ownership

```text
sources/  Original lecture PDFs and supplied source materials
drafts/   Claude Code planning documents and first drafts
reviews/  Codex PDF comparison and integrated review documents
posts/    Claude Code final posts
archive/  Superseded versions retained for history
```

Do not overwrite files from another stage.

## File Naming

```text
sources/DAY[N]_[머신러닝|딥러닝]_[주제].pdf
drafts/DAY[N]_[머신러닝|딥러닝]_[주제]_초안_v1.md
reviews/DAY[N]_[머신러닝|딥러닝]_[주제]_Codex_통합검토안.md
posts/DAY[N]_[머신러닝|딥러닝]_[주제]_최종본.md
```

When another revision is needed, increment the draft version instead of replacing it:

```text
drafts/..._초안_v2.md
```

## Claude Code Stage 1

When creating a draft:

- Read the complete PDF, including screenshot-only pages.
- First make a page-to-topic map and writing plan.
- Create one complete beginner-oriented post.
- Save it under `drafts/`.
- Do not create a file under `reviews/` or `posts/`.
- Do not invent a next-DAY topic when it is unknown.

## Codex Stage 2

When reviewing:

- Compare the PDF and latest draft section by section.
- Check missing PDF content, conceptual accuracy, runnable code, framework consistency, tensor shapes, data leakage, evaluation choices, and beginner readability.
- Separate required corrections from optional additions.
- Include paste-ready corrections when useful.
- Create one integrated review under `reviews/`.
- Preserve the draft and PDF unchanged.
- Do not write the final post unless explicitly requested.

## Claude Code Stage 3

When producing the final post:

- Read the PDF, latest draft, and Codex integrated review.
- Apply high-priority accuracy and code corrections first.
- Apply medium and low-priority suggestions only when they improve the introductory scope.
- Resolve conflicts in this order:
  1. Verified source facts and runnable behavior
  2. Codex accuracy and safety findings
  3. Existing draft wording and organization
- Save one polished final post under `posts/`.
- Preserve the draft and review.

## Version Safety

- Before editing, identify the latest file by explicit version number, not modification time alone.
- Never edit the original PDF.
- Never silently replace a draft, review, or final post.
- Move obsolete deliverables to `archive/` only when the user requests it.
- If Claude Code and Codex may work concurrently, use separate Git branches or worktrees.

## Final Checks

- Confirm Korean text is UTF-8.
- Check Markdown headings, tables, links, and code fences.
- Report the exact output path.
- State when code could not be executed or a screenshot could not be reliably recovered.

