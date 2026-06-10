---
name: write-ml-blog-post
description: Convert machine-learning or deep-learning lecture PDFs and optional Markdown drafts into beginner-friendly Korean Velog posts, or review and revise existing DAY-series posts. Use when the user asks to create, review, supplement, merge, or rewrite an ML/DL blog post from lecture material while checking PDF coverage, conceptual accuracy, runnable Python code, framework consistency, data leakage, evaluation choices, and practical caveats.
---

# Write ML Blog Post

Create or review Korean machine-learning and deep-learning posts for readers who know basic Python but are new to the subject. Treat the supplied PDF as the primary source, preserve its educational intent, and clearly separate source-derived content from useful additions.

Read [references/quality-checklist.md](references/quality-checklist.md) for every task. Read [references/series-style.md](references/series-style.md) when working on the user's DAY-series posts.

Read [references/deep-learning-checklist.md](references/deep-learning-checklist.md) when the material includes neural networks, TensorFlow, Keras, PyTorch, tensors, GPU setup, image datasets, training loops, or deep-learning framework installation.

## Select The Task Mode

Infer the mode from the user's request. Do not ask when the intent is clear.

- **Generate**: Create a complete post from a PDF.
- **Review**: Compare an existing Markdown draft with its source PDF and create a review report.
- **Revise**: Apply the review and produce a polished replacement post.
- **Merge**: Combine separate review notes or revisions into one integrated proposal.

When the user says only "검토해줘", create a separate review file and preserve the original. Overwrite the original only when explicitly requested.

## Inspect The Inputs

1. Confirm that every supplied file exists and identify the PDF, draft, and related previous posts.
2. Determine the DAY number, topic, intended audience, and requested output.
3. Inspect the PDF page count, table of contents, headings, and text-layer quality.
4. Extract text with an available structured PDF tool such as `pypdf` or `pdftotext`.
5. If extraction is sparse or garbled, render pages and use OCR or visual inspection. Ask for another source only after available extraction methods fail.
6. Visually inspect pages containing diagrams, tables, formulas, code screenshots, or unusually little extracted text. Text extraction alone can omit important teaching content.
7. Build a short page-to-topic map before writing.

Treat consecutive low-text pages as a high-risk block. Inspect every page in the block rather than sampling only the first page, because lecture PDFs often store complete code examples as screenshots.

Do not copy long passages from the PDF. Paraphrase, reorganize, and explain in original language.

## Compare PDF And Draft

For review or revision tasks, compare the files section by section.

Classify findings as:

- **Missing**: Present in the PDF but absent from the draft.
- **Needs detail**: Present but too brief for a beginner.
- **Needs correction**: Inaccurate, overgeneralized, misleading, or incompatible with the code.
- **Useful addition**: Not required by the PDF but improves understanding or practical use.
- **Can reduce**: Repetitive, off-topic, too advanced, or already covered in another DAY.

Record PDF page numbers when useful. Never claim that an added explanation came from the PDF when it did not.

## Plan The Teaching Flow

Prefer this progression for each major concept:

1. Explain why the concept is needed.
2. Give an intuitive explanation.
3. Define the technical term and notation.
4. Explain the mechanism step by step.
5. Show a minimal, relevant Python example.
6. Interpret the output or visualization.
7. Explain when to use it, its assumptions, and its limitations.
8. Compare it with nearby alternatives only when the comparison helps selection.

Preserve the lecture's scope. Add prerequisites and common failure modes, but do not turn an introductory post into an exhaustive textbook chapter.

Use analogies, diagrams, tables, formulas, and callouts only when they clarify the concept. Do not force one into every section.

## Write And Validate Code

Make each code block either:

- independently runnable, with imports and data preparation included; or
- explicitly labeled as a partial snippet or pseudocode.

Apply these rules:

- Use stable public datasets or small generated examples.
- Set `random_state` where randomness affects reproducibility.
- Include required imports and installation notes for optional packages.
- Explain important output instead of ending immediately after code.
- Do not invent execution results.
- Run representative examples when the environment permits.
- Verify version-sensitive APIs against official documentation when needed.
- Keep one framework consistent within a complete example. Clearly label framework changes when the lecture uses both PyTorch and TensorFlow/Keras.
- Check tensor shapes, data types, label formats, model outputs, loss functions, and devices together.

Apply data splitting conditionally:

- For supervised prediction, split data before fitting learned preprocessing and fit transformations only on training data.
- For cross-validation, keep preprocessing and the estimator in a `Pipeline`.
- For exploratory clustering or visualization, do not force `train_test_split`.
- If an unsupervised transform or cluster feature feeds a later predictive model, fit it on training data only.
- For time-ordered data, use chronological validation instead of random splitting.

Use only evaluation methods that fit the task. Do not force a confusion matrix into an unsupervised post or a residual plot into a classification post.

## Produce The Requested Output

### Generate Or Revise

Produce a complete Markdown post following the established series style. Include:

- a clear title and short scope-setting introduction;
- logically numbered sections;
- concept, code, interpretation, and caveats;
- a concise final summary;
- references that were actually used.

Include a next-DAY preview only when the next topic is known. Do not invent one.

### Review

Create one integrated review document with:

1. Overall assessment
2. High-priority corrections
3. Missing PDF content
4. Explanations that need more detail
5. Useful additions
6. Content to reduce or remove
7. Paste-ready replacement or addition blocks
8. A priority table
9. Final recommendation

Prefer actionable revisions over general comments. Include code or wording that can be inserted directly when practical.

## Save Safely

Follow these filename patterns unless the user specifies another:

```text
DAY[N]_[머신러닝|딥러닝]_[주제].md
DAY[N]_[머신러닝|딥러닝]_Velog_통합검토안.md
DAY[N]_[머신러닝|딥러닝]_Velog_수정본.md
```

Preserve the source file by default. Save the deliverable in the workspace or an explicitly requested writable location, then report its exact path.

## Final Quality Gate

Before finishing:

1. Confirm that all major PDF topics were included, intentionally deferred, or listed as omissions.
2. Check technical claims, formulas, terminology, and algorithm assumptions.
3. Check that code and prose describe the same preprocessing and model.
4. Check task-appropriate evaluation and data-leakage safeguards.
5. Check that every added claim is presented as an addition rather than PDF content.
6. Remove unsupported absolutes such as "always", "perfectly", or "guarantees".
7. Remove padding, repeated summaries, fake quotations, and unused references.
8. Confirm headings, code fences, tables, links, and Korean text render correctly.
9. Confirm the final artifact exists and is readable.
10. For deep-learning posts, complete the framework, shape, device, loss/output, dataset, and training-state checks in the deep-learning checklist.
