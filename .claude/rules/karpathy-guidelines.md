# Careful And Focused Work Guidelines

Apply these principles together with the project workflow in `AGENTS.md`.
These rules favor accuracy and restraint over speed. Use proportionate judgment for trivial tasks.

## 1. Resolve Uncertainty Before Writing

- Do not silently guess the PDF's meaning, the requested DAY number, or the intended output stage.
- State material assumptions when the source is ambiguous.
- When multiple interpretations would change the post substantially, explain the alternatives and ask the user.
- Prefer the simplest interpretation that is supported by the PDF and project files.
- Do not hide unreadable pages, missing code, uncertain OCR, or conflicting explanations.
- When a claim cannot be verified, mark it as uncertain or omit it.

For this project, asking is especially appropriate when:

- it is unclear whether the task is drafting, reviewing, or final revision;
- deep-learning DAY numbering may be separate from the machine-learning series;
- a screenshot-only code block cannot be recovered reliably;
- the PDF and Codex review appear to conflict;
- the next DAY topic has not been provided.

## 2. Keep The Post As Simple As The Lesson Allows

- Add only explanations that help a Python beginner understand the supplied lesson.
- Do not expand an introductory post into a complete textbook chapter.
- Do not add advanced models, mathematical proofs, libraries, or deployment topics unless they support the lesson.
- Avoid abstractions or helper functions used only once when direct code is clearer.
- Prefer one coherent runnable example over several incomplete examples.
- Use diagrams, analogies, tables, and callouts only when they improve understanding.
- Remove repeated comparisons and summaries.

Before adding a section, ask:

> Does this help the reader understand, run, interpret, or safely use the concept in this PDF?

If not, leave it out or mark it as an optional extension.

## 3. Change Only The Current Stage's Files

- Follow the directory ownership rules in `AGENTS.md`.
- During drafting, write only the requested file under `drafts/`.
- During final revision, write only the requested file under `posts/`.
- Do not edit PDFs or Codex review files.
- Do not rewrite unrelated posts, naming conventions, or project instructions.
- Match the established style of neighboring DAY posts.
- Preserve existing source, draft, review, and final files unless the user explicitly requests replacement.
- Clean up only temporary artifacts created by the current task.

Every changed file must have a direct reason traceable to the user's request.

## 4. Define Completion In Verifiable Terms

For multi-step work, establish a short plan with a check for each step.

Example:

```text
1. Map PDF pages to topics
   Check: every major section and screenshot-only block is represented.

2. Write or revise the post
   Check: concepts, code, output interpretation, and cautions are included.

3. Validate the artifact
   Check: Markdown, UTF-8, code consistency, links, and output path are correct.
```

Before declaring a draft complete, verify:

- every major PDF topic is covered or intentionally deferred;
- PDF-derived content and outside additions are distinguishable;
- code uses one clearly identified framework per complete example;
- tensor shapes, labels, outputs, and loss functions are compatible;
- the post does not invent results, sources, or future topics;
- the file was saved to the correct stage directory;
- the source PDF and previous artifacts remain unchanged.

When verification cannot be completed, report exactly what remains unverified.

## 5. Prefer Corrections Over Polished Guesswork

- Accuracy takes priority over smooth wording.
- A short, correct explanation is better than a confident but unsupported one.
- Do not preserve an attractive analogy if it creates a technical misconception.
- Do not claim a code example works unless it was executed or carefully validated.
- When the Codex review identifies a high-priority correctness issue, resolve it before stylistic improvements.

