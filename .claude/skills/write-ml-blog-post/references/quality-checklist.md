# ML Blog Post Quality Checklist

Use this checklist selectively. Apply only criteria relevant to the post's problem type.

## 1. Source Fidelity

- Create a page-to-topic map before drafting.
- Compare the PDF table of contents with the post headings.
- Inspect diagrams, formulas, tables, and screenshots visually when text extraction may omit them.
- Distinguish three content types:
  - PDF-derived explanation
  - prerequisite explanation added for beginners
  - practical or advanced addition
- Do not attribute outside knowledge to the PDF.
- Do not reproduce long source passages verbatim.
- Mark content already covered in another DAY as intentionally deferred rather than missing.

For review tasks, use a compact coverage table when helpful:

| PDF topic/page | Draft status | Action |
|---|---|---|
| Topic A, p. 3-5 | Covered | Keep |
| Topic B, p. 6 | Too brief | Expand |
| Topic C, p. 9 | Missing | Add |
| Topic D, p. 12 | Covered in DAY6 | Link only |

## 2. Beginner Readability

- Assume basic Python knowledge but no ML background.
- Define each technical term on first use.
- Explain symbols in formulas and state what larger or smaller values mean.
- Connect array shapes to concepts when shape errors are likely.
- Introduce prerequisites before using them.
- Follow code with interpretation of output, graph axes, or metrics.
- Use one consistent dataset through a section when possible.
- State what the reader should notice in each example.
- Avoid unexplained library defaults.
- Avoid excessive history, trivia, and advanced variants.

## 3. Technical Accuracy

Check:

- whether the algorithm solves classification, regression, clustering, dimensionality reduction, association analysis, or anomaly detection;
- assumptions such as linearity, independence, scale sensitivity, cluster shape, or class balance;
- whether a score is optimized upward or downward;
- whether model output is a label, probability, distance, score, embedding, or reconstruction;
- whether a hyperparameter is an exact quantity, an estimate, a bound, or a threshold control;
- whether correlation or association is incorrectly described as causation;
- whether visualization is being mistaken for quantitative validation;
- whether cluster IDs are incorrectly treated as ordered labels;
- whether unsupervised output is presented as ground truth rather than a hypothesis requiring interpretation.

Prefer qualified wording:

```text
Avoid: "This always prevents overfitting."
Use:   "This can reduce overfitting, depending on the data and settings."
```

## 4. Code And Reproducibility

- Include imports, data loading, and variable definitions.
- Set `random_state` for random splits, stochastic models, and generated data.
- Use explicit parameters when library defaults are version-sensitive.
- Label shortened snippets and pseudocode clearly.
- Add installation instructions only for nonstandard dependencies.
- Avoid `try/except ImportError` unless the example genuinely provides a fallback path.
- Prefer structured APIs over manual parsing.
- Check feature names, target shape, missing values, and categorical encoding.
- Ensure plots have titles, axis labels, legends, and readable dimensions when those labels carry meaning.
- Execute representative code where possible.
- Do not publish fabricated output values.

## 5. Data Leakage

For supervised workflows:

- Split before fitting scalers, imputers, encoders, selectors, PCA, clustering, or resampling.
- Put learned preprocessing inside a `Pipeline`.
- Apply SMOTE or other resampling only to training folds.
- Keep the target out of feature engineering unless the method is explicitly supervised and validation-safe.
- Use chronological validation for time-dependent data.

For unsupervised workflows:

- A split is not mandatory for pure exploratory analysis.
- If embeddings, PCA components, anomaly scores, or cluster IDs become predictive features, fit their transformers on training data only.
- Explain the distinction instead of declaring that all preprocessing requires a train/test split.

## 6. Evaluation By Task

### Classification

Choose metrics based on class balance and error costs:

- confusion matrix
- precision, recall, F1
- ROC-AUC or PR-AUC
- log loss or calibration when probabilities matter

Accuracy alone is insufficient for imbalanced or cost-sensitive problems.

### Regression

Use relevant metrics:

- MAE
- RMSE
- R-squared
- residual plot

Explain units and sensitivity to outliers.

### Clustering

Use a combination of:

- silhouette score
- inertia/elbow for K-Means
- cluster counts and sizes
- cluster profile tables
- visualization when dimensions permit
- domain usefulness and stability

Do not compare cluster IDs directly with class IDs without alignment. State that cluster numbers are arbitrary.

### Dimensionality Reduction

Consider:

- explained variance for PCA
- reconstruction error for autoencoders
- neighborhood preservation or qualitative structure for visualization methods
- downstream task performance when reduction is used as preprocessing

Do not interpret t-SNE or UMAP axis values as original feature meanings.

### Anomaly Detection

When labels exist, consider precision, recall, F1, PR-AUC, and error costs.

When labels do not exist, use sample review, domain validation, sensitivity analysis, and comparison with known incidents or rules.

Clarify that `contamination` controls a decision threshold or expected fraction; it is not an anomaly probability.

### Association Rules

Explain:

- support
- confidence
- lift
- actual occurrence counts
- direction of the rule

State that association does not establish causation. Mention FP-Growth only when scale or performance is relevant.

### Neural Networks

Show:

- train/validation behavior
- loss and metric curves
- output-layer and loss-function compatibility
- overfitting controls

Do not claim that one regularization technique fully eliminates overfitting or vanishing gradients.

For framework installation, tensor operations, image datasets, or training code, also apply [deep-learning-checklist.md](deep-learning-checklist.md).

## 7. Structure And Style

- Preserve the established DAY-series title and numbering when applicable.
- Keep heading depth shallow and consistent.
- Use tables for comparisons, not for long prose.
- Use diagrams only when they improve understanding.
- Keep emoji use restrained and consistent with the existing series.
- Avoid arbitrary minimum length. Cover the topic fully without padding.
- Avoid repeating the same comparison table or summary in multiple sections.
- Include a next-DAY preview only when known.
- Link only references actually consulted.
- Prefer official documentation for library behavior and version-sensitive details.

## 8. Review Report Standard

Lead with correctness and learning risks, then completeness and presentation.

Use priority levels:

| Priority | Meaning |
|---|---|
| High | Incorrect, misleading, non-runnable, leaked, or central PDF content missing |
| Medium | Important beginner explanation, practical caveat, or evaluation guidance missing |
| Low | Optional extension, style refinement, or advanced reference |

For each finding, state:

1. where it occurs;
2. why it matters;
3. what to change;
4. a paste-ready example when useful.

## 9. Final Artifact Check

- Open the saved Markdown as UTF-8.
- Check that Korean characters are not corrupted.
- Check code-fence pairing and table alignment.
- Check local and web links.
- Check that the filename matches the requested DAY and topic.
- Report the exact saved path to the user.
