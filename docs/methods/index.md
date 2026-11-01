# Methods

The methods pages describe each method of the library as the library implements it: what the method is, the algorithm with its constants, what it assumes and which errors it corrects, its parameters, a small example on the toy corpus, and the literature it rests on. The methods follow the accompanying methods paper (under review). Where the library departs from the paper's analysis code, the page says what the library does and what the analysis code did.

The examples run on a laptop CPU, with the fake backend for the LLM coders and the small encoder MiniLM for the embeddings, like the [tutorials](../getting-started.md). The fake backend answers from a table of cue words, so its scores show how a method works, not how well a real model codes your texts.

## Detection methods

| Page | What the method does |
|------|----------------------|
| [Two-stage LLM coder](two-stage-llm-coder.md) | Extracts verbatim passages per marker, verifies them against the text, then scores the attribute from the verified passages alone |
| [One-stage LLM coder](one-stage-llm-coder.md) | Scores each attribute in one model call, without passages |
| [Similarity coder](similarity-coder.md) | Scores a text by the cosine of its embedding to reference texts of the two poles |
| [Hybrid coder](hybrid-coder.md) | Rebuilds each text from the verified passages of a two-stage run and scores it by similarity |
| [NLP-ML baseline](nlp-ml-baseline.md) | Predicts the human values of the human-coded texts from embeddings with a ridge regression, out of fold, as the benchmark the coders are read against; it scores no other text, and [embedding-augmented calibration](corrections/embedding-augmented-calibration.md) scores a corpus from the same kind of fit |

[The validity panel](validity-panel.md) checks any of these scores against human values: rank validity, the four score errors, a reliability ceiling, and intervals from a bootstrap of whole groups.

## The four score errors

A machine score can be wrong in four ways. The validity panel measures each on a 0 to 1 scale on which the score and the human values are each scaled by their own observed range; [its page](validity-panel.md) says what that scale shows and what it hides.

- **Level displacement**: the scores are too high or too low throughout.
- **Calibration shortfall**: the scores are squeezed toward the middle (or spread too far), so they move less (or more) than the human value.
- **Covariate-linked error**: the error changes with text length or another observed variable.
- **Noise**: the same text gets a different score on another run.

## The four correction targets

A correction is built to reproduce one quantity, its target. The same corrected values can be valid for one target and invalid for another, so each correction page names its target.

| Target | What the correction reproduces | Judged by |
|--------|--------------------------------|-----------|
| A | The best guess of each text's human value, given its score and features | Rank validity, mean absolute error and covariate-linked error on held-out texts |
| B | Plausible draws of each text's human value, combined across draws by Rubin's rules | The coverage of the pooled interval |
| C | An estimate over all texts with the construct as the outcome: a mean, a total, a trend, a difference between groups, a coefficient with the construct as outcome | The bias and coverage of the estimate |
| D | The coefficient of the construct as a predictor in a regression | The bias and coverage of the coefficient |

A correction at the **score** level gives every text a new value: most score-level families aim at target A and multiple imputation at B, while the misclassification matrix also serves C for class shares and embedding-augmented calibration serves C under pseudo-values, as the table below shows. A correction at the **estimate** level leaves the scores unchanged and corrects the estimate computed from them (targets C and D). [Corrections](corrections/index.md) introduces the families, the human sample they use, cross-fitting and the battery that compares them.

## Correction families

| Page | Target | Level |
|------|--------|-------|
| [Regression calibration](corrections/regression-calibration.md) | A | score |
| [Marginal matching](corrections/marginal-matching.md) | A | score |
| [Multiple imputation](corrections/multiple-imputation.md) | B | score |
| [Misclassification matrix](corrections/misclassification-matrix.md) | A, C | score |
| [Pseudo-values and prediction-powered inference](corrections/pseudo-values.md) | C, D | estimate |
| [Latent pooling](corrections/latent-pooling.md) | A | score |
| [Supervised stacking](corrections/supervised-stacking.md) | A | score |
| [Classical corrections](corrections/classical-corrections.md) | A, D | score, estimate |
| [Embedding-augmented calibration](corrections/embedding-augmented-calibration.md) | A, then C | score |
| [Correction ensemble](corrections/correction-ensemble.md) | A | score |
| [Instrumented coding](corrections/instrumented-coding.md) | D | estimate |

## Literature

Every page lists the works it rests on. [References](references.md) gives the full entries, copied from the reference list of the accompanying methods paper (under review).
