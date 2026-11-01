# Embedding-augmented calibration

Embedding-augmented calibration predicts each text's human value from the text's embedding, with or without the coder's score, by a penalised regression fitted on the human-coded sample (Zou and Hastie 2005 for the family of penalised linear models). The embedding carries information about the text that a single score does not, so the prediction can rank the texts better than the score, which no calibration of the score alone can do. Three variants separate what the score and the embedding contribute: the embeddings alone, the embeddings plus the score, and a calibration line of the score plus a model of its residual on the embeddings.

The corrected values aim at best guesses for single texts, which the fitted ridge only approximates. An estimate built from them, such as a difference between groups that the model did not see, can still be biased, so the family is meant to serve as the prediction inside [pseudo-values and prediction-powered inference](pseudo-values.md), the design-based supervised learning of Egami et al. (2023). There the corrected values narrow the interval where the embeddings add signal, while the human-coded sample keeps the point approximately unbiased.

## The algorithm as the library implements it

1. **Embeddings.** The vectors come from one embedding set of the data (`encoder=<name>`) or from every set concatenated (`encoder="stacked"`, the default, which needs at least two sets; with a single encoder, pass its name). Every text of the data must be in each set with a finite vector. Each set is z-scored on the training rows (a constant column keeps standard deviation 1); stacked sets are weighted by `sqrt(mean_dim / dim)` so that each contributes the same total variance.
2. **Variants**, each a ridge regression with an unpenalised intercept:
    - `embeddings_only`: the gold on the embeddings. It reads no score, so it is the yardstick for what the coder's score adds.
    - `embeddings_plus_score`: the gold on the embeddings and the best score column, the score standardised with its mean and population standard deviation over the training rows and penalised like the embeddings.
    - `score_anchored_residual`: the least-squares line of the gold on the best score column, plus a ridge of the line's residual on the embeddings. When the embeddings carry nothing, the chosen penalty shrinks the residual model toward zero and the corrected value stays on the line.
3. **Penalty.** The penalty is one of `ALPHAS` (41 values from 0.01 to 10^8, evenly spaced on the log scale). The groups of the training rows are split into `min(5, groups)` grouped folds; each fold is deleted in turn and the whole pipeline is refitted on the rest: the score column is selected again and the calibration line refitted on the retained rows, so neither the column, nor the line, nor the penalty sees the gold values of the rows whose error decides them. The squared errors of the deleted rows are pooled over all deletions, and the first penalty with the smallest mean squared error is chosen. The deletions follow a seed drawn from the fit's generator (`inner_seed`).
4. **Ridge.** For each deletion and for the final fit, ridge along the whole grid comes from one thin singular value decomposition of the centred design, dropping singular values at or below `1e-12` times the largest; this is scikit-learn's ridge with an intercept and the `svd` solver.
5. **Refit.** The variant is refitted on all training rows at the chosen penalty, with the score column selected on all of them. The provenance records the penalty, whether it sits on an edge of the grid, and the share of deletions whose own best penalty sits on an edge.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

spec = cop.corrections.REGISTRY["embedding_augmented"]
print(spec.target, spec.level)
print(spec.estimate_errors)
for variant in spec.variants:
    print(variant, ", ".join(spec.errors_for(variant)))
```

<!-- output -->
```text
A, then C score
with pseudo-values on top: bias and coverage, narrower confidence intervals
embeddings_only calibration_shortfall, level_displacement, covariate_linked_error
embeddings_plus_score calibration_shortfall, level_displacement, covariate_linked_error
score_anchored_residual calibration_shortfall, level_displacement, covariate_linked_error
```

The family aims at target A, and at target C with pseudo-values on top. It addresses the calibration shortfall, whose ceiling it raises with the information in the text, level displacement and covariate-linked error. It assumes that the human-coded sample comes from the corpus it corrects and holds enough groups for the nested penalty choice; with few groups the chosen penalty often sits on an edge of the grid, which the provenance shows.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `variant` | | `embeddings_only`, `embeddings_plus_score` or `score_anchored_residual` |
| `encoder` | `"stacked"` | One embedding set of the data by name, or every set concatenated; `"stacked"` needs at least two sets, so with one encoder pass its name |
| `CorrectionData(embeddings=...)` | | The embedding sets by encoder name, covering every text |

## Example

The block needs MiniLM. It embeds the toy texts, cross-fits the embeddings-only and the residual variants on a human sample of 27 texts, and then uses the residual variant's corrected values as the prediction of a prediction-powered mean.

<!-- run embed -->
```python
rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
encoder = cop.Encoder("minilm", device="cpu")
texts = encoder.embed(table.frame["text"].tolist(), table.frame["text_id"].tolist())
full = cop.CorrectionData.from_scores(
    cop.export_text_scores("run.sqlite"),
    table,
    table.frame,
    gold_column="gold_constructiveness",
    score_columns=["construct_score"],
    embeddings={"minilm": texts},
)
human = table.frame.set_index("text_id")["gold_constructiveness"]
draw = cop.draw_human_sample(table, n=27, seed=5)
data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)

for variant in ("embeddings_only", "score_anchored_residual"):
    correction = cop.get_correction("embedding_augmented", variant, encoder="minilm")
    corrected = cop.cross_fit_predictions(correction, data)
    print(variant, round((corrected - human).abs().mean(), 1))
estimate = cop.mean_ppi(data, prediction=corrected)
print(round(estimate.point, 1), round(human.mean(), 1))
```

<!-- output -->
```text
embeddings_only 0.7
score_anchored_residual 0.6
3.0 3.0
```

With 27 labelled texts the small encoder adds little: the embeddings alone are off by about 0.7 points, the score with the embeddings by about 0.6. As the prediction of the prediction-powered mean, the corrected values give a point of 3.0, the true mean of the 54 gold values. Whether they also narrow the interval depends on how much signal the embeddings add; check it on your corpus with `repeated_study` or by comparing the standard errors.

## Where the library departs from the paper's analysis code

- `embeddings_plus_score` standardises the appended score on the training rows; the paper's analysis code appended the raw score, which was harmless there because its scores shared one rating scale. The library receives scores on any scale, and a raw column would make the result and the chosen penalty depend on the score's units.
- A non-finite value in an embedding vector is refused.
- The penalty curve of each deletion holds mean squared errors; the paper's analysis code summed the squared errors. The chosen penalty and the edge shares are the same.
- Every variant decomposes the retained design of every deletion afresh, and the variants do not share these decompositions; in the battery, which fits all three variants on every fold, this is the largest cost.

## Literature

Egami et al. (2023); Zou and Hastie (2005). Full entries are in [References](../references.md).
