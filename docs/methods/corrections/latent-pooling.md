# Latent pooling

Latent pooling combines several machine scores of the same texts into one estimate of the construct. It treats the true construct as a latent variable that every score measures with noise, extracts the one factor the scores share (Thurstone 1947), and puts the factor on the scale of the human values with a calibration line fitted on the human-coded sample. The idea has a long history in combining fallible ratings into one estimate (Dawid and Skene 1979) and in drawing inferences about latent variables from noisy indicators (Mislevy 1991).

The scores to pool are the score columns of the data: the same coder under several configurations or models, or several coders. Pooling averages away noise that the columns do not share and can rank the texts better than any single column, which a calibration of one column cannot. The library's estimator is a first version of the idea. It has no separate term for the error that the machine scores share, so a shared error enters the factor as if it were signal, and many machine columns can outweigh the one human anchor.

## The algorithm as the library implements it

1. **Columns.** Every score column of the data is pooled; the family applies only to data with at least two, and `fit` refuses data with fewer with `NotApplicableError`, as `applies` does.
2. **Standardise.** On the training rows each column is centred and divided by its population standard deviation; a constant column keeps a standard deviation of 1, standardises to zeros and gets no weight.
3. **One factor.** From the correlation matrix of the standardised columns, iterated principal-axis factoring extracts one factor. The starting communalities are the squared multiple correlations `1 - 1 / diag(R^-1)`, or, when the matrix is singular, each column's largest absolute correlation with another column; they are kept between 0.001 and 1. Each round puts the communalities on the diagonal and takes the top eigenvector scaled by the square root of its eigenvalue as the loadings, until the communalities change by less than `1e-6` or after 50 rounds.
4. **Orient.** The loadings are flipped when the factor score correlates negatively with the row mean of the standardised scores.
5. **Factor score.** The weighted sum of the standardised scores with the loadings as weights. A factor score that is constant on the training rows is refused.
6. **Anchor.** The least-squares line of the gold values on the factor score, fitted on the training rows, puts the factor on the gold scale; the corrected value of any text is that line applied to its factor score.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

spec = cop.corrections.REGISTRY["latent_pooling"]
print(spec.target, spec.level, spec.estimate_errors)
for variant in spec.variants:
    print(variant, ", ".join(spec.errors_for(variant)))
```

<!-- output -->
```text
A score none
factor_anchor noise, calibration_shortfall
```

The family aims at target A. It addresses noise, by pooling, and the calibration shortfall of a single column, whose ceiling it raises by pooling configurations. It assumes that one factor carries the construct and that the columns' errors are independent; error that the columns share stays in the factor. The factor weights are those of the training rows, and the anchor line holds for the corpus the human-coded sample was drawn from.

## Parameters

The variant is `factor_anchor`; it takes no parameters. Its inputs are the score columns of the data, at least two, and the folds and seed of `cross_fit_predictions`.

## Example

The toy data have one construct score per coder, so the block pools the three attribute scores of the fake backend's run as three indicators of the construct. In a study the columns would be the construct scores of several configurations or coders.

<!-- run -->
```python
import numpy as np

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
full = cop.CorrectionData.from_scores(
    cop.export_text_scores("run.sqlite"),
    table,
    table.frame,
    gold_column="gold_constructiveness",
    score_columns=["specificity", "tone", "reasoning"],
)
human = table.frame.set_index("text_id")["gold_constructiveness"]
draw = cop.draw_human_sample(table, n=27, seed=5)
data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)

pooling = cop.get_correction("latent_pooling", "factor_anchor")
fitted = pooling.fit(data, data.labelled_ids, rng=np.random.default_rng(0))
print(fitted.provenance["n_indicators"], fitted.provenance["score_column"])
corrected = cop.cross_fit_predictions(pooling, data)
print(round((corrected - human).abs().mean(), 1))
```

<!-- output -->
```text
3 all
0.6
```

The pooled and anchored factor is off by 0.6 points of the 1-to-5 scale on average.

## Where the library departs from the paper's analysis code

- A factor score that is constant on the training rows is refused; the paper's analysis code skipped such a fold without a word.
- When the correlation matrix is singular, the starting communality of each column is its largest absolute correlation with another column; the paper's analysis code put 1.0 on the diagonal for a constant column.

## Literature

Dawid and Skene (1979); Mislevy (1991); Thurstone (1947). Full entries are in [References](../references.md).
