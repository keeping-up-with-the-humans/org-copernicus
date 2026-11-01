# Multiple imputation

Multiple imputation treats the human value of every text outside the human-coded sample as missing (Rubin 1987). The sample is drawn at random, so these values are missing at random by design. The family fits a model of the human value given the machine score and the text length on the human-coded texts, and draws several completed columns from it, each a plausible set of human values for every text (Little 1988, Blackwell et al. 2017). An analysis is run on each completed column, and Rubin's rules combine the results into one estimate whose standard error includes the uncertainty of the draws.

The draws are proper when they carry the uncertainty of the model itself, so the family refits the model on a bootstrap of the human-coded texts before every draw and only then adds the draw's own randomness. Values come either from the model, with normal noise, or by predictive mean matching, which fills each value with the observed human value of a text whose prediction is close. As a single corrected column the family reports the mean over the draws; because the model uses the score and the text length, that column behaves like the covariate-augmented line of [regression calibration](regression-calibration.md).

## The algorithm as the library implements it

1. **Model.** On the training rows the family takes the score column with the highest signed Spearman correlation with the gold values; the model is `gold ~ 1 + score + covariate`, with the covariate of the data (the log of the character count by default). A design of less than full rank is refused.
2. **Bootstrap models.** For each of `m_draws` (20) draws, the training rows are resampled with replacement to their own number and the model is refitted by least squares, with the residual variance on `n - 3` degrees of freedom.
3. **Draws.** `model_draws` adds to each text's fitted mean a normal error with the residual variance of that draw's model. `pmm` sorts the training rows by their fitted mean under that draw's model and replaces each text's fitted mean by the observed gold value of one of its `k_donors` (5) nearest training rows by fitted mean, chosen at random, so every draw lies on the observed gold values.
4. **Reproducible draws.** The random numbers of a text come from a generator seeded with a seed stored at fit time and the text's row position in the data, so a text's draws do not depend on which other texts are completed with it.
5. **Output.** `draws(data, ids)` returns all completed columns, of shape `[m_draws, n]`; `predict` returns their mean, the family's corrected column.
6. **Rubin's rules.** `rubin_pool(estimates, variances)` takes one estimate and its variance per draw and returns the mean of the estimates, the within-draw variance `W` (the mean of the variances), the between-draw variance `B` (the variance of the estimates, divisor `m - 1`), the total variance `W + (1 + 1/m) B`, and the degrees of freedom `(m - 1) * (1 + W / ((1 + 1/m) B))^2`, infinite when `B` is 0.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

spec = cop.corrections.REGISTRY["multiple_imputation"]
print(spec.target, spec.level, spec.estimate_errors)
for variant in spec.variants:
    print(variant, ", ".join(spec.errors_for(variant)))
```

<!-- output -->
```text
B score confidence-interval coverage via Rubin's rules
model_draws level_displacement, calibration_shortfall, noise
pmm level_displacement, calibration_shortfall, noise
```

The family aims at target B: plausible draws of each text's human value. It addresses level displacement, calibration shortfall and noise, with the spread restored in each draw, and the coverage of confidence intervals through Rubin's rules. The guarantee holds for analyses whose variables are all in the imputation model; an analysis with a variable outside it, or a single draw used as a text's value, has none. The model is linear in the score and the covariate, so a relation it misses is missing from every draw. Twenty draws serve a point correction. When only 2.5 to 5 percent of the texts are human-coded, the fraction of missing information is high, and inference with Rubin's rules needs 150 draws or more; every fit records this in its provenance.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `variant` | | `model_draws` or `pmm` |
| `m_draws` | 20 | Number of draws |
| `k_donors` | 5 | Nearest donors of `pmm` |
| `rng` of `fit` | | The generator of the bootstraps and of the seed of the draws |

## Example

On the toy corpus, with a human sample of 27 texts drawn by whole groups, the block draws 50 completed columns, keeps the human value of every labelled text, and pools the mean of the construct over the draws. The variance of each draw's mean treats the texts as independent, which is enough for the illustration. The true mean of all 54 gold values is 3.0.

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
    score_columns=["construct_score"],
)
human = table.frame.set_index("text_id")["gold_constructiveness"]
draw = cop.draw_human_sample(table, n=27, seed=5)
data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)

imputation = cop.get_correction("multiple_imputation", "model_draws", m_draws=50)
fitted = imputation.fit(data, data.labelled_ids, rng=np.random.default_rng(3))
draws = fitted.draws(data, data.ids)
completed = np.where(data.labelled_mask, data.gold, draws)
pooled = cop.corrections.rubin_pool(completed.mean(axis=1), completed.var(axis=1, ddof=1) / data.n)
print(draws.shape, pooled.m, round(pooled.estimate, 1))
print(pooled.estimate - 1.96 * pooled.se < human.mean() < pooled.estimate + 1.96 * pooled.se)
```

<!-- output -->
```text
(50, 54) 50 3.0
True
```

Predictive mean matching draws only values that the human coders gave:

<!-- run -->
```python
matching = cop.get_correction("multiple_imputation", "pmm")
donors = matching.fit(data, data.labelled_ids, rng=np.random.default_rng(0)).draws(data, data.ids)
print(sorted(np.unique(donors).tolist()))
```

<!-- output -->
```text
[1.0, 2.0, 3.0, 4.0, 5.0]
```

## Where the library departs from the paper's analysis code

- The draws are keyed to the row: a text's imputations do not depend on the batch of texts completed with it. The random streams are therefore not those of the paper's battery; the model is the same.
- The paper lists predictive mean matching while its analysis battery removed the variant, so the `pmm` results of the library's battery have no counterpart in the paper. The registry carries this note (`REGISTRY["multiple_imputation"].note_for("pmm")`).

## Literature

Blackwell et al. (2017); Little (1988); Rubin (1987). Full entries are in [References](../references.md).
