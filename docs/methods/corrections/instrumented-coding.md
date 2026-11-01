# Instrumented coding

Instrumented coding corrects the coefficient of the construct as a predictor with a second measurement of the construct instead of a model of the error variance. Random error in a measurement pulls a regression coefficient toward zero (Fuller 1987). When two measurements of the same construct, such as the scores of two coders, have errors that are independent of each other, each can serve as an instrument for the other: the part of one measurement that the other predicts is free of the first one's error. Obviously related instrumental variables (ORIV) use both directions at once, in one stacked two-stage least squares (Gillen et al. 2019, equation 6).

The attenuation correction itself uses no human values. The library uses the human-coded sample for one step only: it puts both measurements on the construct's scale before the regression, so that the coefficient is in the units of the human values. The method corrects the association and leaves the score columns unchanged, so it gives no value per text and cannot change rank validity.

## The algorithm as the library implements it

1. **Scale step.** Each measurement is put on the construct's scale by the reverse calibration on the labelled texts: the least-squares line of the score on the gold values, `score = c + d * gold`, inverted to the proxy `W = (score - c) / d`, with sample moments. At least 3 texts must be labelled.
2. **Instrument strength.** The correlation of the two proxies over all texts must reach `weak_instrument_floor` (0.20) in absolute value; a weaker pair is refused.
3. **Stacked two-stage least squares.** The outcome is stacked twice, `(y; y)`, on `(W_A; W_B)` with an intercept for each block, and instrumented by the block-diagonal `[W_B, 0; 0, W_A]`. The coefficient of the stacked measurement is the estimate. A singular system, from a constant measurement or no first-stage signal, is refused.
4. **Standard error.** A stratified bootstrap resamples the labelled and the unlabelled texts separately, each to its own count, and refits both calibration lines and the regression on every resample (`n_boot` 200, seed 42), so the sampling error of the scale step is in the interval. A resample on which the estimate is undefined is counted; at least 50 must be defined. The interval is the estimate plus or minus 1.96 standard errors.
5. **Diagnostics.** The details hold `sandwich_se`, the two-stage sandwich clustered by text, which treats the calibration as known and is too narrow with a small human sample; `instrument_correlation`; the first-stage coefficients and the statistic `(n - 2) r^2 / (1 - r^2)`; and `shared_error_correlation`, the correlation of the two proxies' residuals after regressing each on the gold values over the labelled texts.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

spec = cop.corrections.REGISTRY["instrumented_coding"]
print(spec.target, spec.level, spec.variants)
print(spec.estimate_errors)
```

<!-- output -->
```text
D estimate ('oriv',)
attenuation of the association from two machine measurements; the scale step uses the labelled rows
```

The family aims at target D, the coefficient of the construct as a predictor, and corrects the attenuation that random error causes in it. It assumes that the errors of the two measurements are independent of each other and of the construct. A clearly positive `shared_error_correlation` means that the two measurements share an error, which biases the estimate; two configurations of one coder, or two coders that read the same cues, often do.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `outcome` | | The outcome, a numeric column observed on every text |
| `score_a`, `score_b` | | The two measurements, two different score columns |
| `weak_instrument_floor` | 0.20 | The smallest absolute correlation of the two proxies |
| `n_boot` | 200 | Bootstrap resamples, at least 50 |
| `seed` | 42 | Seed of the bootstrap |

## Example

The toy data offer two measurements: the construct scores of the two-stage and of the one-stage coder, both from the fake backend. The block estimates the slope of text length in words on the construct; the true slope, from all gold values, is 10.6.

<!-- run -->
```python
rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "two.sqlite", cop.CodingSettings(runs=2))
cop.code_texts(table, rubric, backend, "one.sqlite", cop.CodingSettings(runs=2, method="one_stage"))
two = cop.export_text_scores("two.sqlite")[["text_id", "construct_score"]]
one = cop.export_text_scores("one.sqlite")[["text_id", "construct_score"]]
both = two.merge(one.rename(columns={"construct_score": "one_stage_score"}), on="text_id")
full = cop.CorrectionData.from_scores(
    both,
    table,
    table.frame,
    gold_column="gold_constructiveness",
    score_columns=["construct_score", "one_stage_score"],
    extra_columns=["length_words"],
)
human = table.frame.set_index("text_id")["gold_constructiveness"]
draw = cop.draw_human_sample(table, n=27, seed=5)
data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)

result = cop.oriv(data, "length_words", score_a="construct_score", score_b="one_stage_score")
details = result.details
print(round(result.point, 1), details["weak_instrument_floor"], details["n_boot_defined"])
print(round(details["instrument_correlation"], 1), round(details["shared_error_correlation"], 1))
```

<!-- output -->
```text
11.2 0.2 200
1.0 0.9
```

The estimate is 11.2, and every resample was defined. The diagnostics warn, though: the two toy measurements correlate almost perfectly, and their residuals correlate at 0.9, because both coders read the same cue words. Their errors are shared, so the assumption of the method fails here; with real coders, check this correlation before you trust the estimate.

## Where the library departs from the paper's analysis code

- The library's estimator is the stacked two-stage least squares of equation 6, which the paper used in its repeated-study test. The paper's correction battery averaged the two single-direction estimates and chose the pair of configurations itself; in the library the user names the two measurements.
- The standard error is the stratified bootstrap over the human sample and the corpus, with the text-clustered sandwich kept as a diagnostic. The weak-instrument floor of 0.20 applies to the correlation of the two measurements after the scale step, and the scale step uses the labelled texts.
- The bootstrap resamples texts, not groups, so it treats the texts as independent; with groups of correlated texts the standard error is likely too small. This is a limitation of the library.

## Literature

Fuller (1987); Gillen et al. (2019). Full entries are in [References](../references.md).
