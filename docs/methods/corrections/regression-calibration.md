# Regression calibration

Regression calibration replaces each machine score by the expected human value given the score, and given text features when it uses them (Prentice 1982, Carroll et al. 2006). It fits a model of the human value on the score in the human-coded sample and applies that model to every text. The corrected value is meant to be the best guess of each text's human value, its expected value given the score and features; to the extent that the fitted calibrator matches that expectation, the errors among texts with the same score and features average zero, and a fitted line or curve only approximates it. That puts the score on the gold scale and removes level displacement and calibration shortfall.

The library has four calibrators, ordered by how many human-coded texts they need: a straight line, a line that adds the log of the text length, Platt's logistic calibration for a gold with two values (Platt 2000), and isotonic regression, a monotone piecewise-linear map. A calibrator that keeps the order of the scores cannot change rank validity: the line with a positive slope and the logistic curve keep it exactly, isotonic regression changes it only through the ties it creates, and only the line with the covariate adds information beyond the score.

## The algorithm as the library implements it

1. **Score column.** On the training rows the family takes the score column with the highest signed Spearman correlation with the gold values.
2. **Fit**, by variant:
    - `linear`: the least-squares line `gold ~ 1 + score`. A score that is constant on the training rows is refused.
    - `covariate_augmented`: the least-squares fit `gold ~ 1 + score + covariate`, with the data's covariate (the log of the character count by default). A design of less than full rank, such as a constant covariate, is refused.
    - `platt`: scikit-learn's logistic regression, with its default L2 penalty (`C = 1`) and up to 1000 iterations, of the larger of the two gold values on the score. The predicted probability `p` is placed on the gold scale as `low + p * (high - low)`, which is `p` itself for 0 and 1. It applies only when the labelled gold takes exactly two values, and `fit` refuses training texts whose gold takes another number of values with `NotApplicableError` and the reason of `applies`.
    - `isotonic`: scikit-learn's isotonic regression of the gold on the score, a monotone piecewise-linear map. The fit is non-decreasing over the training scores, a score between two of them gets the linearly interpolated value, and a score outside the training range gets the value at the nearer end.
3. **Apply.** The fitted map turns the score (and the covariate) of any text into its corrected value. Applied through `cross_fit_predictions`, every text is corrected by a fit that did not see its gold value. Each fold has its own fit, so the cross-fitted values of different folds can be ordered slightly differently from the raw scores.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

spec = cop.corrections.REGISTRY["regression_calibration"]
print(spec.target, spec.level, spec.estimate_errors)
for variant in spec.variants:
    print(variant, ", ".join(spec.errors_for(variant)))
```

<!-- output -->
```text
A score none
linear level_displacement, calibration_shortfall
covariate_augmented level_displacement, calibration_shortfall, covariate_linked_error
platt level_displacement, calibration_shortfall
isotonic level_displacement, calibration_shortfall
```

The family aims at target A. It addresses level displacement and calibration shortfall, with the spread of the corrected values matched to the signal, and the covariate-augmented line also addresses the error linked to its covariate. It assumes that the relation between score and human value in the human-coded sample holds for every text, which a random sample from the same corpus provides. When the calibration captures the expected human value, the remaining error is uncorrelated with every variable it conditions on, so a regression of the corrected values on covariates approaches the true coefficients provided every covariate is part of the calibration. Corrected values are pulled toward the mean: they are not valid for variances or percentiles, and an estimate built from them, such as a difference between groups the calibration did not model, needs [prediction-powered inference](pseudo-values.md) on top.

## Parameters

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `variant` | `linear`, `covariate_augmented`, `platt`, `isotonic` | The calibrator |
| `CorrectionData.covariate_column` | `log_chars` | The covariate of `covariate_augmented` |
| `cross_fit_predictions(k=..., seed=...)` | 5, 42 | Folds and seed of the cross-fitting |

The calibrators take no further parameters.

## Example

The toy corpus coded with the fake backend, a human sample of 27 texts drawn by whole groups, and the linear calibration fitted once and cross-fitted:

<!-- run -->
```python
import numpy as np
from scipy.stats import spearmanr

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
raw = data.frame.set_index("text_id")["construct_score"]

linear = cop.get_correction("regression_calibration", "linear")
fitted = linear.fit(data, data.labelled_ids, rng=np.random.default_rng(0))
once = fitted.predict(data, data.ids)
print(fitted.provenance["score_column"], round(spearmanr(once, raw).statistic, 1))
for variant in ("linear", "covariate_augmented", "isotonic"):
    calibration = cop.get_correction("regression_calibration", variant)
    corrected = cop.cross_fit_predictions(calibration, data)
    print(variant, round((raw - human).abs().mean(), 1), round((corrected - human).abs().mean(), 1))
```

<!-- output -->
```text
construct_score 1.0
linear 1.3 0.6
covariate_augmented 1.3 0.6
isotonic 1.3 0.6
```

A line fitted once keeps the order of the scores exactly. On the gold's own scale the raw score is off by 1.3 points of the 1-to-5 scale on average; with five folds, every calibrator brings the error to about 0.6. [Correcting scores](../../tutorials/correcting-scores.md) compares the calibrators with other families in the battery.

## Where the library departs from the paper's analysis code

- The `platt` variant maps the class probability onto the gold scale, `low + p * (high - low)`, so it works for any two gold values; for a gold coded 0 and 1 this is the probability, as in the paper's analysis code.

## Literature

Carroll et al. (2006); Platt (2000); Prentice (1982). Full entries are in [References](../references.md).
