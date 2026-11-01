# Pseudo-values and prediction-powered inference

A design-based correction rests on the random draw of the human sample, not on a model of the score's errors. Every text keeps a prediction `ghat` of its human value, and every human-coded text adds the gap between its human value and its prediction, divided by the probability `pi` that it was drawn: the pseudo-value `ghat_i + (D_i / pi_i) * (gold_i - ghat_i)`, with `D_i` 1 for a human-coded text. When `ghat` is fixed before the sample is seen, such as the machine score itself, the mean of the pseudo-values is the Horvitz-Thompson estimate of the mean of the human values (Horvitz and Thompson 1952, Neyman 1938), exactly unbiased over the possible draws whatever the quality of the score. With the score as the prediction this is prediction-powered inference (Angelopoulos et al. 2023a); with a prediction learned by cross-fitting it is design-based supervised learning (Egami et al. 2023).

The estimate-level estimators apply the same idea to a mean, a difference between two groups and a regression coefficient, with a standard error and a 95 percent interval: the estimate computed from the predictions of all texts, corrected by the gap measured on the labelled texts. A poor prediction widens the interval; it does not bias the point, exactly or approximately under the conditions set out below. Power tuning scales the prediction's terms by a coefficient estimated on the labelled texts, so that a weak prediction costs little precision (Angelopoulos et al. 2026). For the construct as a predictor, the `ludwig` variant is the moment correction of Ludwig et al. (2026): the corrected regression equations with the texts outside the human sample, instead of all texts, as the base. A single pseudo-value carries no guarantee and can fall outside the scale; pseudo-values are for aggregates.

## The algorithm as the library implements it

**The pseudo-value column**, `pseudo_values(data, rectifier=..., score_column=..., k=5, seed=42)`:

1. Every labelled text needs a finite `pi`; `draw_human_sample` and `attach_draw` supply `pi = m / G`. Unlabelled texts may carry none.
2. The prediction `ghat`: `identity` takes a score column as it stands (the first one unless named), so it must already be on the gold scale; `calibrated` fits the least-squares line `gold ~ 1 + score` on the labelled texts (with the named column, or the one with the highest signed Spearman correlation there); `crossfit` splits the groups into `k` inner folds and predicts each fold with the line `gold ~ 1 + score + covariate` fitted on the labelled texts of the other folds, the score column chosen on a random half of those folds' groups. `crossfit` refuses fewer groups than folds, a fold whose labelled training texts are fewer than 4 or lie in fewer than 2 groups, and a selection half (half of those groups, at least one) on which no score column is usable.
3. The result holds the column, `ghat`, `mean_plugin` (the mean of `ghat`), `mean_rectified` (the mean of the column), `lambda_star = cov(gold, ghat) / var(ghat)` over the labelled texts with population moments (1 for a constant `ghat`), and `mean_rectified_tuned = mean(gold_lab) + lambda_star * (mean(ghat_all) - mean(ghat_lab))`, missing when `pi` differs between labelled texts.

**The estimators** need at least two labelled texts (two on each side of a difference) with the same `pi`; a sample with unequal `pi` is refused with a pointer to `pseudo_values`. The prediction `f` is the first score column, never chosen on gold values, or a series on the gold scale.

4. `mean_ppi`: the point is `mean(y_lab) + lambda * (mean(f_all) - mean(f_lab))`, with `lambda` 1, or `lambda_star` under power tuning. The standard error is `sqrt(se(lambda * f - y, lab)^2 + lambda^2 * se(f, all)^2)`, with `se = sd / sqrt(n)` (divisor `n - 1`).
5. `group_difference_ppi`: the same with the difference of the two group means in place of the mean, and `sqrt(var_a / n_a + var_b / n_b)` as the standard error of a difference.
6. `ols_outcome_ppi`: the construct is the outcome of a design `X` (a formula or a matrix): `M_all beta = c_all(f) - c_lab(f) + c_lab(y)`, with `M = mean(x x')` and `c = mean(x v)`, and the sandwich variance `H^-1 [Cov(psi_f) / N + Cov(psi_f - psi_y) / n] H^-1`. It reports the first coefficient after the intercept unless told otherwise.
7. `ols_predictor_ppi`: the construct is a predictor of an outcome observed on every text, with optional covariates: `[M_base(f) - M_lab(f) + M_lab(y)] beta = c_base(f) - c_lab(f) + c_lab(y)`, where the base is every text (`ppi`) or the texts outside the human sample (`ludwig`). A rectified moment matrix that is not positive definite leaves the estimate undefined.
8. **Clustering.** With `cluster=None` (the default) the variance of each term is clustered whenever a group holds more than one labelled text, as under a whole-group draw: the contributions of the term's texts are centred at their mean and summed within groups, and the variance is `G / (G - 1) * sum_g S_g^2 / n^2` over the `G` groups. Clustering needs two groups among the texts of each term and among the labelled texts, and a clustered difference needs labelled texts from two groups on each side; otherwise the estimator refuses and names the remedies (draw more groups, or `cluster=False`, which treats the texts as independent).
9. The interval is `point +/- 1.96 * SE`.

## Exactness and limitations

- `mean_rectified` with the `identity` prediction is exactly unbiased under any draw with the recorded `pi`. The `calibrated` and `crossfit` predictions learn from the labelled texts and leave a small bias of the order of one over the sample size, in exchange for a smaller variance.
- The point of `mean_ppi` is exactly unbiased over repeated whole-group draws when every group of the corpus holds the same number of texts (groups of one text included). The point of `group_difference_ppi` is exactly unbiased when, on each side of the split, every group of the corpus that has texts on that side holds the same number of them. Groups of equal size that each lie wholly on one side are one such case, not the only one: groups that each hold one text on one side and two on the other are another. In a mixed layout, where some groups straddle the split and others lie on one side, this holds only provided no draw is refused: the estimator refuses a draw with fewer than two labelled texts on a side (or, clustered, fewer than two labelled groups), and conditioning on the accepted draws shifts which groups are drawn. In one enumeration of every three-group draw of a ten-group corpus, 41 of the 120 draws were refused and the accepted ones had a bias of −0.56 in the gold's units. Otherwise, as with groups of unequal size or a split such as long against short texts that gives the groups different numbers of a side's texts, the means over the labelled texts are ratio estimators and the point is approximately unbiased, with a bias of the order of one over the number of drawn groups. The coefficients and the power-tuned points are approximately unbiased.
- The standard error counts a variance for the machine term, the mean of the predictions over all texts, although, for a prediction fixed before the draw, that mean does not vary over draws from a fixed corpus (a learned `prediction=` does vary), and it applies no finite-population correction. Against the corpus's own full-gold value the intervals are therefore conservative, and more so as the labelled share grows: in simulations a nominal 95 percent interval of a mean covered about 0.966 with 11 percent of the texts labelled and 0.98 to 0.995 with 46 percent. The covariance between the two terms is left out as well; it can take either sign.
- With few drawn groups a clustered interval covers less than 95 percent, because the variance of the labelled term rests on as many group sums as there are drawn groups: in simulations with groups of ten texts sharing a machine error, the interval of a mean covered about 0.88 with five drawn groups, 0.92 with ten and 0.96 with twenty.
- With a noisy prediction and few drawn groups, the estimates of `ols_predictor_ppi` have heavy tails: the rectified moment matrix that the coefficient is divided by can come close to singular in some draws, so the point swings far from the truth there, and its normal interval is not a reliable guide. Draw more groups, or use a less noisy prediction, before you rely on that interval.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

for family in ("pseudo_values", "ludwig_moment"):
    spec = cop.corrections.REGISTRY[family]
    print(family, spec.target, spec.level, spec.variants)
    print(spec.estimate_errors)
```

<!-- output -->
```text
pseudo_values C estimate ('identity', 'calibrated', 'crossfit')
estimate off and confidence-interval misses, for means and for slopes on any covariate with the construct as outcome
ludwig_moment C / D estimate ('mean', 'difference', 'ols_outcome', 'ols_predictor')
bias and coverage of the construct's coefficient as predictor (D) or of means and differences (C)
```

The pseudo-values aim at target C, any estimate linear in the construct with the construct as the outcome, and the registry family `ludwig_moment` lists the four estimators, at target C for means and differences and D for the coefficient of the construct as a predictor. They correct an estimate that is off and an interval that misses. The guarantee rests on the draw: a random sample of whole groups from the corpus under study, with every text of a drawn group coded and its inclusion probability recorded. It needs no assumption about the score's errors when the prediction is fixed before the draw; a learned prediction adds the assumptions of cross-fitting.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `pseudo_values(rectifier=...)` | `"identity"` | `identity`, `calibrated` or `crossfit` |
| `pseudo_values(score_column=..., k=..., seed=...)` | `None`, 5, 42 | The column of `identity` and `calibrated`; the folds and seed of `crossfit` |
| `prediction` | `None` | The prediction of every text on the gold scale; `None` for the first score column |
| `power_tuning` | `False` | Scale the prediction's terms by `lambda_star` (`mean_ppi`, `group_difference_ppi`) |
| `cluster` | `None` | Cluster by group when a group holds several labelled texts; `True` or `False` forces it |
| `formula` or `X` | | The design of `ols_outcome_ppi`; `covariates` or `X` of `ols_predictor_ppi` |
| `variant` | `"ppi"` | `ppi` or `ludwig` for `ols_predictor_ppi` |

## Example

On the toy corpus, with a human sample of 9 of the 18 groups (27 texts), the pseudo-values move the mean from the raw score's 1.7 to 3.1; the true mean of all 54 gold values is 3.0. Every toy group holds three texts, so the prediction-powered mean equals the Horvitz-Thompson mean exactly.

<!-- run -->
```python
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

values = cop.pseudo_values(data)
print(round(values.mean_plugin, 1), round(values.mean_rectified, 1), round(human.mean(), 1))
estimate = cop.mean_ppi(data)
print(round(estimate.point, 1), abs(estimate.point - values.mean_rectified) < 1e-9)
print(estimate.details["clustered"], estimate.details["n_clusters_labelled"])
tuned = cop.mean_ppi(data, power_tuning=True)
print(round(tuned.details["lambda_star"], 1), round(tuned.point, 1))
```

<!-- output -->
```text
1.7 3.1 3.0
3.1 True
True 9
1.9 3.0
```

The standard error is clustered by the 9 drawn groups. [Correcting estimates](../../tutorials/correcting-estimates.md) runs the difference and both regressions, and checks the estimators with a repeated study.

## Where the library departs from the paper's analysis code

- The power-tuning coefficient `lambda_star` uses population moments in both its covariance and its variance; the paper's analysis code divided a sample covariance by a population variance, a factor of `n / (n - 1)`.
- The inclusion probability is required on the labelled texts only.
- Power tuning is not implemented for `ols_outcome_ppi`; for a mean it is available through `mean_ppi(power_tuning=True)` and `pseudo_values` (`mean_rectified_tuned`).
- The OLS estimators accept formulas in the syntax of `formulaic`; an error inside a stateful transform of a formula is reported as a configuration error.

## Literature

Angelopoulos et al. (2023a); Angelopoulos et al. (2026); Egami et al. (2023); Horvitz and Thompson (1952); Ludwig et al. (2026); Neyman (1938). Full entries are in [References](../references.md).
