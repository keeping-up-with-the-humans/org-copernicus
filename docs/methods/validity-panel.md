# Validity panel

The validity panel compares a machine score with human values of the same texts and reports how far the score can be trusted, one row per target (the construct, an attribute, or a roll-up). Its headline is rank validity, the absolute Spearman correlation between score and human value: how closely the score orders the texts the way the human values do. A correlation with a criterion is the classical criterion-validity coefficient (Carmines and Zeller 1979), and agreement between two methods that measure one construct is the convergent evidence of Campbell and Fiske (1959). It is one strand of construct validity, which rests on the accumulated evidence that a measure captures the intended construct (Cronbach and Meehl 1955, Messick 1995). Rank validity ignores the level and the spread of the scores, so it is lowered only by noise and by error that follows something other than the human value.

Beside it the panel reports four score errors, which come from the linear error model `score = a + b * gold + lambda * X + e` of the measurement-error literature (Fuller 1987, Carroll et al. 2006): level displacement (`a`), calibration shortfall (`b - 1`), covariate-linked error (`lambda`, with `X` the log of the text length by default) and noise (`e`). The model is a guide, not fitted as a whole. The errors matter downstream in different ways: error unrelated to the true value weakens a predictor's coefficient, error related to the true value or to other variables can bias it either way (Hausman 2001), and a noisy estimate that passes a significance test in a small sample overstates the effect (Loken and Gelman 2017). The human values carry error of their own, so no score can correlate with them above the square root of their reliability, the reliability ceiling (Spearman 1904). The panel reports that ceiling and the share of it that the score reaches, with Cronbach's alpha (Cronbach 1951) as one source of the reliability, whose limits as a reliability estimate are well documented (Sijtsma and Pfadt 2021). How far the human values deserve their role as gold depends on the coding behind them: trained coders, written rules and agreement checks (Krippendorff 2018, Lombard et al. 2002, Hayes and Krippendorff 2007).

## The algorithm as the library implements it

For each target:

1. **Scores per text.** The score is averaged over runs per text. A text without a score, a gold value or a covariate value is dropped from every metric of the target and counted in `n_dropped_missing`; nothing is imputed.
2. **Direction.** `positive` uses the score as given, `negative` negates it, and `auto` negates it when its signed Spearman correlation with the gold is negative.
3. **Association.** The signed Spearman correlation, rank validity (its absolute value), the Pearson correlation and Kendall's tau-b; for a gold with two values, the area under the ROC curve of the score for the larger value and Somers' D (`2 * AUC - 1`). With fewer than 3 usable texts, or a score or gold that is constant, the status says so and the associations are missing. A set of values counts as constant when its range is at most `1e-12` times the larger of 1 and its largest absolute value.
4. **Score errors.** Score and gold are each min-max scaled to 0 to 1 with their own observed minimum and maximum over the used texts. Level displacement is the mean of `score01 - gold01`; calibration shortfall is `cov(score01, gold01) / var(gold01) - 1`, negative when the score is squeezed relative to the gold; covariate-linked error is the least-squares slope of `score01 - gold01` on the covariate, per standard deviation of the covariate (`standardized`, the default) or per unit (`raw`), positive when the score rises relative to the gold as the covariate rises. The mean absolute error `mae_01` is the mean of `|score01 - gold01|`.
5. **Noise.** The run scores of the used texts are min-max scaled to 0 to 1 over all their runs; noise is the standard deviation across runs per text (divisor `n - 1`), averaged over the texts with at least two runs. With a single run it is missing.
6. **Intervals.** A percentile bootstrap that resamples whole groups, `n_boot` (1000) times with seed 42, gives 95 percent intervals (the 2.5 and 97.5 percentiles) for rank validity, the Pearson correlation, `mae_01` and the first three errors, all from the same resamples. A metric undefined on more than 10 percent of the resamples stops the panel with an error. Rank validity is an absolute value, so its interval is a percentile interval of the absolute Spearman correlation: it is bounded below by 0 and does not test zero validity, because a score unrelated to the gold still gets an interval above 0. The interval of the signed Pearson correlation (`pearson_low`, `pearson_high`) can contain 0. Without a group column every text is its own group, and the bootstrap resamples single texts.
7. **Ceiling.** With a reliability `r`, the ceiling is `sqrt(r)` and `share_of_ceiling` is rank validity divided by it. A number applies to every row; a mapping from gold column to reliability sets the rows of those columns only; `items` (one column per human coder, for example) give Cronbach's alpha for the construct row.

## Assumptions and what it corrects

The panel diagnoses; it corrects nothing. It assumes that the gold values measure the construct the score is meant to measure, on the same texts. Because score and gold are scaled by their own observed ranges, the errors describe the score relative to its own range: a score that is an exact shift or stretch of the gold shows no error at all, and level displacement measures whether the score's values sit higher or lower within their range than the gold values sit within theirs. A shift in the gold's own units shows in the native-scale errors of the corrections instead. The four diagnostics cannot isolate error that is nonlinear or that follows a variable the panel does not see, and the ceiling formula holds for the Pearson correlation, so a ceiling for rank validity is approximate.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `PanelSettings.covariate` | `None` | Covariate of the covariate-linked error; `None` uses the log of the table's `n_chars` |
| `PanelSettings.covariate_scale` | `"standardized"` | Slope per standard deviation, or `"raw"` per unit |
| `PanelSettings.n_boot` | 1000 | Bootstrap resamples of whole groups |
| `PanelSettings.seed` | 42 | Seed of the bootstrap |
| `PanelSettings.direction` | `"positive"` | `"positive"`, `"negative"` or `"auto"` |
| `PanelSettings.reliability` | `None` | A number for every row, or a mapping from gold column to reliability |
| `PanelSettings.items` | `None` | Items of the gold instrument, for Cronbach's alpha of the construct row |

## Example

Three made-up scores of eight texts against their gold values. An exact stretch of the gold shows no error, a noisy score shows a calibration shortfall, and a score that drifts with text length shows covariate-linked error.

<!-- run -->
```python
import numpy as np

import copernicus as cop

gold = np.array([1, 2, 3, 4, 5, 2, 4, 3.0])
length = np.log([40, 90, 60, 300, 220, 150, 80, 120.0])
scores = {
    "stretched": 2 + 0.5 * gold,
    "noisy": gold + np.array([0.6, -0.9, 0.8, -0.4, -1.2, 0.7, 0.3, 0.9]),
    "drifting": gold + 2.0 * (length - length.mean()),
}
for name, score in scores.items():
    errors = cop.score_errors(score, gold, length)
    level = round(errors.level_displacement, 1)
    shortfall = round(errors.calibration_shortfall, 1)
    linked = round(errors.covariate_linked_error, 1)
    print(name, [level, shortfall, linked])
print(cop.reliability_ceiling(0.81))
```

<!-- output -->
```text
stretched [0.0, 0.0, 0.0]
noisy [0.1, -0.1, -0.1]
drifting [0.0, -0.1, 0.1]
0.9
```

The panel computes the same errors for a coder's scores, with intervals. Here the fake backend's scores of the toy corpus, with 200 resamples:

<!-- run -->
```python
rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
panel = cop.validity_panel(
    cop.export_scores("run.sqlite"),
    table.frame,
    table,
    targets={"construct": "gold_constructiveness"},
    settings=cop.PanelSettings(n_boot=200),
)
row = panel.frame.iloc[0]
names = ["rank_validity", "level_displacement", "calibration_shortfall", "covariate_linked_error"]
print([round(float(row[name]), 1) for name in names], round(float(row["noise"]), 1))
```

<!-- output -->
```text
[0.9, -0.1, 0.0, 0.1] 0.0
```

The noise is 0 because the fake backend gives the same score on every run. [The validity panel](../tutorials/validity-panel.md) reads the report, the intervals, the ceiling and the command `copernicus validity panel`.

## Where the library departs from the paper's analysis code

- Noise is computed on the 0 to 1 scale over the texts that entered each target, as the standard deviation across runs per text averaged over the texts with at least two runs. This is the paper's definition.
- A constant score keeps its row, with the status `constant_score`: its calibration shortfall is exactly -1, its covariate-linked error is minus the slope of the scaled gold on the covariate (on both scales), its level displacement and mean absolute error are missing, and its noise is computed as for any score. The paper's analysis code reported all four errors as missing.
- The covariate-linked error is reported on both scales, as `covariate_linked_error_standardized` and `covariate_linked_error_raw`; the headline column and its interval follow the chosen scale.
- A reliability ceiling belongs to a gold column. A single number applies to every row, which is right only when all targets share one gold instrument.
- A set of values counts as constant when its range is at most `1e-12` times the larger of 1 and its largest absolute value, in every association, slope and min-max scaling; the paper's analysis code tested for an exact zero range.

## Literature

Campbell and Fiske (1959); Carmines and Zeller (1979); Carroll et al. (2006); Cronbach (1951); Cronbach and Meehl (1955); Fuller (1987); Hausman (2001); Hayes and Krippendorff (2007); Krippendorff (2018); Loken and Gelman (2017); Lombard et al. (2002); Messick (1995); Sijtsma and Pfadt (2021); Spearman (1904). Full entries are in [References](references.md).
