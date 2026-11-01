# Classical corrections

The classical corrections come from the literature on measurement error (Fuller 1987, Carroll et al. 2006). Two give each text a new score: Fuller's best linear predictor, which centres the score on the mean of the human values and shrinks each score's distance from the mean by the reliability ratio, and empirical-Bayes smoothing by Tweedie's formula, which replaces each score by its estimated posterior mean (Efron 2011). Two leave the scores unchanged and correct the coefficient of the construct when it is a predictor in a regression, where random error in the score pulls the coefficient toward zero: moment subtraction (Fuller 1987) and simulation-extrapolation, SIMEX (Cook and Stefanski 1994). Blind transport asks whether a correction fitted on other corpora carries over to a corpus without human values.

Moment subtraction and SIMEX need a classical proxy: a measurement on the construct's own scale that equals the construct plus error independent of it. A calibrated score is not one, because it is the best prediction of the construct, not the construct plus noise. The library therefore regresses the score on the human value in the human-coded sample, `score = c + d * gold + u`, and inverts the line: the proxy `W = (score - c) / d = gold + u / d` is on the construct's scale, and its error is uncorrelated with the human value on the labelled texts by construction.

## The algorithm as the library implements it

**Fuller's shrinkage** (`classical`, `fuller_shrinkage`). On the training rows, the best score column, `kappa = cov(gold, score) / var(score)` with population moments, and the corrected value `mean(gold) + kappa * (score - mean(score))`. This is the least-squares line, so a positive `kappa` keeps the order of the scores.

**Tweedie smoothing** (`classical`, `tweedie`). On the training rows, the best score column must covary positively with the gold values; a zero or negative covariance is refused (reverse the score first). With `beta = cov(score, gold) / var(gold)`, the proxy `X = mean(gold) + (score - mean(score)) / beta` has the error variance `sigma2 = max(var(X) - var(gold), 0)`, and the posterior mean under normal error is `X + sigma2 * d/dX log f(X)`, with `f` a Gaussian kernel density of the training proxy at the bandwidth `1.06 * sd(X) * n^(-1/5)`. An increasing isotonic regression of these posterior means on the score makes the map monotone in the score and applicable to any score. All moments are population moments.

**Moment subtraction** (`moment_subtraction(data, outcome)`). The reverse calibration on the labelled texts gives the proxy `W` of every text and its error variance `sigma2 = var(W - gold)` on the labelled texts. The slope of the outcome on the construct is `cov(W, y) / (var(W) - sigma2)` over all texts, undefined when `var(W)` does not exceed `sigma2`. Moments are sample moments; at least 3 texts must be labelled.

**SIMEX** (`simex(data, outcome)`). The same proxy and `sigma2`. At each multiple `lambda` of 0, 0.5, 1, 1.5 and 2, `reps` (25) replicates add normal noise of variance `lambda * sigma2` to `W` and the slopes of the outcome on the noisy proxy are averaged. The curve is fitted by `a + b / (c + lambda)` (least squares, the pole `c` searched on a log grid of 181 values from 0.001 to 10^6 and refined) and evaluated at `lambda = -1`. A pole at or below 1 leaves the estimate undefined (`pole_le_1`), as does a non-finite curve (`no_fit`); a flat curve returns its mean (`constant`), and a pole at the grid's cap gives the straight-line extrapolation (`linear_limit`).

**Standard errors of both.** A stratified bootstrap resamples the labelled and the unlabelled texts separately, each to its own count, and refits the reverse calibration, `sigma2` and the estimate on every resample (`n_boot` 200 for moment subtraction, 100 for SIMEX with `boot_reps` 10 noise replicates). The standard error is the standard deviation of the defined estimates; it needs at least 50 of them.

**Blind transport** (`blind_transport(sources, target)`). Within each source corpus, the human values and every score column shared by all corpora are z-scored on the labelled texts with the population standard deviation, and the z-scores are pooled across sources. The shared column with the highest pooled signed Spearman correlation with the pooled human values is chosen, and the least-squares line of the pooled human values on that column is applied to the target's column, z-scored over the target's texts. The target's human values are never read.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

for family in ("classical", "moment_subtraction", "simex", "blind_transport"):
    spec = cop.corrections.REGISTRY[family]
    errors = {variant: ", ".join(spec.errors_for(variant)) for variant in spec.variants}
    print(family, spec.target, spec.level, errors, spec.estimate_errors)
```

<!-- output -->
```text
classical A score {'fuller_shrinkage': 'calibration_shortfall', 'tweedie': 'calibration_shortfall, noise'} none
moment_subtraction D estimate {'moment_subtraction': ''} attenuation of the construct's coefficient as predictor
simex D estimate {'simex': ''} attenuation of the construct's coefficient as predictor
blind_transport A score {'transport': 'level_displacement, calibration_shortfall'} none
```

The two score corrections aim at target A: Fuller's shrinkage addresses the calibration shortfall, with the spread matched to the signal, and Tweedie smoothing the calibration shortfall and noise. Under a normal distribution of the proxy, Tweedie smoothing reduces to the least-squares line. Moment subtraction and SIMEX aim at target D, the attenuation of the construct's coefficient as a predictor; they assume classical error in the proxy, independent of the construct, the outcome and the covariates, and their guarantee holds only as far as that error model is right. The labelled texts serve only to fit the error model, so they must be a random sample of the texts. Blind transport addresses level displacement and calibration shortfall with another corpus's human values; its output is in standard-deviation units of the sources' human values, so it supports ranking and standardised comparisons, not the level of any one gold scale.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `classical` variant | | `fuller_shrinkage` or `tweedie` |
| `outcome` | | The outcome of the regression, a numeric column observed on every text |
| `score_column` | `None` | The score to put on the construct's scale; `None` for the first |
| `n_boot` | 200 (moment subtraction), 100 (SIMEX) | Bootstrap resamples, at least 50 |
| `reps`, `boot_reps` | 25, 10 | SIMEX noise replicates per multiple, for the estimate and in each resample |
| `seed` | 42 | Seed of the bootstrap and of the SIMEX noise |
| `sources`, `target` | | Blind transport's corpora with human values, and the corpus to correct |

## Example

On the toy corpus with a human sample of 27 texts, the score corrections are cross-fitted, and the coefficient of the construct as a predictor of text length in words is corrected. The true slope, from all gold values, is 10.6 words per point; the slope of the raw score is 25.9.

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
    extra_columns=["length_words"],
)
human = table.frame.set_index("text_id")["gold_constructiveness"]
draw = cop.draw_human_sample(table, n=27, seed=5)
data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)

for variant in ("fuller_shrinkage", "tweedie"):
    corrected = cop.cross_fit_predictions(cop.get_correction("classical", variant), data)
    print(variant, round((corrected - human).abs().mean(), 1))
moment = cop.moment_subtraction(data, "length_words")
print(round(moment.details["naive_slope"], 1), round(moment.point, 1))
extrapolated = cop.simex(data, "length_words")
print(extrapolated.details["lambdas"], round(extrapolated.point, 1))
```

<!-- output -->
```text
fuller_shrinkage 0.6
tweedie 0.6
11.2 13.2
(0.0, 0.5, 1.0, 1.5, 2.0) 12.6
```

On the proxy's scale the uncorrected slope is 11.2; moment subtraction gives 13.2 and SIMEX 12.6, both above the true 10.6 here. The toy proxy's error is not classical: it rises with text length, the outcome of this regression, so the uncorrected slope on the proxy already exceeds the truth, at 10.9 even with the reverse calibration fitted on all 54 gold values. The scale step fitted on the 27 labelled texts adds the rest, to 11.2. The attenuation correction, which assumes classical error, then amplifies both. Blind transport needs other corpora; the block treats the first 9 toy groups as a source with human values and the other 9 as a target without them.

<!-- run -->
```python
import numpy as np

groups = sorted(set(full.groups))
first = full.frame["group_id"].isin(groups[:9])
source = cop.CorrectionData(
    frame=full.frame[first], score_columns=("construct_score",), gold_column="gold_constructiveness"
)
target = cop.CorrectionData(
    frame=full.frame[~first].assign(gold_constructiveness=np.nan),
    score_columns=("construct_score",),
    gold_column="gold_constructiveness",
)
moved = cop.corrections.blind_transport([source], target)
print(moved.score_column, round(moved.selection_rho, 1), round(moved.slope, 1))
```

<!-- output -->
```text
construct_score 0.7 0.9
```

## Where the library departs from the paper's analysis code

- Fuller's `kappa` and Tweedie's proxy use population moments in both the covariance and the variance; the paper's analysis code mixed sample and population moments.
- Tweedie smoothing refuses a score that covaries negatively with the human values; the paper's analysis code returned a constant for it.
- The bootstrap of moment subtraction and SIMEX resamples texts, not groups, so it treats the texts as independent; with groups of correlated texts the standard errors are likely too small. This is a limitation of the library.
- Blind transport's column is in standard-deviation units of the sources' human values and is meant for ranking.

## Literature

Carroll et al. (2006); Cook and Stefanski (1994); Efron (2011); Fuller (1987). Full entries are in [References](../references.md).
