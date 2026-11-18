# Corrections

Machine-coded values are predictions, and their errors bias an analysis as measurement error does (Mullainathan and Spiess 2017, Fuller 1987, Carroll et al. 2006). The errors of language-model labels need not be classical: they can move with the covariates of the regression they enter, in ways that depend on the prompt and the model (Ludwig et al. 2026). A sample of texts coded by humans lets a study measure these errors and correct what they do. The library's corrections start from such a sample, drawn from the corpus under study; blind transport, which borrows the samples of other corpora, is the one exception.

The four score errors of the [validity panel](../validity-panel.md) turn into three ways a published estimate can be wrong: the estimate is off (a mean by the level displacement; a slope or a difference by calibration shortfall or covariate-linked error; a predictor's coefficient pulled toward zero), the confidence interval is too wide (noise and a small human sample), and the interval misses its target (bias that remains, or a standard error that treats corrected values as known). A correction at the score level gives every text a new value. Most score-level families aim at target A and multiple imputation at B, and two of them also serve target C, as the table below shows: the misclassification matrix for class shares (`A, C`) and embedding-augmented calibration under pseudo-values (`A, then C`). A correction at the estimate level corrects the estimate itself and aims at target C or D. The [methods overview](../index.md#the-four-correction-targets) defines the targets.

## The families

The registry lists every family with its target and level. `list_corrections()` gives one row per variant, with the score errors it addresses.

<!-- run -->
```python
import copernicus as cop

listing = cop.list_corrections()
families = listing.groupby("family", sort=False).agg(
    variants=("variant", "size"), target=("target", "first"), level=("level", "first")
)
print(families.reset_index().to_string(index=False))
```

<!-- output -->
```text
                  family  variants    target    level
  regression_calibration         4         A    score
               classical         2         A    score
     embedding_augmented         3 A, then C    score
     correction_ensemble         4         A    score
     multiple_imputation         2         B    score
          latent_pooling         1         A    score
       marginal_matching         2         A    score
misclassification_matrix         1      A, C    score
           pseudo_values         3         C estimate
     supervised_stacking         3         A    score
         blind_transport         1         A    score
           ludwig_moment         4     C / D estimate
      moment_subtraction         1         D estimate
                   simex         1         D estimate
     instrumented_coding         1         D estimate
```

| Families | Page |
|----------|------|
| `regression_calibration` | [Regression calibration](regression-calibration.md) |
| `marginal_matching` | [Marginal matching](marginal-matching.md) |
| `multiple_imputation` | [Multiple imputation](multiple-imputation.md) |
| `misclassification_matrix` | [Misclassification matrix](misclassification-matrix.md) |
| `pseudo_values`, `ludwig_moment` (the estimators `mean_ppi`, `group_difference_ppi`, `ols_outcome_ppi`, `ols_predictor_ppi`) | [Pseudo-values and prediction-powered inference](pseudo-values.md) |
| `latent_pooling` | [Latent pooling](latent-pooling.md) |
| `supervised_stacking` | [Supervised stacking](supervised-stacking.md) |
| `classical`, `moment_subtraction`, `simex`, `blind_transport` | [Classical corrections](classical-corrections.md) |
| `embedding_augmented` | [Embedding-augmented calibration](embedding-augmented-calibration.md) |
| `correction_ensemble` | [Correction ensemble](correction-ensemble.md) |
| `instrumented_coding` | [Instrumented coding](instrumented-coding.md) |

A score-level family that is fitted per text is built with `get_correction(family, variant, **params)` and applied with `cross_fit_predictions`. The correction ensemble and blind transport work on whole columns or corpora and are called as functions; the estimate-level families are functions that return an `Estimate`. The details of every `Estimate` end with `se_exactly_zero`, true when its standard error is exactly 0, so that its interval has zero width; the estimate keeps that standard error, and one warning says so.

## The human sample

`draw_human_sample` draws whole groups (the persons, conversations or documents whose texts must stay together) by simple random sampling without replacement. For a quota of `n` analysis units (texts by default), it draws `m = min(G, ceil(n * G / n_units))` of the `G` groups, with `numpy.random.default_rng(seed)` from the sorted group ids. The number of groups depends only on the structure of the corpus, never on gold values, so every text of a drawn group has the exact inclusion probability `pi = m / G`. `attach_draw` sets the human values of the drawn texts on the correction data together with `pi`, and refuses a draw made on another table or one whose groups are not coded whole. A draw binds to its table through `population_hash`, a SHA-256 fingerprint of the table's texts and their groups, so a text added, removed or regrouped after the draw, or a renamed group, counts as another table. The design-based estimators divide by `pi`, so they refuse a labelled sample without it. With groups of unequal size the number of texts drawn varies with the groups drawn, and a draw of a few groups can miss a group that holds a large share of the texts; [the repeated study](#the-repeated-study) says what that can do to the estimates.

`text_count_variation(group_id, m)` measures that variation: `c = (sd_n / mean_n) * sqrt((1 - m / G) / m)`, the coefficient of variation (the standard deviation divided by the mean) of the number of texts of a draw over all draws of `m` of the `G` groups, with `n_g` the number of texts of group `g`, `mean_n` their mean and `sd_n` their standard deviation over the `G` groups with divisor `G - 1`. For a mean over the drawn texts of a quantity fixed before the draw, the bias is at most `c` times the standard deviation, both taken over all possible draws of `m` of the `G` groups. The reason: such a mean `R`, times the mean number of texts of a drawn group, is the mean of the drawn groups' totals, whose expectation does not depend on the draw, so `E[R] - R_corpus = -Cov(R, mean texts per drawn group) / mean_n`; the Cauchy-Schwarz inequality bounds that covariance by `sd(R)` times `c * mean_n`, the standard deviation of the mean texts per drawn group. Of the library's estimators this covers the point of `mean_ppi` without power tuning and with a prediction fixed before the draw, such as the score, which is such a mean, of the gap between the human value and the prediction, plus the mean of the prediction over all texts; and it covers it only when no draw is refused (the estimator refuses a draw with fewer than two labelled texts and, when it clusters, one whose labelled texts lie in one group). The bound does not cover a summary over the accepted draws only, since a refused draw changes which draws count, nor the bias that the repeated study reports, the mean over the defined draws of a finite run, nor the power-tuned or the cross-fitted mean, nor the coverage of any interval. Nor does `c` bound the bias of `group_difference_ppi`. For that estimator without power tuning and with a prediction fixed before the draw, provided it refuses no draw, the bias is at most `c_A * sd(R_A) + c_B * sd(R_B)`. Here `R_A` and `R_B` are each side's mean, over its drawn texts, of the gap between the human value and the prediction, both standard deviations are taken over all possible draws, and `c_A` is the formula above with `n_g` the number of the group's texts on side A, counted over all `G` groups of the corpus, a group without a text on that side counting 0 (`text_count_variation` applied to the ids of one side's texts alone leaves out the groups without a text on that side, so keep those groups when computing `c_A`). `draw_human_sample` logs one warning when `c` exceeds `TEXT_COUNT_VARIATION_THRESHOLD`, 0.1, the value up to which the bias of such a mean is at most a tenth of its standard deviation: a signal that the number of drawn texts varies much, not a bound for every estimator. The warning names `c`, `m`, `G` and the share of the texts that the largest group holds, and the decision compares the square of `c`, computed exactly from the counts, with 1/100. `c` counts texts, the denominator of the estimators, also when the quota counts another unit, and it is 0 with groups of equal size, with one text per group and for the full draw. Where the warning appears, check the design with [the repeated study](#the-repeated-study), or estimate the mean with `pseudo_values(data).mean_rectified`, which, with its default `identity` rectifier, the recorded `pi = m / G` and every text of a drawn group coded, is exactly unbiased over draws. A corpus of one group of 200 texts and 200 groups of one text shows how far `c` can rise:

<!-- run -->
```python
import numpy as np

groups = np.array(["large"] * 200 + [f"single{index}" for index in range(200)])
for quota in (24, 300):
    m = cop.corrections.groups_for_quota(quota, groups, np.arange(groups.size))
    print(quota, m, round(cop.corrections.text_count_variation(groups, m), 3))
```

<!-- output -->
```text
24 13 1.892
300 151 0.286
```

A quota of 24 texts draws 13 of the 201 groups, and the bias of a mean over the drawn texts of a quantity fixed before the draw can be up to 1.9 times its standard deviation over all possible draws; a quota of 300 draws 151 of the groups, and the bound is still 0.29 times the standard deviation.

`plan_sample_size(rho, n_human, n_machine)` turns the Pearson correlation `rho` of the score with the human value, computed from the data, into the factor `1 / ((1 - rho^2) + rho^2 * n_human / n_machine)`. The human sample then counts as `factor * n_human` human-coded texts. The upper limit of this multiplier, for a growing corpus, is `1 / (1 - rho^2)`. A rank correlation is another number than the Pearson correlation.

The factor is the variance ratio of a prediction-powered mean whose prediction enters with the weight `cov(gold, score) / var(score)`. The power-tuned `mean_ppi(power_tuning=True)` gives the score that weight, and so does the `calibrated` rectifier of `pseudo_values`, which puts the score on the gold scale first. When some scored texts have no human value, the default weight of 1 on the raw score reaches the factor only when `cov(gold, score) / var(score)` is 1. With another slope it falls short, and the spreads of score and gold alone do not show that slope. When every scored text is human-coded, every weight gives the mean of the human values.

With `rho = 0.91`, `c = 0.5`, `n_human = 100` and `n_machine = 5000`, where `c` is the standard deviation of the score over that of the human value, the plan promises an effective size of 531 texts and the default weight is worth 283, from `n_human / (1 + (1 - n_human / n_machine) * (c^2 - 2 * rho * c))`. Both are large-sample values. They assume that the scored texts include the human-coded ones, that the human-coded texts are a simple random subset of independent texts, and that precision is that of the estimate of the mean of the population the texts come from. So a study sized by the planner should estimate with power tuning.

The factor is a large-sample number that treats the weight of the score as known. Power tuning estimates the weight on the human-coded texts, which lowers the realised factor. For independent texts with a jointly normal score and human value, of which a random subset of more than three is human-coded and gives the weight, the estimate of the mean of the population the texts come from falls short of the plan by at most `1 / (n_human - 2)` of it. That is an upper value, approached when the human-coded texts are a small share of the corpus.

For a coefficient of a regression with the construct as the outcome, call a residual what the linear fit on the regressors leaves unexplained, of the human value or of the score. The same large-sample factor holds with the partial correlation, the correlation of the two residuals, when the prediction is the least-squares fit of the human value on the score and the regressors and when the mean squares of the two residuals and their mean product, at given values of the regressors, do not change with the regressors. Jointly normal human value, score and regressors satisfy this condition on the residuals. It is not enough that each residual is, on its own, unrelated to the regressors. When the size of the score's error depends on a regressor, the corrected coefficient can be less precise than the human-coded sample alone, as the planner's docstring says.

The planner counts the texts as independent and does not model the dependence of texts within a group. What matters is the dependence, within a group, of what each text contributes to the estimate: a positive correlation of these contributions lowers the effective sample size below that of independent texts, and a negative one raises it. Check the design with [the repeated study](#the-repeated-study).

## The raw column and cross-fitting

The score-level families read one score column unless they pool several: the column with the highest signed Spearman correlation with the gold values on the training rows, skipping a column that is constant there or has fewer than 3 labelled rows (the first column wins a tie). The estimate-level estimators use the first score column unless told otherwise, a choice that never looks at gold values.

A correction fitted on labelled texts and applied to the same texts looks better than it is. `cross_fit_predictions` splits the groups into `k` (5) grouped folds and predicts each fold with a correction fitted on the labelled texts of the other folds, so the corrected value of every text comes from a fit that never saw its gold value or that of any text of its group (Chernozhukov et al. 2018). A fold needs at least `MIN_TRAIN` (10) labelled training texts, and a family's conditions (`applies`) are checked on the whole data and again on each fold's training rows. Called directly, a family's `fit` checks the same conditions, on its training rows where they concern the labelled texts, and raises `NotApplicableError` with the reason `applies` gives.

## The battery

`run_battery` compares the score-level corrections out of sample on a corpus where more texts carry gold values than a study would code. For each seed, scheme and size, it splits the texts into grouped folds (`kfoldK`) or one grouped 80/20 split (`holdout`), draws a labelled sample of whole groups from the training texts that carry a gold value, fits every entry on the drawn texts and scores it on the held-out texts with a gold value, beside the raw baseline: the best column on the drawn texts. A size is a number of texts (at least 2) or a share of the training pool (at most 0.5); a quota above half the pool is not drawn. The defaults are the sizes 50, 100, 250, 500 and 1,000 texts and 1, 5, 10 and 25 percent, the schemes `kfold5`, `kfold10` and `holdout`, and the seeds 42, 43 and 44. `summarize_battery` reports the mean gain of every entry over the raw score in rank validity, in the 0 to 1 and the native mean absolute error, and in the absolute covariate-linked error, with standard errors clustered by seed. A positive gain is an improvement. Each 0 to 1 scale is the variable's own training range, so an increasing line leaves the 0 to 1 errors unchanged; the native error shows what it does to level and spread. A fit or a combination that keeps a value the data leave open records it and, called on its own, logs one warning per case (see the pages of multiple imputation, latent pooling and the correction ensemble); in the battery these warnings are counted instead, and after its seeds the battery logs one warning for each entry and size whose calls met such cases, with the count of each case, while the provenance of every scored fold holds the records of its fit.

## The repeated study

`repeated_study` checks the estimate-level estimators on a corpus where every text has a human value. It draws many human samples of each size the way `draw_human_sample` does, estimates the mean and the long-short difference, the texts at or above the median of the covariate (by default the longer texts) against the texts below it, with each estimator (`raw`, `calibrated`, `ppi_identity`, `ppi_crossfit`), and reports the bias, the share of 95 percent intervals that contain the full-gold value, and the mean standard error. A row undefined on more than 10 percent of its draws is reported as missing. Ties at the median can make the two sides unequal, and a constant covariate leaves the side below the median empty, which the study refuses: each side needs at least two texts. A draw with fewer than two human-coded texts, or for the difference fewer than two on a side, is undefined for `ppi_identity` and `ppi_crossfit` alike, whatever the value of `cluster`: a side that the draws rarely reach shows as undefined draws, not as an estimate from that side's uncorrected predictions. A clustered difference of `ppi_crossfit` also needs at least two groups among the texts of each side; when every text of a side lies in one group, the study refuses the difference on every clustered draw. No finite-population correction is applied. This can make the intervals conservative at large sampling fractions, but it does not guarantee coverage: when the groups differ in size, a whole-group draw can miss a group that holds a large share of the texts in many draws, and the [ratio bias of the mean and the difference](pseudo-values.md#exactness-and-limitations) and a severe undercoverage of the normal interval can then remain although every draw is defined. Before the draws of each size whose `c` (see [the human sample](#the-human-sample)) exceeds 0.1, the study logs one such warning, which names the size, whatever the number of draws. Over all possible draws of that size, and provided no draw is refused, the bias of the mean of `ppi_identity` is at most `c` times its standard deviation: that mean, times the mean number of texts of a drawn group, has an expectation that does not depend on the draw, so its bias is minus a covariance divided by the mean number of texts per group, and the Cauchy-Schwarz inequality bounds the covariance by the product of the two standard deviations. The bound does not cover the study's own bias column, the mean over the defined draws of a finite run, in which a refused draw changes which draws count; nor the difference, whose two sides each vary on their own; nor `calibrated` or `ppi_crossfit`. Read both the bias and the coverage of the rows of that size, and the reasons of their undefined draws: the bound says nothing about coverage. Where the rows show a bias, estimate a mean with `pseudo_values(data).mean_rectified`. A row's status `ok` certifies only that the share of undefined draws is within the limit, not that the interval covers: read its coverage. The study counts in the same way the warnings that its cross-fits and the estimates of its draws log on their own for the cases of the [pseudo-values](pseudo-values.md#the-algorithm-as-the-library-implements-it), a fold of numerical rank below 3 and a standard error of exactly 0, and logs one warning for each size whose draws met them, with the count of each case, a cross-fit of a draw counted once for both estimands. The column `n_rank_deficient_folds` counts for `ppi_crossfit` the inner folds, over a row's defined draws, whose design `[1, score, covariate]` had a numerical rank below 3, as `numpy.linalg.lstsq` with `rcond=None` returns it, where the fold kept the coefficients that `lstsq` returns (see [the pseudo-values](pseudo-values.md#the-algorithm-as-the-library-implements-it)); it is 0 for the other estimators, which fit no folds. The last column, `n_se_exactly_zero`, counts for every estimator the defined draws of a row whose computed standard error is exactly 0, as it is for a plugged-in score of 2 on every text (for a score of 0.1 on every text rounding can leave a tiny positive standard error instead, depending on the number of texts); the interval of such a draw has zero width, and it enters the row's coverage and mean standard error like any other.

## Where the library departs from the paper's analysis code

- `draw_human_sample` refuses only a quota below 1 or above the number of units, and warns above half of the units; the battery keeps the strict refusal of sizes above half the units.
- Each metric of the battery's tables is a mean of per-fold values weighted by the fold's scored texts. The native error is the mean absolute error per text. The battery applies no table of minimum numbers of human-coded texts for the families; a correction still refuses what its own fit cannot use, such as the misclassification matrix with fewer than 30 labelled texts in a category. The ensemble pools the entries fitted in each fold, so its members can change from fold to fold.
- The battery keys the random generator of a fit to the seed, the scheme, the size, the fold and the family, not to the variant, so the variants of a family start from the same state of that generator. A variant may derive its own inner stream from it, as the lasso and elastic-net variants of supervised stacking derive their inner splits. The paper's analysis code keyed its generator also to the dataset, the target and the coder, and for the classical corrections to each method separately, so the same seed gives other draws in the library. The estimate-level families and blind transport are not part of the library's battery; the paper's battery also ran pseudo-values, moment subtraction, SIMEX, instrumented coding and blind transport.
- The default covariate of `CorrectionData.from_scores` is the log of `n_chars`, the number of characters of the table text with every run of whitespace collapsed to one space. The paper's analysis code counted the characters of the stored text, markup and whitespace included, and took the log of the count in its correction battery and `log(1 + count)` in its repeated study. To use the paper's covariate, pass a column of your own with `covariate=`.
- `repeated_study` draws whole groups, as `draw_human_sample` does, with `pi = m / G`, and by default clusters the standard errors of `ppi_identity` and `ppi_crossfit` by group when a drawn group holds more than one text. The paper's repeated-study code drew single texts by simple random sampling, used the groups only in its cross-fitting and treated the texts as independent in its standard errors. On a corpus in which some group holds more than one text the two codes therefore draw by different designs.
- The `calibrated` estimator of the repeated study uses the least-squares line of the gold on the score; the paper's analysis code divided a sample covariance by a population variance, so its line was `n / (n - 1)` times as steep for `n` labelled texts, and its long-short difference `n / (n - 1)` times the library's.
- The best score column needs at least three labelled texts with a finite score; the paper's analysis code also chose a column on two.
- In the battery an entry whose predictions on the drawn texts do not vary is not scored on that fold (`constant_training_range`); the paper's analysis code mapped such predictions to 0 on the 0 to 1 scale and kept the fold in its error metrics.

## Literature

Carroll et al. (2006); Chernozhukov et al. (2018); Fuller (1987); Ludwig et al. (2026); Mullainathan and Spiess (2017). Full entries are in [References](../references.md).
