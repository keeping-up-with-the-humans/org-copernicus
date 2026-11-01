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

A score-level family that is fitted per text is built with `get_correction(family, variant, **params)` and applied with `cross_fit_predictions`. The correction ensemble and blind transport work on whole columns or corpora and are called as functions; the estimate-level families are functions that return an `Estimate`.

## The human sample

`draw_human_sample` draws whole groups (the persons, conversations or documents whose texts must stay together) by simple random sampling without replacement. For a quota of `n` analysis units (texts by default), it draws `m = min(G, ceil(n * G / n_units))` of the `G` groups, with `numpy.random.default_rng(seed)` from the sorted group ids. The number of groups depends only on the structure of the corpus, never on gold values, so every text of a drawn group has the exact inclusion probability `pi = m / G`. `attach_draw` sets the human values of the drawn texts on the correction data together with `pi`, and refuses a draw made on another table or one whose groups are not coded whole. A draw binds to its table through `population_hash`, a SHA-256 fingerprint of the table's texts and their groups, so a text added, removed or regrouped after the draw, or a renamed group, counts as another table. The design-based estimators divide by `pi`, so they refuse a labelled sample without it.

`plan_sample_size(rho, n_human, n_machine)` turns the correlation `rho` of the score with the human value into the factor `1 / ((1 - rho^2) + rho^2 * n_human / n_machine)`: the human sample counts as `factor * n_human` human-coded texts. Its limit for a growing corpus is `1 / (1 - rho^2)`. The factor is the variance ratio of a prediction-powered mean whose prediction enters with the weight `cov(gold, score) / var(score)`, as in the power-tuned `mean_ppi(power_tuning=True)` or with the `calibrated` rectifier of `pseudo_values`, which puts the score on the gold scale first. The default weight of 1 on the raw score reaches the factor only when the score already lies on the gold scale with that slope; otherwise it falls short (in one simulation with a score of correlation 0.91 and less than half the gold's spread, the plan promised 539 texts, the default estimator gave 296 and the power-tuned one 619), so a study sized by the planner should estimate with power tuning. The planner counts texts as independent. Under whole-group draws with correlated texts, the effective size `factor * n_human` is therefore an upper bound, and the number of texts it tells you to code is a lower bound.

## The raw column and cross-fitting

The score-level families read one score column unless they pool several: the column with the highest signed Spearman correlation with the gold values on the training rows, skipping a column that is constant there or has fewer than 3 labelled rows (the first column wins a tie). The estimate-level estimators use the first score column unless told otherwise, a choice that never looks at gold values.

A correction fitted on labelled texts and applied to the same texts looks better than it is. `cross_fit_predictions` splits the groups into `k` (5) grouped folds and predicts each fold with a correction fitted on the labelled texts of the other folds, so the corrected value of every text comes from a fit that never saw its gold value or that of any text of its group (Chernozhukov et al. 2018). A fold needs at least `MIN_TRAIN` (10) labelled training texts, and a family's conditions (`applies`) are checked on the whole data and again on each fold's training rows. Called directly, a family's `fit` checks the same conditions, on its training rows where they concern the labelled texts, and raises `NotApplicableError` with the reason `applies` gives.

## The battery

`run_battery` compares the score-level corrections out of sample on a corpus where more texts carry gold values than a study would code. For each seed, scheme and size, it splits the texts into grouped folds (`kfoldK`) or one grouped 80/20 split (`holdout`), draws a labelled sample of whole groups from the training texts that carry a gold value, fits every entry on the drawn texts and scores it on the held-out texts with a gold value, beside the raw baseline: the best column on the drawn texts. A size is a number of texts (at least 2) or a share of the training pool (at most 0.5); a quota above half the pool is not drawn. The defaults are the sizes 50, 100, 250, 500 and 1,000 texts and 1, 5, 10 and 25 percent, the schemes `kfold5`, `kfold10` and `holdout`, and the seeds 42, 43 and 44. `summarize_battery` reports the mean gain of every entry over the raw score in rank validity, in the 0 to 1 and the native mean absolute error, and in the absolute covariate-linked error, with standard errors clustered by seed. A positive gain is an improvement. Each 0 to 1 scale is the variable's own training range, so an increasing line leaves the 0 to 1 errors unchanged; the native error shows what it does to level and spread.

## The repeated study

`repeated_study` checks the estimate-level estimators on a corpus where every text has a human value. It draws many human samples of each size the way `draw_human_sample` does, estimates the mean and the difference between the longer and the shorter half of the texts with each estimator (`raw`, `calibrated`, `ppi_identity`, `ppi_crossfit`), and reports the bias, the share of 95 percent intervals that contain the full-gold value, and the mean standard error. A row undefined on more than 10 percent of its draws is reported as missing. No finite-population correction is applied, so at large sampling fractions the coverage of the design-based estimators is conservative.

## Where the library departs from the paper's analysis code

- `draw_human_sample` refuses only a quota below 1 or above the number of units, and warns above half of the units; the battery keeps the strict refusal of sizes above half the units.
- The battery's tables are not the paper's battery tables and do not reproduce its numbers. Each metric is a mean of per-fold values weighted by the fold's scored texts, where the paper pooled the held-out texts of all folds for its error metrics; the native error is the mean absolute error per text, where the paper reported the error of the mean; no fold is dropped for holding too few units; and the ensemble pools the entries fitted in each fold, so its members can change from fold to fold.
- The battery keys each family's random generator to the family, so the variants of a family share their inner draws, as in the paper's analysis code. Estimate-level families and blind transport are not part of the battery.
- The `calibrated` estimator of the repeated study uses the least-squares line of the gold on the score; the paper's analysis code divided a sample covariance by a population variance.

## Literature

Carroll et al. (2006); Chernozhukov et al. (2018); Fuller (1987); Ludwig et al. (2026); Mullainathan and Spiess (2017). Full entries are in [References](../references.md).
