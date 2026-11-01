# Correcting scores

A machine score has errors that a human sample can reveal: a shift in level, a spread that is too small, a drift with text length, noise. A **score-level correction** uses the human sample to repair the score text by text, so that the corrected value lies on the gold scale. This tutorial applies one correction with cross-fitting, then compares several corrections in the battery. To estimate a mean, a difference or a coefficient instead of repairing each score, read [Correcting estimates](correcting-estimates.md).

## What you need

- The base install. No GPU, no API key. (Corrections that read embeddings need the `embed` extra.)
- Machine scores for all texts and a human sample with gold values: see [The human sample](human-sample.md).
- The toy corpus supplies both here. The scores come from the fake backend.

To code with a real model instead of the fake backend, replace the backend line of the code below. This is a snippet to read, not to run:

```python
backend = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
```

## The families

`cop.list_corrections()` lists every correction, the errors it addresses and its level. A **score**-level correction maps each text's score to a corrected value. An **estimate**-level estimator corrects a statistic directly (see [Correcting estimates](correcting-estimates.md)).

| Family | Variants | What it does |
|--------|----------|--------------|
| `regression_calibration` | `linear`, `covariate_augmented`, `platt`, `isotonic` | Maps the best score column to the gold scale with a line, a line with the covariate, a logistic curve (binary gold) or a monotone piecewise-linear map (isotonic regression) |
| `classical` | `fuller_shrinkage`, `tweedie` | Classical measurement-error corrections of one score column |
| `marginal_matching` | `mean_sd`, `quantile` | Gives the score the mean and spread, or the whole distribution, of the gold; a tool for comparability, not a correction of single texts |
| `supervised_stacking` | `ridge`, `lasso`, `enet` | A penalised linear model over all score columns, the covariate and the spread of the scores |
| `embedding_augmented` | `embeddings_only`, `embeddings_plus_score`, `score_anchored_residual` | A ridge on text embeddings, with or without a score |
| `latent_pooling` | `factor_anchor` | One factor over several score columns, anchored to the gold |
| `correction_ensemble` | `mean`, `mode`, `factor`, `pc1` | Pools corrected columns of several families into one |
| `multiple_imputation` | `model_draws`, `pmm` | Draws the gold value several times and pools with Rubin's rules |
| `misclassification_matrix` | `posterior` | Discretises the score and inverts the confusion matrix |
| `pseudo_values` | `identity`, `calibrated`, `crossfit` | Spreads the human sample over all texts with known inclusion probabilities |
| `blind_transport` | `transport` | Applies a correction fitted on other corpora to a corpus without gold; the output is rank-only |

Most families read the best of the score columns. `supervised_stacking` and `latent_pooling` need at least two score columns. The ensemble pools the corrected columns of at least two other entries.

## Steps

### Step 1: build the data

<!-- run -->
```python
import numpy as np
from scipy.stats import spearmanr

import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
text_scores = cop.export_text_scores("run.sqlite")
full = cop.CorrectionData.from_scores(
    text_scores,
    table,
    table.frame,
    gold_column="gold_constructiveness",
    score_columns=["construct_score"],
)
human = table.frame.set_index("text_id")["gold_constructiveness"]
draw = cop.draw_human_sample(table, n=27, seed=5)
data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)
print(data.n, data.n_labelled)
```

<!-- output -->
```text
54 27
```

`full` has the gold value of every toy text; `data` keeps only the 27 texts of the draw, as a real study would.

### Step 2: fit and apply a correction with cross-fitting

`cross_fit_predictions` splits the groups into folds. Each fold is predicted by a correction fitted on the labelled texts of the other folds. So the corrected value of every text, labelled or not, comes from a fit that never saw that text's gold value.

<!-- run -->
```python
linear = cop.get_correction("regression_calibration", "linear")
corrected = cop.cross_fit_predictions(linear, data, k=3, seed=42)
raw = data.frame.set_index("text_id")["construct_score"]
truth = human.reindex(corrected.index)
print(len(corrected), int(corrected.isna().sum()))
print(round((raw - truth).abs().mean(), 1), round((corrected - truth).abs().mean(), 1))
print(round(spearmanr(raw, truth)[0], 2), round(spearmanr(corrected, truth)[0], 2))
```

<!-- output -->
```text
54 0
1.3 0.5
0.9 0.85
```

The raw score sits far below the gold on the gold's own scale: its mean absolute error is 1.3 points of the 1-to-5 scale. After the correction the error is 0.5. The rank correlation barely moves, and it can even fall a little, because the correction is fitted on a small sample. A linear correction with a positive slope cannot reorder the texts, so it cannot raise rank validity.

### Step 3: compare corrections in the battery

Which correction should you use, and how large must your human sample be? The battery answers that on a corpus where you have more gold values than you would use. It draws labelled samples of several sizes from the training part of each fold, fits every correction, and scores it on the held-out texts. The block compares three corrections at two sizes on the whole toy corpus, with two seeds.

<!-- run -->
```python
families = [
    ("regression_calibration", "linear"),
    ("marginal_matching", "mean_sd"),
    ("supervised_stacking", "ridge"),
]
settings = cop.BatterySettings(
    families=families, sizes=(9, 30), schemes=("kfold3",), seeds=(42, 43)
)
result = cop.run_battery(full, settings)
summary = cop.summarize_battery(result)
print(
    summary[["family", "variant", "gain_rank_validity", "n_rows"]].round(1).to_string(index=False)
)
scored = result.results[~result.results["diagnostic"]]
means = scored.groupby("variant")[["mae_native_raw", "mae_native_corrected"]].mean()
print(means.round(1).to_string())
```

<!-- output -->
```text
family variant gain_rank_validity n_rows
regression_calibration linear 0.0 2
marginal_matching mean_sd 0.0 2
supervised_stacking ridge NaN 0
mae_native_raw mae_native_corrected
variant
linear 1.2 0.6
mean_sd 1.2 0.6
```

`summarize_battery` reports the mean **gain** of each correction over the raw score: in rank validity, in the 0-to-1 mean absolute error, in the native-scale mean absolute error, and in the covariate-linked error. A positive gain for the errors means that the correction reduced them. Both corrections gain nothing in rank validity, as they should not. They halve the native-scale error, from about 1.2 to about 0.6 points of the 1-to-5 scale. The ridge stacking has no rows: the toy data have a single score column.

### Step 4: read the skipped fits

The battery never fits a size it cannot support. The `skipped` table says why.

<!-- run -->
```python
reasons = result.skipped["reason"].str.split(":").str[0].value_counts().to_dict()
print(reasons)
```

<!-- output -->
```text
{'quota_exceeds_half_units': 20, 'not_applicable': 5, 'EstimatorError': 3}
```

- `quota_exceeds_half_units`: a size above half of the texts of the training pool. Here a size of 30 does not fit a pool of 36 texts.
- `not_applicable`: the correction does not apply to the data, for example the stacking with one score column.
- `EstimatorError`: a fit failed; the reason names the cause, such as a score column that is constant on the drawn texts.

The battery also adds a `pseudo_values` row with `diagnostic=True`. It scores the pseudo-value estimate on the whole data and is not a competitor of the score corrections; it appears only when every text has a gold value.

## On the command line

```bash
copernicus store export run.sqlite --what text --out text_scores.csv
copernicus corrections list
copernicus corrections battery --scores text_scores.csv --texts toy_corpus.csv --gold toy_corpus.csv \
    --group-column group_id --gold-column gold_constructiveness --score-columns construct_score \
    --families regression_calibration:linear --families marginal_matching:mean_sd \
    --sizes 9,30 --schemes kfold3 --seeds 42,43 --out battery.csv
```

The command writes `battery.csv`, and beside it `battery_skipped.csv`, `battery_raw.csv` and `battery_summary.csv`.

## What the output means

- **Rank validity does not move under a monotone correction.** A line with a positive slope, and the other monotone maps, keep the order of the texts. Corrections help the level and the spread of the score, and they help a covariate-linked error only if they model the covariate. Raising rank validity needs a different or an additional score, such as several scores pooled by the stacking or the ensemble.
- **Use the native-scale error to see a level shift.** The 0-to-1 errors of the validity panel scale the score and the gold by their own ranges, so an affine correction leaves them unchanged. The battery's `mae_native` measures the error on the gold scale, where the shift shows.
- **The battery tables are not the paper's tables.** The battery averages per-fold values, weighted by the fold's scored texts. The paper pooled the held-out texts of all folds. Do not compare the numbers with the paper's tables. See the [methods pages](../methods/index.md).
- **Marginal matching is a comparability tool.** It gives a score the gold's mean and spread. It does not make a single text's value more accurate.

## Pitfalls

- **Fit and apply with cross-fitting.** Fitting a correction on the labelled texts and applying it to the same texts overstates its quality. `cross_fit_predictions` and the battery do it correctly.
- **A correction needs enough labelled texts.** Cross-fitting needs at least 10 labelled training rows in each fold, and families with more parameters need more.
- **Do not take the best correction on a tiny validation set.** The battery's gains carry a seed-clustered standard error; with two seeds it is rough. Use more seeds on real data.
- **A correction does not carry over to another corpus.** It is fitted to one coder, one rubric and one population. Blind transport applies a correction to a corpus without gold, and its output is rank-only.

## Next

[Correcting estimates](correcting-estimates.md) estimates a statistic from the human sample directly.
