# NLP-ML baseline

The NLP-ML baseline is the yardstick for the coders. It uses no rubric and no prompt: an encoder turns each text into an embedding, and a ridge regression, a linear regression whose coefficients are shrunk toward zero so that it fits less noise, learns the human value from the embedding on a sample of human-coded texts. Ridge regression is one member of the family of penalised linear models that also holds the lasso and the elastic net (Zou and Hastie 2005).

A detection method that needs no human-coded texts should be judged against what a model trained on such texts reaches, and supervised models trained on human-coded texts are an established way of measuring constructs in management research (Choudhury et al. 2019, Miric et al. 2023). Their output is a prediction, with the strengths and limits of prediction rather than of estimation (Mullainathan and Spiess 2017). The baseline reports how well the predictions rank held-out texts at each size of the training sample, so a study can see what its human-coded texts would buy if it trained a model on them. A second variant adds a coder's score to the embeddings and shows whether the score carries information beyond them.

## The algorithm as the library implements it

1. **Folds.** The groups (the persons or documents whose texts must stay together) are split at random into `k_folds` (5) outer folds; a group never straddles a split.
2. **Training sizes.** For each seed and each size, each fold's training pool is cut down to a draw of whole groups, added in random order until the draw holds the size's number of analysis units (it can overshoot by less than one group); `"full"` takes the whole pool. The default sizes are 50, 100, 150, 200, 250, 500 and 1,000 units and `"full"`. A size is skipped for the seed, and recorded with its reason, when its quota exceeds half the units (`quota_exceeds_half_units`), when some fold's pool holds fewer units than the quota (`pool_below_quota`), or when some draw holds fewer units than the floor of `floor_units` (50) units (`below_floor`), the full pool included.
3. **Features.** The embedding columns are z-scored on the training draw (a column whose computed standard deviation is exactly zero is divided by 1). Several encoders can be stacked; each block is then weighted by `sqrt(mean_dim / dim_block)`, with `dim_block` counting every column of the block, so that every encoder contributes the same total variance when no column is constant on the training draw.
4. **Ridge and penalty.** A ridge regression with an unpenalised intercept is fitted on the draw. The eigenvalues of the centred design's Gram matrix (its squared singular values) at or below `1e-10` times the larger of the largest eigenvalue and `1e-300` are dropped, and the fit refuses a design with none left. The penalty comes from `ALPHAS`, 41 values from 0.01 to 10^8 evenly spaced on the log scale, and is the one with the smallest leave-one-group-out error inside the draw. That error is exact, computed from the hat matrix of a single decomposition of the design, not approximated by refits. Whether the chosen penalty sits on an edge of the grid is recorded.
5. **The score variant.** With a coder score, `embeddings_plus_score` standardises the score on the draw and appends it as one more penalised column, through an exact rank-one update of the same decomposition.
6. **Metrics.** The test fold's predictions and gold values are averaged within each analysis unit, and the metrics are computed on these unit means: Spearman and absolute Spearman correlation, Pearson correlation, the mean absolute error on the 0 to 1 scale of the training draw's gold range and on the gold's own scale, and for binary gold the area under the ROC curve and Somers' D (`2 * AUC - 1`). Fold values are weighted by the fold's units. `curves` reports the rank validity, the mean over seeds of the weighted absolute Spearman correlation, by training size.

**Folds and draws of your own.** `run_ridge_baseline(..., splits=BaselineSplits(folds, draws))` takes the folds of step 1, and the draws of step 2 unless `draws` is `None`, from two tables in place of the random generators. `folds` gives the test fold (0 to `k_folds - 1`) of every analysis text for every seed, in the columns `seed`, `text_id` and `fold`. `draws` gives the texts of the training draw of each fold at each size, in the columns `seed`, `size`, `fold` and `text_id`, with `size` written as the results table writes it (`"50"`, `"full"`); a `"full"` draw may be left out, as it is the whole pool. With `draws=None` the draws come from the generators, on the supplied folds. Before any fit the run refuses splits that would break a guarantee of the steps above, with a message that names the seed, the size and the fold where they apply and the first ten offending ids: seeds other than those of the settings; a fold label outside 0 to `k_folds - 1`, or a fold without texts; folds of a seed that miss an analysis text, hold one twice or hold a text that is not an analysis text; a group with texts in two folds of a seed; and, for a seed and size whose draws the run consults (to fit them, or to check them against the floor), a fold without a draw, a draw that holds a text outside the fold's training pool, part of a group, or fewer units than the size names while the pool holds more, a `"full"` draw that is not the whole pool, or draws for a seed, a size or a fold that the settings do not ask for. The three skip reasons apply as to drawn splits. A size skipped because its quota exceeds half the units or a pool lies below it needs no draw, and draws supplied for it are not used, with a warning. A count size for which the splits hold no draw at all needs none either when its smallest training pool holds fewer units than the floor, as no draw from that pool can reach the floor: it is skipped with the reason `below_floor(<floor>):min_pool_units=<units>`, where a size whose draws were consulted names its smallest draw (`min_draw_units`). The draw receipts are computed from the supplied draws: a supplied draw may overshoot its size by several groups; the row then keeps the size's label (`size`, `size_units`) while the receipts (`draw_units_min`, `draw_units_mean`, `draw_units_max`, `draw_overshoot_max`) show how many units were fitted on. `result.splits` holds the folds and every draw that a run consulted: the draws of every fitted seed and size and of every one then skipped below the floor, those of size `"full"` included. Passed as `splits` to a run on the same inputs and settings they give the same folds, draws and skips, under any name of the gold column, and on the same machine with the same thread count the same tables bit for bit. `write_splits` and `read_splits` keep them in a JSON file that holds the two tables under `splits` beside the library version, as the draw file of the human sample holds its draw; on the command line, `--write-splits` and `--splits` do the same. The `seed` columns of `results`, `oof_predictions` and `skipped` hold the seeds' integer values; their dtype can be a signed integer, an unsigned integer or object, so do not assume `int64`.

The defaults are the paper's settings: 5 folds, the seeds 42, 43 and 44, the sizes above and the penalty grid above. To reuse the folds and draws of an earlier run, pass them as `splits`.

## Assumptions and what it corrects

The baseline produces predictions; it corrects nothing. In the library it predicts the human-coded texts only, each out of fold, so it is a benchmark, not a score for the rest of a corpus; a cross-fitted `embeddings_only` variant of [embedding-augmented calibration](corrections/embedding-augmented-calibration.md) scores a corpus from its human-coded texts (see the departures below). It assumes that the human-coded texts are a sample of the corpus they stand for, and that the analysis unit, the level at which the gold value is defined, is named correctly: a person, a conversation or a text (by default the group). Its out-of-fold predictions are a score like any other and carry bias into an estimate unless they are corrected. The absolute Spearman correlation inflates weak predictors, because a fold with a negative correlation still adds to it; compare it with the signed `spearman` column. The folds are only as good as the group column: when one person stands behind several groups, such as several conversations, group by the person; otherwise the person's other texts can sit in the training folds, and the out-of-fold values can overstate what the baseline reaches on new persons.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `BaselineSettings.k_folds` | 5 | Grouped outer folds |
| `BaselineSettings.seeds` | (42, 43, 44) | Seeds of the folds and the draws |
| `BaselineSettings.sizes` | (50, 100, 150, 200, 250, 500, 1000, `"full"`) | Training sizes in analysis units |
| `BaselineSettings.floor_units` | 50 | The fewest units a training draw may hold |
| `BaselineSettings.alphas` | `ALPHAS` | The penalty grid |
| `BaselineSettings.encoder_sets` | (`"single"`,) | One set per encoder, `"stacked"`, or one encoder's name |
| `BaselineSettings.binary` | `None` | Treat the gold as binary; `None` infers it from the gold values |
| `unit_id` | `None` | Column of the analysis unit; `None` uses the group |
| `coder_score` | `None` | A score by text id, which adds the `embeddings_plus_score` variant |
| `splits` | `None` | Folds, and training draws unless they are `None`, of your own (`BaselineSplits`), in place of the random generators |

## Example

The block prints the constants named above.

<!-- run -->
```python
import copernicus as cop

print(len(cop.baseline.ALPHAS), cop.baseline.ALPHAS[0], cop.baseline.ALPHAS[-1])
print(cop.baseline.K_FOLDS, cop.baseline.FLOOR_UNITS, cop.baseline.SIZES)
```

<!-- output -->
```text
41 0.01 100000000.0
5 50 (50, 100, 150, 200, 250, 500, 1000, 'full')
```

The baseline on the toy corpus needs MiniLM. The toy corpus has 18 groups, so the block uses three folds, one seed and a floor of 5 units.

<!-- run embed -->
```python
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
encoder = cop.Encoder("minilm", device="cpu")
texts = encoder.embed(table.frame["text"].tolist(), table.frame["text_id"].tolist())
settings = cop.BaselineSettings(sizes=(6, "full"), seeds=(42,), k_folds=3, floor_units=5)
result = cop.run_ridge_baseline(
    {"minilm": texts}, table.frame, table, settings, gold_column="gold_constructiveness"
)
print(cop.curves(result)[["size", "rank_validity"]].round(1).to_string(index=False))
```

<!-- output -->
```text
size  rank_validity
   6            0.9
full            0.9
```

[The NLP-ML baseline](../tutorials/nlp-ml-baseline.md) adds a coder's score, reads the skipped sizes and the out-of-fold predictions, and runs the command `copernicus baseline ridge`.

## Where the library departs from the paper's analysis code

- `run_ridge_baseline` predicts the human-coded texts only. The paper's analysis code also predicted every text of a dataset-construct combination with a ridge cross-fitted over all its texts, and corrected the estimates built on those predictions with prediction-powered inference. In the library the `embeddings_only` variant of embedding-augmented calibration takes that step, cross-fitted with `cross_fit_predictions` and passed to `mean_ppi(data, prediction=...)`; it uses the whole embedding and chooses among the 41 penalties of `ALPHAS` on grouped inner folds, where the paper's analysis code reduced the embedding to at most 128 principal components and chose among five penalties by the leave-one-group-out error.
- A binary gold may use any two values, such as 1 and 2; the paper's analysis code required 0 and 1. A binary gold that varies within an analysis unit is refused.
- The floor applies to the size `"full"` too, as in the paper's analysis code. A size whose quota some fold's training pool cannot reach is skipped with the reason `pool_below_quota`; the paper's analysis code fitted such a point on a smaller sample under the size's label.
- The `size` column is a string label (`"50"`, `"full"`) with the integer quota in a separate `size_units` column.
- In the fold metrics, as in every association, slope and min-max scaling of the library, a set of values counts as constant when its range is at most `1e-12` times the larger of 1 and its largest absolute value; the paper's analysis code tested for an exact zero range.
- The random generators of the folds and of the training draws are keyed by the seed and the name of the gold column, a draw also by its size written as a number (`50`); the paper's analysis code keyed them by the seed, the dataset, the target and the cohort, with a size written as `n50`. With the same seed the two codes can therefore select other folds and other training texts. These are the folds and draws of the default, `splits=None`. Folds and draws saved from an earlier run can be passed as `splits` instead: they fix which texts form each test fold and each training draw and nothing else, so the scaling, the fit, the choice of the penalty, the metrics, the skip rules and the other departures on this page stay the library's own.
- `curves` averages over the seeds without skipping a missing value, so a seed without a value leaves the size's mean missing; the paper's analysis code averaged over the seeds that had one.
- The library leaves the thread count of the numerical libraries as it finds it. The paper's runner set the count itself, to one unless the environment named another number and, inside each fit, to a number given on its command line. The predictions can depend on it in their last digits, so to reproduce stored predictions bit for bit, run with the thread count they were made with.

## Literature

Choudhury et al. (2019); Miric et al. (2023); Mullainathan and Spiess (2017); Zou and Hastie (2005). Full entries are in [References](references.md).
