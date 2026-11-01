# NLP-ML baseline

The NLP-ML baseline is the yardstick for the coders. It uses no rubric and no prompt: an encoder turns each text into an embedding, and a ridge regression, a linear regression whose coefficients are shrunk toward zero so that it fits less noise, learns the human value from the embedding on a sample of human-coded texts. Ridge regression is one member of the family of penalised linear models that also holds the lasso and the elastic net (Zou and Hastie 2005).

A detection method that needs no human-coded texts should be judged against what a model trained on such texts reaches, and supervised models trained on human-coded texts are an established way of measuring constructs in management research (Choudhury et al. 2019, Miric et al. 2023). Their output is a prediction, with the strengths and limits of prediction rather than of estimation (Mullainathan and Spiess 2017). The baseline reports how well the predictions rank held-out texts at each size of the training sample, so a study can see what its human-coded texts would buy if it trained a model on them. A second variant adds a coder's score to the embeddings and shows whether the score carries information beyond them.

## The algorithm as the library implements it

1. **Folds.** The groups (the persons or documents whose texts must stay together) are split at random into `k_folds` (5) outer folds; a group never straddles a split.
2. **Training sizes.** For each seed and each size, each fold's training pool is cut down to a draw of whole groups, added in random order until the draw holds the size's number of analysis units (it can overshoot by less than one group); `"full"` takes the whole pool. The default sizes are 50, 100, 150, 200, 250, 500 and 1,000 units and `"full"`. A size is skipped for the seed, and recorded with its reason, when its quota exceeds half the units (`quota_exceeds_half_units`), when some fold's pool holds fewer units than the quota (`pool_below_quota`), or when some draw holds fewer units than the floor of `floor_units` (50) units (`below_floor`), the full pool included.
3. **Features.** The embedding columns are z-scored on the training draw (a constant column keeps standard deviation 1). Several encoders can be stacked; each block is then weighted by `sqrt(mean_dim / dim_block)` so that every encoder contributes the same total variance.
4. **Ridge and penalty.** A ridge regression with an unpenalised intercept is fitted on the draw. The penalty comes from `ALPHAS`, 41 values from 0.01 to 10^8 evenly spaced on the log scale, and is the one with the smallest leave-one-group-out error inside the draw. That error is exact, computed from the hat matrix of a single decomposition of the design, not approximated by refits. Whether the chosen penalty sits on an edge of the grid is recorded.
5. **The score variant.** With a coder score, `embeddings_plus_score` standardises the score on the draw and appends it as one more penalised column, through an exact rank-one update of the same decomposition.
6. **Metrics.** The test fold's predictions and gold values are averaged within each analysis unit, and the metrics are computed on these unit means: Spearman and absolute Spearman correlation, Pearson correlation, the mean absolute error on the 0 to 1 scale of the training draw's gold range and on the gold's own scale, and for binary gold the area under the ROC curve and Somers' D (`2 * AUC - 1`). Fold values are weighted by the fold's units. `curves` reports the rank validity, the mean over seeds of the weighted absolute Spearman correlation, by training size.

The defaults are the paper's settings: 5 folds, the seeds 42, 43 and 44, the sizes above and the penalty grid above.

## Assumptions and what it corrects

The baseline produces predictions; it corrects nothing. It predicts the human-coded texts only, each out of fold, so it is a benchmark, not a score for the rest of a corpus; to score a corpus from its human-coded texts, cross-fit the `embeddings_only` variant of [embedding-augmented calibration](corrections/embedding-augmented-calibration.md). It assumes that the human-coded texts are a sample of the corpus they stand for, and that the analysis unit, the level at which the gold value is defined, is named correctly: a person, a conversation or a text (by default the group). Its out-of-fold predictions are a score like any other and carry bias into an estimate unless they are corrected. The absolute Spearman correlation inflates weak predictors, because a fold with a negative correlation still adds to it; compare it with the signed `spearman` column.

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

- A binary gold may use any two values, such as 1 and 2; the paper's analysis code required 0 and 1. A binary gold that varies within an analysis unit is refused.
- The floor applies to the size `"full"` too, as in the paper's analysis code. A size whose quota some fold's training pool cannot reach is skipped with the reason `pool_below_quota`; the paper's analysis code fitted such a point on a smaller sample under the size's label.
- The `size` column is a string label (`"50"`, `"full"`) with the integer quota in a separate `size_units` column.
- In the fold metrics, as in every association, slope and min-max scaling of the library, a set of values counts as constant when its range is at most `1e-12` times the larger of 1 and its largest absolute value; the paper's analysis code tested for an exact zero range.

## Literature

Choudhury et al. (2019); Miric et al. (2023); Mullainathan and Spiess (2017); Zou and Hastie (2005). Full entries are in [References](references.md).
