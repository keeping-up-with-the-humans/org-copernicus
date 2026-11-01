# The NLP-ML baseline

The NLP-ML baseline answers a plain question: how well does a simple supervised model rank your texts when it learns from gold values? It fits a ridge regression on the embeddings of the texts and tests it on held-out groups. Its rank validity at each training size is the reference point against which you read the LLM coders. This tutorial runs it on the toy corpus and explains its sizes, its floor and its skips.

## What you need

- The `embed` extra: `pip install ".[embed]"` in the package's root folder (see [Installation](../installation.md)).
- **No GPU and no API key.** MiniLM runs on the CPU; the baseline itself is a ridge regression.
- Embedded texts, and a numeric gold value for every text you analyse. The baseline does not drop texts with a missing gold value; it needs all of them.

To code with a real model instead of the fake backend, replace the backend line of the code below. This is a snippet to read, not to run:

```python
backend = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
```

## How the baseline works

For each seed the groups are split into grouped folds, so that no group lies on both sides of a split. For each requested training size, the training part of a fold is cut down to a draw of whole groups holding that many **analysis units**. The ridge penalty is chosen by exact grouped leave-one-group-out tuning on the draw, and the held-out fold is scored on **unit means**.

The analysis unit is the group by default. A text-level gold value is averaged within its unit. Sizes count units, not texts. Pass `unit_id="text_id"` to make every text a unit.

## Steps

### Step 1: embed the texts

<!-- run embed -->
```python
import copernicus as cop

table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
encoder = cop.Encoder("minilm", device="cpu")
texts = encoder.embed(table.frame["text"].tolist(), table.frame["text_id"].tolist())
print(len(table), table.frame["group_id"].nunique())
```

<!-- output -->
```text
54 18
```

### Step 2: run the baseline

The toy corpus has 18 groups, so the default sizes (50 units and up) and the default floor of 50 units do not fit. The block asks for sizes of 4, 8 and 12 units and for the full training pool, and lowers the floor to 4 units.

<!-- run embed -->
```python
settings = cop.BaselineSettings(sizes=(4, 8, 12, "full"), seeds=(42, 43), k_folds=3, floor_units=4)
result = cop.run_ridge_baseline(
    {"minilm": texts}, table.frame, table, settings, gold_column="gold_constructiveness"
)
print(result.unit_id, result.binary)
print(result.results[["variant", "seed", "size", "size_units"]].to_string(index=False))
```

<!-- output -->
```text
group_id False
variant seed size size_units
embeddings_only 42 4 4
embeddings_only 43 4 4
embeddings_only 42 8 8
embeddings_only 43 8 8
embeddings_only 42 full <NA>
embeddings_only 43 full <NA>
```

`run_ridge_baseline` takes the embedding sets by encoder name, a frame with `text_id` and the gold column (the table's own frame works), the text table, the settings, and the gold column's name. The result holds four tables.

| Table | Content |
|-------|---------|
| `results` | One row per encoder set, variant, seed and size: the fold metrics weighted by units (`spearman`, `spearman_abs`, `pearson`, `mae_01`, `mae_native`, and `auc` and `somers_d` for binary gold), the median penalty, the folds with a penalty on the edge of the grid, and the size of the draws |
| `oof_predictions` | The out-of-fold prediction of every text at size `"full"`, with the penalty used in its fold |
| `skipped` | The sizes that were not fitted, with the reason |
| `curves` (through `cop.curves(result)`) | Rank validity by training size, averaged over seeds |

### Step 3: read the skips

The size of 12 units was not fitted. The `skipped` table says why.

<!-- run embed -->
```python
print(result.skipped[["seed", "size", "reason"]].to_string(index=False))
```

<!-- output -->
```text
seed size reason
42 12 quota_exceeds_half_units(quota=12,units=18)
43 12 quota_exceeds_half_units(quota=12,units=18)
```

A size is skipped, never fitted on less than it names. There are three reasons:

- `quota_exceeds_half_units`: the size is more than half of the units in the cohort.
- `pool_below_quota`: some fold's training pool holds fewer units than the size.
- `below_floor`: some fold's draw holds fewer units than `floor_units`. The floor also applies to the size `"full"`.

### Step 4: read the curves

<!-- run embed -->
```python
curves = cop.curves(result)
print(
    curves[["size", "size_units", "rank_validity", "spearman", "n_seeds"]]
    .round(1)
    .to_string(index=False)
)
```

`rank_validity` is the mean over seeds of the unit-weighted absolute Spearman correlation of the held-out fold. `spearman` is the signed one. Expect about 0.8 to 0.9 on the toy corpus, rising with the training size.

### Step 5: add a coder score as a feature

If you pass the score of a coder, the baseline also fits the variant `embeddings_plus_score`: the score is standardised on the draw and appended to the embeddings as one more penalised column. The comparison with `embeddings_only` shows what the coder adds beyond the embeddings.

<!-- run embed -->
```python
backend = cop.FakeBackend(
    cop.cue_responder(
        cop.load_rubric(cop.rubric.loader.toy_rubric_path()), cop.assets.load_toy_cues()
    )
)
rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=1))
coder_score = cop.export_text_scores("run.sqlite").set_index("text_id")["construct_score"]
both = cop.run_ridge_baseline(
    {"minilm": texts},
    table.frame,
    table,
    settings,
    gold_column="gold_constructiveness",
    coder_score=coder_score,
)
print(sorted(both.results["variant"].unique()))
```

<!-- output -->
```text
['embeddings_only', 'embeddings_plus_score']
```

## On the command line

```bash
copernicus embed texts --texts toy_corpus.csv --encoder minilm --out texts_minilm --group-column group_id --device cpu
copernicus baseline ridge --embeddings minilm=texts_minilm --texts toy_corpus.csv --gold toy_corpus.csv \
    --gold-column gold_constructiveness --group-column group_id --sizes 4,8,full \
    --floor-units 4 --k-folds 3 --out baseline.csv
```

The command writes `baseline.csv` and, beside it, `baseline_oof.csv`, `baseline_skipped.csv` and `baseline_curves.csv`, with a description in `baseline.csv.json`. In `--embeddings NAME=STEM`, `NAME` must be the encoder recorded in the set. Add `--coder-scores` for the plus-score variant.

## What the output means

- **The baseline is a reference point, not a competitor to beat in every case.** An LLM coder that does not reach the baseline's rank validity at the size of gold data you have is not adding much; one that does, with no training data, is.
- **`rank_validity` is the absolute Spearman correlation.** A fold whose out-of-fold correlation is negative still adds to it, so an encoder without signal plots above zero. Compare `spearman` with `rank_validity`: if `spearman` is well below, some folds are anti-correlated.
- **`mae_01` and `mae_native` measure error on two scales.** `mae_01` uses the training draw's gold range for prediction and gold. The validity panel scales each variable by its own observed range instead, so the two numbers are not comparable.
- **`n_edge_folds` counts the folds whose chosen penalty sits on an edge of the grid.** Many such folds suggest that the grid (`alphas` in the settings) does not cover the best penalty.

## Pitfalls

- **The default floor is 50 units.** A small corpus needs `floor_units` and smaller `sizes`, as above. Otherwise every size is skipped and the result is empty.
- **The gold must be complete.** A text with a missing gold value is an error, not a dropped row.
- **Units must nest in groups.** All texts of a unit belong to one group, because the folds split groups.
- **Folds and draws are keyed to the seed and to the name of the gold column.** Renaming the gold column changes them. Run several seeds and read the spread (`rank_validity_sd`).
- **The sizes count units.** If your units are groups of several texts, a size of 100 means 100 groups.

## Next

[The validity panel](validity-panel.md) compares any score, including the baseline's out-of-fold predictions, with the gold values.
