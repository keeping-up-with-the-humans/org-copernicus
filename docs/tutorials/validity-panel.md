# The validity panel

The validity panel compares a machine score with a gold value. It answers two questions. How well does the score rank the texts? And how does the score go wrong: by a shift, by a compressed spread, by a link to text length, or by noise? This tutorial builds a panel for the toy corpus, reads the report, and shows the settings that change the answer.

## What you need

- The base install. No GPU, no API key, no encoder.
- Scores: the long table of `export_scores`, or the frame of the similarity coder or the hybrid coder.
- Gold values: a frame with a `text_id` column and one numeric column per gold variable. The toy corpus carries them in its `gold_*` columns.
- A text table, which supplies the groups for the bootstrap and the text length for the covariate. Without a group column every text is its own group: the library logs a warning, the bootstrap resamples single texts, and the report says `groups: one per text (no group column)`.

The tutorial codes the toy corpus with the fake backend, so the scores are not those of a real model.

To code with a real model instead of the fake backend, replace the backend line of the code below. This is a snippet to read, not to run:

```python
backend = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
```

## Steps

### Step 1: code and build the panel

<!-- run -->
```python
import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
scores = cop.export_scores("run.sqlite")

targets = {
    "construct": "gold_constructiveness",
    "specificity": "gold_specificity",
    "tone": "gold_tone",
    "reasoning": "gold_reasoning",
}
settings = cop.PanelSettings(n_boot=200)
panel = cop.validity_panel(scores, table.frame, table, targets=targets, settings=settings)
print(panel.report())
```

The `targets` mapping names what to score and what to compare it with. A key is an attribute id, or `construct` for the mean of the attribute scores per text and run. A score frame whose attribute ids include `construct`, or an id ending in `__rollup`, is refused, since the naming rule of rubrics reserves those names for the targets the library generates. The value is a gold column. The report prints one row for each target. With the toy corpus you should see rank validities between about 0.6 and 0.9 and noise of 0.00, because the fake backend gives the same score on every run.

### Step 2: read the frame

`panel.frame` has one row per target and many columns. The report picks the main ones.

<!-- run -->
```python
columns = ["target", "n_used", "rank_validity", "rank_validity_low", "rank_validity_high"]
print(panel.frame[columns].round(1).to_string(index=False))
print(panel.n_used, panel.n_dropped_missing)
```

<!-- output -->
```text
target n_used rank_validity rank_validity_low rank_validity_high
construct 54 0.9 0.8 0.9
specificity 54 0.9 0.9 1.0
tone 54 0.6 0.4 0.8
reasoning 54 0.9 0.8 0.9
54 0
```

A text with a missing score, gold value or covariate is dropped from every metric of that target and counted in `n_dropped_missing`. Nothing is imputed.

### Step 3: set the direction, the covariate and the intervals

<!-- run -->
```python
flipped = scores.assign(score=6 - scores["score"].astype(float))
auto = cop.validity_panel(
    flipped,
    table.frame,
    table,
    targets={"construct": "gold_constructiveness"},
    settings=cop.PanelSettings(n_boot=200, direction="auto"),
)
print(auto.frame[["target", "direction_applied"]].to_string(index=False))
```

<!-- output -->
```text
target direction_applied
construct negative
```

`direction="positive"` (the default) uses the score as given. `"negative"` negates it before every metric. `"auto"` negates when the signed Spearman correlation is negative, as for a score where low means good. The covariate for the covariate-linked error is `log(n_chars)` of the table; `PanelSettings(covariate="length_words", covariate_scale="raw")` uses a column of the gold frame or the table, per unit instead of per standard deviation. `n_boot` and `seed` set the bootstrap.

### Step 4: add a reliability ceiling

Gold values are themselves measured with error, so no score can correlate perfectly with them. If you know the reliability of the gold, the panel reports the ceiling `sqrt(reliability)` on the rank validity and the share of the ceiling that the score reaches.

<!-- run -->
```python
with_ceiling = cop.validity_panel(
    scores,
    table.frame,
    table,
    targets={"construct": "gold_constructiveness"},
    settings=cop.PanelSettings(n_boot=200, reliability=0.85),
)
print(with_ceiling.frame[["target", "ceiling", "share_of_ceiling"]].round(2).to_string(index=False))
```

<!-- output -->
```text
target ceiling share_of_ceiling
construct 0.92 0.97
```

A number applies to every target, which is right only when every gold column comes from one instrument. A mapping from gold column to reliability, as `{"gold_tone": 0.8}`, sets the ceiling of those columns only. If you have the ratings of several human coders, pass them as `items` (a table with one row per rated unit and one column per coder): the panel computes Cronbach's alpha and uses it for the construct row.

### Step 5: read a failed row

A score that does not vary cannot be ranked. The panel keeps its row and says so.

<!-- run -->
```python
flat = scores.assign(score=3)
failed = cop.validity_panel(
    flat, table.frame, table, targets={"construct": "gold_constructiveness"}, settings=settings
)
print(failed.frame[["target", "status", "calibration_shortfall"]].round(2).to_string(index=False))
```

<!-- output -->
```text
target status calibration_shortfall
construct constant_score -1.0
```

The other statuses are `too_few` (fewer than three finite pairs) and `constant_gold`. For a status other than `ok`, the rank validity, the intervals and the 0-to-1 errors are missing. A constant score still has two errors that the panel reports: a calibration shortfall of -1 and a covariate-linked error of minus the slope of gold on the covariate.

### Panels of the similarity and hybrid coders

The similarity and hybrid frames have a `target_id` column instead of `attribute_id`. Pass them the same way, with the target keys `construct`, an attribute id, or `<rubric_id>__rollup`. Filter a grid of settings to one setting before the call, because the panel needs one score per text and target.

## On the command line

```bash
copernicus validity panel --store run.sqlite --texts toy_corpus.csv --gold toy_corpus.csv \
    --group-column group_id --target construct=gold_constructiveness --target tone=gold_tone \
    --n-boot 200 --out panel.csv --report panel.txt
```

Use `--scores similarity.csv` instead of `--store` for a similarity or hybrid table.

## What the output means

**Rank validity** is the absolute Spearman correlation of the score with the gold value. It asks only whether the score orders the texts as the gold does. The bracket is a 95 percent percentile interval from a bootstrap that resamples whole groups, so texts of one group are never split. Because rank validity is an absolute value, the bracket is bounded below by 0 and does not test zero validity: a score unrelated to the gold still gets a bracket above 0, and the report says so in a note under the table. To check for zero validity, read the interval of the signed Pearson correlation in `panel.frame` (`pearson_low`, `pearson_high`), which can contain 0.

The four score errors describe how a score fails beyond ranking. Each one is computed after scaling the score and the gold value to the unit interval by their observed ranges. The report shows the point estimates. Because the score and the gold value are each scaled to their own range, a pure shift or stretch of the gold shows zero error on all four measures; the [methods page](../methods/validity-panel.md) explains what the errors then describe.

| Error | Column | Meaning |
|-------|--------|---------|
| Level displacement | `level_displacement` | The mean of score minus gold. Negative: the score sits below the gold on average. |
| Calibration shortfall | `calibration_shortfall` | The covariance of score and gold over the variance of gold, minus 1. Negative: the score is compressed, so its spread is smaller than the gold's. Zero: score and gold rise together one for one. |
| Covariate-linked error | `covariate_linked_error` | The slope of score minus gold on the covariate. Positive: the score rises relative to gold as the covariate (text length) rises. |
| Noise | `noise` | The standard deviation of a text's score across runs, averaged over texts, on the 0-to-1 scale. It needs at least two runs. |

`mae_01` is the mean absolute difference of the scaled score and gold. `share of ceiling` is the rank validity divided by the ceiling.

## Pitfalls

- **The 0-to-1 scaling hides some shifts.** Each variable is scaled by its own observed range, so a score that is an exact linear function of the gold shows no level displacement and no calibration shortfall, even if it lives on another scale. The errors of the panel describe the shape of the relation, not the scale. To see the score on the gold scale, use the correction tools; the native-scale error of [the correction battery](correcting-scores.md) does.
- **One run gives no noise.** The noise is missing with a single run, and it is zero when every run agrees.
- **A reliability number is a claim about your gold.** Apply a single number to every target only when all gold columns come from one instrument. A wrong reliability moves the ceiling and the share of it.
- **The gold must be numeric.** The panel refuses text and other types.
- **Do not compare panels with different text sets.** Each target uses the texts that have a score, a gold value and a covariate. `n_used` tells you which.
- **Rank validity is a magnitude.** The sign is in `spearman`. A score with the wrong sign has the same rank validity as one with the right sign; use `direction` deliberately. Under the default direction the report adds a warning line for every target whose signed Spearman correlation is negative.

## Next

[The human sample](human-sample.md) draws the texts a human should code, and the correction tutorials use that sample to correct the scores.
