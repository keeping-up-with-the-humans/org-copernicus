# The human sample

The corrections of this library need a sample of texts that a human has coded. The sample must be drawn in a way the library can account for. This tutorial draws a sample of whole groups, hands the drawn texts out, takes the human values back, and attaches them to the machine scores. The corrections that use the sample are in the tutorials that follow.

## What you need

- The base install. No GPU, no API key, no encoder.
- A text table with a `group_id`: the persons, conversations or documents whose texts must not be split. Without a group column every text is its own group: the library logs a warning, the draw takes texts independently (its log line says `groups: one per text (no group column)`), and the standard errors of the estimators are not clustered.
- Machine scores for all texts, from any coder. The tutorial uses the fake backend.
- In a real study, a human coder. In the tutorial the gold columns of the toy corpus stand in for the human values.

To code with a real model instead of the fake backend, replace the backend line of the code below. This is a snippet to read, not to run:

```python
backend = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
```

## How the draw works

The draw takes **whole groups** by simple random sampling without replacement. If the corpus has `G` groups and the draw takes `m` of them, every text of a drawn group has the **inclusion probability** `pi = m / G`. The estimators divide by that probability, so a sample without it cannot be used by them. `draw_human_sample` supplies it.

You ask for a quota: `n` analysis units, or a `share` of them. The number of groups is `m = min(G, ceil(quota * G / n_units))`, so it depends only on the structure of your corpus, never on gold values. An analysis unit is a text by default; pass `unit_id` to name a column that defines the unit.

## Steps

### Step 1: draw

<!-- run -->
```python
import numpy as np

import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
draw = cop.draw_human_sample(table, n=10, seed=11)
print(draw.quota, draw.m, draw.n_groups, len(draw.ids), round(draw.pi, 3))
```

<!-- output -->
```text
10 4 18 12 0.222
```

The toy corpus has 18 groups of three texts. A quota of 10 texts needs `ceil(10 * 18 / 54) = 4` groups, which hold 12 texts, so the draw returns 12 texts with `pi = 4 / 18`. The draw rounds the number of groups up, never down. Ask for a share instead and the same rule applies:

<!-- run -->
```python
share = cop.draw_human_sample(table, share=0.25, seed=11)
print(share.quota, share.m, len(share.ids), round(share.pi, 3))
```

<!-- output -->
```text
14 5 15 0.278
```

A share of 0.25 of 54 texts is a quota of 14; that needs 5 groups, so 15 texts.

### Step 2: check the rules

<!-- run -->
```python
for options in ({"n": 0}, {"n": 55}, {"n": 5, "share": 0.1}):
    try:
        cop.draw_human_sample(table, seed=11, **options)
    except cop.errors.CopernicusError as error:
        print(type(error).__name__, "-", error)
```

<!-- output -->
```text
ConfigurationError - the quota of 0 units must lie between 1 and the 54 units of the pool
ConfigurationError - the quota of 55 units must lie between 1 and the 54 units of the pool
SettingsError - give exactly one of n and share
```

The quota must lie between 1 and the number of units, and you give exactly one of `n` and `share`. A quota equal to the number of units is the full draw: every group, `pi = 1`. A quota above half of the units, short of the full draw, is allowed but logged as a warning: the correction battery refuses such sizes, and the correction has little left to work on.

The draw is a function of the table, the quota and the seed:

<!-- run -->
```python
again = cop.draw_human_sample(table, n=10, seed=11)
other = cop.draw_human_sample(table, n=10, seed=12)
print(np.array_equal(draw.ids, again.ids), np.array_equal(draw.ids, other.ids))
```

<!-- output -->
```text
True False
```

Record the seed with your results. Draw before you look at any gold value, and do not redraw until the sample looks good: a redraw chosen for its look is not a random sample.

### Step 3: hand out the texts, take the values back

Give the coders the drawn texts, with the group and `pi` beside them if you want to keep the probability with the file.

<!-- run -->
```python
batch = table.subset(draw.ids).frame[["text_id", "group_id", "text"]].assign(pi=draw.pi)
print(batch.shape, sorted(batch["group_id"].unique()))
```

<!-- output -->
```text
(12, 4) ['w03', 'w09', 'w14', 'w16']
```

Coding takes time, so keep the draw in a file until the values come back. `write_draw` writes it as the draw file of the command line does, and `read_draw` reads it back field for field, refusing a file whose fields contradict each other:

<!-- run -->
```python
cop.corrections.write_draw(draw, "draw.json")
kept = cop.corrections.read_draw("draw.json")
print(np.array_equal(kept.ids, draw.ids), kept.population_hash == draw.population_hash)
```

<!-- output -->
```text
True True
```

The human codes every drawn text. Here the gold column of the toy corpus plays the human:

<!-- run -->
```python
human = table.frame.set_index("text_id")["gold_constructiveness"].loc[draw.ids]
print(human.shape, human.index.is_unique)
```

<!-- output -->
```text
(12,) True
```

### Step 4: attach the values to the scores

Code the texts with a coder, take the text scores, and join them with the table and the gold frame into `CorrectionData`. Then attach the draw.

<!-- run -->
```python
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
text_scores = cop.export_text_scores("run.sqlite")
data = cop.CorrectionData.from_scores(
    text_scores,
    table,
    table.frame,
    gold_column="gold_constructiveness",
    score_columns=["construct_score"],
)
print(data.n, data.n_labelled)
data = cop.corrections.attach_draw(data, human, draw)
print(data.n, data.n_labelled, round(float(data.pi[data.labelled_mask][0]), 3))
```

<!-- output -->
```text
54 54
54 12 0.222
```

`from_scores` joined the toy corpus's own gold column here, so all 54 texts were labelled to start with. `attach_draw` replaces the gold values by the human values of the draw: 12 texts keep a value, the others are missing, and the column `pi` holds the inclusion probability. In a real study the gold frame holds only the human-coded texts, and `from_scores` leaves the other texts unlabelled from the start.

`attach_draw` refuses a draw that does not match the data. The draw binds to the exact table it was made on: it records a fingerprint of the table's texts and their groups, `draw.population_hash`, and the correction data must hold the same texts in the same groups. A text added, removed or moved to another group after the draw makes it a draw of another table, and so do renamed groups, even when they only relabel the same partition, because `pi = m / G` and the clustering by group describe the table that was drawn. So draw after the table is final, from the table the correction data are built from: drop the texts you will not score before you draw, and score every text that remains. `attach_draw` also refuses a drawn text without a finite human value and a gold series that repeats an id.

<!-- run -->
```python
incomplete = human.iloc[:-1]
renamed = cop.CorrectionData(
    frame=data.frame.assign(group_id="person " + data.frame["group_id"]),
    score_columns=data.score_columns,
    gold_column=data.gold_column,
)
for target, values in ((data, incomplete), (renamed, human)):
    try:
        cop.corrections.attach_draw(target, values, draw)
    except cop.errors.ConfigurationError as error:
        print(error)
```

<!-- output -->
```text
no finite human value for drawn texts 't048'
the draw was made on a different table than the data (different texts or groups): draw on the table the correction data are built from, then attach that draw; the draw records n_units = 54 and n_groups = 18, the data hold 54 texts in 18 groups
```

The second message gives the counts of both tables. Here they agree, because only the names of the groups changed; a text added or removed shows in the counts, and a drawn text that the data lack is named.

## On the command line

```bash
copernicus corrections sample --texts toy_corpus.csv --group-column group_id --n 10 --seed 11 --out drawn.csv
copernicus corrections estimate --kind mean --scores text_scores.csv --texts toy_corpus.csv \
    --group-column group_id --gold drawn.csv --gold-column gold_constructiveness \
    --score-columns construct_score --draw drawn.csv.json --out estimate.json
```

The file `drawn.csv` holds the drawn texts with their group and `pi`. `drawn.csv.json` is the draw file: the drawn ids and groups, `pi` and the fingerprint of the table, `population_hash`. Add the human values to `drawn.csv`, pass it as `--gold`, and pass the draw file as `--draw`.

The draw file is what binds the coded sample to the texts file. With `--draw`, `corrections estimate` makes the checks of `attach_draw`: it refuses a texts file whose texts or groups changed after the draw, with the message above, and a drawn text without a human value, and it takes `pi` from the draw; a `pi` column in the gold file must agree with it. Pass `--draw` whenever the sample came from `corrections sample`. Without it, the prediction-powered kinds (`mean`, `difference`, `ols-outcome` and `ols-predictor`) take `pi` from `--pi-column` and check only that the coded texts cover whole groups, so a texts file regrouped after the draw goes unnoticed; moment subtraction, SIMEX and ORIV use no `pi` and check nothing of the draw.

## What the output means

- **`pi` is not a weight you may drop.** The design-based estimators use it. A human-coded sample without `pi` is refused by the prediction-powered estimators.
- **Every text of a drawn group is coded.** The draw takes whole groups; a group you code partly breaks the design and the clustered standard errors.
- **`m` can give more texts than you asked for.** The quota is met by rounding the groups up.

## Pitfalls

- **Draw from the same corpus that you score.** The correction learns how the machine score relates to the human value in the population you drew from. A sample from another corpus, or a convenience sample, teaches it the wrong relation. See the [FAQ](../faq.md#why-must-the-human-sample-come-from-the-same-corpus).
- **Do not pick the texts yourself.** Hand-picked "typical" texts bias the correction.
- **Do not draw by texts when the texts of a person correlate.** Draw by groups, and name the person or document as the group.
- **Do not swap or redraw texts after the draw.** A redraw changes the inclusion probabilities, so a text that a coder cannot code stays in the draw and is recorded as uncodable, with the reason, not replaced by another draw. `attach_draw` refuses a drawn text without a value, so give it the value that your coding rules set for a text that shows nothing of the construct (the two-stage coder's rule is the lowest level), and report how many drawn texts were uncodable.
- **A small number of groups limits the estimate.** The standard errors are clustered by group. With few labelled groups, the estimates are noisy and some estimators refuse to run.

## Next

[Correcting scores](correcting-scores.md) uses the sample to repair the machine scores text by text. [Correcting estimates](correcting-estimates.md) estimates a mean, a difference or a coefficient directly. [Sample size](sample-size.md) tells you how many texts to draw.
