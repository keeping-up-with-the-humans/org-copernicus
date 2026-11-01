# Correcting estimates

Often you do not need a corrected score for every text. You need a number: the mean of the construct, the difference between two groups, or a regression coefficient. The estimate-level estimators compute that number from the machine scores of all texts and the human values of a sample, with a standard error and a 95 percent interval. This tutorial runs each estimator on the toy corpus and checks the estimators with a repeated study.

## What you need

- The base install. No GPU, no API key.
- Machine scores for all texts, and a human sample drawn by whole groups with its inclusion probability: see [The human sample](human-sample.md). The toy corpus stands in for the human coder.

To code with a real model instead of the fake backend, replace the backend line of the code below. This is a snippet to read, not to run:

```python
backend = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
```

## Which estimator for which question

| Question | Function | Command-line `--kind` |
|----------|----------|-----------------------|
| What is the mean of the construct? | `mean_ppi` | `mean` |
| How much higher is the construct in one group of texts than in another? | `group_difference_ppi` | `difference` |
| How does the construct depend on covariates (the construct is the outcome)? | `ols_outcome_ppi` | `ols-outcome` |
| How does an outcome depend on the construct (the construct is a predictor)? | `ols_predictor_ppi`, `moment_subtraction`, `simex`, `oriv` | `ols-predictor`, `moment-subtraction`, `simex`, `oriv` |

The prediction-powered estimators (`*_ppi`) use the machine score of every text as a prediction and correct it with the mean gap between the human value and the prediction on the labelled texts. They need a human sample in which every text had the same inclusion probability, such as the group draw of [The human sample](human-sample.md). Unbiasedness is a property of an estimator over repeated draws of the human sample, not of one estimate, and over those draws the quality of the machine score does not bias the point: a poor score widens the interval, a good one narrows it. The point of a mean is exactly unbiased over repeated whole-group draws when every group of the corpus holds the same number of texts (groups of one text included). The point of a difference is exactly unbiased when, on each side of the split, every group of the corpus that has texts on that side holds the same number of them; groups of equal size that each lie wholly on one side are one such case, not the only one. Where some groups straddle the split and others lie on one side, this holds only provided no draw is refused, because the estimator refuses a draw with too few labelled texts or groups on a side, and keeping only the accepted draws shifts which groups are drawn. Otherwise, as with groups of unequal size or a split such as long against short texts that gives the groups different numbers of a side's texts, the means over the labelled texts are ratio estimators and the point is approximately unbiased, with a bias of the order of one over the number of drawn groups. Regression coefficients are approximately unbiased. `pseudo_values(data).mean_rectified`, with its default `identity` rectifier, is a mean that is exactly unbiased under any such draw; it is usually more variable and has no standard error. Power tuning estimates its coefficient on the labelled texts, so a tuned point is approximately unbiased as well. The estimators that treat the construct as a predictor (`moment_subtraction`, `simex`, `oriv`) correct the attenuation that measurement error causes in a regression coefficient.

## Steps

### Step 1: build the data with a human sample

<!-- run -->
```python
import numpy as np

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
    extra_columns=["length_words"],
)
human = table.frame.set_index("text_id")["gold_constructiveness"]
draw = cop.draw_human_sample(table, n=21, seed=3)
data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)
print(draw.m, draw.n_groups, data.n_labelled, round(draw.pi, 2))
```

<!-- output -->
```text
7 18 21 0.39
```

`extra_columns=["length_words"]` carries a column of the table into the data, as the outcome of a regression below. The draw takes 7 of 18 groups, 21 texts, each with the inclusion probability 0.39.

### Step 2: estimate a mean

<!-- run -->
```python
truth = human.mean()
estimate = cop.mean_ppi(data)
print(
    round(table.frame["gold_constructiveness"].mean(), 1),
    round(text_scores["construct_score"].mean(), 1),
)
print(estimate.method, round(estimate.point, 1), estimate.ci_low < truth < estimate.ci_high)
print(estimate.details["clustered"], estimate.details["n_clusters_labelled"])
```

<!-- output -->
```text
3.0 1.7
ppi_mean 3.0 True
True 7
```

The true mean over all 54 texts is 3.0. The mean of the raw machine scores is 1.7, far below it. The corrected estimate is 3.0, and its interval contains the truth. An `Estimate` holds `point`, `se`, `ci_low`, `ci_high`, the number of labelled and all texts, the method, and a `details` mapping that each estimator documents.

**The standard error is clustered by group.** The texts of a drawn group enter together, so their errors are correlated. With `cluster=None` (the default) the estimator sums the estimating functions within each group whenever a group holds more than one labelled text. Switching it off gives a smaller, wrong standard error:

<!-- run -->
```python
naive = cop.mean_ppi(data, cluster=False)
print(naive.se < estimate.se)
```

<!-- output -->
```text
True
```

Power tuning scales the machine terms by a coefficient `lambda_star` that is estimated on the labelled texts, and shrinks the interval when the machine score is informative.

<!-- run -->
```python
tuned = cop.mean_ppi(data, power_tuning=True)
print(tuned.se < estimate.se, round(tuned.details["lambda_star"], 1))
```

<!-- output -->
```text
True 2.1
```

### Step 3: estimate a difference

The difference compares two groups of texts. `long_short_split` marks the texts whose length (the covariate `log_chars`) lies above the median.

<!-- run -->
```python
long_texts = cop.corrections.long_short_split(data)
difference = cop.group_difference_ppi(data, long_texts)
true_difference = human[full.ids[long_texts]].mean() - human[full.ids[~long_texts]].mean()
print(round(difference.point, 1), round(true_difference, 1))
```

<!-- output -->
```text
1.6 1.7
```

### Step 4: estimate a regression

With the construct as the outcome, `ols_outcome_ppi` fits `construct ~ 1 + log_chars`. The reported coefficient is the first one after the intercept.

<!-- run -->
```python
outcome = cop.ols_outcome_ppi(data, formula="1 + log_chars")
print(outcome.details["columns"], round(outcome.point, 1))
```

<!-- output -->
```text
('Intercept', 'log_chars') 2.0
```

With the construct as a predictor, the machine score is an error-laden measure and a naive regression is attenuated, or distorted. The block regresses `length_words` on the construct. The true slope, from all gold values, is 10.6 words per point of the construct. The slope of the raw score is 25.9, because the raw score has a small range.

<!-- run -->
```python
outcome_values = full.frame["length_words"].to_numpy()
print(round(np.polyfit(human.loc[full.ids], outcome_values, 1)[0], 1))
print(round(np.polyfit(full.frame["construct_score"], outcome_values, 1)[0], 1))
for estimator in (cop.ols_predictor_ppi, cop.moment_subtraction, cop.simex):
    result = estimator(data, "length_words")
    print(result.method, round(result.point, 1))
```

<!-- output -->
```text
10.6
25.9
ppi_ols_predictor 10.3
moment_subtraction 12.4
simex 11.6
```

The three corrected slopes lie near the true 10.6 and far from the naive 25.9. Their standard errors differ; compare `result.se`. `ols_predictor_ppi` has the variants `ppi` and `ludwig`. `oriv` needs two independent measurements of the construct (`score_a` and `score_b`, for example two coders); the data here have one, so this is a snippet to read:

```python
result = cop.oriv(data, "length_words", score_a="coder_a_score", score_b="coder_b_score")
```

### Step 5: check the estimators with a repeated study

How do you know the correction works for your scores? `repeated_study` takes a corpus where all gold values are known, draws many human samples of several sizes, applies each estimator to each draw, and reports its bias and the coverage of its interval against the truth.

<!-- run -->
```python
study = cop.repeated_study(
    full,
    truth_gold="gold_constructiveness",
    sizes=(12, 24),
    draws=20,
    estimators=("raw", "calibrated", "ppi_identity"),
)
raw_rows = study[study["estimator"] == "raw"]
ppi_rows = study[study["estimator"] == "ppi_identity"]
print((raw_rows["coverage"] == 0).all(), (ppi_rows["bias"].abs() < 0.2).all())
```

<!-- output -->
```text
True True
```

The raw score's interval never covers the true mean, because the score is biased. The prediction-powered estimate is close to the truth on average. The study has two estimands, the `mean` and the `long_short_difference`, and a row for each estimator and size. Use at least a few hundred draws on real data; 20 keep the tutorial fast.

## On the command line

```bash
copernicus store export run.sqlite --what text --out text_scores.csv
copernicus corrections estimate --kind mean --scores text_scores.csv --texts toy_corpus.csv \
    --gold human_values.csv --gold-column gold_constructiveness --score-columns construct_score \
    --draw drawn.csv.json --group-column group_id --out estimate.json
```

The command prints the estimate and writes it, with its details, to a JSON file. The gold file holds `text_id` and the human value of each coded text. `--draw` names the draw file that `corrections sample` (or `write_draw`) wrote: it binds the coded sample to the texts file and gives `pi`, as [The human sample](human-sample.md#on-the-command-line) explains. Without a draw file, the prediction-powered kinds read the inclusion probability of each coded text from the column of the gold file that `--pi-column` names, and check only that the coded texts cover whole groups; moment subtraction, SIMEX and ORIV use no `pi` and check nothing of the draw. `--kind difference` takes `--long-short` or `--group-column-split`; `--kind ols-outcome` takes `--formula`; the predictor kinds take `--outcome` and optionally `--covariates`.

## What the output means

- **The point estimate is on the gold scale.** The corrected mean of 3.0 is a mean of the gold values, not of the machine scores.
- **The interval reflects the human sample.** With few labelled groups the interval is wide. It is wide on purpose: it says how much the sample can tell you.
- **A narrower interval needs a better score or more human values.** [Sample size](sample-size.md) quantifies the trade.
- **Compare estimators on the question, not on the number.** For a mean or a difference, use the prediction-powered estimators. For a coefficient of the construct as a predictor, compare `ols_predictor_ppi`, `moment_subtraction`, `simex` and, with two measurements, `oriv`.

## Pitfalls

- **The prediction-powered estimators need `pi`.** They refuse a labelled sample without an inclusion probability, or with unequal probabilities among the labelled texts.
- **The classical estimators (`moment_subtraction`, `simex`) resample texts, not groups.** Their bootstrap standard errors ignore the clustering, so they are likely too small when groups are large.
- **Few labelled groups.** Clustering needs at least two labelled groups, and some estimators need more labelled texts than that. The error message names the shortfall.
- **A clustered difference needs two labelled groups on each side.** Whenever the standard error is clustered, by `cluster=True` or by the automatic choice, `group_difference_ppi` refuses a difference whose labelled texts on one side all lie in one group, because that side would add none of its variation between groups to the standard error. Draw more groups, or pass `cluster=False` (`--no-cluster` on the command line), whose independent standard error ignores the correlation of the texts within a group.
- **The clustered standard error carries a small-sample factor.** The clustered variance of each term is the sum of its squared group sums, divided by the squared number of its texts and multiplied by G/(G − 1), with G the number of groups among those texts. The factor is 2 with two groups and 1.25 with five.
- **The intervals grow conservative as the labelled share grows.** The standard error counts a variance for the prediction's mean over all texts, although, for a prediction fixed before the draw, that mean does not vary over draws from a fixed corpus (a learned `prediction=` does vary), and it applies no finite-population correction. Against the corpus's own full-gold value the intervals are therefore conservative, and more so as the labelled share grows. With a small labelled share this changes little. With about half of the texts labelled, a nominal 95 percent interval covers the truth in about 98 to 99.5 percent of draws.
- **A weak or reversed score.** If the machine score does not track the gold, the prediction-powered point keeps the unbiasedness described above, but its interval is no narrower than that of the human values alone, and without power tuning it is wider. With `power_tuning=True`, `lambda_star` comes out near 0 for a useless score and negative for a score that falls as the gold rises.
- **Do not reuse the sample for model choice and for the estimate.** If you choose among corrections on the same labelled texts, the interval is too optimistic. Use the battery on a separate corpus for the choice.

## Next

[Sample size](sample-size.md) tells you how many texts to code.
