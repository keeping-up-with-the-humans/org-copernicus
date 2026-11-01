# Sample size

How many texts must a human code? The answer depends on how well the machine score tracks the human value, and on how many texts you score by machine. This tutorial reads the planner of the library, which turns a correlation and two sample sizes into an effective-size factor, and shows how to find the smallest human sample that reaches a target.

## What you need

- The base install. No GPU, no API key. No model is called anywhere in this tutorial.
- A guess of the correlation `rho` between the machine score and the human value. A pilot gives it: code a small human sample, then read the Pearson correlation from the [validity panel](validity-panel.md).

## What the planner computes

For a correction that uses `n_machine` machine-scored texts and `n_human` human-coded texts, the planner returns three numbers.

| Number | Formula | Meaning |
|--------|---------|---------|
| `factor` | `1 / ((1 - rho^2) + rho^2 * n_human / n_machine)` | The human sample counts as `factor * n_human` human-coded texts |
| `limit` | `1 / (1 - rho^2)` | The factor when the number of machine-scored texts grows without bound |
| `uncorrected_equivalent` | `1 / rho^2` | A companion number, defined in the API reference of `SizePlan` |

With `rho = 0.8`, 20 human-coded texts and 500 machine-scored texts, the factor is 2.6. The 20 coded texts then give an estimate about as precise as 52 human-coded texts would give without a machine score. A better score raises the factor; a larger machine-scored corpus pushes it toward the limit, 2.8 for `rho = 0.8`.

**The factor assumes an estimator that uses the score at its best.** It is the gain of a prediction-powered mean whose prediction enters with the weight `cov(gold, score) / var(score)`: the power-tuned `mean_ppi(data, power_tuning=True)`, or a score put on the gold scale first by the `calibrated` rectifier of `pseudo_values`. The default `mean_ppi(data)` adds the raw score with the weight 1. It reaches the factor only when the score already lies on the gold scale with that slope; a score with a smaller or larger spread than the gold is worth fewer human texts. In one simulation with a score of correlation 0.91 and less than half the gold's spread, 100 human texts and 5,000 machine-scored ones, the plan promised an effective size of 539 texts, the default estimator gave 296 and the power-tuned one 619. So estimate a study sized with this page with power tuning, as Step 5 does.

## Steps

### Step 1: read a plan

<!-- run -->
```python
import copernicus as cop

plan = cop.plan_sample_size(0.8, 20, 500)
print(round(plan.factor, 1), round(plan.limit, 1), round(plan.n_human * plan.factor))
```

<!-- output -->
```text
2.6 2.8 52
```

### Step 2: compare several designs

`plan_table` returns one row for each pair of human and machine sample sizes.

<!-- run -->
```python
table_of_plans = cop.corrections.plan_table(0.8, [20, 50, 100], [500, 5000])
print(table_of_plans[["n_human", "n_machine", "factor"]].round(1).to_string(index=False))
```

<!-- output -->
```text
n_human n_machine factor
20 500 2.6
20 5000 2.8
50 500 2.4
50 5000 2.7
100 500 2.0
100 5000 2.7
```

The factor falls as the human sample grows, because a larger human sample is a larger share of the machine-scored texts. It rises with the machine-scored corpus. Even with 5,000 machine-scored texts it never passes the limit of 2.8.

### Step 3: find the smallest human sample for a target

Suppose you want an estimate as precise as 100 human-coded texts would give. Find the smallest `n_human` whose effective size reaches that.

<!-- run -->
```python
def smallest_sample(rho: float, target: float, n_machine: int) -> int:
    for n_human in range(2, n_machine + 1):
        plan = cop.plan_sample_size(rho, n_human, n_machine)
        if n_human * plan.factor >= target:
            return n_human
    raise ValueError("the target cannot be reached with this corpus")


for rho in (0.5, 0.7, 0.8, 0.9):
    print(rho, smallest_sample(rho, target=100, n_machine=5000))
```

<!-- output -->
```text
0.5 76
0.7 52
0.8 37
0.9 20
```

A score with a correlation of 0.9 needs 20 human-coded texts for the precision of 100. A score with 0.5 needs 76. Improving the score pays: from 0.5 to 0.8 halves the human sample.

### Step 4: turn texts into groups

A human sample is drawn by whole groups, so the number of texts follows from the number of groups. Ask for a quota and see what the draw takes. The toy corpus has 18 groups of three texts.

<!-- run -->
```python
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
for quota in (10, 12, 20):
    draw = cop.draw_human_sample(table, n=quota, seed=1)
    print(quota, draw.m, len(draw.ids))
```

<!-- output -->
```text
10 4 12
12 4 12
20 7 21
```

The quota is met by rounding the number of groups up, so you may code more texts than you asked for. Budget for the texts of the drawn groups, not for the quota.

### Step 5: estimate with the estimator the plan assumes

The plan describes the power-tuned prediction-powered mean, so estimate with it. Here the fake backend scores the toy corpus, and its gold column stands in for the human values of a draw for a quota of 20 texts (7 groups, 21 texts).

<!-- run -->
```python
rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=1))
data = cop.CorrectionData.from_scores(
    cop.export_text_scores("run.sqlite"),
    table,
    table.frame,
    gold_column="gold_constructiveness",
    score_columns=["construct_score"],
)
draw = cop.draw_human_sample(table, n=20, seed=1)
human = table.frame.set_index("text_id")["gold_constructiveness"].loc[draw.ids]
data = cop.corrections.attach_draw(data, human, draw)
tuned = cop.mean_ppi(data, power_tuning=True)
print(tuned.method, round(tuned.details["lambda_star"], 1), round(tuned.point, 1))
```

<!-- output -->
```text
ppi_mean_tuned 2.2 3.1
```

`lambda_star`, 2.2 here, is the weight the score gets. The fake backend's scores spread less than the gold values, so the default weight of 1 would leave much of what the score knows unused; the true mean of the 54 gold values is 3.0.

## On the command line

```bash
copernicus corrections plan --rho 0.8 --n-human 20,50,100 --n-machine 500,5000 --out plan.csv
```

With a single value for each of `--n-human` and `--n-machine` the command prints the plan; with lists it writes the table.

## What the output means

- **The factor is a ratio of precision, not a guarantee.** It tells you how many human-coded texts the corrected estimate is worth. It does not say that the interval has a given width.
- **Where the factor comes from.** It is the formula of the table above, applied to the numbers you give. The planner does not look at your data.
- **The limit caps the gain.** With `rho = 0.8` a corrected estimate is worth at most 2.8 times its human sample, however many texts you score by machine.

## Pitfalls

- **The planner counts texts as independent.** Whole-group draws with correlated texts carry less information per text than independent draws. Treat the planned number as a lower bound, and check the design with [`repeated_study`](correcting-estimates.md) on a corpus where you know the truth.
- **Use the correlation of the score you will use.** A pilot correlation from a few coded texts is itself noisy. Plan with a cautious `rho`, below the pilot's value.
- **`rho` must lie strictly between 0 and 1, and `n_human` at least 2.** The planner refuses other values, and refuses `n_machine` below `n_human`.
- **The battery needs room.** The correction battery refuses sizes above half the texts of a training pool. A pool is the training part of a fold, so the validation corpus must hold well over twice the largest size you want to test.
- **There is no universal number.** Ten coded texts can be enough for a very good score and a coarse question; hundreds may be needed for a weak score and a small effect. See the [FAQ](../faq.md#how-many-human-coded-texts-do-i-need).

## Next

Return to [Correcting estimates](correcting-estimates.md) to apply the sample, or read the [FAQ](../faq.md).
