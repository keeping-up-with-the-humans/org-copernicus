# Correction ensemble

The correction ensemble combines the corrected columns of several correction families into one column, by their mean, their mode, a one-factor score, or their first principal component. Combining forecasts can do better than any one of them: an optimally weighted combination has an error variance no larger than its best member's (Bates and Granger 1969), and combining fitted predictors is the idea behind stacking (Breiman 1996). The library's combinations are unweighted, and how they compare with their members depends on how much the members share their errors: members with independent errors can pool into a column that ranks the texts better than the best of them, and members that share their errors gain nothing and can come out slightly below every member, as here. The ensemble serves as a check on the families and as a practical default for ranking.

The members are standardised before they are combined, so the result is not on the scale of the human values: its mean can lie further from the mean of the human values than any member's. The ensemble's output is therefore rank-only. It supports rank validity and correlations, not the level or the spread of the human values. The `mode` combiner is the exception: it returns gold values, but only the discrete classes of the gold.

## The algorithm as the library implements it

1. **Members.** A mapping from member names to corrected columns of the same texts, each a series indexed by text id, such as the cross-fitted predictions of several families. A member that lacks a requested id is an error; a member with a non-finite value or no variation over the requested ids is dropped and named in `dropped_members`. At least two members must remain.
2. **Standardise.** Each member is centred and divided by its population standard deviation over the requested ids.
3. **Combine**, by variant:
    - `mean`: the mean of the standardised members.
    - `mode`: the members, each value rounded to the nearest gold class, and the most frequent class per text (a tie goes to the smaller class). It needs the gold classes.
    - `factor`: the regression factor score of a one-factor principal-axis solution over the standardised members, solved with `1e-3` added to the diagonal of the correlation matrix.
    - `pc1`: the first principal component of the standardised members.
4. **Orient.** `factor` and `pc1` have an arbitrary sign, set so that the result does not correlate negatively with the mean of the standardised members.

The ensemble works on columns, not on texts, so it has no factory: build it as `cop.corrections.CorrectionEnsemble(variant)` and call `combine(members, ids)`. In the battery it pools the held-out predictions of the entries fitted in each fold, and only its rank validity and signed Spearman correlation are scored.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

spec = cop.corrections.REGISTRY["correction_ensemble"]
print(spec.target, spec.level, spec.estimate_errors)
for variant in spec.variants:
    print(variant, ", ".join(spec.errors_for(variant)))
```

<!-- output -->
```text
A score none
mean noise, calibration_shortfall
mode noise, calibration_shortfall
factor noise, calibration_shortfall
pc1 noise, calibration_shortfall
```

The family aims at target A in the rank-only sense. It addresses the noise of the members and raises the ceiling of the calibration shortfall by pooling corrected columns. It assumes that the members' errors are not all the same: members that share their errors pool to the same errors.

## Parameters

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `variant` | `mean`, `mode`, `factor`, `pc1` | The combiner |
| `members` | | Corrected columns by name, indexed by text id |
| `ids` | | The text ids to combine, unique |
| `gold_classes` | `None` | The discrete gold values, needed by `mode` only |

## Example

Three cross-fitted corrections of the fake backend's toy scores, with a human sample of 27 texts, pooled by their mean and by their mode:

<!-- run -->
```python
import numpy as np
from scipy.stats import spearmanr

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
full = cop.CorrectionData.from_scores(
    cop.export_text_scores("run.sqlite"),
    table,
    table.frame,
    gold_column="gold_constructiveness",
    score_columns=["construct_score"],
)
human = table.frame.set_index("text_id")["gold_constructiveness"]
draw = cop.draw_human_sample(table, n=27, seed=5)
data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)

members = {
    f"{family}/{variant}": cop.cross_fit_predictions(cop.get_correction(family, variant), data)
    for family, variant in [
        ("regression_calibration", "linear"),
        ("regression_calibration", "isotonic"),
        ("classical", "tweedie"),
    ]
}
pooled = cop.corrections.CorrectionEnsemble("mean").combine(members, data.ids)
print(pooled.provenance["n_members"], pooled.provenance["dropped_members"])
truth = human.loc[data.ids]
pooled_rho = spearmanr(pooled.column, truth).statistic
member_rhos = [spearmanr(member, truth).statistic for member in members.values()]
print(round(pooled_rho, 1), pooled_rho < min(member_rhos))
print(round(pooled.column.std(ddof=0), 1), round(human.std(ddof=0), 1))
voted = cop.corrections.CorrectionEnsemble("mode").combine(
    members, data.ids, gold_classes=np.arange(1, 6)
)
print(sorted(set(voted.column)), round((voted.column - human).abs().mean(), 1))
```

<!-- output -->
```text
3 ()
0.8 True
1.0 1.4
[1.0, 2.0, 3.0, 4.0, 5.0] 0.5
```

The mean of the standardised members has a rank correlation of 0.8 with the gold, a little below each of its three members: all three are monotone corrections of the one toy score column, so they share their errors and pooling has nothing to average away. It lives on a standardised scale: its standard deviation is about 1, the gold's 1.4. The mode returns gold classes.

## Where the library departs from the paper's analysis code

- In the battery the ensemble pools the entries fitted in each fold, so its members can change from fold to fold; the paper admitted only variants fitted on every fold.

## Literature

Bates and Granger (1969); Breiman (1996). Full entries are in [References](../references.md).
