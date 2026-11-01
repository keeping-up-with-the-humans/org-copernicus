# Marginal matching

Marginal matching maps the distribution of the machine score onto the distribution of the human values in the human-coded sample, the way test equating links the scores of two test forms (Kolen and Brennan 2014). The `mean_sd` variant gives the score the mean and standard deviation of the human values; the `quantile` variant gives it their whole distribution, by sending each score to the human value at the same quantile.

Matching marginals is not the same as estimating each text's human value. The corrected scores have the distribution of the human values whatever the relation between score and human value of a single text, so the spread is matched to the human values rather than to the signal the score carries, and an error that moves with text length or another variable stays where it was. The family is therefore a tool for putting scores of different coders or settings on a comparable scale, and each fit says so in its provenance (`comparability_only`). Use [regression calibration](regression-calibration.md) to correct single values.

## The algorithm as the library implements it

1. **Score column.** On the training rows the family takes the score column with the highest signed Spearman correlation with the gold values.
2. **`mean_sd`.** The line with slope `sd(gold) / sd(score)` and intercept `mean(gold) - slope * mean(score)`, with population standard deviations over the training rows. A score that is constant on the training rows is refused.
3. **`quantile`.** Each distinct training score gets its mid-rank quantile `(average rank - 0.5) / n` among the `n` training scores. A score is mapped to a quantile by linear interpolation between the distinct training scores (a score beyond them gets the quantile of the nearer end), and the quantile to a gold value by the Hazen quantile of the training gold values.
4. **Apply.** The fitted map turns the score of any text into its matched value; `cross_fit_predictions` applies it without letting a text see its own gold value.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

spec = cop.corrections.REGISTRY["marginal_matching"]
print(spec.target, spec.level, spec.estimate_errors)
for variant in spec.variants:
    print(variant, ", ".join(spec.errors_for(variant)))
```

<!-- output -->
```text
A score none
mean_sd level_displacement, calibration_shortfall
quantile level_displacement, calibration_shortfall
```

The family is filed under target A and addresses level displacement and calibration shortfall in one sense only: the level and the spread of the corrected scores are those of the human values. It assumes that the human-coded sample has the distribution of the corpus, which a random sample from the same corpus provides. In a fit, `mean_sd` keeps the order of the texts exactly. `quantile` never reverses two texts, but with tied gold values it can send distinct scores to the same gold value, so it changes rank validity only through the ties it creates. Both leave covariate-linked error in place.

## Parameters

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `variant` | `mean_sd`, `quantile` | Match the mean and standard deviation, or every quantile |
| `cross_fit_predictions(k=..., seed=...)` | 5, 42 | Folds and seed of the cross-fitting |

## Example

On the toy corpus, with a human sample of 27 texts drawn by whole groups, both variants are fitted on the labelled texts. `mean_sd` reproduces the mean and standard deviation of their gold values exactly. `quantile` does not, because the toy scores have many ties: tied scores share one quantile and so one matched value, and the matched values have a standard deviation of 1.1 against the gold's 1.4.

<!-- run -->
```python
import numpy as np

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

gold = data.gold[data.labelled_mask]
print("gold", round(gold.mean(), 1), round(gold.std(), 1))
for variant in ("mean_sd", "quantile"):
    matching = cop.get_correction("marginal_matching", variant)
    fitted = matching.fit(data, data.labelled_ids, rng=np.random.default_rng(0))
    matched = fitted.predict(data, data.labelled_ids)
    print(variant, round(matched.mean(), 1), round(matched.std(ddof=0), 1))
print(fitted.provenance["comparability_only"])
```

<!-- output -->
```text
gold 3.1 1.4
mean_sd 3.1 1.4
quantile 3.2 1.1
True
```

## Where the library departs from the paper's analysis code

- `mean_sd` refuses a score that is constant on the training rows; the paper's analysis code set the slope to 0 without a word, which returns the gold mean for every text.
- `quantile` maps tied scores through their mid-rank quantile and the Hazen quantile of the gold values. The paper's analysis code interpolated the sorted scores against the sorted gold values, which sends every text of a block of tied scores to the gold value at the top edge of the block and so pushes the mapped level up. On scores without ties the two maps coincide.

## Literature

Kolen and Brennan (2014). The full entry is in [References](../references.md).
