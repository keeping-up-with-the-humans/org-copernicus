# Misclassification matrix

When the human value is a category, a machine score that is cut into the same categories misclassifies some texts, and the share of texts in each category is off (Bross 1954). The misclassification matrix estimates on the human-coded sample how often each true category ends up in each predicted category. Inverting that matrix corrects the shares; for two categories this is the Rogan-Gladen estimator, `(observed share + specificity - 1) / (sensitivity + specificity - 1)` (Rogan and Gladen 1978). Bayes' rule turns the same matrix into each text's probability of belonging to each category, and the family's corrected value of a text is the mean of the categories under those probabilities.

The inversion divides by sensitivity plus specificity minus one, Youden's J, so its variance grows as the score separates the categories less well. The family therefore applies only when the score separates the categories at least to a floor and every category has enough human-coded texts. Its corrected values relabel the predicted categories: every text of a predicted category gets the same value, and with two categories the correction cannot reverse the order of two texts.

## The algorithm as the library implements it

1. **Applicability.** The categories are the distinct labelled gold values; there must be between 2 and `max_classes` (5) of them, each with at least `per_class_floor` (30) labelled texts. On the best score column, the informedness of the cut score, `(k * balanced accuracy - 1) / (k - 1)` for `k` categories (Youden's J for two), must reach `youden_floor` (0.3). Otherwise the family does not apply and says why. `applies` judges all labelled texts; `fit` makes the same check on the training texts it is given and raises `NotApplicableError` with the same reason when they fail it, also after `applies` passed on a larger sample. A lower floor is an explicit setting of the correction, passed to `get_correction`.
2. **Cutting the score.** On the training rows, the best score column is cut at its quantiles at the cumulative shares of the gold categories, so the cut score has the category shares of the gold.
3. **Confusion matrix.** `P(predicted i | true j)` is estimated on the training rows with one added count per cell, `(hits + 1) / (n_j + k)`. The priors are the category shares of the training gold.
4. **Posterior means.** For each predicted category, Bayes' rule gives the probability of each true category, and the corrected value is the mean of the category values under it. `predict` returns the posterior mean of a text's predicted category.
5. **Category shares.** `class_shares(data, ids)` returns the share of each predicted category among the given texts and, for two categories, the sensitivity and specificity of the training rows and the Rogan-Gladen share of the category with the larger value, clipped to 0 to 1 (missing, with a note, when sensitivity equals one minus specificity).

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

spec = cop.corrections.REGISTRY["misclassification_matrix"]
print(spec.target, spec.level, spec.estimate_errors)
for variant in spec.variants:
    print(variant, ", ".join(spec.errors_for(variant)))
```

<!-- output -->
```text
A, C score bias of class shares
posterior level_displacement
```

The family aims at target A at the level of categories, through the probability of each category for each text, and at target C for the category shares. It addresses level displacement in that sense and the bias of category shares. It assumes that the misclassification rates of the human-coded sample hold for the whole corpus, which a random sample from the same corpus provides, and that the human categories are the truth. It needs a categorical gold with few, well-populated categories.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `youden_floor` | 0.3 | Smallest informedness of the cut score on the labelled texts (`applies`) and on the training texts (`fit`) |
| `per_class_floor` | 30 | Fewest labelled texts in each category, on both |
| `max_classes` | 5 | Most categories |

The variant is `posterior`; pass the parameters to `get_correction("misclassification_matrix", "posterior", ...)`.

## Example

The toy gold has five values. The block turns it into two categories, values of 3 and above against the rest, and uses a human sample of 27 texts. With the default floor of 30 texts per category the family does not apply, so the block lowers the floor to 10 on purpose, to show the method on a toy sample. `fit` checks the floors on its training texts too, as the second line shows, so the lower floor has to be a setting of the correction.

<!-- run -->
```python
import numpy as np

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
frame = table.frame.assign(high=(table.frame["gold_constructiveness"] >= 3).astype(float))
full = cop.CorrectionData.from_scores(
    cop.export_text_scores("run.sqlite"),
    table,
    frame,
    gold_column="high",
    score_columns=["construct_score"],
)
high = frame.set_index("text_id")["high"]
draw = cop.draw_human_sample(table, n=27, seed=5)
data = cop.corrections.attach_draw(full, high.loc[draw.ids], draw)

default = cop.get_correction("misclassification_matrix", "posterior")
print(default.applies(data))
try:
    default.fit(data, data.labelled_ids, rng=np.random.default_rng(0))
except cop.errors.NotApplicableError as error:
    print(error)
matrix = cop.get_correction("misclassification_matrix", "posterior", per_class_floor=10)
fitted = matrix.fit(data, data.labelled_ids, rng=np.random.default_rng(0))
shares = fitted.class_shares(data, data.ids)
sensitivity, specificity = shares["sensitivity"], shares["specificity"]
print(fitted.provenance["youden_j"], round(sensitivity, 1), round(specificity, 1))
print(round(shares["raw"][1.0], 1), round(shares["rogan_gladen"], 1), round(high.mean(), 1))
print(np.unique(fitted.predict(data, data.ids).round(1)).tolist())
```

<!-- output -->
```text
(False, 'class 0 has 10 labelled rows; at least 30 are needed')
misclassification_matrix/posterior does not apply to the training rows: class 0 has 10 labelled rows; at least 30 are needed
1.0 0.9 0.9
0.6 0.5 0.6
[0.1, 1.0]
```

The cut score separates the two categories perfectly on the drawn texts (J of 1.0); the added counts keep sensitivity and specificity below 1. The share of high texts among all 54 is 0.6; the raw share of the cut score is 0.6 and the Rogan-Gladen share 0.5, so with so few texts the inversion moves the share a little the wrong way. Every text gets one of two posterior means.

## Where the library departs from the paper's analysis code

- The library requires at least 30 labelled texts in each category (`per_class_floor`) and computes the informedness on the labelled texts, and again on the training texts of every fit; the paper's analysis code required at least 30 times `k` labelled texts in all, for `k` categories, and computed Youden's J on every text of the corpus with a human value rather than on the labelled texts alone. The default floors are the values the paper states: an informedness of 0.3 and 30 human-coded texts per category.

## Literature

Bross (1954); Rogan and Gladen (1978). Full entries are in [References](../references.md).
