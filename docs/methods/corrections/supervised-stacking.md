# Supervised stacking

Supervised stacking learns how to combine everything the machine outputs say about a text into one prediction of its human value (Wolpert 1992, Breiman 1996). Its inputs are the scores of every configuration and model, the spread of those scores, and the length of the text. A penalised linear regression, ridge, lasso or elastic net (Zou and Hastie 2005), maps them to the human value, with the strength of the penalty chosen on held-out groups of the training data so that the combination does not fit the noise of a small sample.

Because the prediction combines several signals and need not keep the order of any single score, stacking can rank the texts better than the best of its inputs. It also puts the prediction on the gold scale, and with the text length among its inputs it can absorb an error that moves with length. It needs at least two score columns and human-coded texts from at least two groups; `fit` refuses fewer score columns, or training texts from fewer groups, with `NotApplicableError`, as `applies` reports them.

## The algorithm as the library implements it

1. **Features.** Every score column of the data, the covariate (the log of the character count by default), and the spread of the scores: the row-wise population standard deviation of the score columns after standardising each with its mean and standard deviation on the training rows.
2. **Standardise.** Each feature is centred and divided by its population standard deviation on the training rows; a constant feature keeps a standard deviation of 1 and gets no weight. A training gold that is constant, or features that are all constant, are refused.
3. **Penalty**, by variant, always with an unpenalised intercept:
    - `ridge`: the penalty of `ALPHAS` (41 values from 0.01 to 10^8, evenly spaced on the log scale) with the smallest exact leave-one-group-out error, from the same ridge as the [NLP-ML baseline](../nlp-ml-baseline.md).
    - `lasso`: scikit-learn's `LassoCV` over a path of 12 penalties (`max_iter` 1500, `tol` 0.001), validated on `min(5, groups)` grouped inner splits.
    - `enet`: scikit-learn's `ElasticNetCV` with an L1 share of 0.5 or 0.9 on splits of the same kind.
4. **Inner seed.** The inner splits come from a seed drawn as the first number of the fit's generator and recorded as `inner_seed`; `ridge` needs no splits but draws and records the seed as well.
5. **Predict.** The fitted stacker keeps the intercept and the coefficients on the standardised features and predicts `intercept + features @ coefficients` for any text.

## Assumptions and what it corrects

<!-- run -->
```python
import copernicus as cop

spec = cop.corrections.REGISTRY["supervised_stacking"]
print(spec.target, spec.level, spec.estimate_errors)
for variant in spec.variants:
    print(variant, ", ".join(spec.errors_for(variant)))
```

<!-- output -->
```text
A score none
ridge calibration_shortfall, level_displacement, covariate_linked_error
lasso calibration_shortfall, level_displacement, covariate_linked_error
enet calibration_shortfall, level_displacement, covariate_linked_error
```

The family aims at target A. It addresses the calibration shortfall, with its ceiling raised by the information it adds, level displacement, and covariate-linked error through the covariate among its inputs. It assumes that the relation between the machine outputs and the human value in the human-coded sample holds for the corpus, and that the sample is large enough for the penalised fit: with few groups the chosen penalty is noisy and the stacker can do worse than a single calibrated column.

## Parameters

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `variant` | `ridge`, `lasso`, `enet` | The penalty |
| `CorrectionData.score_columns` | at least two | The scores to combine |
| `CorrectionData.covariate_column` | `log_chars` | The covariate among the features |

## Example

The block stacks the fake backend's construct score and its three attribute scores on the toy corpus, with a human sample of 27 texts in 9 groups.

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
    score_columns=["construct_score", "specificity", "tone", "reasoning"],
)
human = table.frame.set_index("text_id")["gold_constructiveness"]
draw = cop.draw_human_sample(table, n=27, seed=5)
data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)

ridge = cop.get_correction("supervised_stacking", "ridge")
fitted = ridge.fit(data, data.labelled_ids, rng=np.random.default_rng(0))
print(fitted.provenance["feature_names"])
print(fitted.provenance["alpha_selection"], fitted.provenance["n_inner_groups"])
for variant in ("ridge", "lasso", "enet"):
    corrected = cop.cross_fit_predictions(cop.get_correction("supervised_stacking", variant), data)
    print(variant, round((corrected - human).abs().mean(), 1))
```

<!-- output -->
```text
['construct_score', 'specificity', 'tone', 'reasoning', 'log_chars', 'score_spread']
grouped_logo 9
ridge 0.6
lasso 0.6
enet 0.6
```

## Where the library departs from the paper's analysis code

- A training gold that is constant, or features that are all constant, are refused; the paper's analysis code predicted the constant.
- The spread feature is the standard deviation of the scores after standardising each column on the training rows, so the units of a column do not change the predictions; the paper's analysis code took the spread of the raw scores.
- The inner path of penalties of `lasso` and `enet` has 12 penalties.

## Literature

Breiman (1996); Wolpert (1992); Zou and Hastie (2005). Full entries are in [References](../references.md).
