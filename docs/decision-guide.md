# Decision guide

A study that codes a construct from text makes three choices: a detection method for its kind of text, a correction for the estimate it will publish, and the number of texts a human codes. This page answers each choice with the route that came closest to the human-coded values in the accompanying methods paper (under review), and points to the methods page and the tutorial of each tool. A route is a starting point, not a guarantee. Confirm it on a human-coded subset of your own corpus with the [validity panel](methods/validity-panel.md) before you commit to it.

## Which detection method for which kind of text

| Your texts | Start with | Why |
|------------|------------|-----|
| Texts that people wrote or spoke in answer to a question or a task: interview answers, open survey answers, essays, written feedback | The [two-stage LLM coder](methods/two-stage-llm-coder.md) | It reads each attribute against the rubric, scores only the passages it verified in the text, and shows you those passages |
| Short posts written for an audience, in the style of microblogs | The [similarity coder](methods/similarity-coder.md) | It scores a text by how close it lies to texts written for each pole of the construct, and needs no model call per text |
| Any texts, when you already have a few hundred human-coded texts from the same corpus and want a value for every text | [Embedding-augmented calibration](methods/corrections/embedding-augmented-calibration.md), variant `embeddings_only`: `get_correction("embedding_augmented", "embeddings_only")` with `cross_fit_predictions` | A ridge regression on embeddings learns the human values from the coded texts, needs no rubric, and, cross-fitted, predicts every text of the corpus |

In the paper, a small decision tree over text genre, construct anchoring, text length and language sent microblog posts to the similarity coder and every other kind of text to the two-stage LLM coder. For traits of a speaker, measured by a questionnaire and scored from the turns of a conversation, no coder was reliably better. The tree was checked on combinations of the paper's benchmark that it had not seen, not on new corpora. So treat the table as the first guess, and compare the candidates on one human-coded subset of your own texts.

Two more coders complete the set. The [hybrid coder](methods/hybrid-coder.md) scores the verified passages of a two-stage run by similarity, so a change of references or encoder costs no new model call. The [one-stage LLM coder](methods/one-stage-llm-coder.md) scores each attribute in one call without passages; it is the baseline that the two-stage coder is compared with.

The [NLP-ML baseline](methods/nlp-ml-baseline.md) fits the same kind of ridge regression but predicts only the human-coded texts, each out of fold: it is the benchmark that the coders are read against, not a score for the corpus, so it cannot feed the prediction-powered estimators of a corpus. The cross-fitted predictions of embedding-augmented calibration can: they are a score like any other, so an estimate built from them still goes through prediction-powered inference, as below, with `mean_ppi(data, prediction=...)`. Cross-fitting predicts every text with a fit that never saw its own human value.

## Which correction for which published estimate

| You will publish | Use | Functions |
|------------------|-----|-----------|
| A mean, a total, a trend, a difference between groups, or a regression coefficient with the construct as the outcome | [Pseudo-values and prediction-powered inference](methods/corrections/pseudo-values.md) | `mean_ppi`, `group_difference_ppi`, `ols_outcome_ppi`, `pseudo_values` |
| A value for each text, to rank, list or merge the texts | [Embedding-augmented calibration](methods/corrections/embedding-augmented-calibration.md), then prediction-powered inference on any estimate built from the corrected values | `get_correction("embedding_augmented", ...)` with `cross_fit_predictions`, then `mean_ppi(data, prediction=...)` |
| A regression coefficient with the construct as a predictor | Prediction-powered inference of the predictor's coefficient | `ols_predictor_ppi` |

The prediction-powered estimators take the machine score of every text and correct it with the gap between the human values and the score on the labelled texts. Their point estimate rests on the random draw of the human sample, not on a model of the score's errors, so a poor score widens the interval instead of biasing the point. The [pseudo-values page](methods/corrections/pseudo-values.md) states when the point is exactly unbiased and when approximately.

Corrected values for single texts aim at best guesses: a fitted line, or the ridge of embedding-augmented calibration, approximates the expected human value given the score and the features, so that among texts with the same score and features the errors average roughly zero. That gets the mean about right, but a difference between groups that the correction did not model can stay biased. So an estimate built from corrected values goes through prediction-powered inference again, with the corrected values as the prediction.

For the construct as a predictor, the library also has [moment subtraction and SIMEX](methods/corrections/classical-corrections.md) and [instrumented coding](methods/corrections/instrumented-coding.md). They correct the coefficient through a model of the measurement error and hold only as far as that model holds. [Correcting estimates](tutorials/correcting-estimates.md) runs every estimator on the toy corpus.

## How many texts a human must code

1. **Plan with the effective-size factor.** `plan_sample_size(rho, n_human, n_machine)` turns the correlation of the score with the human value into the number of human-coded texts that the corrected estimate is worth, for an estimator that uses the score at its best: `mean_ppi(data, power_tuning=True)`, or the `calibrated` rectifier of `pseudo_values`. The default `mean_ppi(data)` weights the raw score by 1 and can be worth far fewer texts, so estimate with power tuning. Take `rho` from a pilot and be cautious with it. [Sample size](tutorials/sample-size.md) finds the smallest human sample for a target.
2. **More human-coded texts keep helping.** In the paper, the gain in rank validity of the strongest score-level corrections kept growing up to 1,000 human-coded texts, the largest number tested. A corrected column is reused in every later analysis, so a study keeps gaining past the 125 to 250 texts that already correct one regression coefficient well (Ludwig et al. 2026).
3. **The sample must come from the same corpus.** A correction learns how the score relates to the human value in the corpus the sample was drawn from. In the paper, a correction fitted on other corpora alone lost rank validity on the corpus it was applied to. Draw the sample from the corpus you will estimate on, by whole groups, with `draw_human_sample`, before you look at any gold value; the estimators need the inclusion probability that the draw records.
4. **Respect the floors of the code.** The prediction-powered estimators need at least two labelled texts and, with clustering, two labelled groups; a group difference needs both on each of its sides. Cross-fitting needs at least 10 labelled training texts in each fold. Check a design with `repeated_study` on a corpus where every text has a human value.

## Where each tool is shown

| If you want to | Methods page | Tutorial |
|----------------|--------------|----------|
| Code texts with a model and read the passages behind each score | [Two-stage LLM coder](methods/two-stage-llm-coder.md) | [The two-stage coder](tutorials/two-stage-coder.md) |
| Code texts at a lower cost per text, without passages | [One-stage LLM coder](methods/one-stage-llm-coder.md) | [The two-stage coder](tutorials/two-stage-coder.md#the-one-stage-coder) |
| Score a corpus by similarity to reference texts | [Similarity coder](methods/similarity-coder.md) | [The similarity coder](tutorials/similarity-coder.md) |
| Score the extracted passages by similarity | [Hybrid coder](methods/hybrid-coder.md) | [The hybrid coder](tutorials/hybrid-coder.md) |
| Know what a supervised model reaches on your gold data | [NLP-ML baseline](methods/nlp-ml-baseline.md) | [The NLP-ML baseline](tutorials/nlp-ml-baseline.md) |
| Check a score against gold values | [Validity panel](methods/validity-panel.md) | [The validity panel](tutorials/validity-panel.md) |
| Draw the texts a human should code | [Corrections](methods/corrections/index.md) | [The human sample](tutorials/human-sample.md) |
| Put each score on the gold scale | [Corrections](methods/corrections/index.md) | [Correcting scores](tutorials/correcting-scores.md) |
| Estimate a mean, a difference or a coefficient with an interval | [Pseudo-values](methods/corrections/pseudo-values.md) | [Correcting estimates](tutorials/correcting-estimates.md) |
| Decide how many texts to code by hand | [Corrections](methods/corrections/index.md) | [Sample size](tutorials/sample-size.md) |
