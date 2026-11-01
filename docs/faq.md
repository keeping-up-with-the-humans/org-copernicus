# FAQ

## How much will a run cost?

The library has no price table, and the prices of providers change. It tells you the size of a run before you spend anything, and you multiply by the prices you read off your provider's list.

**Reference texts.** `copernicus references generate --dry-run` builds no backend and calls no model. It prints the planned calls, the estimated prompt tokens, and the estimated completion tokens:

```text
planned calls: 24
estimated prompt tokens: 15632 (tiktoken o200k_base)
estimated completion tokens: 28800 (120 per requested text)
```

The prompt tokens are those of the rendered prompts of every first attempt, counted with the `o200k_base` encoding of `tiktoken` when it is installed and with the number of characters divided by four when it is not. The line names the counter. The completion tokens are assumed to be 120 per requested text. Retries and surplus texts come on top. Pass both prices to see a cost: `--price-input` and `--price-output` take US dollars per million tokens.

The cost is

```text
cost = (prompt_tokens * price_input + completion_tokens * price_output) / 1,000,000
```

with both prices in the same currency per million tokens. In Python, `cop.references.estimate_tokens(rubric, settings, n_writers=...)` returns the same numbers, and `estimate.cost(price_input, price_output)` applies the formula.

<!-- run -->
```python
import json
from pathlib import Path

import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
settings = cop.GeneratorSettings(texts_per_call=2)
estimate = cop.references.estimate_tokens(rubric, settings)
print(estimate.calls, estimate.completion_tokens)
print(estimate.cost(price_input=1.0, price_output=2.0) > 0)
```

<!-- output -->
```text
24 5760
True
```

The prices in the block are placeholders, not prices of any provider.

**Coding.** The two-stage LLM coder makes one extraction call and, when a verified passage exists, one scoring call for each text, run and attribute. So there are at most two requests per attribute call, plus retries. A text shorter than `min_chars` makes none. The token counts depend on your rubric and your texts. Measure them instead of guessing: code a small slice with the real model and read `prompt_tokens` and `completion_tokens` from the `CodingSummary`, then scale by the number of texts.

```python
summary = cop.code_texts(
    table, rubric, backend, "pilot.sqlite", cop.CodingSettings(runs=2, text_range=(0, 20))
)
print(summary.prompt_tokens, summary.completion_tokens)
```

A dry run with the fake backend counts the requests: `len(backend.calls)` after `code_texts`. The count is exact for the extraction calls, one per text, run and attribute; the scoring calls follow the passages that the model finds, so a real model can make more or fewer of them than the fake backend. The fake backend counts words, not tokens, in `prompt_tokens`.

**Everything else is free to compute.** The similarity coder, the hybrid coder, the baseline, the validity panel and the corrections make no model call.

## Do I need a GPU?

No. The LLM coders send requests to a backend. An OpenAI-compatible server runs elsewhere. The `VLLMBackend` runs an open-weight model on your own GPU and needs the `vllm` extra. The encoders run on a CPU: `minilm` and `mpnet` are small; `bge_m3` is slower; `gte_qwen2_7b` has 7 billion parameters and needs a GPU in practice. `gte_qwen2_7b` is the only registered encoder that runs the model repository's own code; the registry pins it to one commit, and a specification that allows remote code must name the full 40-character commit hash as its revision. Without a device argument the encoder uses CUDA, then Apple's MPS, then the CPU. Pass `device="cpu"` to force the CPU. Everything after the embedding step is numpy, scipy and scikit-learn on a CPU.

## Does it work for languages other than English?

The code does not assume English, with these exceptions.

- The prompt templates are in English. The model reads your rubric and texts in their own language, and a capable model handles that, but check its passages and scores in your language with the validity panel.
- Passage verification works on text normalised by Unicode NFC, case folding and unified quotation marks and dashes, so it works for scripts with case or without. It compares character sequences.
- The `min_words` of the hybrid coder and the `n_words` of the text table count whitespace-separated words, which is not meaningful for languages written without spaces. The two-stage coder's `min_chars` counts characters.
- `minilm` and `mpnet` are English encoders. For other languages use a multilingual encoder, such as the registry's `bge_m3`, or pass the Hugging Face id of another one.
- Read the generated reference texts yourself to check that they are in the language of your texts.

## How do I resume, shard and merge?

- **Coding.** `code_texts` writes the run store every `save_every` texts. Run the same call again and it codes only the cells the store lacks. To code in parts, give each part a `text_range` (`--range A:B` on the command line; positions in text-id order) and its own store, then `cop.merge_stores([...], target)` (`copernicus store merge`). The parts must belong to one run. A store continues only with the same method, model, temperature, seed, rubric and settings. To add runs under `temperature="auto"`, see the pitfall in [The two-stage coder](tutorials/two-stage-coder.md).
- **Changed texts.** A store binds each text id to the text it was coded on: every row records the hash of its text. A rerun stops before any backend call when a text has changed under its id, and the merge refuses parts that hold different texts under one id. Code the changed texts into a new store, or restore them.
- **Failed cells.** Cells whose attributes ended in `extraction_failed` or `scoring_failed` stay in the store. The failure check at the end of every rerun over their texts and runs counts them, so a failed run does not turn into a success by being run again. The checks after each batch judge the calls of the current invocation only, so a failing backend still stops early on a rerun over a large store. `retry_failed=True` (`--retry-failed`) codes them again and replaces them; a higher `max_failure_share` accepts them. A retry uses the same prompts, seeds and temperatures, so it repairs failures of the backend (answers that came back empty or malformed on every attempt) and not a model that fails the same text every time; such a text needs a changed setting of the coder or another model, and so a new store. A timeout that outlasts the backend's own retries is not stored as a failed cell: it stops the run with a `BackendError`, and a plain rerun codes the cell. The hybrid coder still refuses a text whose extraction failed, unless the text is short, because its passages are incomplete.
- **Reference texts.** Pass a resume file, `resume=Path("refs.csv")` or `--resume refs.csv`. Every finished call is written to it at once. A rerun with the same file makes only the missing calls.
- **Embeddings.** `Encoder.embed(..., checkpoint=Path("stem"))` and `--checkpoint stem` make embedding resumable.

<!-- run -->
```python
def respond(request):
    return json.dumps([{"text": f"Placeholder note {request.seed} number {i}."} for i in range(2)])


writer = cop.FakeBackend(respond, model_name="placeholder-writer")
first = cop.generate_references(rubric, [writer], settings, resume=Path("refs.csv"))
second = cop.generate_references(rubric, [writer], settings, resume=Path("refs.csv"))
print(len(first.frame), len(writer.calls))
```

<!-- output -->
```text
48 24
```

The second call found every call in the resume file and made none: the writer received 24 requests in all. The placeholder writer is a fake; its texts are not meaningful references.

## What limits reproducibility?

**What is reproducible.** Every random step takes a seed and records it: the wording and marker order of the coders (`CodingSettings.seed`), the human-sample draw (`draw_human_sample(seed=...)`), the folds and draws of the baseline and the battery (`seeds`), the bootstrap of the panel (`PanelSettings.seed`), and the reference generation (`GeneratorSettings.seed`). The same seed on the same inputs gives the same draw, folds and bootstrap. The run store keeps the provenance of every score: the model, temperature, seed, wording variants, template hashes, rubric hash, the hash of the text it was coded on, and the library version.

**What is not.**

- **Language models.** A provider may not honour a seed, and a model behind an API can change under the same name. The provenance records whether the backend supports a seed and the temperature it applied. A fake backend is exactly reproducible; a real model is reproducible only as far as its provider allows. Run several times (`runs`) and read the noise in the validity panel.
- **Other hardware.** Embeddings computed on another CPU, a GPU, or with another version of torch or transformers differ in the last digits. The differences are small, but they can reorder near ties and change the digits of a score, a ridge penalty or a bootstrap interval. Random draws made with numpy do not depend on the hardware, but they can change with a major release of numpy.
- **Models on the hub.** The weights behind a name can change. The sidecar of every embedding set records the commit that was loaded (`resolved_revision`), with the torch and transformers versions. It also names the arrays it was written with (`arrays_hash`), so a load refuses arrays of another save beside it. Pin the commit for a study you want to rerun: pass `dataclasses.replace(cop.ENCODERS["mpnet"], revision="<commit>")` to `Encoder`, or add `--revision <commit>` to `copernicus embed texts`, `copernicus embed references` and `copernicus hybrid score`. Keep the sidecar files with your results.

## Why does a script crash at exit with `python -W error`?

On macOS with torch and sentencepiece installed, a script run with `python -W error` can end in a segmentation fault at interpreter exit, after all the work has completed and its results are written. Torch leaves a temporary directory behind, and the warning that its clean-up raises becomes an exception during shutdown; sentencepiece's module destructor then crashes. Run scripts without `-W error`. The test suite's `-W error` is not affected, because pytest applies it during the tests only.

## Why does an attribute without a passage score 1?

In the two-stage LLM coder, the scoring stage sees only the verified passages. An attribute for which no marker kept a verified passage has nothing to score. By default the coder records the lowest level, 1, with the status `no_evidence`: a text that shows none of the attribute's markers is a text at the bottom of the scale.

This choice has consequences. Terse texts get low scores, and the construct score is a mean that includes these 1s. Two remedies exist. The status column separates them: filter the scores on `status == "ok"`. Or leave them missing with `TwoStageConfig(no_evidence_score=None)`, as shown in [The two-stage coder](tutorials/two-stage-coder.md); then the panel and the exports leave them out of every mean. The hybrid coder marks the same texts with `fallback == "no_evidence"` and scores the whole text.

## Why must the human sample come from the same corpus?

A correction learns how the machine score relates to the human value in the population you drew the sample from. It then applies that relation to all the texts you scored. If the human texts come from another corpus, the learned relation is the wrong one: the score may be biased differently there, texts may be longer or shorter, and the coder may behave differently. The design-based estimators also rest on the inclusion probability of each text, `m / G` for a draw of `m` of `G` groups. That probability exists only for the corpus you drew from. A draw therefore records a fingerprint of its table's texts and their groups, and `attach_draw` refuses correction data with other texts or groups, renamed groups included.

So draw the human sample from the corpus you will estimate on, by whole groups, before you look at any gold value. Blind transport, which applies a correction fitted on other corpora, is a separate tool with rank-only output.

## How many human-coded texts do I need?

There is no universal number. It depends on how well the score tracks the human value, how many texts you score, and how precise the answer must be. Use the planner:

```python
plan = cop.plan_sample_size(rho=0.8, n_human=40, n_machine=5000)
print(plan.factor * plan.n_human)  # effective number of human-coded texts
estimate = cop.mean_ppi(data, power_tuning=True)  # the estimator the factor describes
```

`plan_sample_size` turns the correlation of the score with the human value into a factor: the human sample counts as `factor * n_human` human-coded texts for an estimator that uses the score at its best, the power-tuned `mean_ppi(data, power_tuning=True)` or the `calibrated` rectifier of `pseudo_values`. The default `mean_ppi(data)` weights the raw score by 1; when the score's spread differs from the gold's it is worth fewer human texts than the plan says, so estimate a study sized this way with power tuning. [Sample size](tutorials/sample-size.md) shows how to find the smallest sample for a target and how the number of texts follows from the number of groups. Take `rho` from a pilot with a small human sample and be cautious with it. Check the design with `repeated_study` on a corpus where you know the truth.

The code sets lower bounds of its own. The prediction-powered estimators need at least two labelled texts (for a group difference, two on each side) and, with clustering, two labelled groups; a clustered group difference needs two labelled groups on each side. Cross-fitting needs at least 10 labelled training texts in each fold. The baseline's default floor is 50 training units.

## Why are the standard errors clustered under whole-group draws?

A human sample is drawn by whole groups: all texts of a drawn group enter together. The texts of one group are alike, because one person or one document wrote them, so their errors are correlated. A standard error that treats them as independent is too small, and intervals built from it cover the truth less often than they claim.

The prediction-powered estimators therefore sum the contributions of the texts within each group before they compute the variance, whenever a group holds more than one labelled text. That makes the group the unit of the standard error, and the variance carries the small-sample factor G/(G − 1) for G groups. `cluster=False` switches it off and gives the smaller, wrong error; the [tutorial](tutorials/correcting-estimates.md) shows the difference. The cost of clustering is that few labelled groups give a noisy standard error. Clustered, the estimators refuse to run with fewer than two labelled groups, and a group difference refuses to run when the labelled texts of one side all lie in one group, because that side would add none of its variation between groups to the standard error; this holds under the automatic choice too. The remedies are to draw more groups, or to pass `cluster=False`, whose independent standard error ignores the correlation of the texts within a group. The classical estimators (`moment_subtraction`, `simex`) bootstrap texts, not groups, and do not have this protection.

Clustering changes the standard error, not the point. Over repeated whole-group draws, the point of a prediction-powered mean is exactly unbiased when every group of the corpus holds the same number of texts, and the point of a group difference when, on each side of the split, every group of the corpus that has texts on that side holds the same number of them (groups of equal size that each lie wholly on one side are one such case, not the only one). Where some groups straddle the split and others lie on one side, this holds only provided no draw is refused: keeping only the draws with enough labelled texts or groups on each side shifts which groups are drawn. Otherwise the means over the labelled texts are ratio estimators, and the point is approximately unbiased. The quality of the machine score does not bias the point. `pseudo_values(data).mean_rectified`, with its default `identity` rectifier, is exactly unbiased under any whole-group draw.
