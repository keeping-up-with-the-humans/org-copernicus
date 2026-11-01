# One-stage LLM coder

The one-stage LLM coder scores one attribute of one text in a single call to a language model. The prompt holds the attribute, its markers, the level descriptions of every marker and the glossary of the rubric, and the model reads the whole text and returns a score from 1 to 5 with an explanation. There is no extraction stage and no check of passages against the text, so the coder records no evidence.

This is the common way of coding text with a language model, as in studies that let a model annotate texts directly (Gilardi et al. 2023, Ziems et al. 2024), and it is the baseline that the [two-stage LLM coder](two-stage-llm-coder.md) is compared with. The library runs it inside the same machinery as the two-stage coder: the same rubric, the same seeded rotation of the prompt wording, the same repeated runs and the same run store. A comparison of the two coders on the same texts therefore differs only in the coding procedure.

## The algorithm as the library implements it

For each text, run and attribute:

1. **Length check.** A text shorter than `min_chars` (20) characters is not sent to the model; the attribute gets the status `too_short` and no score.
2. **Wording and marker order.** The wording and the marker order are drawn exactly as for the two-stage coder, with `random.Random(seed + text_key * 1000 + run_number)`; the one-stage coder uses the `one_stage` texts of the role, task and level wordings.
3. **The call.** The system prompt holds the role and task wording, the attribute, its markers in the drawn order, each marker's level descriptions as a mapping from level 5 down to level 1, the glossary, and the level wording. The user message is `Text to analyze:` followed by the text.
4. **The answer.** It must be a JSON object `{"dimension_result": {"result": 1 to 5, "explanation": "..."}}`. The result is checked under the rules of the two-stage coder's scoring stage: a whole-number float or a digit string counts, a boolean or a value outside 1 to 5 does not. An unusable answer is retried, up to `max_retries` (3) attempts in all, with the seed plus the attempt number and an unchanged temperature. After the last failure the status is `scoring_failed` and the score is missing, never 0.
5. **Scores of a text.** As for the two-stage coder: the construct score of a run is the mean of the attribute scores that exist, and a text's score is the mean over runs.

The statuses are `ok`, `too_short` and `scoring_failed`. The status `no_evidence` does not occur, because the coder never looks for evidence. The runner's failure ceiling, temperature rule and provenance are those of the two-stage coder.

## Assumptions and what it corrects

The coder produces scores; it corrects nothing. It assumes that a model can weigh the whole text against the rubric in one pass. Nothing ties its score to a passage of the text, so a reader cannot audit why a text received its score, and the [hybrid coder](hybrid-coder.md), which needs passages, refuses a one-stage store.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `OneStageConfig.max_retries` | 3 | Attempts, the first included |
| `OneStageConfig.max_new_tokens` | 2048 | Generation budget of the call |
| `OneStageConfig.min_chars` | 20 | Shorter texts are not coded |
| `CodingSettings(method="one_stage")` | | Selects the one-stage coder; the other settings are those of the two-stage coder |

## Example

The fake backend answers the one-stage prompt from the same cue words as the two-stage prompts. Six texts, one run, three attributes:

<!-- run -->
```python
import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
settings = cop.CodingSettings(runs=1, method="one_stage")
summary = cop.code_texts(table.subset(table.ids()[:6]), rubric, backend, "one.sqlite", settings)
print(summary.n_attribute_calls, summary.status_counts["ok"], len(backend.calls))
print(len(cop.export_evidence("one.sqlite")))
```

<!-- output -->
```text
18 18 18
0
```

Eighteen attribute calls make eighteen requests, one each, and the evidence table stays empty. On the command line the coder is `copernicus code one-stage`, with the options of `copernicus code two-stage`.

## Where the library departs from the paper's analysis code

- The library checks that the result is an integer from 1 to 5 and retries otherwise; the paper's analysis code accepted the result without a range check.
- The library's vLLM backend refuses a prompt longer than the model's context with an error; the paper's analysis code returned an empty string for it.

## Literature

Gilardi et al. (2023); Ziems et al. (2024). Full entries are in [References](references.md).
