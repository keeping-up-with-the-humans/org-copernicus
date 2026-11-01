# Two-stage LLM coder

The two-stage LLM coder scores one attribute of one text in two calls to a language model, the way a trained coder works with a codebook: first find the evidence, then judge it. The first call, the extraction stage, receives the text, the attribute's markers and the level descriptions of each marker, and returns for every marker the passages of the text that bear on it, copied word for word, with no score and no comment. The library checks every returned passage against the text and keeps only the passages it finds there verbatim or nearly verbatim. The second call, the scoring stage, receives the markers that kept a passage, the attribute's rubric built from those markers, and the kept passages, in the model's wording, as one list in the order in which they occur in the text. It does not see the text and is not told which marker a passage was found for. It returns a score from 1 to 5 with a short explanation.

The split makes every score auditable: a reader can open the passages behind it, and the score rests only on passages that match the text verbatim or nearly so. Reliable content analysis rests on explicit coding instructions grounded in theory (Krippendorff 2018), and the rubric's levels describe observable evidence in the manner of behaviourally anchored rating scales (Smith and Kendall 1963). A rubric names the literature it rests on for the construct as a whole, for each attribute, or both; the sources document the rubric and enter its content hash, and no prompt shows them. Asking for intermediate steps before an answer is related to chain-of-thought prompting (Wei et al. 2022), and scoring a text against a rubric with calibrated outputs to the approach of Hashemi et al. (2024). Language models code texts at scale (Gilardi et al. 2023, Ziems et al. 2024), but their scores shift with the wording of the prompt and the choice of model (Carlson and Burbano 2026). The coder therefore rotates the wording of its prompts and records it, so that this variation shows up as noise in the [validity panel](validity-panel.md) instead of hiding in one fixed prompt.

## The algorithm as the library implements it

For each text, run and attribute:

1. **Length check.** The text table collapses whitespace. A text shorter than `min_chars` (20) characters is not sent to the model; every attribute gets the status `too_short` and no score.
2. **Wording and marker order.** For each text and run, `random.Random(seed + text_key * 1000 + run_number)` draws one role, one task and one level wording from the shipped pools (three role, three task and four level wordings), and the same seed shuffles the marker order. A string text id becomes the 31-bit integer `text_key` through a BLAKE2b hash. The sampling seed of a run is `seed + run_number`.
3. **Extraction.** The system prompt holds the role and task wording, the attribute, its markers in the shuffled order, each marker's level descriptions from level 5 down to level 1, and the glossary. The user message is `Text to analyze:` followed by the text. The answer must be a JSON object with one entry per marker, keyed by its name or id, each `{"evaluable": true or false, "quotes": [...]}`, with `evaluable` true exactly when there is a quote. A wrapping `marker_analysis` object is accepted. An unusable answer is retried, up to `max_retries` (3) attempts in all, with the seed plus the attempt number and a temperature that rises by 0.3 per retry, capped at 1.0. After the last failure the attribute gets the status `extraction_failed`, no score and no scoring call.
4. **Verification.** The text and every passage are normalised: Unicode NFC, curly quotes and dashes unified, whitespace collapsed, case folded. An exact substring is verified (`verified = 1`, match score 100). Otherwise a passage of at least `MIN_FUZZY_WORDS` (3) words is verified when its best local alignment with the text scores at least `VERIFY_THRESHOLD` (88 of 100) (`verified = 2`); a shorter passage must match exactly, and a passage longer than the text is judged on the ratio of the whole strings. Every passage is stored with its score and its position in the text, verified or not.
5. **Deduplication.** Within a marker, a passage equal to an earlier one after normalisation is a repeat and is dropped. Across markers, each distinct passage is sent once and is credited to the marker under which it starts first in the text (ties by marker order). A marker without a kept passage is not evaluable.
6. **Scoring.** If no marker kept a passage, there is no scoring call: the attribute gets the status `no_evidence` and the score `no_evidence_score` (1 by default). Otherwise the system prompt holds the role and task wording, the level wording, the markers to rate and the glossary. The user message holds the rubric as five lines `Level N: ...`, each joining the level descriptions of the markers to rate, and then the kept passages as a list in text order, without marker labels. The answer must be `{"result": 1 to 5, "explanation": "..."}`; a whole-number float or a digit string counts, a boolean does not. An unusable answer is retried with a new seed and the same temperature. After the last failure the status is `scoring_failed` and the score is missing, never 0.
7. **Scores of a text.** The construct score of a run is the mean of the attribute scores that exist. The text score of an attribute, and of the construct, is the mean over runs.

The runner around the coder resolves the temperature `"auto"` to 0.0 with one run and 0.3 with several, sends `top_p` 0.9 with a non-zero temperature to models that accept it, writes every result to a resumable run store with its provenance, and stops with an error when more than 10 percent of the attribute results end in `extraction_failed` or `scoring_failed`. After each batch, once 30 calls and one full batch are done, it judges the calls of the current invocation, so a failing backend stops early however many good results the store holds. At the end, also when a rerun codes nothing, it judges all attribute results of the requested texts and runs, the results already in the store included. Failed cells are coded again only under `retry_failed`.

The identity of a run store, which a resumed or merged run must match, is the method, model, temperature, seed and rubric hash, every setting of the coder config, the hashes of the prompt templates and of the wording, the verifier version, the marker shuffling, and the provenance of the backend. For an OpenAI-compatible server the provenance holds the endpoint as scheme, host and port, and a digest of the endpoint's path without its trailing slashes (`endpoint_path_digest`), so two deployments on one host are two identities while no user name, password, query or path is stored. Every result row also records the text hash of the text it was coded on: the SHA-256 of the text as the coder receives it, after the text table has collapsed its whitespace. It binds each text id of the store to its text, so a rerun refuses a text that changed under its id before any call, and the merge refuses stores that hold different texts under one id, a check it runs again inside the transaction that copies each store. Each row records, too, what the backend reports it applied: the temperature and `top_p` of the scoring calls (`temperature_applied`, `top_p_applied`) and, for each attribute, the temperature of the extraction attempt whose answer was accepted (`stage1_temperature_applied`), since the requested temperature rises by `retry_temperature_step` (0.3) with every retry, up to `retry_temperature_cap` (1.0). A value is missing when the backend reports none, when no scoring call was made, or, for the extraction, when no attempt was accepted; the one-stage coder records its single call as the scoring call.

## Assumptions and what it corrects

The coder produces scores; it corrects nothing. It assumes that the construct can be read from passages of the text, that the rubric's markers and levels describe that evidence, and that the model copies passages faithfully. Verification enforces the last assumption only in part. Wholly invented passages are stopped, but a near-verbatim passage of three or more words can differ from the text by small insertions, even a negation, and still pass; and the scoring stage receives the model's wording of a passage, not the matched part of the text.

A text without evidence for an attribute scores 1 by default, the bottom of the scale. Terse texts therefore score low, which can add level displacement and covariate-linked error with text length. Set `no_evidence_score=None` to leave such attributes missing, or filter on `status == "ok"`.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `TwoStageConfig.max_retries` | 3 | Attempts per stage, the first included |
| `TwoStageConfig.verify_threshold` | 88.0 | Alignment score a near-exact passage needs to count as verified |
| `TwoStageConfig.min_fuzzy_words` | 3 | Shortest passage, in words, that may match approximately |
| `TwoStageConfig.min_chars` | 20 | Shorter texts are not coded |
| `TwoStageConfig.no_evidence_score` | 1 | Score of an attribute without a kept passage; `None` leaves it missing |
| `TwoStageConfig.stage2_evidence` | `"unlabelled"` | `"labelled"` groups the passages under their markers |
| `TwoStageConfig.stage1_max_new_tokens`, `stage2_max_new_tokens` | 8192, 2048 | Generation budgets of the two calls |
| `TwoStageConfig.retry_temperature_step`, `retry_temperature_cap` | 0.3, 1.0 | Temperature added per extraction retry, and its cap |
| `CodingSettings.runs` | 2 | Repeated codings of every text |
| `CodingSettings.temperature` | `"auto"` | 0.0 with one run, 0.3 with several, or a number from 0 to 2 |
| `CodingSettings.seed` | 42 | Seed of the wording rotation and the sampling |
| `CodingSettings.max_failure_share` | 0.10 | Share of failed attribute results that stops the run: of the invocation's calls after each batch, of the requested texts and runs with the stored results at the end |
| `CodingSettings.retry_failed` | `False` | `True` codes the stored cells with a failed attribute again and replaces them |

## Example

The block checks the verification rule on one toy text, which opens with "Thank you for sending the draft." An exact passage is verified; a near-exact one of at least three words is verified by alignment, even when it inserts a "not" that the text does not hold; a paraphrase scores below 88; and a two-word passage cannot be verified by alignment even at a score above 88.

<!-- run -->
```python
import copernicus as cop
from copernicus.coders import MIN_FUZZY_WORDS, VERIFY_THRESHOLD, match_quote, normalize

table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
text = normalize(table.frame.set_index("text_id").loc["t003", "text"])
print(VERIFY_THRESHOLD, MIN_FUZZY_WORDS)
for passage in (
    "Thank you for sending the draft.",
    "Thank you for sending this draft.",
    "Thank you for not sending the draft.",
    "Thanks for sending the draft.",
    "the drafts",
):
    match = match_quote(passage, text)
    print(match.verified, round(match.score))
```

<!-- output -->
```text
88.0 3
1 100
2 95
2 94
0 86
0 90
```

The coder itself runs through `code_texts`. With the fake backend, which answers from cue words, it codes the toy corpus without a model:

<!-- run -->
```python
rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
summary = cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
print(summary.n_attribute_calls, summary.status_counts["ok"], summary.status_counts["no_evidence"])
```

<!-- output -->
```text
324 220 104
```

[The two-stage coder](../tutorials/two-stage-coder.md) reads the scores, the passages and the provenance, resumes and shards a run, and shows the one line that switches to a real model.

## Where the library departs from the paper's analysis code

- The library's vLLM backend refuses a prompt longer than the model's context with an error; the paper's analysis code returned an empty string for it.

## Literature

Carlson and Burbano (2026); Gilardi et al. (2023); Hashemi et al. (2024); Krippendorff (2018); Smith and Kendall (1963); Wei et al. (2022); Ziems et al. (2024). Full entries are in [References](references.md).
