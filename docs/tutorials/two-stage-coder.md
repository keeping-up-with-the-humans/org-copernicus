# The two-stage coder

The two-stage LLM coder is the library's main coder. It reads each text against the rubric one attribute at a time, in two calls to a model. This tutorial codes the toy corpus into a run store, reads the results, resumes and shards a run, and shows the one line that switches to a real model. The [one-stage LLM coder](#the-one-stage-coder) is at the end.

## What you need

- The base install. No GPU, no API key.
- A rubric and a text table. The tutorial uses the toy ones that ship with the package.
- **A backend.** The tutorial uses `FakeBackend`, which answers from a table of cue words. It calls no model and costs nothing. Its scores show how the coder works. They are not the scores of a real model.

To code with a real model, build `OpenAICompatibleBackend` instead of `FakeBackend`. This is a snippet to read, not to run:

```python
backend = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
# A server of your own that speaks the same protocol:
# backend = cop.OpenAICompatibleBackend(model="my-model", base_url="http://localhost:8000/v1")
# A model on your own GPU, through vLLM (needs the vllm extra):
# backend = cop.VLLMBackend(...)
```

Nothing else in this tutorial changes. Set `concurrency` in `CodingSettings` to send several attribute evaluations at once to an OpenAI-compatible server; keep it at 1 for vLLM.

## How the coder works

For each text, run and attribute the coder makes up to two kinds of call.

1. **Stage 1, extraction.** The model gets the text and the markers of the attribute. It returns verbatim passages for each marker, or none.
2. **Verification.** The library matches every returned passage against the text. An exact substring counts as verified. A near-exact alignment counts as verified for passages of at least three words, at a threshold of 88 on a 0 to 100 scale. Everything else is recorded as unverified and never reaches stage 2.
3. **Stage 2, scoring.** The model gets the verified passages and the level descriptions and returns an integer from 1 to 5 with a short explanation.

A text shorter than 20 characters is not coded. An attribute with no verified passage gets no stage 2 call and receives the score 1 (the [FAQ](../faq.md#why-does-an-attribute-without-a-passage-score-1) says why, and how to turn it off). The wording of the prompts rotates by a seeded rule over several role, task and level variants, and the marker order is shuffled, so repeated runs differ in wording.

## Steps

### Step 1: code the corpus

<!-- run -->
```python
import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
summary = cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
print(summary.n_texts, summary.n_runs, summary.n_attribute_calls)
print(dict(summary.status_counts))
print(len(backend.calls))
```

<!-- output -->
```text
54 2 324
{'ok': 220, 'no_evidence': 104, 'too_short': 0, 'extraction_failed': 0, 'scoring_failed': 0}
544
```

The summary counts what this call did. There are 324 attribute calls (54 texts, 2 runs, 3 attributes). The backend received 544 requests: 324 extraction calls and 220 scoring calls, one for each attribute call with a verified passage. `CodingSettings(runs=2)` codes every text twice. With several runs the temperature `"auto"` resolves to 0.3, which lets the runs differ. With one run it resolves to 0.0.

### Step 2: read the scores

<!-- run -->
```python
scores = cop.export_scores("run.sqlite")
print(
    scores[(scores["text_id"] == "t003") & (scores["run_number"] == 1)][
        ["attribute_id", "score", "status", "explanation"]
    ].to_string(index=False)
)
```

<!-- output -->
```text
attribute_id  score      status           explanation
 specificity      1 no_evidence no_markers_evaluable
        tone      2          ok            2 passages
   reasoning      1 no_evidence no_markers_evaluable
```

`export_scores` returns one row per text, run and attribute, with the provenance of the call: the wording variants, the model, the temperature and the seed. The other exports average or reshape the same data:

| Function | Rows | Use |
|----------|------|-----|
| `export_scores` | text, run, attribute | The long table. The validity panel reads it. |
| `export_construct_scores` | text, run | The construct score: the mean of the attribute scores that exist |
| `export_text_scores` | text | The scores averaged over runs, one column per attribute |
| `export_evidence` | passage | Every passage the extraction stage returned, with its verification |
| `export_provenance` | key | The settings of the run |

### Step 3: read the evidence

<!-- run -->
```python
evidence = cop.export_evidence("run.sqlite")
one = evidence[(evidence["text_id"] == "t003") & (evidence["run_number"] == 1)]
print(
    one[
        ["attribute_id", "marker_id", "quote_norm", "verified", "match_score", "used_in_stage2"]
    ].to_string(index=False)
)
```

<!-- output -->
```text
attribute_id   marker_id                                       quote_norm  verified  match_score  used_in_stage2
        tone tone_respect                 Thank you for sending the draft.         1        100.0               1
        tone tone_balance However, some parts are weak and need more work.         1        100.0               1
```

Every score has its passages. `verified` is 1 for an exact match, 2 for a near-exact match and 0 for a passage that is not in the text. `match_score` is the alignment score, 100 for an exact match. `used_in_stage2` is 1 for the passages that reached the scoring stage; a passage that repeats an earlier one does not.

### Step 4: resume and shard

Run the same call again. The coder finds every cell in the store and calls nothing.

<!-- run -->
```python
again = cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=2))
print(again.n_texts, again.n_attribute_calls, again.resumed)
```

<!-- output -->
```text
0 0 108
```

The 108 resumed cells are the 54 texts times 2 runs. An interrupted run resumes in the same way: the coder writes the store every `save_every` texts (10 by default) and at the end, so a stop loses at most the current batch.

The store binds each text id to the text it was coded on: every row records the hash of its text, the `text_hash` column of `export_scores`. If a text has changed under its id since it was coded, the call stops with a `StoreError` that names the changed ids, before it calls the backend. Code the changed texts into a new store, or restore the texts. Runs of whitespace collapse to one space before hashing, so a change in the amount of whitespace between words is no change; inserting or removing a space between letters is one: `ab` and `a b` are two texts.

Cells with a failed attribute (`extraction_failed` or `scoring_failed`) stay in the store and are not coded again by a plain rerun. The failure check at the end of every call over their texts and runs counts them, so a rerun of a failed run stops with the same error instead of passing as a success. `CodingSettings(retry_failed=True)` (`--retry-failed` on the command line) codes those cells again and replaces their rows and passages; the summary counts them in `retried`. A retry codes a cell with the same prompts, seeds and temperatures, so it repairs failures of the backend, such as answers that came back empty or malformed on every attempt, and not a model that fails the same text every time. Such a text fails again; it needs a changed setting of the coder, such as a larger `stage1_max_new_tokens` for an answer that is cut off, or another model, and with either a new store, since both belong to the identity of a store. A request that fails at the transport level even after the backend's own retries, such as a timeout, is not stored at all: it stops the run with a `BackendError`, and a plain rerun codes its cell.

To code in parts, give each part a text range and its own store, then merge. `text_range` counts positions in text-id order and does not belong to the identity of a store.

<!-- run -->
```python
first = cop.code_texts(
    table, rubric, backend, "part_a.sqlite", cop.CodingSettings(runs=2, text_range=(0, 27))
)
second = cop.code_texts(
    table, rubric, backend, "part_b.sqlite", cop.CodingSettings(runs=2, text_range=(27, 54))
)
merged = cop.merge_stores(["part_a.sqlite", "part_b.sqlite"], "merged.sqlite")
print(first.n_texts, second.n_texts, merged["results"], len(cop.export_scores("merged.sqlite")))
```

<!-- output -->
```text
27 27 108 324
```

`merge_stores` refuses stores of different runs: they must agree on the method, model, temperature, seed, rubric and settings. It also refuses stores that hold different texts under one text id, so only parts coded on the same texts merge. It reports conflicting cells in its result.

### Step 5: leave texts without evidence unscored

By default an attribute without a verified passage scores 1. To leave it missing instead, set `no_evidence_score=None`.

<!-- run -->
```python
config = cop.TwoStageConfig(no_evidence_score=None)
cop.code_texts(
    table, rubric, backend, "missing.sqlite", cop.CodingSettings(runs=1, two_stage=config)
)
missing = cop.export_scores("missing.sqlite")
print(missing["status"].value_counts().to_dict(), int(missing["score"].isna().sum()))
```

<!-- output -->
```text
{'ok': 110, 'no_evidence': 52} 52
```

The `no_evidence` rows now have a missing score. The validity panel and the exports leave missing scores out of every mean.

### Step 6: read the provenance

<!-- run -->
```python
provenance = cop.export_provenance("run.sqlite")
print(
    {
        key: provenance[key]
        for key in ("method", "model_name", "temperature", "seed", "verify_threshold")
    }
)
```

<!-- output -->
```text
{'method': 'two_stage', 'model_name': 'fake', 'temperature': '0.3', 'seed': '42', 'verify_threshold': '88.0'}
```

The store records the backend (for an OpenAI-compatible server its endpoint and a digest of the endpoint's path, never a key), the prompt template hashes, the rubric hash, the verifier version and the library version, and every row records the hash of the text it was coded on. A store continues only with the same identity and the same texts.

## The one-stage coder

The one-stage LLM coder scores an attribute in a single call, without the extraction stage. It is cheaper and records no passages.

<!-- run -->
```python
small = table.subset(table.ids()[:3])
one_stage = cop.CodingSettings(runs=1, method="one_stage")
summary = cop.code_texts(small, rubric, backend, "one_stage.sqlite", one_stage)
print(summary.n_attribute_calls, len(cop.export_evidence("one_stage.sqlite")))
```

<!-- output -->
```text
9 0
```

The evidence table is empty. The [hybrid coder](hybrid-coder.md) needs evidence and refuses a one-stage store.

## On the command line

```bash
copernicus code two-stage --texts toy_corpus.csv --rubric rubric.json --out run.sqlite \
    --backend fake --group-column group_id
copernicus code two-stage --texts texts.csv --rubric rubric.json --out run.sqlite \
    --model gpt-4.1-mini --runs 2 --concurrency 8 --range 0:500
copernicus store merge part_a.sqlite part_b.sqlite --out merged.sqlite
copernicus store export run.sqlite --what scores --out scores.csv
```

`--backend fake` works only with the toy rubric, unless you pass a script of answers with `--fake-script`. `--what` takes `scores`, `construct`, `text`, `evidence` or `provenance`. `--range A:B` is the command-line form of `text_range`.

## What the output means

- **`status`** tells you what happened to each attribute call: `ok`, `no_evidence`, `too_short`, `extraction_failed` or `scoring_failed`. Only `ok` and `no_evidence` carry a score (the latter under the default setting).
- **The failure ceiling protects a run.** If more than 10 percent of the attribute results end in `extraction_failed` or `scoring_failed`, the coder stops with a `RunFailureCeilingError` after writing what it has. During the run it judges the calls of the current invocation after each batch, so a failing backend stops early even on a rerun over a large store. At the end it judges all results of the requested texts and runs, those already in the store included, so the error comes back on a rerun until `retry_failed=True` codes the failed cells again; raise `max_failure_share` to accept them, or code into a new store.
- **Scores from different wording variants differ on purpose.** Repeated runs use different role, task and level wording. The spread across runs is the noise that the validity panel reports.

## Pitfalls

- **One run, then more.** Under `temperature="auto"`, a store of one run (temperature 0.0) cannot be extended to two runs (0.3). Pass an explicit `temperature` in both calls to extend a store.
- **A fake backend's scores say nothing about a real model.** Always check a real model's scores with the [validity panel](validity-panel.md).
- **Texts shorter than 20 characters are skipped** (`too_short`). Change `min_chars` in `TwoStageConfig` if your texts are short.
- **Ids stay strings.** The table reader keeps the ids as written (`"007"` stays `"007"`) and the library compares them as strings, so `"007"` and `"7"` are different ids.
- **A changed rubric, model, setting or text needs a new store.** The store refuses to mix runs and to keep results of an earlier text under an id. A store whose tables lack a column this version of the library writes, such as `text_hash`, or hold one it does not write, is refused too, whether an earlier version made it or someone changed it by hand: code into a new store.

## Next

[The similarity coder](similarity-coder.md) scores the same texts from embeddings, with no model calls.
