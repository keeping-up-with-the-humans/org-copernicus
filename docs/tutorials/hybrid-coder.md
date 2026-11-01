# The hybrid coder

The hybrid coder joins the two ideas of the library. It first reads the verified passages that a two-stage run extracted. It then rebuilds each text from those passages and scores that evidence text with the similarity coder. The coder spends model calls once, in the two-stage run, and every later change of references or settings costs only a pass of the encoder.

## What you need

- The `embed` extra: `pip install ".[embed]"` in the package's root folder (see [Installation](../installation.md)).
- **No GPU and no API key.** The two-stage run uses the fake backend; the encoder is MiniLM on the CPU.
- A **two-stage** run store, because the hybrid coder needs the passages. A one-stage store has none, and the coder refuses it.
- Reference embeddings: the hand-written set of [the similarity coder](similarity-coder.md), built again below.

To use a real model for the two-stage run, replace the backend line of the code below. This is a snippet to read, not to run:

```python
backend = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
```

The hybrid step itself calls no model.

## How the hybrid coder works

1. For each text, the coder takes the verified, non-repeated passages of the chosen run and joins them with line breaks, in the order they occur in the text. This is the **evidence text**.
2. A text with fewer words than `min_words` (default 40), or a text with no verified passage, is used as it is. The `fallback` column says which: `short`, `no_evidence`, or empty when the passages were used. A text that is not short and whose extraction failed gets no such label: the coder refuses it (see the pitfalls).
3. The encoder embeds the evidence texts, and the similarity scorer compares them with the reference poles.

The hybrid coder has its own defaults, which reproduce the paper's hybrid coder: the vector of each whole evidence text (`long_text="pooled_embedding"`), and each pole reduced once over the references of all writers (`writer_pooling="pooled"`), with the `pca` reduction and the net polarity. The similarity coder's defaults differ (`writer_pooling="mean"`, `long_text="max_chunk"`). `cop.hybrid.HYBRID_DEFAULTS` holds the hybrid settings.

## Steps

### Step 1: code, embed the references

<!-- run embed -->
```python
from pathlib import Path

import pandas as pd

import copernicus as cop
from copernicus.references.reference_set import FRAME_COLUMNS

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
store = Path("two_stage.sqlite")
cop.code_texts(table, rubric, backend, store, cop.CodingSettings(runs=1))

encoder = cop.Encoder("minilm", device="cpu")
poles = {
    ("construct", "positive"): [
        "Thank you for the draft. The second table is unclear, so please add the units, because readers will otherwise misread it.",
        "I liked the summary. Section 3 would be stronger with one example, for instance the pilot data, and a short note on why it matters.",
    ],
    ("construct", "negative"): [
        "This is bad. Do it again.",
        "I did not like it. It is sloppy and you should know better.",
    ],
    ("specificity", "positive"): [
        "On page 4 the second paragraph needs a figure, for example the 2019 totals.",
        "Try moving the table in section 2 before the text that cites it.",
    ],
    ("specificity", "negative"): ["It needs work overall.", "Some parts are not good."],
    ("tone", "positive"): [
        "Thank you for the effort. The structure is strong, although the ending could be sharper.",
        "Nice work on the introduction; one small point on the method follows.",
    ],
    ("tone", "negative"): [
        "You were careless and lazy with this.",
        "I expected more from you. This is sloppy.",
    ],
    ("reasoning", "positive"): [
        "The claim needs the data from the survey, because otherwise the reader cannot check it.",
        "Instead of a single average, another option is to show both groups, since they differ.",
    ],
    ("reasoning", "negative"): ["Just change it.", "I do not like the result and that is all."],
}
rows = [
    {
        "rubric_id": rubric.rubric_id,
        "target": target,
        "polarity": polarity,
        "level": level,
        "style": "neutral",
        "writer_model": "hand",
        "text": text,
        "call_index": 0,
        "attempt": 0,
        "seed": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "temperature_applied": None,
        "top_p_applied": None,
        "prompt_hash": "hand",
        "prompt_version": "hand",
        "screened_out": False,
        "screen_reason": "",
    }
    for (target, polarity), pole_texts in poles.items()
    for level, text in zip(("top", "mid"), pole_texts, strict=True)
]
refset = cop.ReferenceSet.for_rubric(
    pd.DataFrame(rows, columns=list(FRAME_COLUMNS)), rubric, cop.GeneratorSettings()
)
refs = cop.embed_references(refset, encoder)
print(len(refs))
```

<!-- output -->
```text
16
```

### Step 2: look at the evidence texts

<!-- run -->
```python
evidence = cop.hybrid.evidence_texts(store, rubric, table, granularity="construct", min_words=5)
print(
    evidence.summary["n_texts"], evidence.summary["n_evidence"], evidence.summary["n_no_evidence"]
)
for text_id, fallback, text in list(zip(evidence.ids, evidence.fallback, evidence.texts))[2:3]:
    print(text_id, repr(fallback))
    print(text)
```

<!-- output -->
```text
54 46 8
t003 ''
Thank you for sending the draft.
However, some parts are weak and need more work.
```

Text `t003` is rebuilt from its two verified passages. Eight texts have no verified passage in run 1 and keep their own text. `min_words=5` lowers the default of 40 because the toy texts are short. With the default, most toy texts would count as `short` and the coder would embed them whole.

### Step 3: score

<!-- run embed -->
```python
result = cop.score_hybrid(store, table, rubric, encoder, refs, min_words=5)
frame = result.frame
print(frame.shape, sorted(frame["method"].unique()))
print(
    frame.loc[frame["target_id"] == "construct", "fallback"]
    .replace("", "passages")
    .value_counts()
    .to_dict()
)
```

<!-- output -->
```text
(216, 13) ['hybrid_construct']
{'passages': 46, 'no_evidence': 8}
```

The frame has the columns of the similarity coder plus `method` (`hybrid_construct` or `hybrid_attribute`) and `fallback`. `result.summary` describes the evidence texts, and `result.provenance` records the settings, the encoder of the evidence texts, the references, the rubric and the store.

### Step 4: score at attribute level

With `granularity="attribute"` the coder builds one evidence text per attribute from that attribute's markers, scores each against the attribute's own poles, and rolls the attribute scores up into one construct row.

<!-- run embed -->
```python
by_attribute = cop.score_hybrid(
    store, table, rubric, encoder, refs, granularity="attribute", min_words=5
).frame
print(by_attribute["target_id"].unique().tolist(), sorted(by_attribute["method"].unique()))
```

<!-- output -->
```text
['specificity', 'tone', 'reasoning', 'constructive_feedback_v1__rollup'] ['hybrid_attribute']
```

### Step 5: read a one-stage store

A one-stage store has no passages. The coder says so:

<!-- run -->
```python
one_stage = cop.CodingSettings(runs=1, method="one_stage")
cop.code_texts(table.subset(table.ids()[:3]), rubric, backend, "one_stage.sqlite", one_stage)
try:
    cop.hybrid.evidence_texts(
        "one_stage.sqlite", rubric, table.subset(table.ids()[:3]), granularity="construct"
    )
except cop.errors.ConfigurationError as error:
    print(error)
```

<!-- output -->
```text
the store one_stage.sqlite was coded with method 'one_stage'; the hybrid coder needs the verified passages of a 'two_stage' run
```

## On the command line

```bash
copernicus hybrid score --store two_stage.sqlite --texts toy_corpus.csv --rubric rubric.json \
    --references-embeddings refs_minilm --encoder minilm --min-words 5 --device cpu --out hybrid.csv
```

Add `--granularity attribute` for the attribute level and `--run-number` to read the passages of another run.

## What the output means

- **The hybrid score is a similarity score of the evidence.** It ranks the texts by how close their extracted passages lie to the positive pole compared with the negative pole.
- **A `no_evidence` fallback means the coder scored the whole text.** For those texts the hybrid score is a similarity score of the raw text, not of evidence. Read `fallback` before you interpret a low score.
- **The score is only as good as the passages.** If the two-stage run extracts the wrong passages, the hybrid coder scores the wrong passages.

## Pitfalls

- **The default `min_words` is 40.** Short texts are then scored whole. Set `min_words` to the length below which your texts carry too little to rebuild.
- **One run at a time.** `run_number` picks the run whose passages are used (default 1). The store may hold several runs; the hybrid coder does not combine them.
- **The store must match the table and the rubric.** The coder checks the rubric hash, that every text of the table has a result in the run, and that every text is the one the store coded under its id: a text changed since the coding is refused. It checks all this before anything is embedded.
- **A failed extraction is refused, not read as missing evidence.** A text whose extraction failed for an attribute in the run lacks that attribute's passages, so the coder refuses it before anything is embedded, unless the text is short and keeps its raw text anyway. Recode the failed cells with `cop.CodingSettings(retry_failed=True)` (`--retry-failed` on the command line), or score a table without those texts. The message names the first ten; `cop.export_scores(store)` lists all of them, as its rows of the run with status `extraction_failed`. A failed scoring leaves the passages in place, so such a text is scored as usual.
- **Use the same encoder for the references.** As for the similarity coder, a mismatch of encoder or model commit is refused.
- **The references must fit the rubric as it reads now.** The coder refuses references whose rubric hash differs from the rubric's, before it reads the store. After an edit of the rubric, regenerate the references, or build a hand-written set again with `ReferenceSet.for_rubric`.

## Next

[The NLP-ML baseline](nlp-ml-baseline.md) fits a supervised model on the same embeddings as a reference point.
