# The similarity coder

The similarity coder scores a text by how close its embedding lies to two sets of reference texts: texts that show the positive pole of the construct, and texts that show the negative pole. It makes no model call at scoring time and costs nothing to rerun. This tutorial embeds the toy corpus, builds a reference set by hand, scores the texts, and sweeps a few settings.

## What you need

- The `embed` extra: `pip install ".[embed]"` in the package's root folder (see [Installation](../installation.md)).
- **No GPU and no API key.** The encoder is MiniLM on the CPU; the first run downloads its weights (about 90 MB). The reference texts are written by hand here, so no writer model is called.
- A rubric and a text table: the toy ones that ship with the package.

For a real study, a writer model produces the reference texts. This is a snippet to read, not to run; it makes paid calls when the backend is a paid one:

```python
writer = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
refset = cop.generate_references(
    rubric, [writer], cop.GeneratorSettings(), resume=Path("refs.parquet")
)
```

## How the similarity coder works

1. A **reference set** holds short texts for the positive and the negative pole of the construct and of each attribute, at three intensity levels (`top`, `mid`, `low`), written by one or more writer models.
2. The encoder embeds the texts and the references into unit-length vectors.
3. For a pole, the cosines between a text and the references of that pole are reduced to one number per text (`mean`, `max` or `pca`).
4. The **net** score is the positive pole minus the negative pole.

## Steps

### Step 1: embed the texts

<!-- run embed -->
```python
import pandas as pd

import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
encoder = cop.Encoder("minilm", device="cpu")
texts = encoder.embed(table.frame["text"].tolist(), table.frame["text_id"].tolist())
texts.save("texts_minilm")
print(texts.vectors.shape, texts.encoder)
```

<!-- output -->
```text
(54, 384) minilm
```

`save` writes `texts_minilm.npz` and `texts_minilm.json`. The description records the encoder and its settings, and the commit of the model that was loaded, so a later run can tell whether two sets are comparable. It also records the hash of the arrays (`arrays_hash`): `cop.EmbeddingSet.load("texts_minilm")` reads the set back and refuses arrays that the description does not name.

### Step 2: estimate the cost of a real reference set

Before you pay for reference texts, count what a run would send. The estimate renders every prompt without calling a model.

<!-- run -->
```python
estimate = cop.references.estimate_tokens(rubric, cop.GeneratorSettings(), n_writers=2)
print(estimate.calls)
```

<!-- output -->
```text
48
```

There are 24 cells (the construct and three attributes, two poles, three levels) for each of two writers. The estimate also holds `prompt_tokens`, `completion_tokens` and, through `estimate.cost(price_input, price_output)`, a cost for the per-million-token prices that you read off your provider's price list. The [FAQ](../faq.md#how-much-will-a-run-cost) has the formula. On the command line, `copernicus references generate --dry-run` prints the same numbers and stops.

### Step 3: write a reference set by hand

A reference set is a table with one row per text. The block writes two texts for each pole and builds the table with the columns that a generated set has. `ReferenceSet.for_rubric` binds the table to the rubric: the set records the rubric's id and content hash, and the coders accept it only with a rubric of that content. A generated set is bound to the rubric it was written for in the same way, and `refset.bound_with` says which of the two happened: `for_rubric` here, `generated` for a generated set.

<!-- run embed -->
```python
from copernicus.references.reference_set import FRAME_COLUMNS

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
print(refset.summary()["balance"])
```

<!-- output -->
```text
{'balanced': True, 'complete': True, 'problems': []}
```

`summary` checks that every writer has as many positive as negative texts for each target (`balanced`) and that no pole is empty (`complete`). `refset.save("refs.csv")` writes the table and a description file beside it, with the rubric hash, the number of rows and the hash of the table (`table_hash`); `cop.ReferenceSet.load("refs.csv")` reads them back and refuses a table that the description does not name.

### Step 4: embed the references

<!-- run embed -->
```python
refs = cop.embed_references(refset, encoder)
refs.save("refs_minilm")
print(len(refs), refs.targets(), refs.writers())
```

<!-- output -->
```text
16 ['construct', 'reasoning', 'specificity', 'tone'] ['hand']
```

The reference embeddings carry the encoder identity of the set. `score_similarity` refuses texts and references from different encoders, or from different commits of one model.

They carry the rubric hash of the set as well. The texts express the rubric as it read when they were written, so every scorer refuses them with a rubric of other content, also one edited under the same ids:

<!-- run embed -->
```python
import dataclasses

edited = dataclasses.replace(rubric, concept_definition="An edited definition of the construct.")
try:
    cop.score_similarity(texts, refs, edited, cop.SimilaritySettings())
except cop.errors.ConfigurationError as error:
    print(error)
```

<!-- output -->
```text
the reference embeddings were generated for rubric constructive_feedback_v1 with hash 4bdee2efb58b0003, but the rubric now has hash d51920267b611de1: the references describe the earlier definition; regenerate them (or rebuild a hand-written set with ReferenceSet.for_rubric)
```

After an edit of the rubric, regenerate a generated set. For a hand-written set, check that its texts still fit the edited rubric, build it again with `ReferenceSet.for_rubric`, and embed it again.

### Step 5: score the texts

<!-- run embed -->
```python
scores = cop.score_similarity(texts, refs, rubric, cop.SimilaritySettings())
print(scores.shape)
print(scores.columns.tolist())
print(sorted(scores["target_id"].unique()))
```

<!-- output -->
```text
(216, 11)
['text_id', 'target_level', 'target_id', 'polarity', 'reduction', 'writer_pooling', 'long_text', 'style', 'encoder', 'score', 'n_references']
['construct', 'reasoning', 'specificity', 'tone']
```

The table has one row per text and target: 54 texts and four targets (the construct and the three attributes). The columns after `target_id` repeat the settings, so a table of scores always says how it was made. `n_references` is the size of the pole: four here, two positive and two negative.

At target level `attribute` the coder scores the attributes and rolls them up into one construct score, the row `<rubric_id>__rollup`:

<!-- run embed -->
```python
rolled = cop.score_similarity(texts, refs, rubric, cop.SimilaritySettings(target_level="attribute"))
print(rolled["target_id"].unique().tolist())
```

<!-- output -->
```text
['specificity', 'tone', 'reasoning', 'constructive_feedback_v1__rollup']
```

The `construct` row (a score against the poles of the construct) and the roll-up row (the combination of the attribute scores) are different estimators. The validity panel treats them as different targets.

### Step 6: sweep the settings

`SimilaritySettings` has five axes.

| Axis | Values | Meaning |
|------|--------|---------|
| `polarity` | `net`, `positive`, `negative` | Positive pole minus negative pole, or one pole alone |
| `reduction` | `mean`, `max`, `pca` | How the cosines to the references of a pole become one number |
| `writer_pooling` | `mean`, `max`, `pooled` | Reduce per writer and pool the writers with `mean` or `max`, or reduce all references of the pole at once |
| `long_text` | `max_chunk`, `mean_chunk`, `pooled_embedding`, `first_window` | How the chunks of a text longer than the encoder's window enter |
| `target_level` | `construct`, `attribute` | Score the construct directly, or score the attributes and roll them up |

The defaults are the paper's configuration of the similarity coder. `run_grid` scores one or more `(encoder, style)` keys under many settings and stacks the results. The block compares the three reductions.

<!-- run embed -->
```python
settings = [cop.SimilaritySettings(reduction=name) for name in ("mean", "max", "pca")]
grid = cop.run_grid({("minilm", "neutral"): (texts, refs)}, rubric, settings)
construct = grid[grid["target_id"] == "construct"]
gold = table.frame.set_index("text_id")["gold_constructiveness"]
for name, part in construct.groupby("reduction", sort=False):
    result = cop.association(part["score"].to_numpy(), gold.loc[part["text_id"]].to_numpy())
    print(name, round(result.rank_validity, 1))
```

`cop.PAPER_GRID` holds the 144 settings of the paper's grid (3 polarities, 3 reductions, 4 long-text modes, 2 writer poolings, 2 target levels). `cop.run_grid(inputs, rubric)` runs all of them.

## On the command line

```bash
copernicus embed texts --texts toy_corpus.csv --encoder minilm --out texts_minilm --group-column group_id --device cpu
copernicus references generate --rubric rubric.json --out refs.parquet --model gpt-4.1-mini --resume refs_progress.parquet
copernicus embed references --references refs.parquet --encoder minilm --out refs_minilm --device cpu
copernicus similarity score --texts-embeddings texts_minilm --references-embeddings refs_minilm \
    --rubric rubric.json --out similarity.csv
```

Add `--grid paper` to `similarity score` for the paper's 144 settings. A description file `similarity.csv.json` records the settings, the encoders, the rubric, the rubric hash of the references and the number of rows.

## What the output means

- **A higher net score means closer to the positive pole than to the negative pole.** The scale depends on the reduction. A `mean` or `max` score is a difference of cosines. A `pca` score is the first principal component of the standardised cosines, signed so that higher means more similar to the pole.
- **A `pca` score depends on the texts you score together.** The standardisation runs over the texts of the call, so `pca` scores of two corpora are not on one scale. `mean` and `max` scores do not depend on the other texts.
- **The score ranks texts; it has no unit.** Use the [validity panel](validity-panel.md) to see how well it ranks against gold values, and the corrections to put it on the gold scale.

## Pitfalls

- **References belong to one content of the rubric.** Edit the rubric, even under the same ids, and the coders refuse its references. Regenerate them, or build a hand-written set again with `ReferenceSet.for_rubric`.
- **Texts and references must come from one encoder and one model commit.** Embed both with the same `Encoder`. A model repository gets a new commit when only its documentation changes, so sets embedded a month apart can disagree; pin a commit (pass `dataclasses.replace(cop.ENCODERS["minilm"], revision="<commit>")` to `Encoder`, or add `--revision <commit>` on the command line), or re-embed.
- **Write balanced, complete references.** Every pole needs texts, and a writer's positive and negative texts for a target should match in number. `summary()["balance"]` tells you.
- **A reference should fit in one window of the encoder.** A longer reference is represented by the token-weighted mean of its chunks, and the library warns.
- **Do not copy the rubric's wording into the references.** Generated sets pass a leakage screen that flags such texts. A hand-written set has no screen.
- **One writer for both poles of a target.** A target whose two poles come from different writers is scored with a warning.

## Next

[The hybrid coder](hybrid-coder.md) scores the verified passages of a two-stage run with the same references.
