# Getting started

This page takes a small corpus through every step of the library: coding, embedding, similarity and hybrid scores, the validity panel, the NLP-ML baseline, a human sample, and one corrected estimate. Read it from top to bottom and run the code blocks in order in one Python session.

## What you need

- The library with the `embed` extra: `pip install ".[embed]"` in the package's root folder. See [Installation](installation.md).
- **No GPU and no API key.** The LLM coder runs against a fake backend that answers from a table of cue words. The embedding steps use the small encoder MiniLM on the CPU. The first run downloads MiniLM once (about 90 MB).
- Time: the code blocks together run in about ten seconds on a laptop CPU once MiniLM is cached.

The corpus is the toy corpus that ships with the package: 54 short feedback texts written by 18 people (three texts each), with a hand-assigned gold value for the construct and for each of its three attributes. The toy rubric describes the construct "constructiveness of written feedback". The fake backend's scores are not scores of a real model. They let you run every step and read every output.

## Step 1: load the rubric and the texts

A rubric file describes the construct. A text table holds the texts, one row per text, with a group id. Texts of the same group stay together in every split and every bootstrap draw of the later steps. Name the group column with `group_id=`. A table with neither that argument nor a column called `group_id` makes every text its own group, with a warning: the human sample then draws texts independently, and the bootstrap intervals and the standard errors of the prediction-powered estimators are not clustered.

<!-- run -->
```python
import pandas as pd
import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
print(len(table), rubric.rubric_id, [a.attribute_id for a in rubric.attributes])
```

<!-- output -->
```text
54 constructive_feedback_v1 ['specificity', 'tone', 'reasoning']
```

## Step 2: code the texts with the two-stage LLM coder

The two-stage LLM coder works on one attribute of one text at a time. Stage 1 asks the model for verbatim passages that show each marker. The library checks every passage against the text and keeps only the verified ones. Stage 2 asks the model for a score from 1 to 5 from the verified passages. The coder writes every result into a run store, a SQLite file that makes the run resumable.

<!-- run -->
```python
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
summary = cop.code_texts(table, rubric, backend, "two_stage.sqlite", cop.CodingSettings(runs=2))
print(summary.n_attribute_calls, dict(summary.status_counts))
```

<!-- output -->
```text
324 {'ok': 220, 'no_evidence': 104, 'too_short': 0, 'extraction_failed': 0, 'scoring_failed': 0}
```

There are 324 attribute calls: 54 texts, 2 runs, 3 attributes. A status of `ok` means the attribute was scored from verified passages. `no_evidence` means no marker kept a verified passage, so the attribute gets the score 1 (see the [FAQ](faq.md)). Run the block again and the coder finds every cell in the store and makes no call.

To use a real model, replace the backend. This is a snippet to read, not to run:

```python
backend = cop.OpenAICompatibleBackend(model="gpt-4.1-mini")  # reads OPENAI_API_KEY
```

Read the results as tables:

<!-- run -->
```python
scores = cop.export_scores("two_stage.sqlite")
text_scores = cop.export_text_scores("two_stage.sqlite")
print(text_scores.head(3).round(2).to_string(index=False))
```

<!-- output -->
```text
text_id  construct_score  specificity  tone  reasoning  n_runs
   t001             1.00          1.0   1.0        1.0       2
   t002             1.00          1.0   1.0        1.0       2
   t003             1.33          1.0   2.0        1.0       2
```

`export_scores` has one row per text, run and attribute. `export_text_scores` averages over runs and adds the construct score, the mean of the attribute scores.

## Step 3: embed the texts with MiniLM

The similarity coder, the hybrid coder and the NLP-ML baseline work on embeddings. The encoder turns each text into a unit-length vector.

<!-- run embed -->
```python
encoder = cop.Encoder("minilm", device="cpu")
texts = encoder.embed(table.frame["text"].tolist(), table.frame["text_id"].tolist())
print(texts.vectors.shape)
```

<!-- output -->
```text
(54, 384)
```

## Step 4: write reference texts, then score with the similarity coder and the hybrid coder

Reference texts show the two poles of the construct and of each attribute. In a real study a writer model produces them (see [The similarity coder](tutorials/similarity-coder.md)). Here you write a few by hand, two per pole. The block builds a reference set from them and embeds it. `ReferenceSet.for_rubric` binds the set to the content of the rubric, so the coders refuse it once the rubric is edited.

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
refs = cop.embed_references(refset, encoder)
print(len(refs), refs.targets())
```

<!-- output -->
```text
16 ['construct', 'reasoning', 'specificity', 'tone']
```

The similarity coder scores each text by its similarity to the two poles. The hybrid coder first rebuilds each text from the verified passages of the two-stage run and scores that evidence text the same way.

<!-- run embed -->
```python
from pathlib import Path

similarity = cop.score_similarity(texts, refs, rubric, cop.SimilaritySettings())
hybrid = cop.score_hybrid(Path("two_stage.sqlite"), table, rubric, encoder, refs, min_words=5)
print(similarity.shape, hybrid.frame.shape)
print(hybrid.summary["n_no_evidence"], hybrid.summary["n_evidence"])
```

<!-- output -->
```text
(216, 11) (216, 13)
8 46
```

Each frame has one row per text and target: the construct and the three attributes, so 54 times 4 rows. The hybrid frame adds the columns `method` and `fallback`. The argument `min_words=5` lowers the default of 40 words, because the toy texts are short: a text with fewer words than this is embedded as it is instead of through its passages. Eight texts have no verified passage in run 1 and fall back to their own text; the other 46 are scored through their passages.

## Step 5: check the scores with the validity panel

The validity panel compares a score with a gold value. Here the gold value is the hand-assigned `gold_constructiveness`. The next block builds one panel per coder and prints the rank validity of the construct score and the width of its 95 percent interval.

<!-- run embed -->
```python
settings = cop.PanelSettings(n_boot=200)
coders = {"two-stage": scores, "similarity": similarity, "hybrid": hybrid.frame}
panels = {
    name: cop.validity_panel(
        frame, table.frame, table, targets={"construct": "gold_constructiveness"}, settings=settings
    )
    for name, frame in coders.items()
}
for name, panel in panels.items():
    row = panel.frame.iloc[0]
    width = row["rank_validity_high"] - row["rank_validity_low"]
    print(name, round(float(row["rank_validity"]), 1), round(float(width), 1))
```

<!-- output -->
```text
two-stage 0.9 0.1
similarity 0.8 0.2
hybrid 0.7 0.2
```

Rank validity is the absolute Spearman correlation between the score and the gold value. The toy corpus gives a rank validity of about 0.9 for the two-stage coder, 0.8 for the similarity coder and 0.7 for the hybrid coder, with intervals one or two tenths wide. The intervals come from a bootstrap that resamples whole groups. The numbers come from cue words and hand-written references, so they do not say how the coders would do on your texts.

Each panel also writes a one-page report. The next block prints the report of the hybrid coder's panel:

<!-- run embed -->
```python
print(panels["hybrid"].report())
```

The report adds the four score errors: level displacement, calibration shortfall, covariate-linked error and noise. [The validity panel](tutorials/validity-panel.md) explains each one.

## Step 6: set a reference point with the NLP-ML baseline

The NLP-ML baseline fits a ridge regression on the embeddings to predict the gold value, with grouped folds, and reports how well it ranks held-out texts. It tells you what a simple supervised model reaches at each training size. The toy corpus has only 18 groups, so the block uses a size of 6 units and a lower floor than the default of 50 units.

<!-- run embed -->
```python
baseline = cop.run_ridge_baseline(
    {"minilm": texts},
    table.frame,
    table,
    cop.BaselineSettings(sizes=(6, "full"), seeds=(42,), k_folds=3, floor_units=5),
    gold_column="gold_constructiveness",
)
print(cop.curves(baseline)[["size", "rank_validity"]].round(1).to_string(index=False))
```

<!-- output -->
```text
size  rank_validity
   6            0.9
full            0.9
```

## Step 7: draw a human sample and correct one estimate

Machine scores carry bias that a human-coded sample can correct. The library draws the sample by whole groups and records the probability that each text was included. You then code the drawn texts by hand and give the values back. Here the toy corpus stands in for the human coder: its gold column holds the human values of the drawn texts.

<!-- run embed -->
```python
text_scores = cop.export_text_scores("two_stage.sqlite")
draw = cop.draw_human_sample(table, n=18, seed=1)
print(f"{draw.m} of {draw.n_groups} groups drawn, {len(draw.ids)} texts to code")
human = table.frame.set_index("text_id")["gold_constructiveness"].loc[draw.ids]
data = cop.CorrectionData.from_scores(
    text_scores,
    table,
    table.frame,
    gold_column="gold_constructiveness",
    score_columns=["construct_score"],
)
data = cop.corrections.attach_draw(data, human, draw)
estimate = cop.mean_ppi(data)
print(round(text_scores["construct_score"].mean(), 1), round(estimate.point, 1))
print(round(estimate.ci_low, 1), round(estimate.ci_high, 1))
print(round(table.frame["gold_constructiveness"].mean(), 1))
```

<!-- output -->
```text
6 of 18 groups drawn, 18 texts to code
1.7 2.8
2.2 3.5
3.0
```

The draw takes 6 of the 18 groups, whose 18 texts the human codes. The mean of the raw machine scores over all 54 texts is 1.7. The corrected estimate of the mean is 2.8, with a 95 percent interval from 2.2 to 3.5. The true mean of the gold values over all 54 texts is 3.0, inside the interval. The corrected estimate uses the machine scores of all texts and the human values of the 18 drawn texts, and it moves the raw mean toward the human scale. [Correcting estimates](tutorials/correcting-estimates.md) explains how.

## Where to go next

- Replace the fake backend with a real model: see [The two-stage coder](tutorials/two-stage-coder.md).
- Write your own rubric: see [The rubric](tutorials/rubric.md).
- Decide how many texts a human must code: see [Sample size](tutorials/sample-size.md).
- Run any step from the shell: every tutorial names the matching `copernicus` command.
