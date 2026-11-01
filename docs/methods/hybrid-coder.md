# Hybrid coder

The hybrid coder joins the two ideas of the library. It takes the passages that the extraction stage of the [two-stage LLM coder](two-stage-llm-coder.md) returned and verified against the text, rebuilds each text from those passages, and scores this evidence text with the [similarity coder](similarity-coder.md). The model calls are made once, in the two-stage run; every later change of encoder, reference set or similarity setting costs only a pass of the encoder.

The evidence text keeps what the model found relevant to the rubric and drops the rest, so the embedding of a long text is no longer dominated by passages that say nothing about the construct. At construct granularity all passages of a text form one evidence text, compared with the poles of the construct and of each attribute. At attribute granularity the passages of each attribute's markers form their own evidence text, compared only with that attribute's poles, and the attribute scores are rolled up into one construct score. Sentence embeddings (Reimers and Gurevych 2019) carry the scoring, as for the similarity coder.

## The algorithm as the library implements it

1. **Read the run.** The coder reads the evidence table of one run (`run_number`, 1 by default) from a two-stage run store. Before it reads the store, it refuses reference embeddings written for another rubric or for another content of this one, and references that lack a pole the granularity scores. It then refuses a store coded with another rubric, a one-stage store, which has no passages, a store without a result in that run for some text of the table, a text of the table that differs from the text the store coded under its id (the text hash of every result), a text that carries several model labels in the run, and a text with at least `min_words` words whose extraction failed for an attribute in the run. Such a text lacks the passages of that attribute, and without any passage it would pass for a text without evidence; recode it with `CodingSettings(retry_failed=True)` or leave it out of the table. A failed scoring leaves the passages in place, and a short text keeps its raw text whatever its passages, so neither is refused. These checks come before anything is embedded.
2. **Build the evidence texts.** In table order, a text with fewer than `min_words` (40) words keeps its raw text, with the fallback `short`. Otherwise the coder takes the passages of the run that were verified and are not repeats within their marker, orders them by their position in the text (start, then end, attribute, marker, passage index), keeps the first of passages that are equal after normalisation, and joins them with line breaks. A text left without a passage keeps its raw text, with the fallback `no_evidence`. At attribute granularity only the passages of the markers of one attribute count, and each attribute gets its own evidence texts.
3. **Embed.** The encoder embeds the evidence texts; at attribute granularity a string shared by several attributes, such as a raw text, is embedded once.
4. **Score.** At construct granularity the evidence texts are scored like any texts by the similarity coder, under the settings given. At attribute granularity each attribute's evidence texts are scored against that attribute's poles only, and the attribute scores are rolled up into `<rubric_id>__rollup` with the reduction of the settings.

The default settings, `HYBRID_DEFAULTS`, are the paper's hybrid coder: net polarity, the `pca` reduction, each pole reduced once over the references of all writers (`pooled`), the vector of the whole evidence text (`pooled_embedding`), and construct level. The result is the similarity coder's table plus the columns `method` (`hybrid_construct` or `hybrid_attribute`) and `fallback`, with a summary of the evidence texts and the provenance of the run.

## Assumptions and what it corrects

The coder produces scores; it corrects nothing. It assumes that the verified passages carry the construct, so the score is only as good as the extraction stage. The texts that fall back to their raw text are scored as raw texts within the same ranking as the evidence texts; read the `fallback` column before you interpret a low score, and set `min_words` to the length below which your texts carry too little to rebuild. The assumptions of the similarity coder hold as well: the same encoder and model commit for evidence texts and references, and `pca` scores that depend on the texts scored together.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `settings` | `HYBRID_DEFAULTS` | The similarity settings of the scoring |
| `granularity` | `"construct"` | `"construct"`, or `"attribute"` for one evidence text per attribute |
| `run_number` | 1 | The run whose passages are read |
| `min_words` | 40 | Texts with fewer words are scored as they are |
| `checkpoint`, `checkpoint_every` | `None`, 2048 | Make the one embedding pass of a run resumable |

## Example

The evidence texts need no encoder. The block codes the toy corpus with the fake backend and counts how the texts are rebuilt, with the default `min_words` and with 5 words, which suits the short toy texts.

<!-- run -->
```python
import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
cop.code_texts(table, rubric, backend, "run.sqlite", cop.CodingSettings(runs=1))
for min_words in (40, 5):
    evidence = cop.hybrid.evidence_texts(
        "run.sqlite", rubric, table, granularity="construct", min_words=min_words
    )
    summary = evidence.summary
    print(min_words, summary["n_evidence"], summary["n_no_evidence"], summary["n_short"])
```

<!-- output -->
```text
40 29 6 19
5 46 8 0
```

With the default of 40 words, 25 of the 54 toy texts keep their raw text: 19 have fewer than 40 words and 6 have no verified passage. With 5 words, 46 texts are rebuilt from their passages and 8 have no verified passage. Scoring needs an encoder and a reference set; this is a snippet to read, not to run, and [The hybrid coder](../tutorials/hybrid-coder.md) runs it with MiniLM:

```python
result = cop.score_hybrid("run.sqlite", table, rubric, encoder, refs, min_words=5)
```

## Where the library departs from the paper's analysis code

- The default settings reproduce the paper's hybrid coder (each pole reduced once over the references of all writers, the vector of the whole evidence text); the similarity coder keeps its own defaults, which reproduce the paper's similarity configuration (per-writer reduction pooled by the mean, the maximum over the windows of a text).
- The library embeds the evidence texts with the encoder's own data type, remote-code setting and padding side; the paper's analysis code ignored these three settings on this path.
- The library refuses a text with at least `min_words` words whose extraction failed for an attribute in the run. The paper's analysis code read such a text like one without verified passages of that attribute: it embedded the passages of the other attributes, or the raw text with the fallback `no_evidence` when no passage remained.

## Literature

Reimers and Gurevych (2019). The full entry is in [References](references.md).
