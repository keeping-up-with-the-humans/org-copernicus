# Hybrid coder

The hybrid coder joins the two ideas of the library. It takes the passages that the extraction stage of the [two-stage LLM coder](two-stage-llm-coder.md) returned and verified against the text, rebuilds each text from those passages, and scores this evidence text with the [similarity coder](similarity-coder.md). The model calls are made once, in the two-stage run; every later change of encoder, reference set or similarity setting costs only a pass of the encoder.

The evidence text keeps what the model found relevant to the rubric and drops the rest, so the embedding of a long text is no longer dominated by passages that say nothing about the construct. At construct granularity all passages of a text form one evidence text, compared with the poles of the construct and of each attribute. At attribute granularity the passages of each attribute's markers form their own evidence text, compared only with that attribute's poles, and the attribute scores are rolled up into one construct score. Sentence embeddings (Reimers and Gurevych 2019) carry the scoring, as for the similarity coder.

## The algorithm as the library implements it

1. **Read the run.** The coder reads the evidence table of one run (`run_number`, 1 by default) from a two-stage run store. Before it reads the store, it refuses reference embeddings written for another rubric or for another content of this one, and references that lack a pole the granularity scores. It then refuses a store coded with another rubric, a one-stage store, which has no passages, a store without a result in that run for some text of the table, a text of the table that differs from the text the store coded under its id (the text hash of every result), a text that carries several model labels in the run, and a text with at least `min_words` words whose extraction failed for an attribute in the run. Such a text lacks the passages of that attribute, and without any passage it would pass for a text without evidence; recode it with `CodingSettings(retry_failed=True)` or leave it out of the table. A failed scoring leaves the passages in place, and a short text keeps its own text whatever its passages, so neither is refused. These checks come before anything is embedded.
2. **Build the evidence texts.** In table order, a text with fewer than `min_words` (40) words keeps its own text, as the table holds it (as given, with only its ends stripped), with the fallback `short`. Otherwise the coder takes the passages of the run that were verified and are not repeats within their marker, orders them by their position in the text (start, then end, attribute, marker, passage index), keeps the first of passages that are equal after normalisation, and joins them with line breaks. A text left without a passage keeps its own text, with the fallback `no_evidence`. At attribute granularity only the passages of the markers of one attribute count, and each attribute gets its own evidence texts.
3. **Embed.** The encoder embeds the evidence texts; at attribute granularity a string that several attributes share, such as the whole text of a short text, is embedded once. When the encoder refuses an evidence text (step 1 of the [similarity coder](similarity-coder.md)), its message names the text, at attribute granularity as `<text_id> (<attribute_id>)` after the first text and attribute that use the string.
4. **Score.** At construct granularity the evidence texts are scored like any texts by the similarity coder, under the settings given. At attribute granularity each attribute's evidence texts are scored against that attribute's poles only, and the attribute scores are rolled up into `<rubric_id>__rollup` with the reduction of the settings.

The default settings, `HYBRID_DEFAULTS`, take from the paper's analysis code how the hybrid coder pools and embeds: each pole reduced once over the references of all writers (`pooled`), and the vector of the whole evidence text (`pooled_embedding`). The paper varied the polarity and the reduction as for the similarity coder; the defaults take net polarity with the `pca` reduction, one of those settings, at construct level. The similarity coder keeps its own defaults, which reproduce the paper's similarity configuration (per-writer reduction pooled by the mean, the maximum over the windows of a text). The result is the similarity coder's table plus the columns `method` (`hybrid_construct` or `hybrid_attribute`) and `fallback`, with a summary of the evidence texts and the provenance of the run. In the summary, `mean_words_evidence` (at attribute granularity one for each attribute, under `by_attribute`) is the mean number of words of the evidence texts that were built from passages; when no text was built from passages it is NaN (test it with `math.isnan`), which `copernicus hybrid score` writes as `null` in its description file and, at construct granularity, prints as `none`.

## Assumptions and what it corrects

The coder produces scores; it corrects nothing. It assumes that the verified passages carry the construct, so the score is only as good as the extraction stage. The texts that fall back to their own text are scored as whole texts within the same ranking as the evidence texts; read the `fallback` column before you interpret a low score, and set `min_words` to the length below which your texts carry too little to rebuild. The assumptions of the similarity coder hold as well: one load of one encoder for evidence texts and references (the same encoder, model commit and major version of transformers, where both sides record them, checked before the embedding pass for the library's own encoder and after it for an encoder object of one's own; a smaller difference of the versions, or a side without recorded versions, is scored with a warning and recorded in the provenance under `load_comparison`), and `pca` scores that depend on the texts scored together.

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

With the default of 40 words, 25 of the 54 toy texts keep their own text: 19 have fewer than 40 words and 6 have no verified passage. With 5 words, 46 texts are rebuilt from their passages and 8 have no verified passage. Scoring needs an encoder and a reference set; this is a snippet to read, not to run, and [The hybrid coder](../tutorials/hybrid-coder.md) runs it with MiniLM:

```python
result = cop.score_hybrid("run.sqlite", table, rubric, encoder, refs, min_words=5)
```

## Where the library departs from the paper's analysis code

- The library embeds the evidence texts with the encoder's own data type, remote-code setting and padding side; the paper's analysis code ignored these three settings on this path.
- The library refuses a text with at least `min_words` words whose extraction failed for an attribute in the run. The paper's analysis code read such a text like one without verified passages of that attribute: it embedded the passages of the other attributes, or its own text with the fallback `no_evidence` when no passage remained.
- A text that falls back to its own text is embedded as the coding table holds it, the text the coders read. The paper's analysis code embedded the stored text from before the markup cleaning of its coders, so the two differ only when the table holds cleaned text; a table built from the uncleaned texts gives the paper's fallback text.
- In this point the library's code agrees with the paper's analysis code: at attribute granularity a string that several attributes share is embedded once and has one vector.
- The `pca` reduction is computed by an exact decomposition, as on the [similarity coder's page](similarity-coder.md#where-the-library-departs-from-the-papers-analysis-code), and evidence texts whose rows in a reduction matrix are equal byte for byte get one score; the paper's analysis code let scikit-learn choose, which for larger matrices took a randomized decomposition without a seed, and took the scores as the decomposition returned them, so there the scores of such evidence texts can differ in their last bits and a rank correlation can treat them as untied. A roll-up reduces the attribute scores as returned. Giving equal rows one score can therefore change every score of a later roll-up, including scores of texts whose rows are not shared. Rounding changes in that input can cause much larger score and rank changes when leading components are nearly tied or the sign anchor is near zero. If its input columns become constant, the roll-up is refused even when numerical residue previously allowed a result.
- The library refuses evidence texts and references whose recorded transformers versions belong to two epochs (the number before `!`, as in `1!4.57.6`) or to two major versions, and a recorded version that cannot be read; for its own encoder it refuses them before the embedding pass. It scores other readable differences of the versions, and a side that records no version, with a warning. The scoring provenance shows the versions compared, or that a comparison was not possible. The paper's analysis code embedded the evidence texts in a pass of its own, recorded no version of torch or transformers, and scored them against the reference vectors of an earlier pass, found by their file name, comparing only the dimension of the vectors.

## Literature

Reimers and Gurevych (2019). The full entry is in [References](references.md).
