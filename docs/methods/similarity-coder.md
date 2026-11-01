# Similarity coder

The similarity coder scores a text without asking a language model about it. An encoder turns the text into an embedding, a vector whose direction reflects what the text says (Reimers and Gurevych 2019), and the coder measures how close that vector lies to the embeddings of reference texts. The reference texts express the high end (the positive pole) and the low end (the negative pole) of the construct and of each of its attributes. The net score of a text is its similarity to the positive pole minus its similarity to the negative pole, where the similarity is the cosine of the angle between two unit vectors.

Writer models produce the reference texts from the rubric, at three intensities per pole and in one of two styles: neutral, or matched to the register of the texts being scored. Once the texts and the references are embedded, scoring costs no model call, so a corpus can be scored under many settings at once. The score has no unit. It ranks texts, and the [validity panel](validity-panel.md) and the corrections put it in relation to the human values.

## The algorithm as the library implements it

1. **Embedding.** The encoder splits each text into consecutive windows of its `max_tokens` minus 2 tokens, the 2 being reserved for the special tokens of the tokenizer. Each window is pooled into one vector (`mean` over the attended tokens, the `cls` token, or the `last_token`) and the chunk vector is scaled to unit length. A text's vector is the mean of its raw chunk vectors weighted by their token counts, scaled to unit length once; with mean pooling this is the mean over all tokens of the text. The registry holds `minilm` (mean pooling, 256 tokens), `mpnet` (mean, 384; the default encoder), `bge_m3` (cls, 8192) and `gte_qwen2_7b` (last token, 8192, bfloat16).
2. **Reference cells.** For the construct and for each attribute, for each pole and for each of three intensities, one cell: `top`, `mid` and `low` are anchored to the rubric levels 5, 4 and 3 for the positive pole and 1, 2 and 3 for the negative pole, and the writer is guided by the level descriptions of the markers at the anchor level. A call asks for `texts_per_call` (10) texts at temperature 0.9. A leakage screen flags every text that names the concept, an attribute or a marker of the rubric (case folded, whole words, terms of four characters or more) and leaves it out by default.
3. **Cosines.** Every text is compared with every reference of a pole. The long-text setting decides how a text with several windows enters: `pooled_embedding` uses the text's vector, `first_window` its first window, and `mean_chunk` and `max_chunk` the mean or the maximum over its windows of the window-by-reference cosines.
4. **Reduction.** The cosines of a text to the references of a pole become one number: their `mean`, their `max`, or `pca`, the first principal component of the standardised cosine columns across the texts scored together, with its sign set so that it correlates positively with the row means. A pole with a single reference returns that cosine.
5. **Writer pooling.** With `mean` or `max`, the references of each writer model are reduced on their own and the per-writer scores are pooled by that function. With `pooled`, all references of the pole are reduced at once.
6. **Polarity.** `net` is the positive pole minus the negative pole; `positive` and `negative` keep one pole.
7. **Target level.** At `construct` level each text is scored against the poles of the construct, and each attribute against its own poles. At `attribute` level the attribute scores are rolled up into one construct score, the row `<rubric_id>__rollup`, with the same reduction.

The defaults of `SimilaritySettings` (`net`, `pca`, writer pooling `mean`, `max_chunk`, `construct`) are the paper's configuration of the similarity coder. `PAPER_GRID` holds the paper's 144 settings: 3 polarities, 3 reductions, 4 long-text modes, 2 writer poolings and 2 target levels; run over 4 encoders and 2 reference styles, they give the paper's 576 cells at each target level.

## Assumptions and what it corrects

The coder produces scores; it corrects nothing. It assumes that the reference texts express the poles of the construct, and that closeness in the embedding space follows the construct rather than the topic, the length or the style of a text. Texts and references must come from the same encoder and the same model commit; the scorer refuses a mismatch of encoder name, specification or recorded commit. A `pca` score depends on the texts scored together, because the standardisation and the component are computed over them; `mean` and `max` scores do not.

The reference texts express the rubric as it read when they were written. A reference set therefore records the content hash of its rubric (`rubric_hash`), its files keep it, and the reference embeddings carry it from the set. The files name their content too: the description of a set file records the digest of its table (`table_hash`) and the sidecar of the reference embeddings the digest of their arrays (`arrays_hash`), so a load refuses new texts or vectors beside the description of an earlier save, as a save stopped between its two writes leaves, instead of reading them under the old hash. Every scoring path, the grid of settings and the hybrid coder included, refuses references whose hash differs from the hash of the rubric it is given, also after an edit that kept every id. A set written by hand is bound to a rubric with `ReferenceSet.for_rubric`. The set records how it came by its hash (`bound_with`): `generated` when the generator wrote the texts for the rubric, `for_rubric` when they were bound to it by that explicit act. A set file of an earlier version, which records no hash, is refused unless `ReferenceSet.load(path, bind_to=rubric)` names the rubric to bind it to, the same explicit act; a recorded hash is never overridden. The description written beside a score table records the id and the hash of the rubric and, under `references`, the hash the references were written for and their `bound_with`; the two hashes agree, since scoring refuses references of another hash.

## Parameters

| Parameter | Values (default first) | Meaning |
|-----------|------------------------|---------|
| `SimilaritySettings.polarity` | `net`, `positive`, `negative` | Both poles, or one |
| `SimilaritySettings.reduction` | `pca`, `mean`, `max` | How the cosines to a pole become one number |
| `SimilaritySettings.writer_pooling` | `mean`, `max`, `pooled` | Pool per-writer scores, or reduce all references at once |
| `SimilaritySettings.long_text` | `max_chunk`, `mean_chunk`, `pooled_embedding`, `first_window` | How the windows of a long text enter |
| `SimilaritySettings.target_level` | `construct`, `attribute` | Score the construct directly, or roll the attributes up |
| `GeneratorSettings.style` | `neutral`, `register` | Leave the medium open, or match the register given in `register_description` |
| `GeneratorSettings.texts_per_call`, `calls_per_cell`, `temperature` | 10, 1, 0.9 | Texts asked for per call, calls per writer and cell, sampling temperature |
| `Encoder(spec)` | `mpnet` | A registered encoder, a Hugging Face id, or an `EncoderSpec` |
| `EncoderSpec.revision` | `None`, the default branch | The commit, branch or tag to load; pin a commit for a study you want to rerun. An encoder that runs the repository's own code needs the full 40-character commit hash |

To pin a commit, replace the revision of a registered specification and pass the result to `Encoder`; building the encoder loads nothing. On the command line, `--revision` does the same for `copernicus embed texts`, `copernicus embed references` and `copernicus hybrid score`.

<!-- run -->
```python
import dataclasses

import copernicus as cop

commit = "0123456789abcdef0123456789abcdef01234567"  # stands in for the commit you reviewed
pinned = dataclasses.replace(cop.ENCODERS["mpnet"], revision=commit)
encoder = cop.Encoder(pinned, device="cpu")
print(encoder.spec.hf_id, encoder.spec.revision == commit)
```

<!-- output -->
```text
sentence-transformers/all-mpnet-base-v2 True
```

## Example

The arithmetic of steps 4 and 6 on four texts and three references per pole, without an encoder. The three reductions give different numbers but, here, the same order of the texts.

<!-- run -->
```python
import numpy as np
from copernicus.similarity import reduce_matrix

positive = np.array(
    [[0.62, 0.58, 0.71], [0.31, 0.37, 0.28], [0.44, 0.52, 0.41], [0.50, 0.47, 0.56]]
)
negative = np.array(
    [[0.21, 0.24, 0.22], [0.57, 0.61, 0.48], [0.41, 0.38, 0.44], [0.32, 0.36, 0.33]]
)
for reduction in ("mean", "max", "pca"):
    positive_score = reduce_matrix(positive, reduction, "positive")
    negative_score = reduce_matrix(negative, reduction, "negative")
    print(reduction, np.round(positive_score - negative_score, 1).tolist())
```

<!-- output -->
```text
mean [0.4, -0.2, 0.0, 0.2]
max [0.5, -0.2, 0.1, 0.2]
pca [4.5, -4.8, -0.7, 0.9]
```

On real data the coder runs on embedding sets. This is a snippet to read, not to run; it needs the `embed` extra, a reference set and a downloaded encoder, as [The similarity coder](../tutorials/similarity-coder.md) shows step by step:

```python
encoder = cop.Encoder("minilm", device="cpu")
texts = encoder.embed(table.frame["text"].tolist(), table.frame["text_id"].tolist())
refs = cop.embed_references(refset, encoder)
scores = cop.score_similarity(texts, refs, rubric, cop.SimilaritySettings())
```

## Where the library departs from the paper's analysis code

- The library emits the roll-up row for every polarity setting; the paper's analysis code rolled up only the net scores. The net roll-up is the same in both.
- The similarity command offers only the reductions that exist; the paper's analysis code still listed reductions that it had removed.
- The library's reference generator screens every text for the rubric's own words and leaves flagged texts out by default; the paper's analysis code had no such screen.
- The registered encoder `gte_qwen2_7b`, the only one that runs the model repository's own code, is pinned to the commit `a8d08b36ada9cacfe34c4d6f80957772a025daf2`, the head of the repository's main branch, unchanged since 2025-03-24, so that loads of the default branch after that date reached the same commit. The paper's analysis code recorded the commit it loaded in a file beside the embeddings at run time instead of pinning it in code. The library refuses an encoder specification that allows remote code without a pinned commit, the full 40-character hash; a branch, a tag or a shortened hash does not pin the code that runs. The pin covers the code in the model's own repository: a model configuration whose `auto_map` names a class in a second repository (`repo--module.Class`) loads that repository's default branch, which the pin does not cover, so review such a configuration before allowing remote code.

## Literature

Reimers and Gurevych (2019). The full entry is in [References](references.md).
