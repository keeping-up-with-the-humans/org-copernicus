# Similarity coder

The similarity coder scores a text without asking a language model about it. An encoder turns the text into an embedding, a vector whose direction reflects what the text says (Reimers and Gurevych 2019), and the coder measures how close that vector lies to the embeddings of reference texts. The reference texts express the high end (the positive pole) and the low end (the negative pole) of the construct and of each of its attributes. The net score of a text is its similarity to the positive pole minus its similarity to the negative pole, where the similarity is the cosine of the angle between two unit vectors.

Writer models produce the reference texts from the rubric, at three intensities per pole and in one of two styles: neutral, or matched to the register of the texts being scored. Once the texts and the references are embedded, scoring costs no model call, so a corpus can be scored under many settings at once. The score has no unit. It ranks texts, and the [validity panel](validity-panel.md) and the corrections put it in relation to the human values.

## The algorithm as the library implements it

1. **Embedding.** The encoder splits each text into consecutive windows of its `max_tokens` minus 2 tokens, the 2 being reserved for the special tokens of the tokenizer. Each window is pooled into one vector (`mean` over the attended tokens, `cls` for the first attended token, or `last_token` for the last) and the chunk vector is scaled to unit length. A text's vector is the mean of its raw chunk vectors weighted by their token counts, scaled to unit length once; with mean pooling this is the mean over all tokens of the text. The encoder stops with an error that counts the texts concerned and names the first ten in three cases. It stops when the tokenizer turns a window into no token, before the model runs on that window's batch: there is then no token vector to average, and no first or last token to take. (The tokenizers of `minilm`, `mpnet` and `bge_m3` add their special tokens to every window, so a window of theirs always has a token.) It stops when the pooled vector of a window is shorter than `1e-12`, and when the vector of a text is: a vector without length has no direction to keep. It stops as well when the computed length of such a vector is not finite: the vector holds a value that is NaN or infinite, or the sum of its squares overflows. Look at the texts that the message names and at what the tokenizer makes of them: leave them out, or change them so that every window has a token; a text whose vector has no length can be left out or embedded with another encoder. The registry holds `minilm` (mean pooling, 256 tokens), `mpnet` (mean, 384; the default encoder), `bge_m3` (cls, 8192) and `gte_qwen2_7b` (last token, 8192, bfloat16).
2. **Reference cells.** For the construct and for each attribute, for each pole and for each of three intensities, one cell: `top`, `mid` and `low` are anchored to the rubric levels 5, 4 and 3 for the positive pole and 1, 2 and 3 for the negative pole, and the writer is guided by the level descriptions of the markers at the anchor level. A call asks for `texts_per_call` (10) texts at temperature 0.9. A leakage screen flags every text that names the concept, an attribute or a marker of the rubric (case folded, whole words, terms of four characters or more) and leaves it out by default. `copernicus.references.screen_report` counts the texts that a screen flagged, in all and by term, and gives their share of the texts of the table, which is NaN for an empty table: its counts are then 0, and nothing has a share. A reference set holds the texts of one generation, in one style; a register-matched set is written for the one register that `register_description` names, so for texts of several registers generate one set per register and score each kind of text against its own set.
3. **Cosines.** Every text is compared with every reference of a pole. The long-text setting decides how a text with several windows enters: `pooled_embedding` uses the text's vector, `first_window` its first window, and `mean_chunk` and `max_chunk` the mean or the maximum over its windows of the window-by-reference cosines.
4. **Reduction.** The cosines of a text to the references of a pole become one number: their `mean`, their `max`, or `pca`, the first principal component of the standardised cosine columns across the texts scored together, with its sign set so that the scores of the decomposition do not correlate negatively with the row means. When the computed standard deviation of the row means is 0, no correlation is consulted and the scores keep the sign that the decomposition gave them; a column of cosines that does not vary is centred and left unscaled, so it adds nothing to the component beyond rounding. The reduction logs a warning where it treats the sign or the direction as open. For the sign it logs one warning when the computed standard deviation of the row means is 0; otherwise, when that of the scores is positive, it logs one when the computed correlation that sets the sign is at most `1e-12` in absolute value (as for two columns of unit variance that correlate negatively, whose component is their difference) and one when the range of the row means is at most `1e-12` times the larger of 1 and their largest absolute value. For the direction it logs one warning, whatever the row means, when the two largest singular values of the centred standardised matrix differ by at most a relative `1e-12`; it computes them for this check with `numpy.linalg.svd` in 64-bit floats, apart from the decomposition that gives the scores, whose own spectrum can differ from them in the last digits, and by much more where the covariance computation underflows; should this second decomposition not converge, the reduction stops with an error. These thresholds are conventions on computed numbers: a value below one can be genuine, and the scores that the reduction returns are as computed. The component is computed exactly, from the covariance matrix when the matrix has at most 1,000 columns and at least ten rows per column (with scikit-learn 1.5 or later) and by the full singular value decomposition otherwise, so two runs give the same scores; the paper's analysis code let scikit-learn choose, which in the version the paper used took a randomized decomposition without a seed for a matrix with more than 500 rows or columns, unless it had at most 1,000 columns and at least ten rows per column. In each matrix passed to the reduction, texts whose rows of cosines are equal byte for byte get one `pca` score: the score of the first of them; the sign is set before this is done. A pole with a single reference returns that cosine.
5. **Writer pooling.** With `mean` or `max`, the references of each writer model are reduced on their own and the per-writer scores are pooled by that function. With `pooled`, all references of the pole are reduced at once.
6. **Polarity.** `net` is the positive pole minus the negative pole; `positive` and `negative` keep one pole.
7. **Target level.** At `construct` level each text is scored against the poles of the construct, and each attribute against its own poles. At `attribute` level the attribute scores are rolled up into one construct score, the row `<rubric_id>__rollup`, with the same reduction.

The defaults of `SimilaritySettings` (`net`, `pca`, writer pooling `mean`, `max_chunk`, `construct`) are the paper's configuration of the similarity coder. `PAPER_GRID` holds the paper's 144 settings: 3 polarities, 3 reductions, 4 long-text modes, 2 writer poolings and 2 target levels; run over 4 encoders and 2 reference styles, they give the paper's 576 cells at each target level.

## Assumptions and what it corrects

The coder produces scores; it corrects nothing. It assumes that the reference texts express the poles of the construct, and that closeness in the embedding space follows the construct rather than the topic, the length or the style of a text. Texts and references must come from one load of one encoder. Every function that receives a text embedding set and reference embeddings (`score_similarity`, `score_similarity_many`, `run_grid`, the pole scorer `PoleScorer`, `scoring_provenance` and the hybrid coder) refuses a mismatch of the encoder's specification, two recorded commits, a recorded version that cannot be read, or recorded transformers versions of two major versions or two epochs; it accepts two releases of one major version of transformers, two versions of torch, or a side that records no version, which scoring logs as a warning and the provenance of the scores records under `load_comparison`, with the versions compared or the note that a comparison was not possible. The scoring functions also refuse a mismatch of the encoder's name; the arithmetic kernel `per_reference_similarity`, which receives the reference vectors as an array, checks nothing. A `pca` score depends on the texts scored together, because the standardisation and the component are computed over them; `mean` and `max` scores do not.

The reference texts express the rubric as it read when they were written. A reference set therefore records the content hash of its rubric (`rubric_hash`), its files keep it, and the reference embeddings carry it from the set. The files name their content too: the description of a set file records the digest of its table (`table_hash`) and the sidecar of the reference embeddings the digest of their arrays (`arrays_hash`), so a load refuses new texts or vectors beside the description of an earlier save, as a save stopped between its two writes leaves, instead of reading them under the old hash. Every scoring function that receives the rubric, the grid of settings and the hybrid coder among them, refuses references whose hash differs from the hash of that rubric, also after an edit that kept every id; the scorer object `PoleScorer` receives only the rubric's id and does not make this check. A set written by hand is bound to a rubric with `ReferenceSet.for_rubric`. The set records how it came by its hash (`bound_with`): `generated` when the generator wrote the texts for the rubric, `for_rubric` when they were bound to it by that explicit act. A set file of an earlier version, which records no hash, is refused unless `ReferenceSet.load(path, bind_to=rubric)` names the rubric to bind it to, the same explicit act; a recorded hash is never overridden. The description written beside a score table records the id and the hash of the rubric and, under `references`, the hash the references were written for and their `bound_with`; the two hashes agree, since scoring refuses references of another hash.

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
| `EncoderSpec.transformers_below` | `None`; `"5"` for `gte_qwen2_7b` | The first version of transformers with which the encoder's own model code does not load; under that version or a later one the encoder stops before anything is downloaded and names the release line to install |

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

- The library emits the roll-up row for every polarity setting; the paper's analysis code rolled up only the net scores. Both reduce the attribute net scores to a construct score, subject to the differences in the `pca` reduction described below.
- The similarity command offers only the reductions that exist; the paper's analysis code still listed reductions that it had removed.
- The library's reference generator screens every text for the rubric's own words and leaves flagged texts out by default; the paper's analysis code had no such screen.
- The `pca` reduction computes the first principal component by an exact decomposition that the library chooses, so two runs give the same scores; the paper's analysis code let scikit-learn choose (step 4), and the scores of a matrix that took its randomized decomposition without a seed vary from run to run. The library also gives texts whose rows of a reduction matrix are equal byte for byte one score, in the reduction of a pole and in the roll-up, which reduces the attribute scores as returned; the paper's analysis code took the scores as the decomposition returned them, so there the scores of such texts can differ in their last bits and a rank correlation can treat them as untied. A roll-up reduces the attribute scores as returned. Giving equal rows one score can therefore change every score of a later roll-up, including scores of texts whose rows are not shared. Rounding changes in that input can cause much larger score and rank changes when leading components are nearly tied or the sign anchor is near zero. If its input columns become constant, the roll-up is refused even when numerical residue previously allowed a result.
- Under `cls` pooling the library takes the first attended token of each window, on either padding side. The paper's analysis code took position 0 of the batch, which is the same token in a batch padded on the right and a padding position for the shorter windows of a batch padded on the left. In a batch padded on the right, position 0 is the first attended token of every window that has one, and the two agree there.
- The encoder refuses a text one of whose windows the tokenizer turns into no token, and a window whose pooled vector is shorter than `1e-12`. The paper's analysis code returned embeddings in both cases when the text had a window with a token and the vector of the text passed the code's check of unit length; under `cls` and last-token pooling it failed with an `IndexError` when all the windows of a batch were empty. A window without a token got the weight 0, so a finite vector of that window added nothing to the weighted sum behind its text's vector; the vector stored for the window was zero under mean pooling when the model's outputs for it were finite and, under `cls` and last-token pooling, the model's output at a padding position divided by its computed length, a length below `1e-12` being replaced by 1, and in a batch with left padding the last-token pooling then read the batch's other windows at their token count minus 1 in place of their last position. A window vector shorter than `1e-12` was left as it was. Both stop for a text without any token and for a text vector without length or with a length that is not finite, the paper's analysis code with a failed assertion.
- The registered encoder `gte_qwen2_7b`, the only one that runs the model repository's own code, is pinned to the commit `a8d08b36ada9cacfe34c4d6f80957772a025daf2`, the head of the repository's main branch, unchanged since 2025-03-24, so that loads of the default branch after that date reached the same commit. The paper's analysis code recorded the commit it loaded in a file beside the embeddings at run time instead of pinning it in code. The library refuses an encoder specification that allows remote code without a pinned commit, the full 40-character hash; a branch, a tag or a shortened hash does not pin the code that runs. The pin covers the code in the model's own repository: a model configuration whose `auto_map` names a class in a second repository (`repo--module.Class`) loads that repository's default branch, which the pin does not cover, so review such a configuration before allowing remote code.
- The library checks the loads of the encoder. An encoder whose specification names a bound, `transformers_below`, stops before anything is downloaded when the installed transformers is not below that bound, and names the release line below it to install. Among the registered encoders only `gte_qwen2_7b` has a bound, 5, because the model code at its pinned commit reads a configuration attribute, `rope_theta`, that transformers 5 no longer provides on the configuration object; under transformers 5 or later it stops with this reason and names `pip install "transformers>=4.56,<5"`. The library refuses texts and references whose recorded transformers versions belong to two epochs (the number before `!`, as in `1!4.57.6`) or to two major versions, and a recorded version that cannot be read. It scores other readable differences of the versions, and a side that records no version, with a warning. The scoring provenance shows the versions compared, or that a comparison was not possible. The paper's analysis code loaded its encoders without checking the installed transformers, recorded no version of torch or transformers beside its embeddings, and paired the files of the texts and of the references by their names, comparing only the dimension of their vectors.

## Literature

Reimers and Gurevych (2019). The full entry is in [References](references.md).
