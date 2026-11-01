# copernicus

copernicus codes constructs from text against a rubric. A rubric names a construct, splits it into attributes, and gives each attribute markers that a reader can look for in a text. The library scores texts against such a rubric with LLM coders, with similarity coders on embeddings, and with hybrid coders that combine the two. It then checks the scores against human gold values with a validity panel, compares them with an NLP-ML baseline, and corrects the bias that remains when the scores are used in a regression or a mean. It reproduces the configurations of the accompanying methods paper (under review) and ships the prompts of that paper.

## What the library does

- **Detection.** The two-stage and one-stage LLM coders read each text against the rubric, record the passages that support each marker, and write scores into a resumable run store. The similarity coder and the hybrid coder score texts from embeddings.
- **Validation.** The validity panel compares scores with gold values: rank validity, error, calibration, noise, and a share of the ceiling that the gold's reliability allows, each with a group-clustered bootstrap interval. The NLP-ML baseline fits a ridge regression on embeddings as the reference point.
- **Correction.** Given a human-coded sample, the correction functions turn machine scores into estimates of a mean, a group difference or a regression coefficient with intervals, plan the size of the human sample, and compare the corrections out of sample.

## Installation

Install the package from its root folder, the one that holds `pyproject.toml`. The name on the package index is `org-copernicus`; the import name and the command are `copernicus`. `pip install org-copernicus` installs it once the first release is published; until then it is installed from its root folder or from a wheel built there:

```
pip install .
```

The base install needs Python 3.10 or later and runs everything except the embedding encoders. Extras add the rest:

| Extra | Adds | Needed for |
|-------|------|------------|
| `embed` | `torch`, `transformers`, `safetensors` | embedding texts and references, the similarity and hybrid coders, the baseline's encoders |
| `vllm` | `vllm` | running an open-weight model locally as the coder's backend |
| `docs` | `mkdocs-material`, `mkdocstrings` | building the documentation site, on Python 3.11 or later (on 3.10 the extra installs nothing) |
| `dev` | `pytest`, `ruff`, `mypy`, `hypothesis`, `build` and others | running the tests and the checks |

Install an extra in the usual way, for example `pip install ".[embed]"`. A wheel built with `python -m build` installs from its path, as in `pip install "dist/org_copernicus-<version>-py3-none-any.whl[embed]"`.

## Quick start

The example below needs no GPU and no API key. It codes the toy corpus that ships with the package, with the fake backend that answers from a table of cue words, and checks the scores against the gold values that the toy corpus carries in its `gold_*` columns.

```python
from pathlib import Path
import copernicus as cop

rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
store = Path("run.sqlite")
cop.code_texts(table, rubric, backend, store, cop.CodingSettings(runs=1))
panel = cop.validity_panel(
    cop.export_scores(store), table.frame, table, targets={"construct": "gold_constructiveness"}
)
print(panel.report())
```

The last line prints the panel: the number of texts that entered, one row for the construct with its rank validity and interval, and a sentence on what was scored. Run it again and the coder finds every cell in `run.sqlite` and makes no call. For a real study, replace `FakeBackend` with `OpenAICompatibleBackend` or `VLLMBackend` and the toy rubric with your own.

## Module map

| Subpackage | Purpose | Command group |
|------------|---------|---------------|
| `copernicus.rubric` | Rubric model, JSON schema and loader | `rubric` |
| `copernicus.data` | Validated text tables from CSV, parquet and JSON lines | |
| `copernicus.llm` | Backends for OpenAI-compatible servers and vLLM, a fake backend, JSON answer parsing | |
| `copernicus.coders` | Two-stage and one-stage LLM coders, the run store, exports | `code`, `store` |
| `copernicus.embeddings` | Encoders, embedding sets, the encoder registry | `embed` |
| `copernicus.references` | Generated reference texts for the poles of each construct and attribute | `references` |
| `copernicus.similarity` | The similarity coder and its grid of settings | `similarity` |
| `copernicus.hybrid` | The hybrid coder: similarity scores on the verified passages of a two-stage run | `hybrid` |
| `copernicus.validity` | The validity panel, score errors, reliability ceilings, the cluster bootstrap | `validity` |
| `copernicus.baseline` | The NLP-ML baseline: ridge regression on embeddings, learning curves | `baseline` |
| `copernicus.corrections` | Human sampling, pseudo-values, prediction-powered estimates, and the other corrections, plus the battery that compares them | `corrections` |
| `copernicus.assets` | The toy rubric, corpus and cue words that ship with the package | |

## Command line

Every step of the library is also a command. Run `copernicus --help` for the list and `copernicus <group> --help` for the commands of a group.

- `copernicus rubric`: check a rubric file against the schema.
- `copernicus code`: code texts with the two-stage or the one-stage coder into a run store.
- `copernicus store`: merge the shard stores of one run and export a table of a store.
- `copernicus references`: generate reference texts for the poles of a rubric.
- `copernicus embed`: embed texts and reference texts with an encoder.
- `copernicus similarity`: score embedded texts with the similarity coder.
- `copernicus hybrid`: score the verified passages of a two-stage run with the hybrid coder.
- `copernicus validity`: compute the validity panel of scores against gold values.
- `copernicus baseline`: fit the ridge baseline over seeds, training sizes and encoders.
- `copernicus corrections`: estimate with a correction, draw and plan the human sample, run the battery, list the families.

## Documentation

The documentation site is built from the `docs/` folder with MkDocs. Install the `docs` extra and run `mkdocs serve` from the package's root folder to read it in a browser.

## How to cite

Please cite the accompanying methods paper (under review) and the software:

> Author names withheld. copernicus: rubric-based construct coding with LLM methods. Version 0.1.0. Software accompanying a methods paper under review.

`CITATION.cff` holds the same entry in a form that reference managers read.

The source is at <https://github.com/keeping-up-with-the-humans/org-copernicus> and the documentation at <https://keeping-up-with-the-humans.github.io/org-copernicus/>.

## License

MIT. See `LICENSE`.
