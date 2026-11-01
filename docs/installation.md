# Installation

## Requirements

The library needs Python 3.10 or later. It needs no GPU and no API key for anything in this documentation.

## Install the package

The name on the package index is `org-copernicus`; the import name and the command are `copernicus`. `pip install org-copernicus` installs it once the first release is published; until then it is installed from its root folder, the one that holds `pyproject.toml`, or from a wheel built there:

```bash
pip install .
```

The distribution name `org-copernicus` differs from the import name `copernicus` in the way scikit-learn differs from sklearn. The index name `copernicus` belongs to a different project: if that project ever publishes a package with the import name `copernicus`, the two cannot be installed into one environment.

The base install runs everything except the embedding encoders. Extras add the rest:

| Extra | Adds | Needed for |
|-------|------|------------|
| `embed` | `torch`, `transformers`, `safetensors` | embedding texts and references, the similarity coder, the hybrid coder, the NLP-ML baseline |
| `vllm` | `vllm` | running an open-weight model on your own GPU as the coder's backend |
| `docs` | `mkdocs-material`, `mkdocstrings` | building this documentation site, on Python 3.11 or later (on 3.10 the extra installs nothing) |
| `dev` | `pytest`, `ruff`, `mypy`, `hypothesis` and others | running the tests and the checks |

Install an extra in the usual way:

```bash
pip install ".[embed]"
```

A wheel built with `python -m build` installs from its path, with or without extras: `pip install "dist/org_copernicus-<version>-py3-none-any.whl[embed]"`. Use `pip install -e ".[dev]"` to work on the library itself.

## Check the install

```bash
python -c "import copernicus; print(copernicus.__version__)"
copernicus --help
```

The second command lists the command groups: `rubric`, `code`, `store`, `references`, `embed`, `similarity`, `hybrid`, `validity`, `baseline` and `corrections`. Every step of the library is also a command. The tutorials show the Python calls; each of them notes the matching command.

## The first embedding run downloads a model

The tutorials that embed texts use the encoder `minilm` (`sentence-transformers/all-MiniLM-L6-v2`). Its weights are about 90 MB. The first call downloads them from the Hugging Face hub into the hub cache. Later calls read the cache.

To work offline after the first download, set the environment variable `HF_HUB_OFFLINE=1`. To keep the weights somewhere else, set `HF_HOME` or `HF_HUB_CACHE`.

The encoder runs on a CPU. Without a device argument it uses a GPU (CUDA, or Apple's MPS) when torch finds one, so pass `device="cpu"` to `Encoder` (or `--device cpu` to the commands) to stay on the CPU. The registry also holds `mpnet`, `bge_m3` and `gte_qwen2_7b`; the last one is a 7-billion-parameter model that needs a GPU in practice. `gte_qwen2_7b` is the only registered encoder that runs the model repository's own code; the registry pins it to one commit, and a specification that allows remote code must name the full 40-character commit hash as its revision.

## Optional: exact token counts

The token estimate of `copernicus references generate --dry-run` counts prompt tokens with the `o200k_base` encoding of `tiktoken` when `tiktoken` is installed. Without it, the estimate falls back to the number of characters divided by four. The estimate names the counter it used.

## Build this documentation

Building the documentation needs Python 3.11 or later; the library itself runs on 3.10.

```bash
pip install ".[docs]"
mkdocs serve
```

Run both commands in the package's root folder, which holds `mkdocs.yml`. `mkdocs build --strict` builds the site once and fails on any warning.
